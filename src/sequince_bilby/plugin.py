"""Plugin for the sequince sampler in bilby."""

import copy
from numbers import Integral
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from bilby.core.result import Result
from bilby.core.sampler.base_sampler import Sampler
from bilby.core.utils import random
from bilby.core.utils.log import logger
from orng import RandomGenerator
from sequince.array_api import determine_backend_name, to_numpy
from sequince.checkpoint import HDF5CheckpointWriter, SMCCheckpoint
from sequince.log import configure_logging
from sequince.prebuilt_samplers import (
    make_persistent_bayesian_smc,
    make_standard_bayesian_smc,
)
from sequince.proposals import draw_initial_state
from sequince.state import PersistentParticleState
from sequince.steps import StepContext
from sequince.utils.random import resolve_rng

from .proposal import BilbyPriorProposal
from .utils import get_sequince_functions


class SequinceSMC(Sampler):
    """Bilby wrapper for the sequential Monte Carlo sample from sequince.

    Sequince: https://github.com/sequince-dev/sequince

    """

    sampler_name = "sequince-smc"
    """
    Name of the sampler.
    """

    _prebuilt_fn = staticmethod(make_standard_bayesian_smc)
    _builder_accepts_n_particles = False

    @property
    def external_sampler_name(self) -> str:
        """The name of package that provides the sampler."""
        return "sequince"

    @property
    def default_kwargs(self) -> dict:
        """Dictionary of default keyword arguments."""
        return dict(
            n_particles=500,
            checkpoint=True,
            checkpoint_filename=None,
            checkpoint_frequency=1,
            resume=True,
            proposal=None,
            sampler_backend="numpy",
            rng=None,
            map_fn=None,
            jit=False,
        )

    @staticmethod
    def _resolve_sampling_rng(
        backend: str,
        rng: RandomGenerator | np.random.Generator | int | None = None,
    ) -> RandomGenerator:
        """Adapt a supplied RNG or seed, defaulting to Bilby's stream."""
        if backend not in ("numpy", "jax", "torch"):
            raise ValueError("sampler_backend must be numpy, jax, or torch.")
        if isinstance(rng, Integral) and not isinstance(rng, bool):
            return resolve_rng(int(rng), backend=backend)
        resolved = resolve_rng(random.rng if rng is None else rng)
        if resolved.backend == backend:
            return resolved
        if resolved.backend != "numpy":
            raise ValueError("rng backend differs from sampler_backend.")
        # A backend-native stream cannot wrap NumPy's generator. Derive one
        # seed from Bilby's stream so its usual seed() interface controls both.
        seed = int(resolved.state().integers(0, 2**32))
        return resolve_rng(seed, backend=backend)

    def _configure_checkpointing(self, kwargs: dict) -> Path:
        """Consume checkpoint options and add the configured writer."""
        checkpoint = kwargs.pop("checkpoint")
        checkpoint_filename = kwargs.pop("checkpoint_filename")
        checkpoint_frequency = kwargs.pop("checkpoint_frequency")

        if not isinstance(checkpoint, bool):
            raise TypeError("checkpoint must be a boolean.")
        if isinstance(checkpoint_frequency, bool) or not isinstance(
            checkpoint_frequency, Integral
        ):
            raise TypeError("checkpoint_frequency must be a positive integer.")
        if checkpoint_frequency <= 0:
            raise ValueError(
                "checkpoint_frequency must be a positive integer."
            )

        if checkpoint_filename is None:
            checkpoint_filename = self.get_checkpoint_file_name(
                self.outdir,
                self.label,
            )
        checkpoint_filename = Path(checkpoint_filename)

        if checkpoint:
            checkpoint_writers = list(
                kwargs.pop("checkpoint_writers", None) or ()
            )
            checkpoint_writers.append(
                HDF5CheckpointWriter(
                    checkpoint_filename,
                    every=int(checkpoint_frequency),
                )
            )
            kwargs["checkpoint_writers"] = checkpoint_writers

        return checkpoint_filename

    @staticmethod
    def _load_checkpoint(checkpoint_filename: Path) -> SMCCheckpoint | None:
        """Load a non-empty checkpoint, or return ``None`` if it is absent."""
        if (
            not checkpoint_filename.is_file()
            or checkpoint_filename.stat().st_size == 0
        ):
            logger.info(
                f"Checkpoint file {checkpoint_filename} does not exist."
            )
            return None

        logger.info(f"Reading checkpoint file {checkpoint_filename}")
        with h5py.File(checkpoint_filename, "r") as checkpoint_file:
            return SMCCheckpoint.load_hdf5(checkpoint_file)

    @staticmethod
    def _restore_target_identity(posterior, checkpoint: SMCCheckpoint) -> None:
        """Make the rebuilt target accept evaluations stored in a checkpoint."""
        evaluation = checkpoint.state.evaluation
        if evaluation is not None:
            object.__setattr__(
                posterior,
                "_evaluation_id",
                evaluation.target_id,
            )

    @staticmethod
    def _checkpoint_is_complete(checkpoint: SMCCheckpoint) -> bool:
        """Return whether a checkpoint already represents a completed run."""
        if isinstance(checkpoint.state, PersistentParticleState):
            return True
        max_temperature = checkpoint.metadata.get("max_temperature")
        temperature = checkpoint.metadata.get("temperature")
        return (
            max_temperature is not None
            and temperature is not None
            and float(temperature) >= float(max_temperature)
        )

    def run_sampler(self) -> Result:
        """Run the sampler and release owned workers, including on failure."""
        user_pool = hasattr(
            self.kwargs.get("pool", getattr(self, "pool", None)), "map"
        )
        try:
            return self._run_sampler()
        finally:
            if not user_pool:
                self._close_pool()

    def _run_sampler(self) -> Result:
        """Configure likelihood mapping and run or resume Sequince."""

        proposal = self.kwargs.get("proposal")
        kwargs = copy.deepcopy(
            {
                key: value
                for key, value in self.kwargs.items()
                if key not in ("proposal", "rng", "map_fn", "pool")
            }
        )
        sampling_rng = self._resolve_sampling_rng(
            kwargs.pop("sampler_backend"), self.kwargs.get("rng")
        )
        if sampling_rng.backend != "numpy":
            if not getattr(random, "BILBY_ARRAY_API", False):
                raise ValueError(
                    "Backend sampling requires array-aware Bilby with "
                    "BILBY_ARRAY_API=1 before import."
                )
            if self.pool is not None or self.npool not in (None, 1):
                raise ValueError(
                    "Backend sampling currently requires npool=1."
                )

        resume = kwargs.pop("resume")
        if not isinstance(resume, bool):
            raise TypeError("resume must be a boolean.")
        n_particles = kwargs.pop("n_particles")

        configure_logging(level=kwargs.pop("sequince_log_level", "INFO"))

        self._setup_pool()

        funcs = get_sequince_functions(
            map_fn=self.kwargs.get("map_fn"),
            backend=sampling_rng.backend,
            pool=self.pool,
            jit=kwargs.pop("jit"),
        )

        # Should handle these properly
        kwargs.pop("npool", None)
        kwargs.pop("pool", None)

        checkpoint_filename = self._configure_checkpointing(kwargs)

        if proposal is None:
            proposal = BilbyPriorProposal(
                self.priors,
                self.search_parameter_keys,
            )
        else:
            kwargs["log_proposal"] = proposal

        if self._builder_accepts_n_particles:
            kwargs["n_particles"] = n_particles

        logger.info(f"Creating sequince instance with kwargs: {kwargs}")

        sampler, posterior, metadata = self._prebuilt_fn(
            log_prior=funcs.log_prior,
            log_likelihood=funcs.log_likelihood,
            **kwargs,
        )

        checkpoint = (
            self._load_checkpoint(checkpoint_filename) if resume else None
        )
        if checkpoint is None:
            context = StepContext(t=0, metadata=metadata)
            initial_state = draw_initial_state(
                proposal,
                n_particles,
                posterior=posterior,
                context=context,
                rng=sampling_rng,
            )
            if (
                determine_backend_name(initial_state.particles)
                != sampling_rng.backend
            ):
                raise ValueError(
                    "Proposal particles and sampling RNG must use the "
                    "same backend."
                )
            result = sampler.run(
                initial_state,
                metadata=metadata,
                rng=sampling_rng,
            )
        else:
            if (
                determine_backend_name(checkpoint.state.particles)
                != sampling_rng.backend
            ):
                raise ValueError(
                    "Checkpoint backend differs from the sampling backend."
                )
            restored_rng = checkpoint.restore_rng()
            if restored_rng is not None:
                if restored_rng.backend != sampling_rng.backend:
                    raise ValueError(
                        "Checkpoint RNG backend differs from the "
                        "sampling backend."
                    )
                sampling_rng = restored_rng
            self._restore_target_identity(posterior, checkpoint)
            if self._checkpoint_is_complete(checkpoint):
                logger.info("Checkpoint contains a completed sampling run.")
                result = checkpoint.result()
            else:
                runtime = sampler.resume_from(checkpoint, rng=sampling_rng)
                result = sampler.run(runtime=runtime)
                sampling_rng = runtime.rng

        result_file = self.get_result_file_name(self.outdir, self.label)
        result.save_to_file(result_file)

        weighted_samples, weights = result.posterior_samples(method="weights")
        weighted_posterior = pd.DataFrame(
            to_numpy(weighted_samples),
            columns=self.search_parameter_keys,
        )
        weighted_posterior["weights"] = to_numpy(weights)
        self.result.nested_samples = weighted_posterior

        posterior_samples, _ = result.posterior_samples(
            method="rejection",
            rng=sampling_rng,
        )
        self.result.samples = to_numpy(posterior_samples)
        self.result.log_evidence = float(to_numpy(result.log_evidence()))
        self.result.log_evidence_err = float(
            to_numpy(result.log_evidence_error())
        )
        self.result.num_likelihood_evaluations = np.nan

        return self.result

    @classmethod
    def get_result_file_name(cls, outdir, label):
        sampler_name = cls.sampler_name.replace("-", "_")
        return Path(outdir) / f"{label}_{sampler_name}_result.h5"

    @classmethod
    def get_checkpoint_file_name(cls, outdir, label):
        sampler_name = cls.sampler_name.replace("-", "_")
        return Path(outdir) / f"{label}_{sampler_name}_checkpoint.h5"

    @classmethod
    def get_expected_outputs(cls, outdir=None, label=None):
        """Get lists of the expected outputs directories and files.

        These are used by :code:`bilby_pipe` when transferring files via
        HTCondor. Both can be empty.

        Parameters
        ----------
        outdir : str
            The output directory.
        label : str
            The label for the run.

        Returns
        -------
        list
            List of file names.
        list
            List of directory names.
        """
        filenames = [
            cls.get_result_file_name(outdir, label),
            cls.get_checkpoint_file_name(outdir, label),
        ]
        dirs = []
        return filenames, dirs

    def _verify_kwargs_against_default_kwargs(self):
        """Check for additional kwargs that are not included in the defaults.

        Since the arguments for sequince depend on the flow being used, arguments
        are not removed if they are not present in the defaults.
        """
        args = self.default_kwargs
        for user_input in self.kwargs.keys():
            if user_input not in args:
                logger.debug(
                    (
                        "Supplied argument '{user_input}' is not a default "
                        f"argument of '{self.__class__.__name__}'. "
                    )
                )

    def _translate_kwargs(self, kwargs):
        """Translate the keyword arguments"""
        aliases = dict.fromkeys(
            ("rng", "random_state", *self.sampling_seed_equiv_kwargs)
        )
        supplied = {key: kwargs.pop(key) for key in aliases if key in kwargs}
        values = [value for value in supplied.values() if value is not None]
        if len(values) > 1:
            raise ValueError(
                "Specify only one of rng, random_state, seed, "
                "sampling_seed, or random_seed."
            )
        if supplied:
            kwargs["rng"] = values[0] if values else None
        if "npool" not in kwargs:
            for equiv in self.npool_equiv_kwargs:
                if equiv in kwargs:
                    kwargs["npool"] = kwargs.pop(equiv)
                    break
            # If nothing was found, set to npool but only if it is larger
            # than 1
            else:
                if self._npool > 1:
                    kwargs["npool"] = self._npool
        super()._translate_kwargs(kwargs)


class SequincePersistentSampling(SequinceSMC):
    """Bilby wrapper for persistent sampling from sequince.

    Sequince: https://github.com/sequince-dev/sequince
    """

    _prebuilt_fn = staticmethod(make_persistent_bayesian_smc)
    _builder_accepts_n_particles = True
    sampler_name = "sequince-ps"
