"""Plugin for the sequince sampler in bilby."""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
from bilby.core.sampler.base_sampler import Sampler
from bilby.core.utils import random
from bilby.core.utils.log import logger
from sequince.log import configure_logging
from sequince.prebuilt_samplers import (
    make_persistent_bayesian_smc,
    make_standard_bayesian_smc,
)
from sequince.state import ParticleState
from sequince.steps import StepContext

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
        return dict(n_particles=500)

    def run_sampler(self) -> dict:
        """Run the sampler."""

        kwargs = copy.deepcopy(self.kwargs)

        kwargs.pop("resume", None)
        n_particles = kwargs.pop("n_particles")

        configure_logging(level=kwargs.pop("sequince_log_level", "INFO"))

        self._setup_pool()

        if self.pool is not None:
            map_fn = self.pool.map
        else:
            map_fn = map

        funcs = get_sequince_functions(map_fn=map_fn)

        # Should handle these properly
        kwargs.pop("npool", None)
        kwargs.pop("pool", None)
        kwargs.pop("sampling_seed", None)

        if self._builder_accepts_n_particles:
            kwargs["n_particles"] = n_particles

        logger.info(f"Creating sequince instance with kwargs: {kwargs}")

        sampler, posterior, metadata = self._prebuilt_fn(
            log_prior=funcs.log_prior,
            log_likelihood=funcs.log_likelihood,
            **kwargs,
        )

        samples_dict = self.priors.sample(n_particles)
        samples = np.array(
            [samples_dict[key] for key in self.search_parameter_keys]
        ).T
        context = StepContext(t=0, metadata=metadata)
        initial_state = ParticleState(
            samples,
            evaluation=posterior.prepare_evaluation(samples, context),
        )

        result = sampler.run(initial_state, metadata=metadata, rng=random.rng)

        result_file = self.get_result_file_name(self.outdir, self.label)
        result.save_to_file(result_file)

        self._close_pool()

        weighted_samples, weights = result.posterior_samples(method="weights")
        weighted_posterior = pd.DataFrame(
            np.asarray(weighted_samples),
            columns=self.search_parameter_keys,
        )
        weighted_posterior["weights"] = np.asarray(weights)
        self.result.nested_samples = weighted_posterior

        posterior_samples, _ = result.posterior_samples(
            method="rejection",
            rng=random.rng,
        )
        self.result.samples = posterior_samples
        self.result.log_evidence = result.log_evidence()
        self.result.log_evidence_err = result.log_evidence_error()
        self.result.num_likelihood_evaluations = np.nan

        return self.result

    @classmethod
    def get_result_file_name(cls, outdir, label):
        sampler_name = cls.sampler_name.replace("-", "_")
        return Path(outdir) / f"{label}_{sampler_name}_result.h5"

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
        filenames = [cls.get_result_file_name(outdir, label)]
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
