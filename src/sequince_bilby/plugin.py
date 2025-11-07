"""Plugin for the sequince sampler in bilby."""

import copy

import numpy as np
from bilby.core.sampler.base_sampler import Sampler
from bilby.core.utils import random
from bilby.core.utils.log import logger
from sequince.logging_utils import configure_logging
from sequince.prebuilt_samplers import make_standard_bayesian_smc
from sequince.state import ParticleState

from sequince_bilby.utils import get_sequince_functions


class SequinceSMC(Sampler):
    """Bilby wrapper for sequince.

    Sequince: https://github.com/mj-will/sequince

    """

    sampler_name = "sequince"
    """
    Name of the sampler.
    """

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

        funcs = get_sequince_functions(
            self.likelihood,
            self.priors,
            self.search_parameter_keys,
            use_ratio=self.use_ratio,
        )

        # Should handle these properly
        kwargs.pop("npool", None)
        kwargs.pop("pool", None)
        kwargs.pop("sampling_seed", None)

        logger.info(f"Creating sequince instance with kwargs: {kwargs}")

        sampler, posterior, context = make_standard_bayesian_smc(
            log_prior=funcs.log_prior,
            log_likelihood=funcs.log_likelihood,
            **kwargs,
        )

        samples_dict = self.priors.sample(n_particles)
        samples = np.array(
            [samples_dict[key] for key in self.search_parameter_keys]
        ).T
        log_weights = np.full((n_particles,), -np.log(n_particles))
        initial_state = ParticleState(
            samples,
            log_weights=log_weights,
        )

        results = sampler.run(
            initial_state, shared_context=context, rng=random.rng
        )

        self._close_pool()

        self.result.samples = results.posterior_samples()
        self.result.log_evidence = results.final_state.log_evidence
        self.result.log_evidence_err = np.nan
        self.result.num_likelihood_evaluations = np.nan

        return self.result

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
        filenames = []
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
