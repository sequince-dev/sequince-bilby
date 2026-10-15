import copy
from collections import namedtuple
from typing import Callable

import numpy as np
from bilby.core.likelihood import Likelihood

try:
    from bilby.core.utils.parallel import sampling_convenience_dump
except ImportError:
    # Bilby 2.8 keeps the worker-local sampling context in base_sampler.
    from bilby.core.sampler.base_sampler import (
        _sampling_convenience_dump as sampling_convenience_dump,
    )

try:
    from bilby.core.likelihood import _safe_likelihood_call as _call_likelihood
except ImportError:

    def _call_likelihood(
        likelihood: Likelihood, parameters: dict[str, float], use_ratio: bool
    ) -> float:
        """Evaluate explicit-parameter likelihoods on the development API."""
        if use_ratio:
            return likelihood.log_likelihood_ratio(parameters=parameters)
        return likelihood.log_likelihood(parameters=parameters)


Functions = namedtuple("Functions", ["log_likelihood", "log_prior"])
"""Container for the log likelihood and log prior functions."""


def _global_log_likelihood(x):
    params = copy.deepcopy(sampling_convenience_dump.parameters)
    theta = dict(zip(sampling_convenience_dump.search_parameter_keys, x))
    params.update(theta)
    return _call_likelihood(
        sampling_convenience_dump.likelihood,
        params,
        use_ratio=sampling_convenience_dump.use_ratio,
    )


def get_sequince_functions(
    likelihood_dtype: str = "float64",
    map_fn: Callable = map,
):
    """Get the log likelihood function for a bilby likelihood object."""

    def log_prior(particles):
        x = dict(
            zip(sampling_convenience_dump.search_parameter_keys, particles.T)
        )
        return sampling_convenience_dump.priors.ln_prob(x, axis=0)

    def log_likelihood(particles):
        logl = -np.inf * np.ones(len(particles), dtype=float)
        # Only evaluate the log likelihood for finite log prior
        logp = log_prior(particles)
        if not any(np.isfinite(logp)):
            return logl
        mask = np.isfinite(logp, dtype=bool)
        x = np.asarray(particles[mask, :], dtype=likelihood_dtype)
        logl[mask] = np.fromiter(
            map_fn(_global_log_likelihood, x),
            dtype=float,
        )
        return logl

    return Functions(log_likelihood=log_likelihood, log_prior=log_prior)
