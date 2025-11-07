import copy
from collections import namedtuple
from typing import Callable

import numpy as np
from bilby.core.sampler.base_sampler import _sampling_convenience_dump

Functions = namedtuple("Functions", ["log_likelihood", "log_prior"])
"""Container for the log likelihood and log prior functions."""


def _global_log_likelihood(x):
    params = copy.deepcopy(_sampling_convenience_dump.parameters)
    theta = dict(zip(_sampling_convenience_dump.search_parameter_keys, x))
    params.update(theta)
    if _sampling_convenience_dump.use_ratio:
        return _sampling_convenience_dump.likelihood.log_likelihood_ratio(parameters=params)
    else:
        return _sampling_convenience_dump.likelihood.log_likelihood(parameters=params)


def get_sequince_functions(
    likelihood_dtype: str = "float64",
    map_fn: Callable = map,
):
    """Get the log likelihood function for a bilby likelihood object."""

    def log_prior(particles):
        x = dict(zip(_sampling_convenience_dump.search_parameter_keys, particles.T))
        return _sampling_convenience_dump.priors.ln_prob(x, axis=0)

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
