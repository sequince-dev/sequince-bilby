from collections import namedtuple
from dataclasses import dataclass
from typing import Callable

import numpy as np
from bilby.core.likelihood import Likelihood
from bilby.core.prior import PriorDict

Functions = namedtuple("Functions", ["log_likelihood", "log_prior"])
"""Container for the log likelihood and log prior functions."""


@dataclass
class GlobalFunctions:
    """Dataclass to store global functions."""

    bilby_likelihood: Likelihood
    bilby_priors: PriorDict
    parameters: list
    use_ratio: bool


_global_functions = GlobalFunctions(None, None, [], False)


def update_global_functions(
    bilby_likelihood: Likelihood,
    bilby_priors: PriorDict,
    parameters: list[str],
    use_ratio: bool,
):
    """Update the global functions for log likelihood and log prior."""
    global _global_functions
    _global_functions.bilby_likelihood = bilby_likelihood
    _global_functions.bilby_priors = bilby_priors
    _global_functions.parameters = parameters
    _global_functions.use_ratio = use_ratio


def _global_log_likelihood(x):
    theta = dict(zip(_global_functions.parameters, x))
    _global_functions.bilby_likelihood.parameters.update(theta)

    if _global_functions.use_ratio:
        return _global_functions.bilby_likelihood.log_likelihood_ratio()
    else:
        return _global_functions.bilby_likelihood.log_likelihood()


def get_sequince_functions(
    bilby_likelihood,
    bilby_priors,
    parameters,
    use_ratio: bool = False,
    likelihood_dtype: str = "float64",
    map_fn: Callable = map,
):
    """Get the log likelihood function for a bilby likelihood object."""

    update_global_functions(
        bilby_likelihood, bilby_priors, parameters, use_ratio
    )

    def log_prior(particles):
        x = dict(zip(parameters, particles.T))
        return bilby_priors.ln_prob(x, axis=0)

    def log_likelihood(particles):
        logl = -np.inf * np.ones(len(particles), dtype=float)
        # Only evaluate the log likelihood for finite log prior
        logp = log_prior(particles)
        mask = np.isfinite(logp, dtype=bool)
        x = np.asarray(particles[mask, :], dtype=likelihood_dtype)
        logl[mask] = np.fromiter(
            map_fn(_global_log_likelihood, x),
            dtype=float,
        )
        return logl

    return Functions(log_likelihood=log_likelihood, log_prior=log_prior)
