import copy
from collections import namedtuple
from typing import Any, Callable

import numpy as np
from bilby.core.likelihood import Likelihood
from sequince.array_api import asarray, namespace, to_numpy

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
        likelihood: Likelihood,
        parameters: dict[str, float | Any],
        use_ratio: bool,
    ) -> float | Any:
        """Evaluate explicit-parameter likelihoods on the development API."""
        if use_ratio:
            return likelihood.log_likelihood_ratio(parameters=parameters)
        return likelihood.log_likelihood(parameters=parameters)


Functions = namedtuple("Functions", ["log_likelihood", "log_prior"])
"""Container for the log likelihood and log prior functions."""


def _global_log_likelihood(x: Any) -> float | Any:
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
) -> Functions:
    """Build eager, array-preserving Bilby density callbacks.

    Parameters
    ----------
    likelihood_dtype : str
        Dtype used for likelihood parameters and returned log likelihoods.
    map_fn : callable
        Ordered map over scalar likelihood calls. Backend sampling uses the
        built-in map; NumPy sampling also supports Bilby's worker pools.

    Returns
    -------
    Functions
        Prior and likelihood callbacks accepting particles of shape
        ``(n_particles, n_parameters)`` and returning arrays of shape
        ``(n_particles,)`` on the input backend and device. The likelihood
        callback evaluates only particles with finite prior density.
    """

    def log_prior(particles: Any) -> Any:
        x = dict(
            zip(sampling_convenience_dump.search_parameter_keys, particles.T)
        )
        return asarray(
            sampling_convenience_dump.priors.ln_prob(x, axis=0),
            like=particles,
        )

    def log_likelihood(particles: Any) -> Any:
        xp = namespace(particles)
        # Only evaluate the log likelihood for finite log prior
        logp = log_prior(particles)
        # This is an eager adapter: transfer the support mask once, keeping
        # parameter arrays and likelihood results on their original device.
        mask = to_numpy(xp.isfinite(logp))
        x = asarray(particles, dtype=likelihood_dtype, like=particles)
        values = iter(
            map_fn(
                _global_log_likelihood,
                (x[index] for index in np.flatnonzero(mask)),
            )
        )
        if len(particles) == 0:
            return asarray([], dtype=likelihood_dtype, like=particles)
        return xp.stack(
            [
                asarray(
                    next(values) if valid else -np.inf,
                    dtype=likelihood_dtype,
                    like=particles,
                )
                for valid in mask
            ]
        )

    return Functions(log_likelihood=log_likelihood, log_prior=log_prior)
