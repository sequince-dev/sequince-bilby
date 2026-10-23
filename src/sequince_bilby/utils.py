import copy
from collections import namedtuple
from collections.abc import Callable, Iterable
from typing import Any, Literal

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

MapFunction = Callable[[Callable[[Any], Any], Any], Iterable[Any]]


def resolve_map_fn(
    map_fn: MapFunction | Literal["map", "vmap"] | None,
    *,
    backend: str,
) -> MapFunction:
    """Resolve an ordered scalar map or an explicit backend vectorizer.

    Parameters
    ----------
    map_fn : callable, {"map", "vmap"}, or None
        An ordered map, Python's map, or backend vectorization. ``None`` selects
        Python's map; the sampler can substitute its initialized pool map.
    backend : str
        Sampling backend. Vectorization requires JAX or Torch.

    Returns
    -------
    callable
        Accepts ``(function, particles)`` and returns one scalar per row in
        order. Vectorization requires a traceable, side-effect-free likelihood.
    """
    if map_fn is None:
        return map
    if callable(map_fn):
        return map_fn
    if isinstance(map_fn, str) and map_fn == "map":
        return map
    if not isinstance(map_fn, str) or map_fn != "vmap":
        raise ValueError("map_fn must be None, 'map', 'vmap', or callable.")
    if backend == "jax":
        from jax import vmap
    elif backend == "torch":
        from torch import vmap
    else:
        raise ValueError("map_fn='vmap' requires the jax or torch backend.")

    def vectorized_map(
        function: Callable[[Any], Any], particles: Any
    ) -> Iterable[Any]:
        return vmap(function)(particles)

    return vectorized_map


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
    map_fn: MapFunction | Literal["map", "vmap"] | None = None,
    *,
    backend: str = "numpy",
    pool: Any | None = None,
) -> Functions:
    """Build array-preserving callbacks from an initialized Bilby context.

    Parameters
    ----------
    likelihood_dtype : str
        Dtype used for likelihood parameters and returned log likelihoods.
    map_fn : callable, {"map", "vmap"}, or None
        Ordered map accepting ``(scalar_function, particles)``. The particles
        are a two-dimensional array containing only rows with finite prior
        density. Return exactly one scalar per row, in the same order. ``None``
        uses the supplied pool's map or Python's map.
    backend : str
        Sampling backend used to resolve named maps.
    pool : object or None
        Initialized Bilby worker pool, used when ``map_fn`` is None.

    Returns
    -------
    Functions
        Prior and likelihood callbacks accepting particles of shape
        ``(n_particles, n_parameters)`` and returning arrays of shape
        ``(n_particles,)`` on the input backend and device. The likelihood
        callback evaluates only particles with finite prior density.
    """
    resolved_map = resolve_map_fn(map_fn, backend=backend)
    if map_fn is None and pool is not None:
        resolved_map = pool.map
    priors = sampling_convenience_dump.priors
    keys = tuple(sampling_convenience_dump.search_parameter_keys)

    def log_prior(particles: Any) -> Any:
        x = dict(zip(keys, particles.T))
        return asarray(
            priors.ln_prob(x, axis=0),
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
        indices = np.flatnonzero(mask)
        if len(indices) == 0:
            return xp.full_like(logp, -xp.inf, dtype=x.dtype)
        all_valid = len(indices) == len(particles)
        selected = x if all_valid else x[asarray(indices, like=x)]
        mapped = resolved_map(_global_log_likelihood, selected)
        # Array-returning maps (including vmap) must stay batched. Iterating
        # their output would dispatch a device operation for every scalar.
        if hasattr(mapped, "shape"):
            mapped = asarray(mapped, dtype=likelihood_dtype, like=particles)
        else:
            values = list(mapped)
            if len(values) != len(indices):
                raise ValueError("map_fn must return one scalar per particle.")
            mapped = xp.stack(
                [
                    asarray(value, dtype=likelihood_dtype, like=particles)
                    for value in values
                ]
            )
        if mapped.shape != (len(indices),):
            raise ValueError("map_fn must return one scalar per particle.")
        if all_valid:
            return mapped
        # The cumulative rank maps original rows to their compacted values.
        # Invalid rows gather a safe index before where replaces them by -inf.
        ranks = asarray(np.maximum(np.cumsum(mask) - 1, 0), like=x)
        return xp.where(
            asarray(mask, like=particles),
            xp.take(mapped, ranks, axis=0),
            asarray(-xp.inf, dtype=likelihood_dtype, like=particles),
        )

    return Functions(log_likelihood=log_likelihood, log_prior=log_prior)
