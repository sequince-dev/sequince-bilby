"""Proposal adapters for Bilby distributions."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
from bilby.core.utils import random
from orng import RandomGenerator
from sequince.array_api import asarray, namespace


class BilbyPriorProposal:
    """Adapt a Bilby prior dictionary to a Sequince proposal."""

    def __init__(self, priors, search_parameter_keys) -> None:
        self.priors = priors
        self.search_parameter_keys = tuple(search_parameter_keys)
        self._accepts_random_state = (
            "random_state" in inspect.signature(priors.sample).parameters
        )

    def sample(
        self,
        n_samples: int,
        context=None,
        *,
        rng=None,
        xp: Any | None = None,
    ) -> Any:
        """Draw particles using the random generator supplied by Sequince."""
        del context
        if isinstance(rng, RandomGenerator):
            if rng.backend != "numpy":
                if not self._accepts_random_state or not getattr(
                    random, "BILBY_ARRAY_API", False
                ):
                    raise ValueError(
                        "Backend prior sampling requires array-aware Bilby "
                        "with BILBY_ARRAY_API=1 before import."
                    )
            else:
                # state() returns the live generator for seeded and restored
                # wrappers too; .generator only records the constructor input.
                rng = rng.state()
        if self._accepts_random_state:
            samples = self.priors.sample(n_samples, random_state=rng)
        else:
            # Bilby 2.8 priors consume its process-global generator. Scope the
            # substitution to this draw, preserving both generators' states.
            # This legacy path is intended for process-based worker pools.
            original_rng = random.Generator.rng
            if rng is not None:
                if not isinstance(rng, np.random.Generator):
                    raise ValueError(
                        "Bilby prior sampling requires a NumPy RNG."
                    )
                random.Generator.rng = rng
            try:
                samples = self.priors.sample(n_samples)
            finally:
                random.Generator.rng = original_rng
        columns = [samples[key] for key in self.search_parameter_keys]
        xp = xp or namespace(*columns)
        return xp.stack([asarray(column, xp=xp) for column in columns], axis=1)

    def log_prob(self, particles: Any, context=None) -> Any:
        """Evaluate the joint Bilby prior density for each particle."""
        del context
        samples = dict(zip(self.search_parameter_keys, particles.T))
        return asarray(self.priors.ln_prob(samples, axis=0), like=particles)

    def sample_and_log_prob(
        self,
        n_samples: int,
        context=None,
        *,
        rng=None,
        xp: Any | None = None,
    ) -> tuple[Any, Any]:
        """Draw particles and return their joint prior densities."""
        particles = self.sample(
            n_samples,
            context=context,
            rng=rng,
            xp=xp,
        )
        return particles, self.log_prob(particles, context=context)
