"""Proposal adapters for Bilby distributions."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
from bilby.core.utils import random
from orng import RandomGenerator


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
    ) -> np.ndarray:
        """Draw particles using the random generator supplied by Sequince."""
        del context, xp
        if isinstance(rng, RandomGenerator):
            if rng.backend != "numpy":
                raise ValueError("Bilby prior sampling requires a NumPy RNG.")
            # Preserve the same generator and its state; do not reseed or draw
            # an intermediate seed when adapting Sequince's RNG wrapper.
            rng = rng.generator
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
        return np.column_stack(
            [samples[key] for key in self.search_parameter_keys]
        )

    def log_prob(self, particles, context=None) -> np.ndarray:
        """Evaluate the joint Bilby prior density for each particle."""
        del context
        samples = dict(
            zip(self.search_parameter_keys, np.asarray(particles).T)
        )
        return np.asarray(self.priors.ln_prob(samples, axis=0))

    def sample_and_log_prob(
        self,
        n_samples: int,
        context=None,
        *,
        rng=None,
        xp: Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw particles and return their joint prior densities."""
        particles = self.sample(
            n_samples,
            context=context,
            rng=rng,
            xp=xp,
        )
        return particles, self.log_prob(particles, context=context)
