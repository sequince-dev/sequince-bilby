"""Proposal adapters for Bilby distributions."""

from __future__ import annotations

from typing import Any

import numpy as np


class BilbyPriorProposal:
    """Adapt a Bilby prior dictionary to a Sequince proposal."""

    def __init__(self, priors, search_parameter_keys) -> None:
        self.priors = priors
        self.search_parameter_keys = tuple(search_parameter_keys)

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
        samples = self.priors.sample(
            n_samples,
            random_state=rng,
        )
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
