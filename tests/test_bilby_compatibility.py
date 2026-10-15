"""Exercise version-specific worker likelihood and RNG compatibility."""

from pathlib import Path

import bilby
import numpy as np
import pytest

from sequince_bilby.plugin import SequincePersistentSampling, SequinceSMC
from sequince_bilby.proposal import BilbyPriorProposal
from sequince_bilby.utils import _global_log_likelihood


class GaussianWithNoise(bilby.core.likelihood.GaussianLikelihood):
    def noise_log_likelihood(self) -> float:
        return 0.0


def model(x: np.ndarray, mu: float) -> np.ndarray:
    return np.full_like(x, mu)


@pytest.mark.parametrize(
    "sampler_class", [SequinceSMC, SequincePersistentSampling]
)
@pytest.mark.parametrize("use_ratio", [False, True])
def test_worker_likelihood(
    sampler_class: type[SequinceSMC], use_ratio: bool, tmp_path: Path
) -> None:
    likelihood = GaussianWithNoise(
        x=np.array([0.0]), y=np.array([0.3]), func=model, sigma=1.0
    )
    sampler = sampler_class(
        likelihood=likelihood,
        priors=bilby.core.prior.PriorDict(
            {"mu": bilby.core.prior.Uniform(-5, 5, name="mu")}
        ),
        outdir=str(tmp_path),
        use_ratio=use_ratio,
        npool=1,
    )
    sampler._setup_pool()
    expected = sampler.log_likelihood([0.7])
    assert _global_log_likelihood(np.array([0.7])) == expected


@pytest.mark.parametrize("fail", [False, True])
def test_legacy_prior_restores_global_rng(fail: bool) -> None:
    from bilby.core.utils import random

    class LegacyPriors:
        def sample(self, size: int) -> dict[str, np.ndarray]:
            values = random.rng.uniform(size=size)
            if fail:
                raise RuntimeError("draw failed")
            return {"x": values}

    original = random.rng
    before = original.bit_generator.state
    proposal = BilbyPriorProposal(LegacyPriors(), ["x"])
    rng = np.random.default_rng(100)
    if fail:
        with pytest.raises(RuntimeError, match="draw failed"):
            proposal.sample(5, rng=rng)
    else:
        samples = proposal.sample(5, rng=rng)
        np.testing.assert_array_equal(
            samples[:, 0], np.random.default_rng(100).uniform(size=5)
        )
    assert random.rng is original
    assert original.bit_generator.state == before
