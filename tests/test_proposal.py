import bilby
import numpy as np
from sequince.utils.random import resolve_rng

from sequince_bilby.proposal import BilbyPriorProposal


def test_bilby_prior_proposal_uses_supplied_rng():
    priors = bilby.core.prior.PriorDict({"x": bilby.core.prior.Uniform(-1, 1)})
    proposal = BilbyPriorProposal(priors, ("x",))
    numpy_rng = np.random.default_rng(1234)
    sampling_rng = resolve_rng(numpy_rng)

    samples = proposal.sample(10, rng=sampling_rng)
    repeated = proposal.sample(
        10,
        rng=resolve_rng(np.random.default_rng(1234)),
    )

    assert sampling_rng.generator is numpy_rng
    np.testing.assert_array_equal(samples, repeated)
    np.testing.assert_array_equal(
        proposal.log_prob(samples),
        priors.ln_prob({"x": samples[:, 0]}, axis=0),
    )


def test_bilby_prior_proposal_forwards_orng_generator():
    class RecordingPriors:
        def __init__(self):
            self.random_state = None

        def sample(self, n_samples, *, random_state=None):
            self.random_state = random_state
            return {"x": np.zeros(n_samples)}

    priors = RecordingPriors()
    proposal = BilbyPriorProposal(priors, ("x",))
    sampling_rng = resolve_rng(np.random.default_rng(1234))

    proposal.sample(2, rng=sampling_rng)

    assert priors.random_state is sampling_rng
