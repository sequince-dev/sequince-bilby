from unittest.mock import patch

import bilby
import h5py
import numpy as np
import pytest
from sequince.checkpoint import SMCCheckpoint

from sequince_bilby.plugin import SequincePersistentSampling, SequinceSMC
from sequince_bilby.proposal import BilbyPriorProposal


def model(x, m, c):
    return m * x + c


def conversion_func(parameters):
    # d = |m| + |c|
    parameters["d"] = abs(parameters["m"]) + abs(parameters["c"])
    return parameters


@pytest.fixture()
def bilby_likelihood(rng):
    x = np.linspace(0, 10, 100)
    injection_parameters = dict(m=0.5, c=0.2)
    sigma = 1.0
    y = model(x, **injection_parameters) + rng.normal(0.0, sigma, len(x))
    likelihood = bilby.likelihood.GaussianLikelihood(x, y, model, sigma)
    return likelihood


@pytest.fixture()
def bilby_priors():
    priors = bilby.core.prior.PriorDict(conversion_function=conversion_func)
    priors["m"] = bilby.core.prior.Uniform(0, 5, boundary="periodic")
    priors["c"] = bilby.core.prior.Uniform(-2, 2, boundary="reflective")
    priors["d"] = bilby.core.prior.Constraint(name="d", minimum=0, maximum=5)
    return priors


@pytest.fixture()
def sampler_kwargs():
    return dict(
        n_particles=10,
        mutation_n_steps=5,
    )


@pytest.fixture()
def ps_sampler_kwargs():
    return dict(
        n_particles=10,
        target_ess=20,
        mutation_n_steps=5,
    )


def test_run_smc_sampler(
    bilby_likelihood,
    bilby_priors,
    tmp_path,
    sampler_kwargs,
):
    outdir = tmp_path / "test_run_sampler"
    outdir.mkdir(parents=True, exist_ok=True)

    bilby.run_sampler(
        likelihood=bilby_likelihood,
        priors=bilby_priors,
        sampler="sequince-smc",
        outdir=outdir,
        **sampler_kwargs,
    )


def test_run_sampler_pool(
    bilby_likelihood,
    bilby_priors,
    tmp_path,
    sampler_kwargs,
):
    from multiprocessing.dummy import Pool

    outdir = tmp_path / "test_run_sampler_pool"
    outdir.mkdir(parents=True, exist_ok=True)
    with patch("multiprocessing.Pool", new=Pool):
        bilby.run_sampler(
            likelihood=bilby_likelihood,
            priors=bilby_priors,
            sampler="sequince-smc",
            outdir=outdir,
            npool=2,
            **sampler_kwargs,
        )


def test_run_persistent_sampler(
    bilby_likelihood,
    bilby_priors,
    tmp_path,
    ps_sampler_kwargs,
):
    outdir = tmp_path / "test_run_ps_sampler"
    outdir.mkdir(parents=True, exist_ok=True)

    bilby.run_sampler(
        likelihood=bilby_likelihood,
        priors=bilby_priors,
        sampler="sequince-ps",
        outdir=outdir,
        **ps_sampler_kwargs,
    )


def test_run_sampler_with_custom_proposal(
    bilby_likelihood,
    bilby_priors,
    tmp_path,
    sampler_kwargs,
):
    proposal = BilbyPriorProposal(bilby_priors, ("m", "c"))
    sampler = SequinceSMC(
        likelihood=bilby_likelihood,
        priors=bilby_priors,
        outdir=tmp_path,
        label="custom_proposal",
        proposal=proposal,
        **sampler_kwargs,
    )

    result = sampler.run_sampler()

    assert result.samples.shape[1] == 2


@pytest.mark.parametrize(
    ("sampler_class", "builder_kwargs"),
    [
        (
            SequinceSMC,
            {"max_delta": 0.1},
        ),
        (
            SequincePersistentSampling,
            {},
        ),
    ],
)
def test_resume_from_checkpoint(
    sampler_class,
    builder_kwargs,
    bilby_likelihood,
    bilby_priors,
    tmp_path,
):
    priors = bilby.core.prior.PriorDict(
        {
            "m": bilby_priors["m"],
            "c": bilby_priors["c"],
        }
    )
    outdir = tmp_path / sampler_class.sampler_name
    outdir.mkdir()
    checkpoint_filename = outdir / "checkpoint.h5"

    def stop_after_first_iteration(state, context, result):
        del state, result
        if context.t == 0:
            return context.request_stop()
        return None

    initial_sampler = sampler_class(
        likelihood=bilby_likelihood,
        priors=priors,
        outdir=outdir,
        label="initial",
        n_particles=10,
        mutation="none",
        logging_frequency=0,
        callbacks=[stop_after_first_iteration],
        checkpoint_filename=checkpoint_filename,
        resume=False,
        **builder_kwargs,
    )
    initial_sampler.run_sampler()

    with h5py.File(checkpoint_filename, "r") as checkpoint_file:
        initial_checkpoint = SMCCheckpoint.load_hdf5(checkpoint_file)
    assert initial_checkpoint.next_iteration == 1

    resumed_sampler = sampler_class(
        likelihood=bilby_likelihood,
        priors=priors,
        outdir=outdir,
        label="resumed",
        n_particles=10,
        mutation="none",
        logging_frequency=0,
        checkpoint_filename=checkpoint_filename,
        resume=True,
        **builder_kwargs,
    )
    result = resumed_sampler.run_sampler()

    with h5py.File(checkpoint_filename, "r") as checkpoint_file:
        resumed_checkpoint = SMCCheckpoint.load_hdf5(checkpoint_file)
    assert (
        resumed_checkpoint.next_iteration > initial_checkpoint.next_iteration
    )
    assert result.samples.shape[1] == 2


@pytest.mark.parametrize(
    "sampler_class", [SequinceSMC, SequincePersistentSampling]
)
@pytest.mark.parametrize("mapping", ["auto", "serial", "custom"])
def test_map_selection(
    sampler_class, mapping, bilby_likelihood, bilby_priors, tmp_path
):
    from multiprocessing.dummy import Pool
    from unittest.mock import Mock

    class CustomMap:
        def __init__(self):
            self.calls = 0

        def __deepcopy__(self, memo):
            raise AssertionError("The custom mapper must not be copied")

        def __call__(self, function, batch):
            assert isinstance(batch, np.ndarray)
            self.calls += 1
            return map(function, batch)

    mapper = CustomMap()
    option = {"auto": None, "serial": "map", "custom": mapper}[mapping]
    with Pool(2) as pool:
        pool.map = Mock(wraps=pool.map)
        sampler = sampler_class(
            likelihood=bilby_likelihood,
            priors=bilby.core.prior.PriorDict(
                {"m": bilby_priors["m"], "c": bilby_priors["c"]}
            ),
            outdir=tmp_path,
            pool=pool,
            map_fn=option,
            n_particles=32,
            mutation="none",
            rng=1234,
            checkpoint=False,
            resume=False,
            logging_frequency=0,
        )
        result = sampler.run_sampler()
        assert pool.map.called == (mapping == "auto")
    assert (mapper.calls > 0) == (mapping == "custom")
    np.testing.assert_allclose(result.nested_samples.weights.sum(), 1.0)
    assert np.isfinite(result.log_evidence)


def test_mapping_failure_closes_owned_pool(
    bilby_likelihood, bilby_priors, tmp_path
):
    from multiprocessing.dummy import Pool

    def failed_map(function, batch):
        raise RuntimeError("mapping failed")

    with patch("multiprocessing.Pool", new=Pool):
        sampler = SequinceSMC(
            likelihood=bilby_likelihood,
            priors=bilby_priors,
            outdir=tmp_path,
            npool=2,
            map_fn=failed_map,
            n_particles=32,
            checkpoint=False,
            resume=False,
        )
        with pytest.raises(RuntimeError, match="mapping failed"):
            sampler.run_sampler()
        assert sampler.pool is None
