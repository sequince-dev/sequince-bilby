"""Exercise array preservation and sampling across optional CPU backends."""

from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import bilby
import h5py
import numpy as np
import pytest
from bilby.core.utils import random
from orng import RandomGenerator
from sequince.array_api import determine_backend_name, namespace, to_numpy
from sequince.checkpoint import SMCCheckpoint

from sequince_bilby.plugin import SequincePersistentSampling, SequinceSMC
from sequince_bilby.proposal import BilbyPriorProposal
from sequince_bilby.utils import (
    get_sequince_functions,
    sampling_convenience_dump,
)


@pytest.fixture(params=["numpy", "jax", "torch"])
def xp(request: pytest.FixtureRequest) -> Iterator[ModuleType]:
    if request.param == "jax":
        jax = pytest.importorskip("jax")
        previous = jax.config.x64_enabled
        jax.config.update("jax_enable_x64", True)
        try:
            yield pytest.importorskip("jax.numpy")
        finally:
            jax.config.update("jax_enable_x64", previous)
    else:
        yield pytest.importorskip(
            "array_api_compat.numpy"
            if request.param == "numpy"
            else "array_api_compat.torch"
        )


class GaussianLikelihood(bilby.Likelihood):
    """Explicit-parameter likelihood supported by both Bilby API versions."""

    def log_likelihood(self, parameters: dict | None = None):
        return -0.5 * (parameters["x"] - parameters["offset"]) ** 2

    def noise_log_likelihood(self) -> float:
        return 2.0

    def log_likelihood_ratio(self, parameters: dict | None = None):
        return self.log_likelihood(parameters) - self.noise_log_likelihood()


class ArrayPriors:
    """Minimal array-aware prior for adapter tests independent of Bilby version."""

    def sample(self, size: int, *, random_state=None) -> dict:
        return {"x": random_state.uniform(-1, 1, size=size)}

    def ln_prob(self, samples: dict, axis: int = 0):
        del axis
        x = samples["x"]
        xp = namespace(x)
        return xp.where(xp.abs(x) <= 1, -xp.log(xp.asarray(2.0)), -xp.inf)


@pytest.mark.parametrize("use_ratio", [False, True])
@pytest.mark.parametrize("values", [[0.0, 2.0, -0.5], [2.0, -2.0], []])
def test_likelihood_preserves_backend_and_support(
    xp: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    use_ratio: bool,
    values: list[float],
) -> None:
    particles = xp.reshape(xp.asarray(values, dtype=xp.float64), (-1, 1))
    monkeypatch.setattr(sampling_convenience_dump, "priors", ArrayPriors())
    monkeypatch.setattr(
        sampling_convenience_dump, "parameters", {"offset": 0.2}
    )
    monkeypatch.setattr(
        sampling_convenience_dump, "search_parameter_keys", ["x"]
    )
    monkeypatch.setattr(sampling_convenience_dump, "use_ratio", use_ratio)
    calls = []

    class RecordingLikelihood(GaussianLikelihood):
        def log_likelihood(self, parameters: dict | None = None):
            calls.append(parameters["x"])
            return super().log_likelihood(parameters)

    monkeypatch.setattr(
        sampling_convenience_dump, "likelihood", RecordingLikelihood()
    )
    functions = get_sequince_functions()
    logp = functions.log_prior(particles)
    logl = functions.log_likelihood(particles)
    assert determine_backend_name(logp) == determine_backend_name(xp=xp)
    assert determine_backend_name(logl) == determine_backend_name(xp=xp)
    expected = np.array(
        [
            -0.5 * (x - 0.2) ** 2 - (2.0 if use_ratio else 0.0)
            if abs(x) <= 1
            else -np.inf
            for x in values
        ]
    )
    np.testing.assert_allclose(to_numpy(logl), expected)
    assert len(calls) == sum(abs(x) <= 1 for x in values)
    for value in calls:
        assert determine_backend_name(value) == determine_backend_name(xp=xp)


def test_prior_preserves_backend_and_rng(
    xp: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(random, "BILBY_ARRAY_API", True, raising=False)
    proposal = BilbyPriorProposal(ArrayPriors(), ["x"])
    rng = RandomGenerator.from_xp(xp, seed=1234)
    first, logp = proposal.sample_and_log_prob(10, rng=rng)
    second = proposal.sample(10, rng=rng)
    repeated = proposal.sample(10, rng=RandomGenerator.from_xp(xp, seed=1234))
    assert determine_backend_name(first) == determine_backend_name(xp=xp)
    assert determine_backend_name(logp) == determine_backend_name(xp=xp)
    np.testing.assert_array_equal(to_numpy(first), to_numpy(repeated))
    assert not np.array_equal(to_numpy(first), to_numpy(second))
    np.testing.assert_allclose(to_numpy(logp), -np.log(2))


def test_default_rng_preserves_bilby_stream() -> None:
    rng = SequinceSMC._resolve_sampling_rng("numpy")
    assert rng.state() is random.rng
    with pytest.raises(ValueError, match="sampler_backend"):
        SequinceSMC._resolve_sampling_rng("invalid")


def test_backend_rng_follows_bilby_seed(xp: ModuleType) -> None:
    backend = determine_backend_name(xp=xp)
    random.seed(1234)
    first = SequinceSMC._resolve_sampling_rng(backend).uniform(size=10)
    random.seed(1234)
    repeated = SequinceSMC._resolve_sampling_rng(backend).uniform(size=10)
    random.seed(9999)
    different = SequinceSMC._resolve_sampling_rng(backend).uniform(size=10)
    assert determine_backend_name(first) == backend
    np.testing.assert_array_equal(to_numpy(first), to_numpy(repeated))
    assert not np.array_equal(to_numpy(first), to_numpy(different))


@pytest.mark.parametrize(
    "sampler_class", [SequinceSMC, SequincePersistentSampling]
)
@pytest.mark.parametrize("mutation", ["none", "minipcn"])
def test_backend_run_and_resume(
    xp: ModuleType,
    sampler_class: type[SequinceSMC],
    mutation: str,
    tmp_path: Path,
) -> None:
    backend = determine_backend_name(xp=xp)
    if backend != "numpy" and not getattr(random, "BILBY_ARRAY_API", False):
        pytest.skip("Requires array-aware Bilby with BILBY_ARRAY_API=1")
    priors = bilby.core.prior.PriorDict(
        {"x": bilby.core.prior.Uniform(-5, 5), "offset": 0.3}
    )
    checkpoint_filename = tmp_path / "checkpoint.h5"
    seen = []

    def observe(state, context, result):
        del context, result
        seen.append(determine_backend_name(state.particles))

    options = dict(
        likelihood=GaussianLikelihood(),
        priors=priors,
        outdir=tmp_path,
        n_particles=128,
        mutation=mutation,
        mutation_n_steps=1,
        logging_frequency=0,
        sampler_backend=backend,
        checkpoint_filename=checkpoint_filename,
        callbacks=[observe],
    )
    try:
        result = sampler_class(
            **options, label="initial", resume=False
        ).run_sampler()
    except TypeError as exc:
        if backend == "torch" and str(exc) == (
            "log(): argument 'input' (position 1) must be Tensor, not int"
        ):
            pytest.xfail(
                "Bilby PriorDict.check_ln_prob calls torch.log on the "
                "Python constraint normalization factor outside prior support"
            )
        raise
    assert seen and set(seen) == {backend}
    assert isinstance(result.samples, np.ndarray)
    assert result.samples.shape[1] == 1
    assert np.isfinite(result.log_evidence)
    np.testing.assert_allclose(result.nested_samples.weights.sum(), 1.0)
    # Integral of exp(-0.5*(x-0.3)**2) over [-5,5], divided by prior width.
    assert result.log_evidence == pytest.approx(
        np.log(np.sqrt(2 * np.pi) / 10), abs=0.5
    )
    with h5py.File(checkpoint_filename, "r") as handle:
        checkpoint = SMCCheckpoint.load_hdf5(handle)
    assert determine_backend_name(checkpoint.state.particles) == backend
    assert checkpoint.restore_rng().backend == backend
    resumed = sampler_class(
        **options, label="resumed", resume=True
    ).run_sampler()
    np.testing.assert_array_equal(
        result.nested_samples.to_numpy(), resumed.nested_samples.to_numpy()
    )
    assert resumed.log_evidence == result.log_evidence


@pytest.mark.parametrize(
    "sampler_class", [SequinceSMC, SequincePersistentSampling]
)
def test_backend_interrupted_resume(
    xp: ModuleType, sampler_class: type[SequinceSMC], tmp_path: Path
) -> None:
    backend = determine_backend_name(xp=xp)
    if backend != "numpy" and not getattr(random, "BILBY_ARRAY_API", False):
        pytest.skip("Requires array-aware Bilby with BILBY_ARRAY_API=1")

    def stop(state, context, result):
        del state, result
        return context.request_stop()

    options = dict(
        likelihood=GaussianLikelihood(),
        priors=bilby.core.prior.PriorDict(
            {"x": bilby.core.prior.Uniform(-5, 5), "offset": 0.3}
        ),
        outdir=tmp_path,
        n_particles=48,
        mutation="none",
        particle_transform=None,
        logging_frequency=0,
        sampler_backend=backend,
    )
    random.seed(1234)
    expected = sampler_class(
        **options,
        label="full",
        rng=1234,
        checkpoint=False,
        resume=False,
    ).run_sampler()
    filename = tmp_path / "interrupted.h5"
    random.seed(1234)
    sampler_class(
        **options,
        label="interrupted",
        rng=1234,
        checkpoint_filename=filename,
        callbacks=[stop],
        resume=False,
    ).run_sampler()
    with h5py.File(filename, "r") as handle:
        before = SMCCheckpoint.load_hdf5(handle)
    assert before.next_iteration == 1
    random.seed(9999)
    resumed = sampler_class(
        **options,
        label="continued",
        rng=9999,
        checkpoint_filename=filename,
        resume=True,
    ).run_sampler()
    with h5py.File(filename, "r") as handle:
        after = SMCCheckpoint.load_hdf5(handle)
    assert after.next_iteration > before.next_iteration
    np.testing.assert_array_equal(expected.samples, resumed.samples)
    np.testing.assert_array_equal(
        expected.nested_samples.to_numpy(), resumed.nested_samples.to_numpy()
    )
    assert expected.log_evidence == resumed.log_evidence


def test_numpy_prior_uses_bilby_seed() -> None:
    random.seed(1234)
    rng = SequinceSMC._resolve_sampling_rng("numpy")
    proposal = BilbyPriorProposal(
        bilby.core.prior.PriorDict({"x": bilby.core.prior.Uniform(-1, 1)}),
        ["x"],
    )
    actual = proposal.sample(10, rng=rng)
    np.testing.assert_array_equal(
        actual[:, 0], np.random.default_rng(1234).uniform(-1, 1, size=10)
    )


def test_bilby_prior_preserves_parameter_order(xp: ModuleType) -> None:
    backend = determine_backend_name(xp=xp)
    if backend != "numpy" and not getattr(random, "BILBY_ARRAY_API", False):
        pytest.skip("Requires array-aware Bilby with BILBY_ARRAY_API=1")
    priors = bilby.core.prior.PriorDict(
        {
            "x": bilby.core.prior.Uniform(-2, -1),
            "y": bilby.core.prior.Uniform(1, 3),
            "fixed": 0.3,
        }
    )
    proposal = BilbyPriorProposal(priors, ["y", "x"])
    samples, logp = proposal.sample_and_log_prob(
        10, rng=RandomGenerator.from_xp(xp, seed=1234)
    )
    assert determine_backend_name(samples) == backend
    assert determine_backend_name(logp) == backend
    assert samples.shape == (10, 2)
    values = to_numpy(samples)
    assert np.all((values[:, 0] >= 1) & (values[:, 0] <= 3))
    assert np.all((values[:, 1] >= -2) & (values[:, 1] <= -1))
    np.testing.assert_allclose(to_numpy(logp), -np.log(2))


def test_prior_requires_array_aware_bilby(
    xp: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    if determine_backend_name(xp=xp) == "numpy":
        return
    monkeypatch.setattr(random, "BILBY_ARRAY_API", False, raising=False)
    proposal = BilbyPriorProposal(ArrayPriors(), ["x"])
    with pytest.raises(ValueError, match="BILBY_ARRAY_API=1"):
        proposal.sample(10, rng=RandomGenerator.from_xp(xp, seed=1234))


@pytest.mark.parametrize(
    "alias", ["rng", "random_state", "seed", "sampling_seed", "random_seed"]
)
@pytest.mark.parametrize(
    "sampler_class", [SequinceSMC, SequincePersistentSampling]
)
@pytest.mark.parametrize("generator", [False, True])
def test_rng_aliases(
    alias: str,
    sampler_class: type[SequinceSMC],
    generator: bool,
    tmp_path: Path,
) -> None:
    supplied = np.random.default_rng(1234) if generator else 1234
    sampler = sampler_class(
        likelihood=GaussianLikelihood(),
        priors=bilby.core.prior.PriorDict(
            {"x": bilby.core.prior.Uniform(-5, 5), "offset": 0.3}
        ),
        outdir=tmp_path,
        soft_init=True,
        n_particles=32,
        mutation="none",
        logging_frequency=0,
        resume=False,
        **{alias: supplied},
    )
    assert sampler.kwargs["rng"] is supplied
    assert not (
        {"random_state", "seed", "sampling_seed", "random_seed"}
        & sampler.kwargs.keys()
    )
    sampler.run_sampler()
    with h5py.File(
        sampler.get_checkpoint_file_name(tmp_path, "label"), "r"
    ) as handle:
        checkpoint = SMCCheckpoint.load_hdf5(handle)
    np.testing.assert_array_equal(
        checkpoint.initial_state.particles[:, 0],
        np.random.default_rng(1234).uniform(-5, 5, size=32),
    )


@pytest.mark.parametrize(
    "alias", ["random_state", "seed", "sampling_seed", "random_seed"]
)
def test_conflicting_rng_aliases(alias: str) -> None:
    sampler = SequinceSMC.__new__(SequinceSMC)
    with pytest.raises(ValueError, match="Specify only one"):
        sampler._translate_kwargs({"rng": 1234, alias: 5678})


def test_explicit_rng_state_is_preserved() -> None:
    numpy_rng = np.random.default_rng(1234)
    wrapped = SequinceSMC._resolve_sampling_rng("numpy", numpy_rng)
    assert wrapped.state() is numpy_rng
    restored = RandomGenerator.from_state_dict(wrapped.state_dict())
    assert SequinceSMC._resolve_sampling_rng("numpy", restored) is restored
    np.testing.assert_array_equal(
        wrapped.uniform(size=10), restored.uniform(size=10)
    )


def test_explicit_seed_selects_backend(xp: ModuleType) -> None:
    backend = determine_backend_name(xp=xp)
    random.seed(1)
    before = random.rng.bit_generator.state
    first = SequinceSMC._resolve_sampling_rng(backend, 1234).uniform(size=10)
    assert random.rng.bit_generator.state == before
    repeated = RandomGenerator.from_xp(xp, seed=1234).uniform(size=10)
    np.testing.assert_array_equal(to_numpy(first), to_numpy(repeated))
    native = RandomGenerator.from_xp(xp, seed=1234)
    assert SequinceSMC._resolve_sampling_rng(backend, native) is native
    if backend != "numpy":
        with pytest.raises(ValueError, match="backend differs"):
            SequinceSMC._resolve_sampling_rng("numpy", native)


def test_numpy_rng_seeds_backend(xp: ModuleType) -> None:
    backend = determine_backend_name(xp=xp)
    first = SequinceSMC._resolve_sampling_rng(
        backend, np.random.default_rng(1234)
    ).uniform(size=10)
    repeated = SequinceSMC._resolve_sampling_rng(
        backend, np.random.default_rng(1234)
    ).uniform(size=10)
    np.testing.assert_array_equal(to_numpy(first), to_numpy(repeated))
