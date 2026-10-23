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


@pytest.fixture()
def mapping_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sampling_convenience_dump, "priors", ArrayPriors())
    monkeypatch.setattr(
        sampling_convenience_dump, "parameters", {"offset": 0.2}
    )
    monkeypatch.setattr(
        sampling_convenience_dump, "search_parameter_keys", ["x"]
    )
    monkeypatch.setattr(sampling_convenience_dump, "use_ratio", True)
    monkeypatch.setattr(
        sampling_convenience_dump, "likelihood", GaussianLikelihood()
    )


@pytest.mark.parametrize(
    "values",
    [[0.0, -0.5], [2.0, 0.0, -2.0, -0.5, 2.0], [2.0, -2.0], []],
)
@pytest.mark.parametrize("mapping", ["custom", "array", "vmap"])
def test_likelihood_mapping(
    xp: ModuleType, mapping_context: None, values: list[float], mapping: str
) -> None:
    backend = determine_backend_name(xp=xp)
    if mapping == "vmap" and backend == "numpy":
        pytest.skip("NumPy does not provide vmap")
    calls = []

    def custom_map(function, batch):
        assert determine_backend_name(batch) == backend
        calls.append(batch.shape)
        if mapping == "array":
            return xp.stack([function(row) for row in batch])
        return map(function, batch)

    mapper = "vmap" if mapping == "vmap" else custom_map
    particles = xp.reshape(xp.asarray(values, dtype=xp.float64), (-1, 1))
    actual = get_sequince_functions(
        map_fn=mapper, backend=backend
    ).log_likelihood(particles)
    expected = [
        -0.5 * (value - 0.2) ** 2 - 2 if abs(value) <= 1 else -np.inf
        for value in values
    ]
    np.testing.assert_allclose(to_numpy(actual), expected)
    assert determine_backend_name(actual) == backend
    if mapping != "vmap":
        valid = sum(abs(value) <= 1 for value in values)
        assert calls == ([(valid, 1)] if valid else [])


@pytest.mark.parametrize(
    "output", [[], [0.0], [0.0, 0.0, 0.0], [[0.0], [0.0]]]
)
def test_map_must_return_one_scalar_per_particle(
    mapping_context: None, output: list
) -> None:
    def bad_map(function, batch):
        return output

    functions = get_sequince_functions(map_fn=bad_map)
    with pytest.raises(ValueError, match="one scalar per particle"):
        functions.log_likelihood(np.array([[0.0], [0.5]]))


def test_map_errors_propagate(mapping_context: None) -> None:
    def failed_map(function, batch):
        raise RuntimeError("mapping failed")

    with pytest.raises(RuntimeError, match="mapping failed"):
        get_sequince_functions(map_fn=failed_map).log_likelihood(
            np.array([[0.0]])
        )


@pytest.mark.parametrize("invalid", ["unknown", 42, False])
def test_invalid_map_option(invalid) -> None:
    from sequince_bilby.utils import resolve_map_fn

    with pytest.raises(ValueError, match="map_fn must be"):
        resolve_map_fn(invalid, backend="numpy")


def test_numpy_rejects_vmap() -> None:
    from sequince_bilby.utils import resolve_map_fn

    with pytest.raises(ValueError, match="requires the jax or torch"):
        resolve_map_fn("vmap", backend="numpy")


@pytest.mark.parametrize("shape", [(), (2, 1), (1,), (3,)])
def test_array_map_requires_scalar_per_particle(
    xp: ModuleType, mapping_context: None, shape: tuple[int, ...]
) -> None:
    def bad_map(function, batch):
        return xp.zeros(shape, dtype=xp.float64)

    functions = get_sequince_functions(map_fn=bad_map)
    with pytest.raises(ValueError, match="one scalar per particle"):
        functions.log_likelihood(xp.asarray([[0.0], [0.5]]))


@pytest.mark.parametrize("mapping", ["custom", "vmap"])
def test_batched_mapping_parameter_order_and_dtype(
    xp: ModuleType,
    mapping_context: None,
    monkeypatch: pytest.MonkeyPatch,
    mapping: str,
) -> None:
    backend = determine_backend_name(xp=xp)
    if mapping == "vmap" and backend == "numpy":
        pytest.skip("NumPy does not provide vmap")

    class OrderedLikelihood(GaussianLikelihood):
        def log_likelihood(self, parameters: dict | None = None):
            residual = (
                parameters["z"] + 3 * parameters["x"] - parameters["offset"]
            )
            return -0.5 * residual**2

    monkeypatch.setattr(
        sampling_convenience_dump, "search_parameter_keys", ["z", "x"]
    )
    monkeypatch.setattr(
        sampling_convenience_dump, "likelihood", OrderedLikelihood()
    )

    def array_map(function, batch):
        return xp.stack([function(row) for row in batch])

    mapper = "vmap" if mapping == "vmap" else array_map
    particles = xp.asarray(
        [[0.2, 0.3], [0.4, 2.0], [0.6, -0.5]], dtype=xp.float64
    )
    actual = get_sequince_functions(
        map_fn=mapper, backend=backend, likelihood_dtype="float32"
    ).log_likelihood(particles)
    np.testing.assert_allclose(
        to_numpy(actual), [-2.405, -np.inf, -2.605], rtol=1e-6
    )
    assert actual.dtype == xp.float32
    assert determine_backend_name(actual) == backend


@pytest.mark.parametrize("compile_in_factory", [False, True])
def test_jitted_likelihood_support_and_reuse(
    mapping_context: None,
    monkeypatch: pytest.MonkeyPatch,
    compile_in_factory: bool,
) -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    traces = []
    evaluated = []

    def record(value) -> None:
        value = float(value)
        assert np.isfinite(value) and abs(value) <= 1
        evaluated.append(value)

    class RecordingLikelihood(GaussianLikelihood):
        def log_likelihood(self, parameters: dict | None = None):
            traces.append(None)
            jax.debug.callback(record, parameters["x"])
            return super().log_likelihood(parameters)

    monkeypatch.setattr(
        sampling_convenience_dump, "likelihood", RecordingLikelihood()
    )
    with jax.enable_x64():
        functions = get_sequince_functions(
            map_fn="vmap", backend="jax", jit=compile_in_factory
        )
        likelihood = functions.log_likelihood
        if not compile_in_factory:
            likelihood = jax.jit(likelihood)
        batches = [
            [2.0, 0.0, np.nan, -0.5],
            [-0.2, 0.1, 0.2, 0.3],
            [2.0, -2.0, np.nan, np.inf],
            [0.5, 2.0, 2.0, 2.0],
        ]
        trace_count = None
        for batch in batches:
            before = len(evaluated)
            actual = likelihood(
                jnp.asarray(batch)[:, None]
            ).block_until_ready()
            expected = [
                -0.5 * (x - 0.2) ** 2 - 2 if abs(x) <= 1 else -np.inf
                for x in batch
            ]
            np.testing.assert_allclose(actual, expected)
            if trace_count is None:
                trace_count = len(traces)
                assert trace_count > 0
            else:
                assert len(traces) == trace_count
            if not any(abs(x) <= 1 for x in batch):
                assert len(evaluated) == before
        assert likelihood(jnp.empty((0, 1))).shape == (0,)


def test_jitted_factories_bind_separate_contexts(
    mapping_context: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    with jax.enable_x64():
        first = get_sequince_functions(map_fn="vmap", backend="jax", jit=True)
        monkeypatch.setattr(
            sampling_convenience_dump, "parameters", {"offset": 0.8}
        )
        second = get_sequince_functions(map_fn="vmap", backend="jax", jit=True)
        # First trace occurs after globals have changed. Each callback must use
        # its own bound context, including when tracing a new batch shape.
        for values in ([0.0, 0.5], [-0.5]):
            x = jnp.asarray(values)[:, None]
            np.testing.assert_allclose(
                first.log_likelihood(x),
                -0.5 * (np.array(values) - 0.2) ** 2 - 2,
            )
            np.testing.assert_allclose(
                second.log_likelihood(x),
                -0.5 * (np.array(values) - 0.8) ** 2 - 2,
            )


@pytest.mark.parametrize(
    "backend,mapping", [("numpy", "map"), ("torch", "vmap"), ("jax", "map")]
)
def test_reject_unsupported_jit(backend: str, mapping: str) -> None:
    with pytest.raises(ValueError, match="jit=True requires"):
        get_sequince_functions(map_fn=mapping, backend=backend, jit=True)


@pytest.mark.parametrize(
    "sampler_class", [SequinceSMC, SequincePersistentSampling]
)
@pytest.mark.parametrize("jit", [False, True])
def test_vectorized_sampler(
    xp: ModuleType, sampler_class: type[SequinceSMC], tmp_path: Path, jit: bool
) -> None:
    backend = determine_backend_name(xp=xp)
    if backend == "numpy" or not getattr(random, "BILBY_ARRAY_API", False):
        pytest.skip("Requires JAX or Torch and array-aware Bilby")
    if jit and backend != "jax":
        pytest.skip("JIT compilation is supported only for JAX")
    options = dict(
        likelihood=GaussianLikelihood(),
        priors=bilby.core.prior.PriorDict(
            {"x": bilby.core.prior.Uniform(-5, 5), "offset": 0.3}
        ),
        outdir=tmp_path,
        sampler_backend=backend,
        rng=1234,
        n_particles=64,
        mutation="none",
        checkpoint=False,
        resume=False,
        logging_frequency=0,
    )
    expected = sampler_class(**options, label="serial").run_sampler()
    actual = sampler_class(
        **options, label="vectorized", map_fn="vmap", jit=jit
    ).run_sampler()
    np.testing.assert_allclose(
        actual.nested_samples.to_numpy(), expected.nested_samples.to_numpy()
    )
    assert actual.log_evidence == pytest.approx(expected.log_evidence)
