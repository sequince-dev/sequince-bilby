# sequince-bilby

Interface and plugin for using [sequince](https://github.com/sequince-dev/sequince)
with [bilby](https://github.com/bilby-dev/bilby).

The plugin provides two bilby samplers:

- `sequince-smc`: standard sequential Monte Carlo
- `sequince-ps`: persistent sampling

## Installation

Install `sequince-bilby` with pip (Python 3.11 or newer is required):

```bash
pip install sequince-bilby
```

This also installs Sequince, Bilby and the other required dependencies.
For development setup, see the [contributing guidelines](CONTRIBUTING.md#development-setup).

## Usage

See the [Bilby documentation](https://bilby-dev.github.io/bilby/) for guidance on
defining likelihoods and priors, running analyses, and working with results.

Once installed, select either sampler with `bilby.run_sampler`:

```python
import bilby

likelihood = ...
priors = ...

result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="sequince-smc",
    n_particles=500,
)
```

Use `sampler="sequince-ps"` for persistent sampling. Keyword arguments are
forwarded to the corresponding sequince prebuilt sampler.

### Array backends and random numbers

Sampling uses NumPy by default. With an array-aware development version of
Bilby, select JAX or PyTorch using `sampler_backend="jax"` or
`sampler_backend="torch"`. Install the corresponding backend package and enable
Bilby's array API support before importing Bilby:

```bash
export BILBY_ARRAY_API=1
export SCIPY_ARRAY_API=1
# Enable double precision when using JAX.
export JAX_ENABLE_X64=1
```

```python
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="sequince-smc",
    sampler_backend="jax",
    sampling_seed=1234,
    npool=1,
)
```

The priors, likelihood, and any custom proposal must support the selected
backend. Likelihood data and waveform models must also be configured for that
backend where required by Bilby. Native backend sampling requires `npool=1`
without a pool. Likelihood mapping can use backend vectorization as described
below; the sampler loop remains eager. CuPy is not currently supported.
Bilby 2.8 remains supported for NumPy sampling.

PyTorch support is experimental: the tested development Bilby checkout raises a
`TypeError` in prior normalization when a mutation proposes particles outside
prior support (`torch.log` receives a Python normalization factor). This requires
a fix in Bilby before general PyTorch sampling can be relied upon.

`sampler_backend` selects the array backend. Pass a seed or generator using
`rng`, `random_state`, `seed`, `sampling_seed`, or `random_seed`; these aliases
are translated to `rng`. Supply only one non-`None` alias. Integer seeds create
a local generator on the selected backend. NumPy generators and matching ORNG
generators are used directly, preserving their current state. When a NumPy
generator is supplied for JAX or PyTorch sampling, one integer is consumed to
seed a backend-native generator. Other generator/backend mismatches are rejected.

Without an explicit RNG or seed, NumPy sampling uses Bilby's generator directly;
JAX and PyTorch sampling consume one integer from it to seed a native stream.
The default can therefore be seeded with `bilby.core.utils.random.seed(...)`.
Runs are reproducible within a backend; different backends use different random
streams. Bilby's initialization and likelihood code may also consume randomness.

Resume with the same backend as the checkpoint. The saved RNG state takes
precedence over any supplied RNG or seed. Bilby result
samples and weights are converted to NumPy for downstream processing.

#### Bilby 3 development example

The optional `bilby-main` Pixi environment installs Bilby from its upstream
`main` branch, adds JAX, and sets the array API and double-precision environment
variables. Run the [Gaussian location example](examples/bilby_array_backends.py)
with either backend and sampler:

```bash
pixi run -e bilby-main example-backends --backend jax
pixi run -e bilby-main example-backends --backend jax --map-fn vmap
pixi run -e bilby-main example-backends --backend jax --map-fn vmap --jit
pixi run -e bilby-main example-backends --backend numpy --sampler sequince-ps
pixi run -e bilby-main test
```

The example selects `vmap` for its traceable JAX likelihood by default. Pass
`--map-fn auto` to use the sampler's default mapping or `--map-fn map` to force
serial evaluation.

The example uses Bilby 3's `log_likelihood(parameters)` interface and compares
the sampled posterior mean with its analytic value. Each invocation creates a
fresh temporary output directory and prints its location. Use `--outdir PATH`
to select a fresh output directory yourself. `bilby-main` follows upstream
development and can expose upstream compatibility failures. Upstream currently
reports a `2.8.0.dev` version for this development API; the `bilby28` environment
remains constrained to released Bilby 2.8.

### Parallelization

By default, likelihood evaluation uses Bilby's pool map when a pool is available,
otherwise Python's `map`. For NumPy sampling, set `npool` to the number of worker
processes:

```python
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="sequince-smc",
    npool=4,
)
```

Override this choice with `map_fn`:

- `None` (default): infer the map from Bilby's pool.
- `"map"`: use Python's ordered, serial map, even if a pool exists.
- `"vmap"`: use `jax.vmap` or `torch.vmap` for the selected sampling backend.
- A callable: use an ordered `map_fn(function, particles)` directly. It receives
  a backend array of shape `(n_valid, n_parameters)` and must return exactly one
  scalar per row, in the same order. Custom worker maps must initialize Bilby's
  worker-local sampling context; unordered maps are unsupported.

For a traceable, side-effect-free JAX likelihood:

```python
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="sequince-smc",
    sampler_backend="jax",
    map_fn="vmap",
    sampling_seed=1234,
)
```

Vectorization is explicit because backend arrays alone do not establish that a
likelihood can be traced. Mapping errors propagate without a silent fallback.
Custom maps and Torch's `vmap` receive only particles with finite prior density,
and no call is made when none are valid. Vectorization does not compile the full
SMC loop.
Array-returning maps keep their results batched; iterator-returning maps remain
supported.

For JAX with `map_fn="vmap"`, the likelihood callback is traceable, including its
prior-support mask. It keeps the original batch shape by replacing invalid rows
with a valid row before vectorized evaluation, then restoring `-inf` outputs.
This can evaluate valid rows more than once, but invalid parameter values never
reach the likelihood at runtime. An all-invalid batch skips runtime likelihood
evaluation. Other mapping configurations use eager masking on the host.

Add `jit=True` to compile this JAX callback once per sampler run:

```python
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="sequince-smc",
    sampler_backend="jax",
    map_fn="vmap",
    jit=True,
    sampling_seed=1234,
)
```

Both the prior evaluation and likelihood must support JAX tracing. The callback
binds its sampling context when constructed; treat the likelihood, its data, and
priors as immutable for its lifetime. Changing only which particles are valid
does not change the batch shape or require recompilation. Different batch shapes
or input dtypes can still trigger compilation. The factory's JAX `"vmap"`
callback also supports wrapping with `jax.jit` directly. `jit=True` is rejected
for other backends or mapping options.

To compare serial mapping and `vmap`, with or without JIT compilation:

```bash
pixi run -e bilby-main python examples/benchmark_mapping.py
pixi run -e bilby-main python examples/benchmark_mapping.py --jit
```

The benchmark uses 64 particles and Gaussian data arrays of 32 and 4,096
observations. It reports first-call times and medians of three repeated timings
after an additional warm-up, synchronizes JAX outputs, and checks agreement of
likelihoods, weighted samples, and evidence. Sampler timings include initialization
and Sequince result writing, excluding imports. Every sampler timing constructs
a new sampler, so `--jit` includes compilation for that run. First-call timings
can reuse JAX caches populated by earlier cases. Use `--scope callback` or
`--scope sampler` to measure each separately; particle counts, data sizes, and
repetitions are configurable. Speedups depend on likelihood cost, batch size,
and device.

### Checkpointing

HDF5 checkpointing and resume are enabled by default. The default checkpoint is
`<outdir>/<label>_<sampler>_checkpoint.h5`.

```python
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="sequince-ps",
    checkpoint=True,
    checkpoint_filename="checkpoint.h5",
    checkpoint_frequency=10,
    resume=True,
)
```

Set `checkpoint=False` or `resume=False` to disable either behavior.

### Samples

`result.posterior` contains equal-weight samples obtained by rejection sampling.
The full particle population and its normalized `weights` are stored in
`result.nested_samples`. The complete sequince result is also saved as an HDF5
file in `outdir`.

## Citing

If you use `sequince-bilby` in scientific work, please cite Sequince and Bilby.
Follow [Bilby's citation guidance](https://bilby-dev.github.io/bilby/citing-bilby.html)
for the Bilby paper, the DOI for the version used, and any additional references
relevant to your analysis.

If you use `sequince-ps`, please also cite the
[persistent sampling paper](https://arxiv.org/abs/2407.20722). For
gravitational-wave analyses, please also cite
[arXiv:2506.18977](https://arxiv.org/abs/2506.18977).

## Contributing

Bug reports, feature requests, and pull requests are welcome via the
[GitHub repository](https://github.com/sequince-dev/sequince-bilby).
Please see the [contributing guidelines](CONTRIBUTING.md) and
[Sequince's AI policy](https://github.com/sequince-dev/sequince/blob/main/AI_POLICY.md).

## Related repositories

The following repositories are part of the `sequince` ecosystem:

- [`sequince`](https://github.com/sequince-dev/sequince): provides modular
  sequential Monte Carlo samplers and reusable sampling components
- [`sequince-density`](https://github.com/sequince-dev/sequince-density): provides
  density estimators for preconditioned SMC and ASPIRE-style analyses
- [`sequince-examples`](https://github.com/sequince-dev/sequince-examples): provides
  examples demonstrating Sequince's features
- [`orng`](https://github.com/sequince-dev/orng): provides cross-backend random
  number generation
- [`acai`](https://github.com/sequince-dev/acai): provides array-API compatible
  control flow and iteration

## License

Licensed under the [MIT License](LICENSE).
