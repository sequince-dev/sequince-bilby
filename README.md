# sequince-bilby

Interface and plugin for using [sequince](https://github.com/sequince-dev/sequince)
with [bilby](https://github.com/bilby-dev/bilby).

The plugin provides the following samplers via bilby:

- `sequince-smc`: standard sequential Monte Carlo
- `sequince-ps`: persistent sampling

## Installation

Install `sequince-bilby` with pip (Python 3.11 or newer is required):

```bash
pip install sequince-bilby
```

For development setup, see the [contributing guidelines](CONTRIBUTING.md#development-setup).

## Usage

See the [Bilby documentation](https://bilby-dev.github.io/bilby/) for guidance on
defining likelihoods and priors, running analyses, and working with results.

Once installed, select a sampler with `bilby.run_sampler`:

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
forwarded to the corresponding Sequince prebuilt sampler. See the
[Bilby documentation](https://bilby-dev.github.io/bilby/) for likelihoods, priors,
and results.

### Parallelization

For NumPy sampling, set `npool` to the number of worker processes:

```python
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="sequince-smc",
    npool=4,
)
```

### Array backends

Sampling uses NumPy by default and supports released Bilby 2.8. JAX and PyTorch
require an array-aware development version of Bilby, the corresponding backend
package, and priors and likelihoods that support that backend.

Enable array API support before importing Bilby:

```bash
export BILBY_ARRAY_API=1
export SCIPY_ARRAY_API=1
export JAX_ENABLE_X64=1  # For JAX double precision.
```

Select the backend with `sampler_backend`:

```python
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="sequince-smc",
    sampler_backend="jax",
    npool=1,
    map_fn="vmap",
    jit=True,
    sampling_seed=1234,
)
```

Backend sampling requires `npool=1` without a pool. Use `map_fn="vmap"` for
vectorized likelihood evaluation; the likelihood must support the backend's
vectorization. For JAX, `jit=True` compiles the callback and requires traceable
priors and likelihoods. These options do not compile the full sampler loop.
Omit `map_fn` and `jit` to use the default mapping.

PyTorch support is experimental: the tested Bilby development version can fail
during prior normalization for particles outside prior support. CuPy is not
supported.

Run the [backend example](examples/bilby_array_backends.py) with the optional
`bilby-main` development environment:

```bash
pixi run -e bilby-main example-backends --backend jax
```

### Reproducibility

Use `sampling_seed` for reproducible runs on a given backend. You can also pass
a generator with `rng`. Without either, sampling uses Bilby's random generator.
Different backends use different random streams.

When resuming, use the same backend as the checkpoint. The saved RNG state takes
precedence over a new seed or generator.

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
`result.nested_samples`. The complete Sequince result is also saved as an HDF5
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
