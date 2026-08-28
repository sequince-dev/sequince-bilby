# sequince-bilby

Interface and plugin for using [sequince](https://github.com/sequince-dev/sequince)
with [bilby](https://git.ligo.org/lscsoft/bilby).

The plugin provides two bilby samplers:

- `sequince-smc`: standard sequential Monte Carlo
- `sequince-ps`: persistent sampling

## Installation

Bilby 3.0 has not yet been released and must currently be installed from
source. Then install `sequince-bilby` from GitHub:

```bash
pip install git+https://github.com/bilby-dev/bilby.git
pip install git+https://github.com/sequince-dev/sequince-bilby.git
```

Python 3.11 or newer and bilby 3.0 or newer are required.

## Usage

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

### Parallelization

Likelihood evaluations support parallelization through bilby's pool. Set
`npool` to the number of worker processes:

```python
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="sequince-smc",
    npool=4,
)
```

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

If you use `sequince-bilby`, please cite sequince and bilby.

If you use `sequince-ps`, please also cite the
[persistent sampling paper](https://arxiv.org/abs/2407.20722). For
gravitational-wave analyses, please also cite
[arXiv:2506.18977](https://arxiv.org/abs/2506.18977).
