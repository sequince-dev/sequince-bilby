# Changelog

Notable changes to sequince-bilby are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.0b1] - 2026-10-28

First public beta release.

### Added

- Bilby sampler plugins: `sequince-smc` for sequential Monte Carlo and
  `sequince-ps` for persistent sampling.
- HDF5 checkpoint saving and resuming.
- Equal-weight posterior samples, weighted particles, and evidence estimates
  in Bilby results.
- NumPy sampling with released Bilby 2.8, including process-pool support.
- JAX sampling with vectorized and JIT-compiled likelihood evaluation using
  array-aware development Bilby. PyTorch support is experimental.

[Unreleased]: https://github.com/sequince-dev/sequince-bilby/compare/v0.1.0b1...HEAD
[0.1.0b1]: https://github.com/sequince-dev/sequince-bilby/releases/tag/v0.1.0b1
