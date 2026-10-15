# Agent instructions

Before making changes, read and follow [CONTRIBUTING.md](CONTRIBUTING.md) and
[Sequince's AI policy](https://github.com/sequince-dev/sequince/blob/main/AI_POLICY.md).
CONTRIBUTING.md defines the development setup, code style, typing, testing,
commit-message, and pull request conventions.

## Working in the repository

- Inspect the working tree before editing. Preserve existing work and do not
  overwrite or revert unrelated changes. When committing, stage only files
  relevant to the requested task.
- Keep changes focused. Follow existing interfaces and helpers, and avoid
  unrelated refactors. Add dependencies or abstractions only when they address
  a concrete need within the task.
- Keep generated lock files, local environments, caches, checkpoints, and analysis
  outputs out of commits unless explicitly requested. Preserve existing local
  files.
- Keep general sampling algorithms in Sequince and Bilby integration logic here.
  Preserve the `sequince-smc` and `sequince-ps` sampler entry points.

## Scientific correctness and validation

- Identify changes that affect target distributions, parameter ordering, prior
  support, likelihood ratios, posterior weights, evidence estimates, RNG behavior,
  or checkpoint compatibility. Explain intentional behavior changes and support
  them with appropriate tests.
- Preserve compatibility with the Bilby APIs described in README.md. Use the
  existing likelihood and prior adapters and supplied RNG. Restore temporary
  global RNG state even when legacy prior sampling raises an exception; that
  legacy path requires process pools rather than concurrent threads.
- Follow CONTRIBUTING.md for validation. Report the checks actually run and their
  results, including failures, skipped checks, and Bilby or Sequince versions that
  could not be tested. Do not describe skipped checks as passing.
- Do not bypass hooks or disable checks to make a commit succeed. Address the
  failure, or report the blocker when it cannot be resolved within the task.
