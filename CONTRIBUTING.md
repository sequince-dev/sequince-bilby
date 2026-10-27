# Contributing to sequince-bilby

Contributions are welcome, including bug reports, documentation, tests, and
improvements to the integration between Sequince and Bilby. These guidelines
follow the conventions used in Sequince, adapted to this repository's setup.

## Community and responsibility

Treat other contributors with respect and follow the
[Python Community Code of Conduct](https://www.python.org/psf/codeofconduct/).
Contributions must comply with the project's [license](LICENSE). Preserve required
attribution when incorporating third-party material.

AI-assisted contributions should follow
[Sequince's AI policy](https://github.com/sequince-dev/sequince/blob/main/AI_POLICY.md).
Contributors remain responsible for understanding, validating, and explaining
their changes, including their numerical correctness and performance.

## Scope of contributions

Keep each pull request focused on one problem or a closely related set of changes.
Discuss substantial API changes, new dependencies, or new algorithms in an issue
before investing significant implementation effort. Small fixes can go directly
to a pull request.

This package provides Bilby sampler plugins and adapters for Sequince. General
sampling algorithms belong in Sequince; application-specific likelihoods and
analysis workflows generally belong in downstream projects. Prefer existing
interfaces and helpers, and avoid unrelated cleanup.

## Development setup

Fork [sequince-dev/sequince-bilby](https://github.com/sequince-dev/sequince-bilby)
on GitHub, then clone your fork and configure local `main` to track upstream:

```bash
git clone git@github.com:YOUR_USERNAME/sequince-bilby.git
cd sequince-bilby
git remote add upstream https://github.com/sequince-dev/sequince-bilby.git
git fetch upstream
git switch main
git branch --set-upstream-to=upstream/main main
```

Keep local `main` for upstream changes and develop on separate branches. Before
starting each contribution:

```bash
git switch main
git pull --ff-only
git switch -c fix-descriptive-name
```

Here, `upstream` is the main repository and `origin` is your fork. Push contribution
branches explicitly to `origin`.

Use Pixi for development. Follow the
[Pixi installation instructions](https://pixi.prefix.dev/latest/installation/)
to install it first. Environments, dependencies, and tasks are defined in
`pyproject.toml`. The repository does not provide a lock file; Pixi resolves
dependencies and generates `pixi.lock` locally. Set up the default environment
and Git hooks:

```bash
pixi install
pixi run pre-commit install
```

The default environment uses Python 3.13, installs sequince-bilby in editable
mode, and includes pytest, HDF5 support, pre-commit, and Ruff. Source changes take
effect without reinstalling. Python 3.11 and 3.12 test environments are also
available. The `test`, `lint`, and `dev` dependency groups can be used with other
tools that support dependency groups.

Pixi installs Sequince from PyPI. To test against a local checkout, temporarily
add the following entry under `[tool.pixi.pypi-dependencies]` in `pyproject.toml`,
then run `pixi install`:

```toml
sequince = { path = "../sequince", editable = true }
```

Keep that local path override out of pull requests. When intentionally changing
dependencies, update `pyproject.toml` and run `pixi install`. Keep local
environments, generated lock files, caches, and analysis outputs out of pull
requests.

## Code style and typing

- Follow the existing module's conventions. Ruff handles formatting and import
  ordering; Python code uses four-space indentation and a 79-character line limit.
- Use descriptive names. Conventional mathematical names are appropriate when
  their meaning is clear locally.
- Document public functions and classes with NumPy-style docstrings, including
  types, array shapes, return values, and important assumptions. Keep docstrings
  consistent with annotations.
- Add parameter and return annotations to new functions and methods, including
  private helpers and tests where practical. Update affected callers when changing
  an interface. Use modern Python 3.11 syntax such as `list[str]` and `float | None`.
- Use `numpy.ndarray` for interfaces that require NumPy, as the Bilby adapters do.
  Preserve Sequince's backend-neutral interfaces where applicable; do not imply
  support for backends the adapter cannot accept.
- Give callbacks explicit signatures. Prefer precise unions or protocols to
  `Any` when the contract is known. Use `TYPE_CHECKING` for annotation-only imports
  when needed to avoid cycles or unnecessary runtime dependencies.
- Explain numerical reasoning and non-obvious compatibility choices in comments.
  Avoid comments that restate the code.
- Preserve parameter ordering, fixed parameters, prior support, likelihood-ratio
  semantics, normalized weights, and evidence conventions when adapting results.
- Use the supplied RNG and existing RNG adapters. Preserve generator state when
  interfacing with legacy Bilby APIs, including restoring temporary global state
  if sampling raises an exception. Legacy prior sampling uses process-global
  state; use process pools rather than concurrent threads for that path.
- Avoid unnecessary array copies, repeated likelihood evaluations, and retention
  of unused histories. Support performance claims with reproducible measurements.

There is currently no configured mypy check or standalone type-check task in this
repository. Annotations complement runtime and numerical tests.

## Checks and tests

Run these checks before requesting review:

```bash
pixi run pre-commit run --all-files
pixi run test
```

Pre-commit runs Ruff linting and formatting, whitespace checks, YAML validation,
and large-file checks. It does not run pytest. Review any automatic fixes and
rerun the checks. Configuration in `.pre-commit-config.yaml` and `pyproject.toml`
is authoritative. Do not bypass hooks to make a commit succeed.

Tests live in `tests/`. Follow the existing fixtures and parameterization. During
development, run the relevant file or test selection:

```bash
pixi run python -m pytest tests/test_proposal.py
pixi run python -m pytest tests/test_bilby_compatibility.py
pixi run python -m pytest tests/test_bilby_integration.py -k resume
```

Add regression tests for bug fixes and tests for new behavior. Test observable
outcomes and numerical invariants rather than implementation details. Use fixed
RNG seeds and justified numerical tolerances for stochastic tests.

Changes to prior or likelihood adapters should cover parameter ordering, prior
support, fixed parameters, likelihood ratios, and RNG behavior as applicable.
Changes to result conversion should check posterior sampling, particle weights,
and evidence estimates. Checkpoint changes should exercise save/resume behavior
for both `sequince-smc` and `sequince-ps`. Keep generated outputs in pytest's
temporary directories.

Standalone Ruff checks are available as `pixi run lint` and
`pixi run format-check`. To check other supported Python versions:

```bash
pixi run -e py311 test
pixi run -e py312 test
pixi run -e py313 test
```

For Bilby compatibility changes, test the released Bilby 2.x API using the pinned
environment:

```bash
pixi run -e bilby28 test
```

The `bilby28` environment constrains Bilby to `>=2.8,<3.0`. The default environment
uses the project's `bilby>=2.8.0.dev0` requirement without an upper bound. Also test
the development API described in [README.md](README.md) when relevant:

```bash
pixi run -e bilby-main test
pixi run -e bilby-main example-backends --backend jax
```

The optional `bilby-main` environment installs upstream Bilby `main` and JAX,
and enables Bilby's array API and JAX double precision before import. Optional
backend tests skip when their dependencies or Bilby APIs are unavailable.
PyTorch tests require a separate installation of PyTorch; the environment does
not install it. Report expected failures from known upstream bugs explicitly.

Report the Python, Bilby, and Sequince versions tested, commands and results, and
any relevant skips or environments that could not be tested. Skipped checks do
not verify compatibility.

GitHub Actions uses separate lint, test, and publish workflows, following
Sequince's layout. The lint workflow runs Ruff linting and formatting checks.
The test workflow runs the `py311`, `py312`, `py313`, `bilby28`, and `bilby-main`
environments on Linux for pull requests and pushes to `main` and `release*`
branches. Like Sequince, it uses `xarray-contrib/pixi-lock` to create and cache a
shared Pixi lock file, restores it for each test job, and runs with `--locked`.
The resolved lock file is also stored as an artifact.
The development Bilby environment includes JAX, but not PyTorch or GPU testing.
Publishing a GitHub release triggers the publish workflow. It builds the source
distribution and a wheel, validates metadata, and stores the distributions as
artifacts before publishing to PyPI using trusted publishing and the GitHub
`release` environment.

## Commit messages

Follow Sequince's uppercase SciPy-style prefixes for commit subjects and pull
request titles:

```text
PREFIX(scope): short imperative summary
PREFIX: short imperative summary
PREFIX/PREFIX(scope): short imperative summary
```

Use one prefix by default, or two separated by `/` when both describe the change.
The optional scope identifies an area such as `proposal`, `compatibility`, or
`checkpoint`. Start scopes with a lowercase letter and use lowercase letters,
digits, dots, underscores, or hyphens.

| Prefix | Use |
| --- | --- |
| `BUG` | Fix incorrect behavior |
| `ENH` | Add or extend functionality |
| `PERF` | Improve runtime or memory use without changing intended behavior |
| `API` | Change a public interface or introduce an incompatibility |
| `DEP` | Deprecate functionality |
| `MAINT` | Refactor or maintain code without changing intended behavior |
| `TST` | Add or improve tests |
| `DOC` | Update documentation |
| `BLD` | Change packaging, dependencies, or build configuration |
| `CI` | Change continuous integration workflows |
| `REL` | Prepare or maintain a release |
| `STY` | Make formatting-only changes |

Use `BUG` for bug fixes, including fixes with regression tests. Keep subjects
concise, preferably within 72 characters. Start the summary after `: ` with a
lowercase letter, use imperative wording, and omit a trailing period. Examples:

```text
BUG(proposal): restore the legacy RNG after failed prior draws
TST(checkpoint): cover persistent sampler resume
DOC: add contribution guidelines
```

When useful, add a body explaining the problem, resulting behavior, and important
tradeoffs. Reference related issues; use `Closes #123` only when fully resolving
that issue. Make commits in coherent units. These message conventions are not
currently enforced by a commit-message hook in this repository.

## Pull requests and review

Push your branch and open a pull request targeting
`sequince-dev/sequince-bilby:main`:

```bash
git push --set-upstream origin fix-descriptive-name
```

Use a draft for work in progress or early feedback. Explain the problem and
resulting behavior, link related issues, and describe relevant API changes,
numerical assumptions, and compatibility effects. Summarize validation and any
limitations; include reproducible benchmark conditions for performance changes.

Respond to reviewer feedback and keep the description aligned with the final
implementation. Reviewers should distinguish required changes from suggestions
and raise unrelated findings in separate issues. Maintainers should resolve
applicable checks and review discussions before merging, and ensure the final
commit title follows the conventions above.

## Releases

Releases are coordinated by maintainers. Versions are derived from Git tags by
`setuptools-scm`; do not maintain a separate hard-coded version number. Before
publishing, verify the intended commit and built distributions in a clean
environment, including sampler entry-point discovery and compatible Bilby and
Sequince versions. Release notes should describe user-visible changes and
incompatibilities. Published distributions cannot be overwritten, so verify tags
and artifacts before publishing.
