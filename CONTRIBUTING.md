# Contributing to pjepa

Thank you for your interest in contributing to **Persistent-JEPA** (`pjepa`).
This document explains the workflow, coding standards, and review process.

## Quick Links

* [Quickstart for Developers](docs/developer/01_quickstart.md)
* [Architecture Overview](docs/developer/02_architecture.md)
* [Eight-Class Test Taxonomy](docs/developer/06_test_taxonomy.md)
* [Code of Conduct](#code-of-conduct)

## Workflow

1. **Find or open an issue.** Check the [issues](../../issues) for current priorities.
2. **Fork the repository** and create a branch off `master`:
   ```bash
   git checkout master
   git pull origin master
   git checkout -b feat/your-feature-name
   ```
3. **Make small, atomic commits.** Each commit should be one logical change.
   Do NOT bundle multiple features or refactors into a single commit.
4. **Update the CHANGELOG.** Every entry includes:
   - Commit SHA (short)
   - UTC date (`YYYY-MM-DD`)
   - Rationale — *why* the change was made
5. **Push to your fork** and open a Pull Request against `master`.
6. **Address review feedback** by pushing additional atomic commits.
7. **Squash-merge** once approved.

## Coding Standards

We follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) plus project-specific rules in [`docs/paper/`](../paper/paper.md). Every public symbol must:

- Have a Google-style docstring with `Args`, `Returns`, `Raises`, and `Example` sections.
- Be listed in the module's `__all__`.
- Have type hints (PEP 484).
- Pass `ruff check` and (eventually) `pytype --strict`.

Forbidden patterns:

- `print()` — use the structured logger (`pjepa.logging_setup.get_logger`).
- Bare `except:` — always specify the exception class.
- Mutable default arguments (`def f(x=[]): ...`).
- Wildcard imports (`from x import *`).
- `_underscore` prefixes unless the symbol is truly module-private.
- Comments in source code that don't explain *why* (we use docstrings for *what*).

## Test Taxonomy

Every public module must have tests in all eight classes:

| Class | Purpose |
|---|---|
| **happy** | Typical inputs produce expected outputs. |
| **bad** | Malformed inputs raise typed errors. |
| **ugly** | Edge cases (NaN, empty, single element) don't crash. |
| **leaky** | Long-running operations don't grow memory unbounded. |
| **round-trip** | Save → load → continue is equivalent to save → continue. |
| **cross-backend** | Same code on MPS/CUDA/CPU gives same output within tolerance. |
| **distributional** | Statistical properties hold across runs. |
| **property** | Hypothesis-driven invariants (submodularity, monotonicity, etc.). |

Coverage targets:

- **Core modules** (`objectives/`, `dynamics/`, `retrieval/`, `rewriting/`, `scheduler/`, `graphs/`): ≥ 90% line, ≥ 85% branch.
- **Encoders, training**: ≥ 85% line, ≥ 80% branch.
- **Auxiliary** (CLI, utils): ≥ 70% line.

## Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short summary>

<optional body explaining the why>

<optional footer>
```

Common types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`.

## Pull Request Checklist

Before requesting review, ensure:

- [ ] All new symbols have Google-style docstrings
- [ ] All new modules have eight-class test coverage
- [ ] `make lint` passes (0 ruff errors)
- [ ] `make test` passes (no test failures, no skipped tests)
- [ ] `make typecheck` passes (or has documented exceptions)
- [ ] CHANGELOG.md is updated with the new commits
- [ ] No `print` statements; no bare `except`; no wildcard imports
- [ ] Code follows Google Python Style

## Review Process

A maintainer reviews your PR within 3 business days. Reviewers check:

1. Correctness — does the code do what it claims?
2. Tests — are the eight classes covered?
3. Documentation — are docstrings complete and accurate?
4. Style — does it follow Google Python Style?
5. Architectural fit — does it follow the patterns established in
   [`docs/developer/02_architecture.md`](docs/developer/02_architecture.md)?
6. Backward compatibility — does it break the public API?

## Adding a New Module

When adding a new public module under `src/pjepa/<package>/`:

1. Define the module's class(es) following the
   [`Encoder` Protocol pattern](docs/developer/03_adding_an_encoder.md).
2. Add to the package's `__all__`.
3. Add tests in `tests/test_<module>.py` with eight-class coverage.
4. Add the module to `src/pjepa/<package>/__init__.py`.
5. Run `make lint && make test` and ensure both pass.

## Adding a New Baseline

See [`docs/developer/04_adding_a_baseline.md`](docs/developer/04_adding_a_baseline.md).

## Release Process

Releases follow Semantic Versioning. A release is cut when:

- All 12 phases of the implementation plan are complete.
- The CI matrix is green on Python 3.10, 3.11, and 3.12 across linux and macOS.
- All experiments reproduce within 2× bootstrap CI of the published numbers.

The maintainer tags the commit with `vX.Y.Z`, builds sdist + wheels via
`python -m build`, and publishes to GitHub Releases.

## Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/).
By participating, you agree to abide by its terms.

## Questions?

Open an issue or start a discussion. We're happy to help new contributors.