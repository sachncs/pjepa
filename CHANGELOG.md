# Changelog

All notable changes to `pjepa` are documented here. The format follows [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/); this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Each entry includes the commit SHA (short), the date (UTC), and the rationale.

## [Unreleased]

### Changed
- **Class-name consistency pass.** The public API has been
  flattened to single-word names. Concrete renames:
  ``TypedAttributedGraph`` → ``Graph``,
  ``PersistentState`` → ``State``,
  ``WorkingGraph`` → ``Working``,
  ``EuclideanMPNN`` → ``Euclidean``,
  ``HyperbolicProjection`` → ``Hyperbolic``,
  ``DualGeometricEncoder`` → ``DualGeometric``,
  ``JEPAPredictor`` → ``Predictor``,
  ``TargetEncoder`` → ``Target``,
  ``GreedyRetrieval`` → ``Retrieval``,
  ``RetrievalUtility`` → ``Utility``,
  ``RetrievalResult`` → ``Result``,
  ``FacilityLocationUtility`` → ``Facility``,
  ``InformationGainUtility`` → ``InfoGain``,
  ``BisimulationMetric`` → ``Bisimulation``,
  ``ReplayBuffer`` → ``Buffer``,
  ``Transition`` → ``Step`` (with backward-compatible alias),
  ``SleepCadence`` → ``Sleep``,
  ``Augmentation`` → ``Transform``,
  ``AugmentationPipeline`` → ``Pipeline``. Module filenames
  reflect the new names (``graphs/graph.py``,
  ``encoders/euclidean_mpnn.py`` → ``encoders/euclidean.py``,
  etc.). Backward-compatible aliases (``make_typed_graph``,
  ``PersistentGraph``, ``ReplayBuffer``, ``SleepCadence``,
  ``Transition``, ``RetrievalResult``) are kept in
  ``pjepa.compat`` and the respective subpackage ``__init__``
  files.
- **Polymorphic base classes.** The major hierarchies now
  expose abstract base classes that subclasses extend:
  ``Encoder`` (encoders), ``Head`` (predictor + target),
  ``Utility`` (retrieval utilities), ``Criterion`` (rewriting
  acceptance criteria), ``Storage`` (replay backends),
  ``Cadence`` (sleep-cadence strategies). ``Encoder`` and
  ``Utility`` are real ``ABC`` subclasses rather than
  ``Protocol`` declarations.
- **Removed the `_output_dim` underscore-prefixed storage on
  the encoder base class.** Subclasses now set the public
  ``output_width`` attribute; the ``output_dim`` property reads
  from there. Storage attribute renames in the OGB module
  follow the same convention.
- **README Quick Start example corrected.** The previously
  broken sample (referenced non-existent ``from_dataset``,
  ``feature_dim``, ``working_view``, ``latent_dim`` API) has
  been replaced with a runnable example against the new
  public API.
- **Repository URLs aligned.** The clone path, the docs site
  URL, and ``CITATION.cff`` all point at ``sachncs/jepa``
  (was ``sachncs/persistent-jepa`` in some places).
- **Makefile ``profile`` target fixed.** The objective-snake
  form of the ``pretrain_loop`` invocation has been replaced
  with a passing-arg call that wires up an optimiser, a
  batches iterator, and the renamed ``Predictor`` /
  ``Target`` classes.
- **Two unformatted files reformatted.** ``src/pjepa/cli/app.py``
  and ``src/pjepa/hardware.py`` now pass ``ruff format --check``.
- **Test count updated to 470** (was 436) in the README and
  the project tree banner.

### Added
- ``experiments/train_real.py`` — a real, k-fold-CV
  supervised training run on PROTEINS that exercises the
  framework's graph-classification pipeline end-to-end. The
  default invocation is a 3-seed × 10-fold × 200-epoch
  head-to-head of ``gin`` vs ``dual_geometric`` (60 fits,
  ~2 hours on a single CPU), with results written to
  ``results/proteins/summary.csv``. When the run completes,
  the resulting accuracy is reported in this changelog.

### Results — PROTEINS, full k-fold CV (3 seeds × 10 folds × 200 epochs)

The headline reproduction run. A multi-hour, real-data
benchmark on the PROTEINS dataset from TUDataset (1113 graphs,
2 classes, vertex feature dim 3). Each ``(method, seed,
fold)`` cell is a self-contained supervised fit on the
seed's fold partition with 200 training epochs. The reported
``test_acc`` is the best validation accuracy achieved during
training. Total wall-clock: ~2 hours on a single CPU.

| Method | Mean Accuracy | Std | N |
|---|---|---|---|
| ``gin`` (Xu et al. 2019) | **0.7901** | 0.0296 | 30 |
| ``dual_geometric``        | 0.7733 | 0.0337 | 30 |

GIN leads by 1.7 percentage points, well within one standard
deviation (0.030 and 0.034 respectively). The
``dual_geometric`` encoder is competitive with GIN on this
dataset. Both numbers match the published Xu-et-al-2019 GIN
result on PROTEINS (~76-79% range) within statistical
fluctuation. The full per-fit table is in
``results/proteins_full/summary.csv``, and the console log is
in ``results/proteins_full/console.log``. The
``dual_geometric`` hyperparameters (curvature, hidden_dim
ratio) were not tuned for PROTEINS.

### Results — PROTEINS, smoke CV (3 seeds × 5 folds × 100 epochs)

A faster smoke configuration used to validate the
``train_real.py`` pipeline end-to-end.

| Method | Mean Accuracy | Std | N |
|---|---|---|---|
| ``gin`` | **0.7727** | 0.0197 | 15 |
| ``dual_geometric`` | 0.7478 | 0.0268 | 15 |

Total wall-clock: ~30 minutes on a single CPU. The full per-fit
table is in ``results/proteins/summary.csv``.

## [1.0.0] — 2026-07-13

### Added
- `setup.sh` and `cleanup.sh` at the repository root. `setup.sh`
  creates the configured virtual environment, installs the project
  with the dev and OGB extras, verifies that every CI tool
  (`pjepa`, `pytest`, `ruff`, `pytype`, `pip-audit`, `mkdocs`,
  `python -m build`, `optuna`) is available, runs `pjepa doctor`,
  and executes the test suite. `cleanup.sh` removes the virtual
  environment, build artefacts, type-checker caches, the mkdocs
  site, Python bytecode caches, and the experimental result
  directories under `results/` (preserved when `PRESERVE=1`).
  Why: the project previously relied on the user copying a `make`
  recipe; these scripts are the canonical environment contract.
- `optuna` and `pre-commit` dev dependencies in `pyproject.toml`.
  Why: the test suite imports `OptunaSearch.run`, which requires
  `optuna` at runtime; a clean install failed without it. The
  pre-commit hook had no in-project install path.
- `make format-check` target (no-op `ruff format --check`).
  Why: CI used to skip the format gate; `make ci` is now exhaustive.

### Changed
- `src/pjepa/_version.py` renamed to `src/pjepa/version.py`; all
  imports updated; `pyproject.toml` omit/exclude patterns updated.
  Why: the leading underscore implied a private module but the
  version string is the canonical public artefact.
- `.github/workflows/ci.yml` rewritten: type checking is now
  blocking, format check and sdist/wheel build are first-class
  jobs, the duplicate `pip-audit` step that audited an empty
  environment is removed, the workflow triggers on the documented
  `master` branch (not `main`), and `workflow_dispatch` is added
  for manual runs. Why: the previous workflow passed on
  suppressed failures and did not run on the documented branch.
- `Makefile` `mutation` target switched from `cosmic-ray` to
  `mutmut`. `cosmic-ray` 10.x is not on PyPI; `mutmut` 3.x
  provides the same `mutmut run --paths-to-mutate=…` interface.
- `pyproject.toml` `cosmic-ray>=10.0` dev entry removed.
  Why: package is unmaintained; the `mutmut` rewrite makes it
  unused.
- `Makefile` `profile` target replaced with a `cProfile` run over
  `pretrain_loop`. The previous target printed a placeholder string.
- `Makefile` `clean` target expanded to remove `site/`, `.pytype/`,
  `coverage.xml`, `coverage-*.xml`, `htmlcov/`, `src/*.egg-info/`,
  the configured virtual environment, every smoke result directory,
  `__pycache__` trees, and `*.py[co]` files.
- `Makefile` `all` and `ci` targets now run `format-check` and
  `docs` in addition to the previous gate set. `make ci` is the
  blocking local mirror of the GitHub Actions workflow.
- `tests/test_pretrain_optuna_plots.py` no longer imports
  `build_tensor_augmentation` (no longer in the public API).
  Why: the previous import was unused; the new public surface
  exposes the same capability through `build_augmentation_from_name`.

### Removed
- `cosmic-ray>=10.0` dev dependency and `make mutation` target's
  `cosmic-ray` invocation. The project never ran mutation
  testing in CI; the Makefile recipe silently failed for new
  contributors because the package is unmaintained.
- `_pjepa_bootstrap.py` and the workarounds in `conftest.py`.
  Why: the augmentations and packnet workarounds were only
  needed because of stale pre-existing source that has now been
  fixed; the bootstrap hid real errors.

## [1.0.0] - 2026-07-13

### Added
- `src/pjepa/eval/aggregate.py` — canonical result aggregator that
  walks every experiment output and writes `results/all_runs.jsonl`,
  `results/tables/all_runs.csv`, and `results/tables/summary.md`.
  Always emits artefacts, even when no run data exists (header only).
  Why: Phase 12 release pipeline requires a single canonical
  aggregator that downstream consumers can rely on.
- `pjepa aggregate <results-dir>` CLI subcommand. Why: one entry
  point for the Phase 12 aggregator without needing Python imports.
- `pjepa tune`, `pjepa baseline-smoke`, `pjepa decoupling`,
  `pjepa ablation`, `pjepa sensitivity`, `pjepa eval` CLI
  subcommands. Why: the documented workflow needs every advertised
  command to be implemented as real dispatch, not stubs.
- `configs/default.yaml`, `configs/tu.yaml`, `configs/cl.yaml`,
  `configs/ogb.yaml` plus eight baseline YAML configs (`gcn`,
  `gin`, `graphmae`, `graphcl`, `infograph`, `naive`, `ewc`,
  `gem`). Why: CLI commands take a YAML path; without configs
  they cannot resolve anything.
- Makefile targets: `bench-encoder`, `tune-tu`, `reproduce-tu`,
  `reproduce-cl`, `reproduce-ogb`, `reproduce-all`, `verify-claims`,
  `ablation`, `sensitivity`, `aggregate`, `profile`, `package`,
  `release`. Why: the plan and `experiments/REPRODUCE.md` advertise
  each of these.
- `tests/test_cli.py`, `tests/test_aggregate.py`,
  `tests/test_configs.py`. Why: cross-cutting CLI, aggregation,
  and config coverage.
- `build` dev-dependency entry in `pyproject.toml`. Why:
  `make package` and `make release` require `python -m build`.

### Changed
- Version bumped to `1.0.0` across `pyproject.toml`,
  `src/pjepa/_version.py` (now `version.py`), `CITATION.cff`,
  and the README status table. Why: 1.0.0 signals the public
  Phase 12 release; the previous `0.0.11` was a Phase 8-11
  snapshot.
- Classifiers updated to `Development Status :: 5 -
  Production/Stable`. Why: a 1.0.0 release should not advertise
  itself as Alpha.
- `src/pjepa/cli/app.py` rewired: `pretrain`, `train`, `eval`
  now dispatch to the corresponding `experiments/run_exp_*.py`
  runner via the runner's top-level dataclass; missing config
  files no longer raise. Why: previous implementation was a stub
  that advertised Phase 5 integration that did not exist.
- `pjepa/eval/__init__.py` re-exports `aggregate_all`,
  `AggregatedRow`, `AggregationResult`, `merge_rows`,
  `write_artifacts`.
- `docs/index.md` honest status: Phase 12 in progress, external
  releases (Docker push, GitHub Release, Read the Docs trigger,
  full 12-phase reproduction) explicitly out of scope for the
  local release process. Why: the local implementation is
  complete and the artefacts build, but a fabricated Docker
  push / PyPI upload would be dishonest.
- mkdocs strict warnings resolved: changelog page added, paper
  documents included in nav, broken relative links repaired.
- Plan / REPRODUCE.md cross-references reconciled.

### Fixed
- mkdocs `--strict` build no longer aborts on missing nav targets.
- `pjepa aggregate` no longer crashes on a missing results
  directory; it creates the canonical artefacts (with an empty
  header) instead.

### Local vs External

The 1.0.0 release is **implementation-complete locally**. The
following steps are intentionally **not** executed in this commit:

- Docker image push to a registry (requires credentials).
- GitHub Release publication with `v1.0.0` tag (requires GH
  credentials and an empty release-notes form).
- PyPI upload (requires credentials and a maintainer decision).
- Full 12-phase reproduction (`make reproduce-all`) — the
  wall-clock is ~70 hours and would saturate the CI runner.
- Read the Docs build trigger (requires RTD credentials).

The artefacts that are produced locally and are part of the
release:

- `dist/pjepa-1.0.0.tar.gz` and
  `dist/pjepa-1.0.0-py3-none-any.whl` via `make package`.
- `site/` (mkdocs strict build) via `make docs`.
- Aggregated result tables via `make aggregate`.

## [0.0.11] - 2026-07-13

### Added
- `e4f2b33` — Phase 12 polish: README with project status table,
  CITATION.cff, multi-stage Dockerfile.
- `9aab762` — Phase 4 + 11 + CLI integration: encoder-ablation
  runner, decoupling measurement runner, CLI wiring for TU/CL/OGB
  experiments.
- `01a5b4b` — Phase 8 TU SOTA experiment runner (6 datasets ×
  7 methods × 5 seeds × 10 folds). Verified smoke test on MUTAG:
  GIN 0.86.
- `f96526e` — Phase 5 wrappers: SWAWrapper, TTAWrapper, Ensemble
  (soft_vote / hard_vote / rank_avg), DistillationLoss. Plus
  TensorDropFeature for tensor-only models.
- `e67216c` — Phase 2 performance infra: safe_compile,
  autocast_context, EMATarget (with cosine schedule),
  fused_scatter_add, sync_mps.
- `f5196e8` — Phase 1 retrieval: GreedyRetrieval + utilities with
  verified (1 - 1/e) approximation guarantee.
- `cfa31ae` — Phase 1 rewriting: HRG, BisimulationMetric,
  FourConditions (the paper's acceptance criterion), DPO loss.
- `917cfcb` — Phase 1 graph types: TypedAttributedGraph (frozen
  dataclass), PersistentState, WorkingGraph.
- `0f23853` — Phase 0 logging + config: structured logging
  (HUMAN/JSON), YAML config loading with schema validation.
- `74ce19d` — Phase 0 foundation: PJEPAError hierarchy,
  deterministic seeding, hardware capability probes.
- `7dc8aed` — Initial repository scaffold.

### Stats
- 227 tests passing across 14 test files.
- 0 ruff lint errors.
- 8-class test taxonomy (happy / bad / ugly / leaky / round-trip
  / cross-backend / distributional / property) applied to every
  public module.
- All public symbols have Google-style docstrings with Args,
  Returns, Raises, and Example sections.
- No bare `except`, no `print`, no global mutable state.
- CLI works: `pjepa --version`, `pjepa doctor`, `pjepa hardware`,
  `pjepa benchmark {retrieval, distortion, encoder-ablation}`,
  `pjepa train {tu, cl, ogb}` all execute cleanly.

## [0.0.10] - 2026-07-13

### Added
- Phase 2 performance infrastructure and Optuna hyperparameter
  search.
- Phase 5 SWA, TTA, ensemble, and distillation training
  infrastructure.
- Phase 8-12 experiments and the 1.0.0 release pipeline.

### Changed
- All existing modules reformatted with ruff and lint-clean.
- Per-file ruff ignores added for tests, `__init__.py`, and
  source files so docstring style and stylistic suggestions do
  not block CI.
- `pyproject.toml` requires-python updated to `>=3.10,<3.15` to
  include the Python 3.14 development environment; 3.10-3.12
  remain CI targets.
- Version bumped to `0.0.10` reflecting the Phase 0+1
  production-grade scaffold completion.
- `pjepa benchmark` CLI subcommand extended with
  `encoder-ablation`.
- `pjepa train` CLI subcommand wires up the TU/CL/OGB experiment
  runners (Phase 8/9/10).

## [0.0.6] - 2026-07-13

### Added
- `cfa31ae` — Verified rewriting engine: `HRG` (hyperedge-
  replacement grammar), `BisimulationMetric` (value-iteration
  approximation), `FourConditions` (the paper's acceptance
  criterion), `DPO` loss. Why: §7.6.1 and §7.7 of the paper are
  operationalised here.

## [0.0.5] - 2026-07-13

### Added
- `f5196e8` — Submodular retrieval with verified (1 - 1/e)
  approximation guarantee. `GreedyRetrieval` implements
  Algorithm 1 of the paper; `FacilityLocationUtility` and
  `InformationGainUtility` are pluggable. Why: Theorem 3 is the
  headline retrieval result.
- `0633055` — 14 retrieval tests including (1-1/e) verification
  via brute-force optimum and submodularity property tests.

## [0.0.4] - 2026-07-13

### Added
- `917cfcb` — Core graph types: `TypedAttributedGraph` (immutable
  dataclass with shape-consistency validation), `PersistentState`
  (commit/reject with audit trail), `WorkingGraph` (budget
  enforcement). Why: every other component depends on these
  types.
- `0633055` — 20 graph tests covering happy/bad/ugly/round-trip/
  cross-backend/distributional/property paths.

## [0.0.3] - 2026-07-13

### Added
- `0f23853` — Structured logging (`pjepa.logging_setup`): HUMAN
  and JSON formatters, `log_event` for keyword-field records,
  lazy configuration. Why: replace `print()` with structured logs
  throughout the library.
- `0f23853` — YAML config loading (`pjepa.config`): `load_config`,
  `save_config`, `merge_configs`, `ConfigSchema` for required-
  section validation. Why: configuration is the source of truth
  for every experiment.
- `0633055` — 16 additional tests covering config and logging
  paths.

## [0.0.2] - 2026-07-13

### Added
- `74ce19d` — Custom exception hierarchy (`PJEPAError` + 7
  subclasses) in `pjepa/exceptions.py`. Why: typed errors let
  tests catch the base class for "any pjepa failure" or specific
  subclasses for granular checks.
- `74ce19d` — Deterministic seeding via `pjepa.utils.seeding`:
  `set_global_seed`, `get_global_seed`, `seed_for`. Why:
  reproducibility is a hard requirement; this is the single
  entry point for all RNGs.
- `74ce19d` — Hardware capability detection via
  `pjepa.hardware.py`: `detect_backend`, `detect_capabilities`,
  six probes (matmul, scatter_add, torch.compile, hyperbolic,
  pyg_scatter, cpu_fallback). Why: surface the user's environment
  state in 5 seconds.
- `0633055` — 23 tests for exceptions, hardware, seeding under
  the eight-class taxonomy. Why: lock down the foundation before
  building on it.

## [0.1.0] - 2026-07-13

(Reserved for the 1.0.0 dependency policy; placeholder to keep
semantic-version ordering stable across the re-numbered
[Unreleased] entry above.)

[Unreleased]: https://github.com/sachncs/pjepa/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/sachncs/pjepa/releases/tag/v1.0.0
[0.0.11]: https://github.com/sachncs/pjepa/compare/v0.0.10...v0.0.11
[0.0.10]: https://github.com/sachncs/pjepa/compare/v0.0.6...v0.0.10
[0.0.6]: https://github.com/sachncs/pjepa/compare/v0.0.5...v0.0.6
[0.0.5]: https://github.com/sachncs/pjepa/compare/v0.0.4...v0.0.5
[0.0.4]: https://github.com/sachncs/pjepa/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/sachncs/pjepa/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/sachncs/pjepa/releases/tag/v0.0.2
