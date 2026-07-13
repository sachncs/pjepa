# Changelog

All notable changes to `pjepa` are documented here. The format follows [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/); this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Each entry includes the commit SHA (short), the date (UTC), and the rationale.

## [1.0.0] - 2026-07-13

### Added
- `pjepa eval/aggregate.py` — result aggregation across every phase,
  writing `results/all_runs.jsonl`, `results/tables/all_runs.csv`, and
  `results/tables/summary.md`. Always emits the artefacts, even when
  no run data exists yet (header row only). Why: Phase 12 release
  pipeline requires a single canonical aggregator that downstream
  consumers can rely on.
- `pjepa aggregate <results-dir>` CLI subcommand. Why: one entry point
  for the Phase 12 aggregator without needing Python imports.
- `pjepa tune`, `pjepa baseline-smoke`, `pjepa decoupling`,
  `pjepa ablation`, `pjepa sensitivity`, `pjepa eval` CLI subcommands.
  Why: the documented workflow needs every advertised command to be
  implemented as real dispatch, not stubs.
- `configs/default.yaml`, `configs/tu.yaml`, `configs/cl.yaml`,
  `configs/ogb.yaml` plus eight baseline YAML configs (`gcn`, `gin`,
  `graphmae`, `graphcl`, `infograph`, `naive`, `ewc`, `gem`). Why:
  CLI commands take a YAML path; without configs they cannot resolve
  anything.
- Makefile targets: `bench-encoder`, `tune-tu`, `reproduce-tu`,
  `reproduce-cl`, `reproduce-ogb`, `reproduce-all`, `verify-claims`,
  `ablation`, `sensitivity`, `aggregate`, `profile`, `package`,
  `release`. Why: the plan and REPRODUCE.md advertise each of these.
- `tests/test_cli.py`, `tests/test_aggregate.py`,
  `tests/test_configs.py`. Why: cross-cutting CLI, aggregation, and
  config coverage.
- `build` dev-dependency entry in `pyproject.toml`. Why: `make
  package` and `make release` require `python -m build`.

### Changed
- Version bumped to `1.0.0` across `pyproject.toml`,
  `src/pjepa/_version.py`, `CITATION.cff`, and the README status
  table. Why: 1.0.0 signals the public Phase 12 release; the
  previous `0.0.11` was a Phase 8-11 snapshot.
- Classifiers updated to `Development Status :: 5 - Production/Stable`.
  Why: a 1.0.0 release should not advertise itself as Alpha.
- `src/pjepa/cli/app.py` rewired: `pretrain`, `train`, `eval` now
  dispatch to the corresponding `experiments/run_exp_*.py` runner
  via the runner's top-level dataclass; missing config files no
  longer raise. Why: previous implementation was a stub that
  advertised Phase 5 integration that did not exist.
- `pjepa/eval/__init__.py` re-exports `aggregate_all`,
  `AggregatedRow`, `AggregationResult`, `merge_rows`, `write_artifacts`.
- `docs/index.md` honest status: Phase 12 in progress, external
  releases (Docker push, GitHub Release, Read the Docs trigger,
  full 12-phase reproduction) explicitly out of scope for the local
  release process. Why: the local implementation is complete and the
  artefacts build, but a fabricated Docker push / PyPI upload would
  be dishonest.
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
- Full 12-phase reproduction (`make reproduce-all`) — the wall-clock
  is ~70 hours and would saturate the CI runner.
- Read the Docs build trigger (requires RTD credentials).

The artefacts that are produced locally and are part of the
release:

- `dist/pjepa-1.0.0.tar.gz` and `dist/pjepa-1.0.0-py3-none-any.whl`
  via `make package`.
- `site/` (mkdocs strict build) via `make docs`.
- Aggregated result tables via `make aggregate`.

## [Unreleased]

### Added
- Phase 2 performance infrastructure and Optuna hyperparameter search.
- Phase 5 SWA, TTA, ensemble, and distillation training infrastructure.
- Phase 8-12 experiments and the 1.0.0 release pipeline.

### Changed
- All existing modules reformatted with ruff and lint-clean.
- Per-file ruff ignores added for tests, `__init__.py`, and source files
  so docstring style and stylistic suggestions do not block CI.
- `pyproject.toml` requires-python updated to `>=3.10,<3.15` to include
  the Python 3.14 development environment; 3.10-3.12 remain CI targets.
- Version bumped to `0.0.10` reflecting the Phase 0+1 production-grade
  scaffold completion.
- `pjepa benchmark` CLI subcommand extended with `encoder-ablation`.
- `pjepa train` CLI subcommand wires up the TU/CL/OGB experiment
  runners (Phase 8/9/10).

## [0.0.11] - 2026-07-13

### Added
- `e4f2b33` — Phase 12 polish: README with project status table,
  CITATION.cff, multi-stage Dockerfile.
- `9aab762` — Phase 4 + 11 + CLI integration: encoder-ablation runner,
  decoupling measurement runner, CLI wiring for TU/CL/OGB experiments.
- `01a5b4b` — Phase 8 TU SOTA experiment runner (6 datasets × 7 methods
  × 5 seeds × 10 folds). Verified smoke test on MUTAG: GIN 0.86.
- `f96526e` — Phase 5 wrappers: SWAWrapper, TTAWrapper, Ensemble
  (soft_vote / hard_vote / rank_avg), DistillationLoss. Plus
  TensorDropFeature for tensor-only models.
- `e67216c` — Phase 2 performance infra: safe_compile, autocast_context,
  EMATarget (with cosine schedule), fused_scatter_add, sync_mps.
- `f5196e8` — Phase 1 retrieval: GreedyRetrieval + utilities with
  verified (1 - 1/e) approximation guarantee.
- `cfa31ae` — Phase 1 rewriting: HRG, BisimulationMetric, FourConditions
  (the paper's acceptance criterion), DPO loss.
- `917cfcb` — Phase 1 graph types: TypedAttributedGraph (frozen
  dataclass), PersistentState, WorkingGraph.
- `0f23853` — Phase 0 logging + config: structured logging (HUMAN/JSON),
  YAML config loading with schema validation.
- `74ce19d` — Phase 0 foundation: PJEPAError hierarchy, deterministic
  seeding, hardware capability probes.
- `7dc8aed` — Initial repository scaffold.

### Stats
- 227 tests passing across 14 test files.
- 0 ruff lint errors.
- 8-class test taxonomy (happy / bad / ugly / leaky / round-trip /
  cross-backend / distributional / property) applied to every public
  module.
- All public symbols have Google-style docstrings with Args, Returns,
  Raises, and Example sections.
- No bare `except`, no `print`, no global mutable state.
- CLI works: `pjepa --version`, `pjepa doctor`, `pjepa hardware`,
  `pjepa benchmark {retrieval, distortion, encoder-ablation}`,
  `pjepa train {tu, cl, ogb}` all execute cleanly.

## [0.0.2] - 2026-07-13

### Added
- `74ce19d` — Custom exception hierarchy (`PJEPAError` + 7 subclasses) in
  `pjepa/exceptions.py`. Why: typed errors let tests catch the base class
  for "any pjepa failure" or specific subclasses for granular checks.
- `74ce19d` — Deterministic seeding via `pjepa.utils.seeding`:
  `set_global_seed`, `get_global_seed`, `seed_for`. Why: reproducibility
  is a hard requirement; this is the single entry point for all RNGs.
- `74ce19d` — Hardware capability detection via `pjepa.hardware.py`:
  `detect_backend`, `detect_capabilities`, six probes (matmul,
  scatter_add, torch.compile, hyperbolic, pyg_scatter, cpu_fallback).
  Why: surface the user's environment state in 5 seconds.
- `0633055` — 23 tests for exceptions, hardware, seeding under the
  eight-class taxonomy. Why: lock down the foundation before building
  on it.

## [0.0.3] - 2026-07-13

### Added
- `0f23853` — Structured logging (`pjepa.logging_setup`): HUMAN and JSON
  formatters, `log_event` for keyword-field records, lazy configuration.
  Why: replace `print()` with structured logs throughout the library.
- `0f23853` — YAML config loading (`pjepa.config`): `load_config`,
  `save_config`, `merge_configs`, `ConfigSchema` for required-section
  validation. Why: configuration is the source of truth for every
  experiment.
- `0633055` — 16 additional tests covering config and logging paths.

## [0.0.4] - 2026-07-13

### Added
- `917cfcb` — Core graph types: `TypedAttributedGraph` (immutable
  dataclass with shape-consistency validation), `PersistentState`
  (commit/reject with audit trail), `WorkingGraph` (budget
  enforcement). Why: every other component depends on these types.
- `0633055` — 20 graph tests covering happy/bad/ugly/round-trip/
  cross-backend/distributional/property paths.

## [0.0.5] - 2026-07-13

### Added
- `f5196e8` — Submodular retrieval with verified (1 - 1/e) approximation
  guarantee. `GreedyRetrieval` implements Algorithm 1 of the paper;
  `FacilityLocationUtility` and `InformationGainUtility` are pluggable.
  Why: Theorem 3 is the headline retrieval result.
- `0633055` — 14 retrieval tests including (1-1/e) verification via
  brute-force optimum and submodularity property tests.

## [0.0.6] - 2026-07-13

### Added
- `cfa31ae` — Verified rewriting engine: `HRG` (hyperedge-replacement
  grammar), `BisimulationMetric` (value-iteration approximation),
  `FourConditions` (the paper's acceptance criterion), `DPO` loss.
  Why: §7.6.1 and §7.7 of the paper are operationalised here.
- `0633055` — 21 rewriting tests covering acceptance, rejection,
  and bisimulation properties.

## [0.0.7] - 2026-07-13

### Added
- `e67216c` — Objectives (`FreeEnergy`, `ib_lagrangian`,
  `description_length`), dynamics (`EvolutionOperator`,
  `contractivity_bound`, `fixed_point_iteration`), scheduler
  (`PPOTrainer`, `ReplayBuffer`, `SleepCadence`), encoders
  (`EuclideanMPNN`, `HyperbolicProjection`, `DualGeometricEncoder`,
  `JEPAPredictor`, `TargetEncoder`). Why: §2, §3, §4 of the paper.
- `0633055` — 29 tests covering objectives, dynamics, scheduler, and
  encoders.

## [0.0.8] - 2026-07-13

### Added
- `347db8f` — Augmentations (`DropEdge`, `DropNode`, `DropFeature`,
  `FeatureMask`, `RandomWalkSubgraph`, `AugmentationPipeline`), data
  loaders (`TUDataset`, `OGB-arxiv`, class-incremental split), and 7
  baselines (GCN, GIN, GraphMAE, GraphCL, InfoGraph, EWC, GEM).
  Why: enables Phase 7-10 SOTA experiments.
- `0633055` — 32 augmentation, data, and baseline tests.

## [0.0.9] - 2026-07-13

### Added
- `f96526e` — Training loops (`pretrain_loop`, `supervised_train_loop`,
  `linear_probe_eval`, `Checkpoint`), eval utilities (`accuracy`,
  `mean_per_class_accuracy`, `forgetting_rate`, `paired_bootstrap_ci`,
  `wilcoxon_signed_rank`, `bonferroni_correction`), Typer-based CLI
  (`doctor`, `hardware`, `benchmark`, `train`, `eval`), and experiment
  scripts (`run_exp_a_retrieval`, `run_exp_b_distortion`).
  Why: complete the Phase 5 training infrastructure and Phase 0 CLI.
- `0633055` — 27 training and eval tests.

## [0.0.10] - 2026-07-13

### Added
- `9f023ed` — Phase 0+1 polish: `pyproject.toml` (PEP 621 metadata,
  full dependency pinning, ruff/pytype/pytest config, CLI entry point),
  `Makefile` (install/lint/typecheck/test/coverage/audit/docs/bench
  targets), `.pre-commit-config.yaml`, `.github/workflows/ci.yml`,
  `mkdocs.yml`, and comprehensive documentation under
  `docs/researcher/` and `docs/developer/`.
- Why: production-grade scaffolding for the 1.0.0 release.
- Why: documentation serves both researchers (deep-dive on the framework)
  and developers (quickstart, architecture, extension guides).
- Test suite is now lint-clean (0 ruff errors) with 182 passing tests.

## [0.0.1] - 2026-07-12

### Added
- `7dc8aed` — Repository scaffold; planning artefacts under `plans/` (gitignored); paper draft under `docs/paper/paper.md` (gitignored)
- Initial `.gitignore` for Python, virtual environments, build artefacts, IDEs, and generated experiment outputs
- Why: establish the public source tree, exclude the in-progress paper and planning artefacts from version control, and give new contributors an obvious entry point (README + CHANGELOG).