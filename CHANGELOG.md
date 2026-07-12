# Changelog

All notable changes to `pjepa` are documented here. The format follows [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/); this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Each entry includes the commit SHA (short), the date (UTC), and the rationale.

## [Unreleased]

### Added
- `src/pjepa/` package skeleton with subpackage markers (Phase 0 in progress)
- `tests/` package marker documenting the eight-class test taxonomy
- README with quickstart, audience-specific documentation pointers, and project-status table
- CHANGELOG entries now include commit SHA and UTC date

## [0.0.1] - 2026-07-12

### Added
- `7dc8aed` — Repository scaffold; planning artefacts under `plans/` (gitignored); paper draft under `docs/paper/paper.md` (gitignored)
- Initial `.gitignore` for Python, virtual environments, build artefacts, IDEs, and generated experiment outputs
- Why: establish the public source tree, exclude the in-progress paper and planning artefacts from version control, and give new contributors an obvious entry point (README + CHANGELOG).