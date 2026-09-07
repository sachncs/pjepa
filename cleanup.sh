#!/usr/bin/env bash
# Cleanup script for pjepa.
#
# Removes every generated artefact created by the development workflow:
# the virtual environment, build outputs, coverage files, type-checker
# caches, the mkdocs site, Python bytecode caches, and the experimental
# result directories under ``results/``. Run with ``bash cleanup.sh``
# from the repository root.
#
# Environment variables:
#   VENV         Virtual environment directory to remove (default ``.venv``).
#   PRESERVE     When set, skip removing ``${VENV}`` and the per-experiment
#                result directories (e.g. ``PRESERVE=1 bash cleanup.sh``).

set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${_SCRIPT_DIR}"

VENV="${VENV:-.venv}"
PRESERVE_RESULTS=0
if [[ "${PRESERVE:-0}" = "1" ]]; then
  PRESERVE_RESULTS=1
fi

if [[ -z "${VENV}" || "${VENV}" = "/" ]]; then
  echo "cleanup.sh: refusing to use unsafe VENV='${VENV}'" >&2
  exit 1
fi

CACHE_DIRS=(
  build dist site
  .pytest_cache .mypy_cache .ruff_cache .pytype
  .coverage coverage.xml coverage-*.xml htmlcov
  src/pjepa.egg-info src/*.egg-info
)

if [[ -d "${VENV}" ]]; then
  echo "cleanup.sh: removing virtual environment '${VENV}'"
  rm -rf "${VENV}"
else
  echo "cleanup.sh: no virtual environment at '${VENV}'"
fi

for dir in "${CACHE_DIRS[@]}"; do
  if [[ -e "${dir}" ]]; then
    echo "cleanup.sh: removing ${dir}"
    rm -rf "${dir}"
  fi
done

find . -name '__pycache__' -prune -exec rm -rf {} +
find . -name '*.py[co]' -delete

if [[ "${PRESERVE_RESULTS}" -eq 0 ]]; then
  RESULTS_PATHS=(
    results/checkpoints results/logs results/optuna results/metrics
    results/tu_smoke results/exp_a_smoke results/exp_b_smoke
    results/exp_c_smoke results/ogb_smoke results/decoupling_smoke
    results/ablation_smoke results/sensitivity_smoke
    results/cl results/ogb results/ablation results/decoupling
    results/sensitivity_B results/tables results/plots
    results/all_runs.jsonl
  )
  for path in "${RESULTS_PATHS[@]}"; do
    if [[ -e "${path}" ]]; then
      echo "cleanup.sh: removing ${path}"
      rm -rf "${path}"
    fi
  done
else
  echo "cleanup.sh: PRESERVE=1 set; leaving results/ in place"
fi

if command -v git >/dev/null 2>&1; then
  if ! git status --short >/dev/null 2>&1; then
    echo "cleanup.sh: 'git status' failed; run from inside the repository" >&2
    exit 1
  fi
  if [[ -n "$(git status --short)" ]]; then
    echo "cleanup.sh: git status is not clean; verify the remaining paths"
    git status --short
  else
    echo "cleanup.sh: git status is clean"
  fi
fi

echo "cleanup.sh: complete."
