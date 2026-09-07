#!/usr/bin/env bash
# Setup script for pjepa.
#
# Creates a local virtual environment (default ``.venv``), installs the
# project with the dev and OGB extras, and verifies that every tool the
# CI workflow depends on is importable. Run with ``bash setup.sh`` from
# the repository root.
#
# Environment variables:
#   PYTHON   Python interpreter to use (default: ``python3.12``).
#   VENV     Virtual environment directory (default: ``.venv``).
#   EXTRAS   Comma-separated pip extras (default: ``dev,ogb``).

set -euo pipefail

PYTHON="${PYTHON:-python3.12}"
VENV="${VENV:-.venv}"
EXTRAS="${EXTRAS:-dev,ogb}"

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${_SCRIPT_DIR}"

if [[ -z "${VENV}" || "${VENV}" = "/" ]]; then
  echo "setup.sh: refusing to use unsafe VENV='${VENV}'" >&2
  exit 1
fi

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  echo "setup.sh: required interpreter '${PYTHON}' not found on PATH" >&2
  exit 1
fi

PYTHON_VERSION="$("${PYTHON}" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
case "${PYTHON_VERSION}" in
  3.10|3.11|3.12) ;;
  *)
    echo "setup.sh: Python ${PYTHON_VERSION} is outside the supported range (3.10-3.12)" >&2
    exit 1
    ;;
esac

if [[ ! -d "${VENV}" ]]; then
  echo "setup.sh: creating virtual environment at '${VENV}' using ${PYTHON}"
  "${PYTHON}" -m venv "${VENV}"
fi

VENV_PY="${VENV}/bin/python"
if [[ ! -x "${VENV_PY}" ]]; then
  echo "setup.sh: virtual environment is missing ${VENV_PY}" >&2
  exit 1
fi

echo "setup.sh: upgrading pip"
"${VENV_PY}" -m pip install --upgrade pip

echo "setup.sh: installing pjepa with extras: ${EXTRAS}"
"${VENV_PY}" -m pip install -e ".[${EXTRAS}]"

echo "setup.sh: verifying required tooling"
for tool in pjepa pytest ruff pytype pip-audit mkdocs; do
  if [[ ! -x "${VENV}/bin/${tool}" ]]; then
    echo "setup.sh: missing required entry point '${VENV}/bin/${tool}'" >&2
    exit 1
  fi
done

if ! "${VENV_PY}" -c "import build" >/dev/null 2>&1; then
  echo "setup.sh: python -m build is unavailable; expected 'build' in dev extras" >&2
  exit 1
fi

if ! "${VENV_PY}" -c "import optuna" >/dev/null 2>&1; then
  echo "setup.sh: optuna is unavailable; expected 'optuna' in dev extras" >&2
  exit 1
fi

echo "setup.sh: pjepa version -> $("${VENV}/bin/pjepa" --version)"
echo "setup.sh: running capability probe (pjepa doctor)"
if ! "${VENV}/bin/pjepa" doctor; then
  echo "setup.sh: pjepa doctor reported a RED capability; review before continuing" >&2
  exit 1
fi

echo "setup.sh: running the test suite (pytest -n auto)"
"${VENV_PY}" -m pytest tests -n auto -q

echo "setup.sh: complete. Activate the venv with 'source ${VENV}/bin/activate'."
