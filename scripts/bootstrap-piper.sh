#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
sdk_root="$(cd -- "${project_root}/.." && pwd)/pyAgxArm"

if [[ ! -f "${sdk_root}/pyproject.toml" && ! -f "${sdk_root}/setup.py" ]]; then
  echo "Expected the Piper SDK at ${sdk_root}." >&2
  echo "Place mp-real and pyAgxArm in the same parent directory, then retry." >&2
  exit 1
fi

cd "${project_root}"
uv sync --extra piper
uv pip install --python "${project_root}/.venv/bin/python" --editable "${sdk_root}"
