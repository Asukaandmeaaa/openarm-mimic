#!/usr/bin/env bash
set -euo pipefail
WORKSPACE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WORKSPACE_ROOT/src:$WORKSPACE_ROOT/vendor:$WORKSPACE_ROOT/integrations/trackerlab_openarm/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$WORKSPACE_ROOT"
if [[ -n "${MIMIC_CONDA_PREFIX:-}" && -x "$MIMIC_CONDA_PREFIX/bin/python" ]]; then
  exec "$MIMIC_CONDA_PREFIX/bin/python" "$@"
fi
exec "${PYTHON:-python3}" "$@"
