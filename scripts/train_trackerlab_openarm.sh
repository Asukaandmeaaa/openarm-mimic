#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
TASK_RUN="${1:-runs/trackerlab_wave_v1}"
if (( $# )); then shift; fi
if [[ -e "$TASK_RUN" ]]; then
  echo "Run directory already exists: $TASK_RUN. Choose a new directory to preserve checkpoints." >&2
  exit 2
fi
./scripts/python.sh integrations/trackerlab_openarm/scripts/train.py \
  --config configs/trackerlab_openarm_wave.json --log_dir "$TASK_RUN" "$@"
# Render the final physical policy evaluation already produced by the training callback.
TASK_EVAL="$(./scripts/python.sh -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); c=json.loads((p/"run_summary.json").read_text()); print(p/("eval_%06d"%c["completed_iterations"])/"trajectory.npz")' "$TASK_RUN")"
if [[ -f "$TASK_EVAL" ]]; then
  ./scripts/python.sh scripts/render_supermimic.py "$TASK_EVAL"
  echo "Video: $(dirname -- "$TASK_EVAL")/comparison.mp4"
else
  echo 'No actor evaluation yet: the requested run ended during world-model warmup, or eval_every=0.'
fi
