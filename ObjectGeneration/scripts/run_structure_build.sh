#!/usr/bin/env bash
# Structure generation, steps 1 and 2: white model (walls + floors) and the agent's evidence. No model, no cost.
#   scripts/run_structure_build.sh [--handoff <walls handoff dir>] [--opening-inset 0.15] [--floor-thickness 0.05] [--height 2.8]
# Writes output/<scene>/structuregen/<stamp>/ and points structuregen/latest at it.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PY="$ROOT/.venv-sam3d/bin/python"
LOG="$(mktemp)"
trap 'rm -f "$LOG"' EXIT
"$PY" -m structuregen.whitebox.build "$@" | tee "$LOG"
RUN="$(sed -n 's/^STRUCTUREGEN_OUT //p' "$LOG" | tail -1)"
[[ -n "$RUN" ]] || { echo "build did not report its run dir" >&2; exit 1; }
"$PY" -m structuregen.whitebox.validate "$RUN"
exec "$PY" -m structuregen.evidence.package "$RUN"
