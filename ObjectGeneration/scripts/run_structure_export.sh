#!/usr/bin/env bash
# Structure generation, step 4: the structure handoff for scene assembly, from a committed agent run.
#   scripts/run_structure_export.sh [agent run dir] [--frame amb3r_floor|scannet_scan] [--out <dir>]
# Default: the newest real agent run under structuregen/latest, in the walls handoff's own frame;
# writes output/<scene>/structuregen/structure_handoff[_<frame>]/. Then rebuilds the result page.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PY="$ROOT/.venv-sam3d/bin/python"
"$PY" -m structuregen.export.handoff "$@"
exec "$PY" "$ROOT/pagegen/build_structuregen_page.py"
