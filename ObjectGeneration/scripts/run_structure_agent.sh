#!/usr/bin/env bash
# Structure generation, step 3: the appearance agent (gpt-6-astra through the gateway; costs money unless --fake).
#   scripts/run_structure_agent.sh [run dir] [--fake] [--max-usd 20] [--resume <replay dir>]
# Default run dir: output/<scene>/structuregen/latest. Writes <run>/agent_<stamp>/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$ROOT/.venv-sam3d/bin/python" -u -m structuregen.agent.run "$@"
