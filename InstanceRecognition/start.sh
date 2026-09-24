#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
DEFAULT_PYTHON="python3"
if [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
  DEFAULT_PYTHON="$SCRIPT_DIR/.venv/bin/python"
fi
exec "${SCENE_PYTHON:-$DEFAULT_PYTHON}" -m web.server "$@"
