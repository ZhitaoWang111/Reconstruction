#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_HOME=/home/seer/wzt/tools/cuda-12.8
export CONDA_PREFIX="$CUDA_HOME"
export PATH="$ROOT/.venv-sam3d/bin:$CUDA_HOME/bin:$PATH"
export OPENBLAS_CORETYPE=Haswell
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export LIDRA_SKIP_INIT=true
export ATTN_BACKEND=sdpa
export SPARSE_ATTN_BACKEND=sdpa
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
SAM3D_CPU_AFFINITY="${SAM3D_CPU_AFFINITY-0}"
if [[ -n "$SAM3D_CPU_AFFINITY" ]]; then
  exec taskset -c "$SAM3D_CPU_AFFINITY" "$ROOT/.venv-sam3d/bin/python" -u -X faulthandler -m objgen.generate.generate "$@"
fi
exec "$ROOT/.venv-sam3d/bin/python" -u -X faulthandler -m objgen.generate.generate "$@"
