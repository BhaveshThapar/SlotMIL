#!/usr/bin/env bash
# Build the SlotMIL virtualenv.
#
# Everything lives in /fs/nexus-scratch -- /nfshomes has <10 GB free and pip
# caches plus torch wheels will blow through that. TORCH_HOME points at the
# pre-existing cache so the DINOv2 ViT-B/14 checkpoint already on disk is reused
# instead of re-downloaded.
#
# Single resolve from PyPI, no --index-url juggling. tron drivers are 595.71.05
# (verified on tron62), which is new enough for the CUDA runtime bundled in
# current torch wheels. An earlier attempt pinned torch to cu124 first, but
# monai>=1.3 requires torch>=2.8 and pip simply clobbered it with the default
# build -- installing two full CUDA stacks for nothing.
set -euo pipefail

SCRATCH=/fs/nexus-scratch/bthapar
PROJ="$SCRATCH/SlotMIL"
VENV="$PROJ/.venv"

export PIP_CACHE_DIR="$SCRATCH/.pip-cache"
export TORCH_HOME="$SCRATCH/.cache/torch"
export TMPDIR="$SCRATCH/.tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

source /usr/share/Modules/init/bash
module load Python3/3.11.11

rm -rf "$VENV"
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

python -m pip install --upgrade pip wheel
python -m pip install -r "$PROJ/env/requirements.txt"
python -m pip install -e "$PROJ"

echo "=== versions ==="
python - <<'PY'
import torch, monai, timm, medmnist, h5py, numpy, sqlalchemy
print("torch      ", torch.__version__, "| cuda:", torch.version.cuda)
print("numpy      ", numpy.__version__)
print("sqlalchemy ", sqlalchemy.__version__)
print("monai      ", monai.__version__)
print("timm       ", timm.__version__)
print("medmnist   ", medmnist.__version__)
print("h5py       ", h5py.__version__)
import pylidc
print("pylidc     ", "imports OK")
PY
echo "=== env ready: $VENV ==="
