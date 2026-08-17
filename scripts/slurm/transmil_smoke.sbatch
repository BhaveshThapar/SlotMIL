#!/bin/bash
#SBATCH --job-name=slotmil-transmil-smoke
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa5000:1
# GPU type pinned in --gres itself: legacygpu12/15 carry mixed Pascal and Turing
# cards, so a node-level --constraint still hands out a card this torch build
# has no kernels for.
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/fs/nexus-scratch/bthapar/SlotMIL/logs/%x-%j.out
#SBATCH --error=/fs/nexus-scratch/bthapar/SlotMIL/logs/%x-%j.err
#
# Does TransMIL train at LIDC bag scale? That is the whole question this job
# exists to answer, and it is the question the arm's pre-committed drop rule
# asks: "LIDC bags are ~43.8k instances, far beyond the WSI bags TransMIL was
# designed for. If it is not training by 2026-09-22, cut it."
#
# Answering it BEFORE promoting the arm in the config is the point. Promotion is
# a config edit, a config edit re-hashes the pre-registration, and an amendment
# supersedes every stamp on disk -- the entire confirmatory sweep would have to
# re-run. Doing that for an arm that then OOMs would be the expensive way to
# learn what this four-hour job learns.
#
# Same shape as aux_arm_smoke.sbatch, and for the same reasons: two epochs, one
# seed, DISCOVERY split, --role exploratory, a fresh root under runs/smoke. Unit
# tests run 12-instance bags and cannot see a memory shape that only appears at
# scale -- which for this arm is the squared sequence, 43812 instances rounding
# up to a 210x210 grid that PPEG convolves over.
#
# Step 1 measures the forward/backward directly and prints peak memory, because
# a plain OOM traceback from inside the training loop does not say WHICH tensor
# was the problem, and the remedy differs: a landmark count is a knob, a PPEG
# grid is not.
#
#   sbatch scripts/slurm/transmil_smoke.sbatch

set -euo pipefail
PROJ=/fs/nexus-scratch/bthapar/SlotMIL
cd "$PROJ"

source /usr/share/Modules/init/bash
module load Python3/3.11.11
# shellcheck disable=SC1091
source "$PROJ/.venv/bin/activate"

export TORCH_HOME=/fs/nexus-scratch/bthapar/.cache/torch
export TMPDIR=/fs/nexus-scratch/bthapar/.tmp
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo none)"

echo "=============== 1/2  forward+backward at real bag scale ==============="
python - <<'PY'
import time

import torch

from slotmil.models.mil import build_model

# 43812 = the largest LIDC bag: 171 slices x 256 patches. batch_size 4 is what
# train_cached.py uses, so this is the real resident shape, not a proxy for it.
N, B, DIN, DIM = 43812, 4, 768, 256
dev = "cuda" if torch.cuda.is_available() else "cpu"
side = int(torch.tensor(N - 1).sqrt().item()) + 1
print(f"bag {N} instances -> squared to {side}x{side} = {side * side} cells")

torch.manual_seed(0)
model = build_model(pooling="transmil", input_dim=DIN, dim=DIM,
                    num_classes=2, num_slots=8, readout="gated").to(dev)
n_par = sum(p.numel() for p in model.parameters())
print(f"parameters {n_par/1e6:.2f}M")

feats = torch.randn(B, N, DIN, device=dev)
pad = torch.ones(B, N, dtype=torch.bool, device=dev)
pad[1, 30000:] = False          # a mixed-depth batch, as collate_bags produces
y = torch.tensor([1, 0, 1, 0], device=dev)

if dev == "cuda":
    torch.cuda.reset_peak_memory_stats()
t0 = time.time()
out = model(feats, pad)
loss = torch.nn.functional.cross_entropy(out["logits"], y)
loss.backward()
if dev == "cuda":
    torch.cuda.synchronize()
dt = time.time() - t0

attn = out["attn"]
print(f"forward+backward {dt:.2f}s  loss {loss.item():.4f}")
print(f"attn {tuple(attn.shape)}  pad max {attn[1, :, 30000:].abs().max().item():.3e}  "
      f"row sum {attn[0].sum().item():.6f}")
if dev == "cuda":
    print(f"peak memory {torch.cuda.max_memory_allocated()/2**30:.2f} GiB of "
          f"{torch.cuda.get_device_properties(0).total_memory/2**30:.2f} GiB")
PY

echo "=============== 2/2  two epochs through the real driver ==============="
python scripts/train_cached.py \
    --cache "$PROJ/data/lidc/features_dinov2_vitb14.h5" \
    --splits "$PROJ/data/lidc/splits.json" \
    --out "$PROJ/runs/smoke_transmil" \
    --arms transmil \
    --seeds 0 --epochs 2 --num-workers 4 \
    --role exploratory

echo "=============== transmil smoke complete ==============="
