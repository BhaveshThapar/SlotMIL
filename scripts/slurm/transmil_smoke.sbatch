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

echo "=============== 1/2  both real shapes ==============="
# TRAIN and EVAL are different shapes and the first version of this job only
# measured one of them, which is why the arm reached the sweep and OOMed at the
# first validation pass. protocol.max_slices subsamples to 48 slices at TRAIN
# time only, so training sees ~12,288 instances at batch 4 while evaluation sees
# the FULL bag -- and the deepest LIDC series is 162,560 instances, not the
# 43,812 that "~43.8k instances" in ENGINEERING.md describes.
python - <<'PY'
import math
import time

import torch

from slotmil.models.mil import build_model

DIN, DIM = 768, 256
dev = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(0)
model = build_model(pooling="transmil", input_dim=DIN, dim=DIM,
                    num_classes=2, num_slots=8, readout="gated").to(dev)
print(f"parameters {sum(p.numel() for p in model.parameters())/1e6:.2f}M")


def run(label, n, b, train):
    side = math.isqrt(n - 1) + 1
    feats = torch.randn(b, n, DIN, device=dev)
    pad = torch.ones(b, n, dtype=torch.bool, device=dev)
    if b > 1:
        pad[1, n // 2:] = False        # mixed depth, as collate_bags produces
    y = torch.zeros(b, dtype=torch.long, device=dev)
    if dev == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.set_grad_enabled(train):
        out = model(feats, pad)
        if train:
            torch.nn.functional.cross_entropy(out["logits"], y).backward()
    if dev == "cuda":
        torch.cuda.synchronize()
    attn = out["attn"]
    peak = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0.0
    total = (torch.cuda.get_device_properties(0).total_memory / 2**30
             if dev == "cuda" else 0.0)
    print(f"{label}: n={n} b={b} -> {side}x{side}  {time.time()-t0:.2f}s  "
          f"peak {peak:.2f} GiB of {total:.2f}")
    print(f"   attn {tuple(attn.shape)}  row sum {attn[0].sum().item():.6f}"
          + (f"  pad max {attn[1, :, n // 2:].abs().max().item():.3e}" if b > 1 else ""))
    model.zero_grad(set_to_none=True)


# 48 slices x 256 patches, the train-time view, with a backward pass.
run("train", 48 * 256, 4, train=True)
# The deepest series in the cohort, at the eval batch train_cached.py gives this
# arm, with grad disabled exactly as run_epoch does for validation.
run("eval ", 162560, 1, train=False)
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
