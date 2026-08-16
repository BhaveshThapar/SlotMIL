#!/bin/bash
#SBATCH --job-name=slotmil-collect-f32
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa5000:1
# GPU type pinned in --gres, not via --constraint: scavenger nodes like
# legacygpu12/15 carry mixed Pascal and Turing cards, so a node-level feature
# match still hands out a card this torch build has no kernels for.
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/fs/nexus-scratch/bthapar/SlotMIL/logs/%x-%j.out
#SBATCH --error=/fs/nexus-scratch/bthapar/SlotMIL/logs/%x-%j.err
#
# Cache one trained checkpoint's attention in float32.
#
# Why this exists as its own job: three of the five discovery dumps in
# runs/nulls are fp16, and protocol.dtype is float32, so those three cannot
# carry a pre-registered estimand. f32_seed0 and f32_seed2 were recollected;
# seed 1 never was. Its fp16 prior-normalised skill reads 0.3159 against H5's
# 0.30 threshold, and fp16 ran LOWER than fp32 on seed 2 (0.7173 vs 0.7508), so
# the valid seed-1 number is plausibly above the threshold and has not been
# looked at. Collect it before the H5 unit is ruled, not after.
#
# fp32 eval collection runs with no autocast, which makes it the tighter memory
# budget of the two paths (training gets AMP for free) -- but this is one model,
# not the 30-model floor fleet, so 48G and one card is ample.
#
#   sbatch scripts/slurm/null_collect_f32.sbatch \
#       --ckpt runs/lidc/slot_div=0.5/seed1/best.pt --tag f32_seed1
#
# The arm is slot:div=0.5 on the DISCOVERY split -- div is a loss-side
# regulariser, so the collector builds plain pooling=slot. That is what
# reproduces f32_seed0 bit-for-bit against
# runs/corrected_alignment.json -> nodule_present/slot_div=0.5/seed0/
# instance_auc_frozen_slot (0.8422927114209887).

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

echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)"

python scripts/null_collect_attn.py \
    --splits data/lidc/splits.json \
    --pooling slot \
    --num-slots 8 \
    --dtype float32 \
    "$@"

echo "=============== collect complete ==============="
