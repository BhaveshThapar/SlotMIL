#!/bin/bash
#SBATCH --job-name=slotmil-aux-smoke
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
# End-to-end smoke for clam_sb and dsmil: two epochs, one seed, DISCOVERY split,
# --role exploratory. Proves the arms train on real 43.8k-instance bags -- the
# unit tests run 12-instance bags and cannot see a memory shape or a loss term
# that only misbehaves at scale.
#
# Deliberately NOT the confirmatory split and NOT runs/lidc: a fresh root under
# runs/smoke, so this can never be mistaken for a sweep result and can never
# collide with the 18 completed discovery seeds. Per-seed resume refuses to
# cross split hashes anyway, but the right answer is a separate root.
#
#   sbatch scripts/slurm/aux_arm_smoke.sbatch

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

python scripts/train_cached.py \
    --cache "$PROJ/data/lidc/features_dinov2_vitb14.h5" \
    --splits "$PROJ/data/lidc/splits.json" \
    --out "$PROJ/runs/smoke" \
    --arms clam_sb dsmil \
    --seeds 0 --epochs 2 --num-workers 4 \
    --role exploratory

echo "=============== aux arm smoke complete ==============="
