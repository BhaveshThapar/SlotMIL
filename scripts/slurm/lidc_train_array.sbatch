#!/bin/bash
#SBATCH --job-name=slotmil-lidc-arm
#SBATCH --account=nexus
#SBATCH --qos=medium
#SBATCH --partition=tron
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-5
#SBATCH --output=/fs/nexus-scratch/bthapar/SlotMIL/logs/%x-%A_%a.out
#SBATCH --error=/fs/nexus-scratch/bthapar/SlotMIL/logs/%x-%A_%a.err
#
# One arm per array task, so the six arms train in parallel.
#
# The first attempt trained all six serially in one job and hit the 12 h wall
# clock partway through the third arm: ~1.6 h per arm/seed x 18 runs is ~29 h.
# As an array each task runs 3 seeds of one arm (~5 h) and the whole sweep
# finishes in roughly the time of its slowest arm.

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

ARMS=(mean gated_abmil mh_abmil slot "slot:div=0.1" "slot:div=0.5")
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}

CACHE=$PROJ/data/lidc/features_dinov2_vitb14.h5
SPLITS=$PROJ/data/lidc/splits.json

echo "array task $SLURM_ARRAY_TASK_ID -> arm '$ARM'"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Splits are built once, by task 0, before any training. The others wait for the
# file rather than racing to write it -- concurrent writers would leave a
# truncated JSON and silently give different arms different partitions.
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    python scripts/make_splits.py --cache "$CACHE" --out "$SPLITS" --seed 0
else
    for _ in $(seq 1 120); do
        [ -s "$SPLITS" ] && break
        sleep 10
    done
    [ -s "$SPLITS" ] || { echo "splits.json never appeared" >&2; exit 1; }
fi

python scripts/train_cached.py \
    --cache "$CACHE" --splits "$SPLITS" --out "$PROJ/runs/lidc" \
    --arms "$ARM" --seeds 0 1 2 --epochs 40 --num-workers 8

echo "arm '$ARM' complete"
