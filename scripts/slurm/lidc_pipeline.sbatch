#!/bin/bash
#SBATCH --job-name=slotmil-pipeline
#SBATCH --account=nexus
#SBATCH --qos=medium
#SBATCH --partition=tron
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/fs/nexus-scratch/bthapar/SlotMIL/logs/%x-%j.out
#SBATCH --error=/fs/nexus-scratch/bthapar/SlotMIL/logs/%x-%j.err
#
# Splits -> train all arms -> slot-to-finding alignment, on the LIDC cache.
#
# Submitted with --dependency=afterok:<extraction job> so SLURM sequences this
# itself rather than relying on anyone watching the queue. If extraction fails,
# this stays PENDING instead of training on a half-filled cache.

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

CACHE=${CACHE:-$PROJ/data/lidc/features_dinov2_vitb14.h5}
SPLITS=$PROJ/data/lidc/splits.json
RUNS=$PROJ/runs/lidc

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=============== 1/3  splits ==============="
# Masked series are pushed out of train so localisation is never scored on data
# the model fitted. On LIDC nearly every series carries a mask, so this is a
# no-op there; it matters for MosMed, where only 50 of 1110 are annotated.
python scripts/make_splits.py --cache "$CACHE" --out "$SPLITS" --seed 0

echo "=============== 2/3  training ==============="
python scripts/train_cached.py \
    --cache "$CACHE" --splits "$SPLITS" --out "$RUNS" \
    --arms mean gated_abmil mh_abmil slot slot:div=0.1 \
    --seeds 0 1 2 --epochs 40 --num-workers 8

echo "=============== 3/3  alignment ==============="
# Run for SlotMIL and for the parameter-matched control. The control is the
# comparison that separates "competition causes specialisation" from "extra
# capacity causes specialisation" -- alignment numbers for SlotMIL alone would
# not answer reviewer objection #1.
# "<run directory>:<pooling name>" -- stated explicitly rather than derived from
# the directory name, which is a munged arm spec (slot:div=0.1 -> slot_div=0.1)
# and does not parse back cleanly.
for PAIR in "slot_div=0.1:slot" "mh_abmil:mh_abmil"; do
    ARM=${PAIR%%:*}
    POOL=${PAIR##*:}
    CKPT="$RUNS/$ARM/seed0/best.pt"
    if [ -f "$CKPT" ]; then
        echo "--- alignment: $ARM"
        python scripts/eval_alignment.py \
            --cache "$CACHE" --checkpoint "$CKPT" --splits "$SPLITS" \
            --pooling "$POOL" --num-slots 8 --dim 256 \
            --out "$RUNS/alignment_$ARM" || echo "alignment failed for $ARM"
    else
        echo "missing checkpoint $CKPT -- skipping"
    fi
done

echo "=============== pipeline complete ==============="
