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
# Note: NO --masked-to-eval here. That flag exists for MosMed, where only 50 of
# 1110 series are annotated and masked scans must be kept out of train. On LIDC
# essentially every series carries a nodule mask, so the flag would move most of
# the training set into val/test; make_splits.py now refuses outright above 50%.
# Localisation on LIDC is evaluated on the held-out split, which is already
# patient-disjoint from train.
python scripts/make_splits.py --cache "$CACHE" --out "$SPLITS" --seed 0

echo "=============== 2/3  training ==============="
python scripts/train_cached.py \
    --cache "$CACHE" --splits "$SPLITS" --out "$RUNS" \
    --arms mean gated_abmil mh_abmil slot slot:div=0.1 slot:div=0.5 \
    --seeds 0 1 2 --epochs 40 --num-workers 8

echo "=============== 3/3  alignment ==============="
# Run for SlotMIL and for the parameter-matched control. The control is the
# comparison that separates "competition causes specialisation" from "extra
# capacity causes specialisation" -- alignment numbers for SlotMIL alone would
# not answer reviewer objection #1.
# Alignment runs on whichever slot arm actually kept its slots distinct.
# Hardcoding div=0.1 would risk scoring a collapsed model, which measures
# nothing: the diversity weight that fixed collapse at K=4 on MedMNIST is not
# guaranteed to hold at K=8 on DINOv2 features.
BEST_SLOT=$(python - <<'EOF'
import json, pathlib
p = pathlib.Path("runs/lidc/summary.json")
if p.exists():
    s = json.load(p.open())["summary"]
    c = {k: v for k, v in s.items() if k.startswith("slot")
         and v.get("max_off_diag_cos", {}).get("mean", 1.0) < 0.9}
    print(max(c, key=lambda k: c[k]["auc"]["mean"]).replace(":", "_") if c else "")
EOF
)
# `if`, not `[ -z ] && ...`: under set -e the latter aborts the job whenever
# BEST_SLOT is non-empty.
if [ -z "$BEST_SLOT" ]; then
    echo "NOTE: no slot arm kept max slot cosine < 0.9 -- slots collapsed on"
    echo "      LIDC at K=8. Alignment below is measuring a degenerate model;"
    echo "      treat the numbers as a collapse report, not an alignment result."
    BEST_SLOT="slot_div=0.5"
fi
echo "alignment slot arm: $BEST_SLOT"

# "<run directory>:<pooling name>" -- stated explicitly; the directory name is a
# munged arm spec (slot:div=0.1 -> slot_div=0.1) and does not parse back.
for PAIR in "$BEST_SLOT:slot" "mh_abmil:mh_abmil"; do
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
