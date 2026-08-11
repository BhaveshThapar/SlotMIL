#!/bin/bash
#SBATCH --job-name=slotmil-lidc-mal
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --requeue
#SBATCH --output=/fs/nexus-scratch/bthapar/SlotMIL/logs/%x-%A_%a.out
#SBATCH --error=/fs/nexus-scratch/bthapar/SlotMIL/logs/%x-%A_%a.err
#
# The malignancy experiment: does a balanced, localisation-dependent bag label
# make slot attention localise?
#
# The nodule_present run was 87% positive, so it exerted almost no pressure to
# find anything, and localisation came out at chance (instance AUC 0.489) even
# though a supervised probe reaches 0.9102 on the same features. Malignancy is
# balanced and cannot be predicted without characterising the nodule, so it
# separates "slot attention cannot localise" from "we asked the wrong question".
#
# Same cached features -- only the label changes -- so no re-extraction.
#
# Four arms rather than six: the two collapsed diversity settings (none, 0.1)
# already answered their question on this data and are not worth the GPU time.
# gated_abmil is kept as the baseline to beat and mh_abmil as the matched control.

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

ARMS=(gated_abmil mh_abmil "slot:div=0.5" mean)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}

CACHE=$PROJ/data/lidc/features_dinov2_vitb14.h5
SPLITS=$PROJ/data/lidc/splits_malignancy.json
LABELS=$PROJ/data/lidc/labels_malignancy.json

echo "array task $SLURM_ARRAY_TASK_ID -> arm '$ARM'"
nvidia-smi --query-gpu=name --format=csv,noheader

# Task 0 builds the splits once; the rest wait, so concurrent writers cannot
# truncate the JSON or hand different arms different partitions.
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    python scripts/make_splits.py --cache "$CACHE" --out "$SPLITS" \
        --labels-from "$LABELS" --seed 0
else
    for _ in $(seq 1 120); do
        [ -s "$SPLITS" ] && break
        sleep 10
    done
    [ -s "$SPLITS" ] || { echo "splits_malignancy.json never appeared" >&2; exit 1; }
fi

python scripts/train_cached.py \
    --cache "$CACHE" --splits "$SPLITS" --out "$PROJ/runs/lidc_malignancy" \
    --arms "$ARM" --seeds 0 1 2 --epochs 40 --num-workers 4

echo "arm '$ARM' complete"
