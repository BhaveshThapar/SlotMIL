#!/bin/bash
#SBATCH --job-name=slotmil-mosmed-pipe
#SBATCH --account=nexus
#SBATCH --qos=medium
#SBATCH --partition=tron
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=/fs/nexus-scratch/bthapar/SlotMIL/logs/%x-%j.out
#SBATCH --error=/fs/nexus-scratch/bthapar/SlotMIL/logs/%x-%j.err
#
# MosMed: train severity classification, then slot-to-finding alignment against
# the 50 expert masks. Independent of LIDC -- the MosMed cache is already
# complete -- so this runs in parallel rather than queueing behind extraction.
#
# Task is 4-class severity: CT-4 holds 2 volumes of 1110, so it is folded into
# CT-3 at split time. Without that, validation contains zero CT-4 and macro AUC
# is undefined for that class.
#
# Per the RESULTS.md audit, MosMed is the *supporting* localisation dataset, not
# the headline: consolidation is a median 2.6% of annotated lesion volume, the
# GGO/consolidation split is our own HU threshold rather than an annotated class,
# and all 50 masked scans are CT-1.

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

CACHE=$PROJ/data/mosmed/features_dinov2_vitb14.h5
SPLITS=$PROJ/data/mosmed/splits.json
RUNS=$PROJ/runs/mosmed

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=============== 1/2  training (4-class severity) ==============="
# div=0.5 included alongside div=0.1: the weight that resolved collapse at K=4 on
# MedMNIST (max slot cosine 0.997 -> 0.139) may not carry to K=8 on real DINOv2
# features, and a 1-epoch probe here still showed 0.996. Cheap to settle by
# measurement rather than assuming the W1 setting transfers.
python scripts/train_cached.py \
    --cache "$CACHE" --splits "$SPLITS" --out "$RUNS" \
    --arms mean gated_abmil mh_abmil slot slot:div=0.1 slot:div=0.5 \
    --seeds 0 1 2 --epochs 40 --num-classes 4 --num-workers 8

echo "=============== 2/2  alignment (50 expert masks) ==============="
# Alignment is run on whichever slot arm actually kept slots distinct --
# scoring a collapsed model would measure nothing.
BEST_SLOT=$(python - <<'EOF'
import json,pathlib
p=pathlib.Path("runs/mosmed/summary.json")
if p.exists():
    s=json.load(p.open())["summary"]
    c={k:v for k,v in s.items() if k.startswith("slot") and
       v.get("max_off_diag_cos",{}).get("mean",1.0) < 0.9}
    print(max(c, key=lambda k: c[k]["auc"]["mean"]).replace(":","_") if c else "")
EOF
)
# NB: written as `if`, not `[ -z ... ] && ... && ...`. Under `set -e` the latter
# returns 1 whenever BEST_SLOT is non-empty (the common case) and kills the job
# right before the alignment step.
if [ -z "$BEST_SLOT" ]; then
    echo "NOTE: no slot arm kept max slot cosine < 0.9 -- slots collapsed on"
    echo "      MosMed at K=8. Alignment below is therefore measuring a"
    echo "      degenerate model; treat the numbers as a collapse report."
    BEST_SLOT="slot_div=0.5"
fi
echo "alignment slot arm: $BEST_SLOT"
for PAIR in "$BEST_SLOT:slot" "mh_abmil:mh_abmil"; do
    ARM=${PAIR%%:*}
    POOL=${PAIR##*:}
    CKPT="$RUNS/$ARM/seed0/best.pt"
    if [ -f "$CKPT" ]; then
        echo "--- alignment: $ARM"
        python scripts/eval_alignment.py \
            --cache "$CACHE" --checkpoint "$CKPT" --splits "$SPLITS" \
            --pooling "$POOL" --num-slots 8 --dim 256 --num-classes 4 \
            --out "$RUNS/alignment_$ARM" || echo "alignment failed for $ARM"
    else
        echo "missing checkpoint $CKPT -- skipping"
    fi
done

echo "=============== mosmed pipeline complete ==============="
