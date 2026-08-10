#!/bin/bash
#SBATCH --job-name=slotmil-lidc-align
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa5000:1
# Scavenger contains Maxwell/Pascal cards (gtxtitanx, gtx1080ti, p6000,
# titanxp, p100) that this torch build has no kernels for -- an untyped
# gpu:1 landed on a GTX TITAN X (sm_52) and died instantly with
# 'no kernel image is available for execution on the device'.
# torch.cuda.get_arch_list() is sm_75/80/86/90/100/120.
#
# A node-level --constraint=Turing|Ampere|Hopper is NOT sufficient: nodes
# like legacygpu12/15 carry both Pascal and Turing cards, so the node
# matched the feature and SLURM still handed out a GTX 1080 Ti. The GPU
# type must be pinned in --gres itself. rtxa5000 (Ampere, 24 GB) is widely
# available in scavenger; since profiling showed 0% GPU utilisation, the
# specific card is irrelevant to throughput -- it only has to run.
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/fs/nexus-scratch/bthapar/SlotMIL/logs/%x-%j.out
#SBATCH --error=/fs/nexus-scratch/bthapar/SlotMIL/logs/%x-%j.err
#
# Merge the training array's results, then run slot-to-finding alignment and
# faithfulness. Submitted with --dependency=afterany on the array: afterok would
# block the whole analysis if a single arm failed, and the comparison is still
# informative with five arms.

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

CACHE=$PROJ/data/lidc/features_dinov2_vitb14.h5
SPLITS=$PROJ/data/lidc/splits.json
RUNS=$PROJ/runs/lidc

echo "=============== 1/3  merge ==============="
python scripts/merge_results.py --runs "$RUNS"

echo "=============== 2/3  alignment ==============="
BEST_SLOT=$(python - <<'EOF'
import json, pathlib
p = pathlib.Path("runs/lidc/summary.json")
if p.exists():
    s = json.load(p.open())["summary"]
    c = {k: v for k, v in s.items() if k.startswith("slot")
         and v.get("max_off_diag_cos", {}).get("mean", 1.0) < 0.9}
    print(max(c, key=lambda k: c[k]["auc"]["mean"]) if c else "")
EOF
)
if [ -z "$BEST_SLOT" ]; then
    echo "NOTE: no slot arm kept max slot cosine < 0.9 -- slots collapsed on LIDC."
    echo "      Alignment below measures a degenerate model; read it as a"
    echo "      collapse report, not an alignment result."
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
            --pooling "$POOL" --num-slots 8 --dim 256 \
            --out "$RUNS/alignment_$ARM" || echo "ALIGNMENT FAILED for $ARM"
    else
        echo "missing checkpoint $CKPT -- skipping"
    fi
done

echo "=============== 3/3  figures + faithfulness ==============="
CKPT="$RUNS/$BEST_SLOT/seed0/best.pt"
if [ -f "$CKPT" ]; then
    python scripts/make_figures.py \
        --cache "$CACHE" --checkpoint "$CKPT" --splits "$SPLITS" \
        --pooling slot --num-slots 8 --dim 256 \
        --out "$RUNS/figures" || echo "FIGURES FAILED"
fi

echo "=============== lidc analysis complete ==============="
