#!/bin/bash
#SBATCH --job-name=slotmil-cond-arm
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa5000:1
# GPU type pinned in --gres itself. A node-level --constraint=Turing|Ampere is
# NOT sufficient: legacygpu12/15 carry both Pascal and Turing cards, so the node
# matches the feature and SLURM still hands out a card this torch build has no
# kernels for.
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-8
#SBATCH --requeue
#SBATCH --output=/fs/nexus-scratch/bthapar/SlotMIL/logs/%x-%A_%a.out
#SBATCH --error=/fs/nexus-scratch/bthapar/SlotMIL/logs/%x-%A_%a.err
#
# The confirmatory sweep for a SECONDARY condition -- malignancy or
# balanced_presence, added by the 2026-08-15 scoping amendment.
#
#   COND=malignancy        sbatch scripts/slurm/lidc_condition_confirmatory_array.sbatch
#   COND=balanced_presence sbatch scripts/slurm/lidc_condition_confirmatory_array.sbatch
#
# Why this is a third file rather than a flag on lidc_confirmatory_array.sbatch:
# that script trains the condition carrying the pre-registered hypothesis family
# and must stay submittable exactly as it is, for the same reason it was itself
# split off from lidc_train_array.sbatch. A flag that changes which split a
# confirmatory job trains on is the kind of thing that fails silently.
#
# Neither condition here carries the hypothesis family. `nodule_present` does,
# and the config says so (`hypothesis_family_condition`). These two are
# consistency checks and are reported per-condition; correcting Holm across all
# three would spend the power of the condition carrying the claim in order to
# protect two that cannot carry it -- balanced_presence has 38 test bags, and
# malignancy's supervised patch ceiling is 0.6516 against nodule presence's
# 0.9102. Ruled 2026-08-15, before any confirmatory number existed.
#
# The split is looked up from the pre-registration by condition name and then
# VERIFIED, never generated. Pointing make_splits.py at one of these paths would
# overwrite a seed-2027 partition and break the hash pinned in the frozen config.

set -euo pipefail
PROJ=/fs/nexus-scratch/bthapar/SlotMIL
cd "$PROJ"

COND=${COND:?set COND=malignancy or COND=balanced_presence}

source /usr/share/Modules/init/bash
module load Python3/3.11.11
# shellcheck disable=SC1091
source "$PROJ/.venv/bin/activate"

export TORCH_HOME=/fs/nexus-scratch/bthapar/.cache/torch
export TMPDIR=/fs/nexus-scratch/bthapar/.tmp
export PYTHONUNBUFFERED=1
# mh_abmil materialises [4, 45312, 2304] intermediates for V(x) and U(x); with
# tanh, sigmoid and their product that is ~6 GB per forward on a 24 GB card.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Exactly the arms declared `status: implemented`, in declaration order --
# identical to lidc_confirmatory_array.sbatch, because a secondary condition
# that trained a different arm set would not be a consistency check on anything.
# transmil is still `status: planned`; append it, and raise --array, only once it
# is promoted through the amendment procedure in ENGINEERING.md.
ARMS=(mean abmil gated_abmil mh_abmil "slot:div=0.5" \
      centre_gaussian "normal_guidance:lam=0.1" clam_sb dsmil)

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#ARMS[@]}" ]; then
    echo "array index $SLURM_ARRAY_TASK_ID beyond ${#ARMS[@]} arms -- --array and ARMS have drifted" >&2
    exit 1
fi
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}

CACHE=$PROJ/data/lidc/features_dinov2_vitb14.h5
OUT=$PROJ/runs/lidc_${COND}_confirmatory

echo "array task $SLURM_ARRAY_TASK_ID -> condition '$COND' arm '$ARM'"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Resolve the split from the frozen config and verify it. A mismatch on any task
# is fatal for that task; this is the same check prereg_freeze.py --check runs.
SPLITS=$(python - "$COND" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, "/fs/nexus-scratch/bthapar/SlotMIL")
from slotmil import prereg

cond_name = sys.argv[1]
pre = prereg.load()
cond = next((c for c in pre.get("conditions") if c["name"] == cond_name), None)
if cond is None:
    declared = ", ".join(c["name"] for c in pre.get("conditions"))
    raise SystemExit(f"condition {cond_name!r} is not pre-registered. Declared: {declared}")
if cond.get("carries_hypothesis_family"):
    raise SystemExit(
        f"{cond_name} carries the hypothesis family; train it with "
        "lidc_confirmatory_array.sbatch, which is pinned to that split."
    )

declared = pre.split(cond["dataset"], cond["split"])
path = Path(declared["path"])
actual = json.loads(path.read_text()).get("hash")
if actual != declared["recorded_hash"]:
    raise SystemExit(
        f"{path} hash {actual} != recorded {declared['recorded_hash']}. The split has "
        "been rewritten. Do NOT train on it -- restore it, or the confirmatory claims "
        "rest on a partition the pre-registration does not describe."
    )
print(str(path))
PY
)
echo "[splits] $COND -> $SPLITS verified"

python scripts/train_cached.py \
    --cache "$CACHE" --splits "$SPLITS" --out "$OUT" \
    --arms "$ARM" --seeds 0 1 2 3 4 --epochs 40 --num-workers 4 \
    --role confirmatory

echo "condition '$COND' arm '$ARM' complete"
