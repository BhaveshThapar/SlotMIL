#!/bin/bash
#SBATCH --job-name=slotmil-mosmed-conf
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
#SBATCH --array=0-9
#SBATCH --requeue
#SBATCH --output=/fs/nexus-scratch/bthapar/SlotMIL/logs/%x-%A_%a.out
#SBATCH --error=/fs/nexus-scratch/bthapar/SlotMIL/logs/%x-%A_%a.err
#
# The confirmatory sweep for `mosmed_severity`, the one condition on the one
# dataset that carries H9 -- and nothing else, as the config's own split note
# says.
#
#   sbatch scripts/slurm/mosmed_confirmatory_array.sbatch
#
# Why this exists at all, given runs/mosmed/ already holds trained MosMed arms:
# those were trained on 2026-08-06, before the pre-registration was frozen. They
# carry no `analysis_role`, no `splits_hash` and no `prereg` block, because
# train_cached.py did not stamp any of that until the 2026-08-15 amendment. H9
# is scored on tags shared between MosMed's template family and LIDC's, so
# reusing them would rest the one MosMed hypothesis on discovery-era models
# whose provenance cannot be checked. They are left in place, untouched.
#
# Why a fourth training file rather than a flag: the same reason
# lidc_condition_confirmatory_array.sbatch is a third one. A flag that changes
# which dataset a confirmatory job trains on is the kind of thing that fails
# silently, and here it would fail by pairing one dataset's checkpoints with
# another's features -- which produces a complete, normal-looking, wrong result.
#
# --num-classes is NOT passed. train_cached.py reads num_classes from the split
# file and overrides the flag when they disagree (scripts/train_cached.py:123),
# so passing 4 here would be a second copy of a number the split already owns.
# MosMed is 4-class severity with CT-4's two volumes folded into CT-3 at split
# time; the split file is where that decision lives.
#
# The arm set is LIDC's, in declaration order, because H9 intersects tag sets
# across the two datasets: an arm trained here and not there contributes no
# matched tag, and choosing a smaller arm set AFTER seeing which tags order
# would be the best-of-k error the template denominator ruling already refuses.
#
# Two shape differences from LIDC worth knowing before reading a log. MosMed
# bags are ~9.7k instances (38 slices x 256) against LIDC's ~43.8k, so this is
# the cheaper sweep. And DSMIL emits `k_eff = num_classes` tokens, so it is K=4
# here against K=2 on LIDC -- which matters because H7's entropy_matched_random
# is undefined for K=1 arms, though H7 is scored on nodule_present alone.

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

# Identical to both LIDC training arrays, in declaration order. This list and
# --array must be raised together; the bounds check below is what catches it
# when they are not.
ARMS=(mean abmil gated_abmil mh_abmil "slot:div=0.5" \
      centre_gaussian "normal_guidance:lam=0.1" clam_sb dsmil transmil)

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#ARMS[@]}" ]; then
    echo "array index $SLURM_ARRAY_TASK_ID beyond ${#ARMS[@]} arms -- --array and ARMS have drifted" >&2
    exit 1
fi
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}

echo "array task $SLURM_ARRAY_TASK_ID -> arm '$ARM'"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Resolve cache, split and run root from the frozen config and verify the split.
# The run root must agree with confirmatory_collect_array.sbatch's resolver, so
# both derive it the same way rather than each spelling it out.
read -r CACHE SPLITS OUT <<<"$(python - <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, "/fs/nexus-scratch/bthapar/SlotMIL")
from slotmil import prereg

name = "mosmed_severity"
pre = prereg.load()
cond = next((c for c in pre.get("conditions") if c["name"] == name), None)
if cond is None:
    raise SystemExit(f"condition {name!r} is not pre-registered")
if cond.get("carries_hypothesis_family"):
    raise SystemExit(f"{name} now carries the hypothesis family; this array assumes it does not")

ds = cond["dataset"]
declared = pre.split(ds, cond["split"])
path = Path(declared["path"])
actual = json.loads(path.read_text()).get("hash")
if actual != declared["recorded_hash"]:
    raise SystemExit(
        f"{path} hash {actual} != recorded {declared['recorded_hash']}. The split has "
        "been rewritten. Do NOT train on it -- restore it, or H9 rests on a partition "
        "the pre-registration does not describe."
    )
print(pre.get("datasets")[ds]["cache"], path, f"runs/{name}_confirmatory")
PY
)"
echo "[splits] mosmed_severity -> $SPLITS verified"

# The 50 annotated scans are the whole reason this condition exists, and the
# pre-registration commits to where they sit: none in train, 28 in val, 22 in
# test. Checked here rather than trusted, because a split that quietly moved one
# into train would leave H9 scoring a template fit on scans the arms had seen.
python - "$SPLITS" "$CACHE" <<'PY'
import json, sys
from pathlib import Path

import h5py

sp = json.loads(Path(sys.argv[1]).read_text())
with h5py.File(sys.argv[2], "r") as f:
    masked = {uid for uid in f if "mask" in f[uid]}
have = {k: len(masked.intersection(sp[k])) for k in ("train", "val", "test")}
want = {"train": 0, "val": 28, "test": 22}
print(f"[masks] masked scans per split: {have}")
if have != want:
    raise SystemExit(f"masked-scan layout {have} != the pre-registered {want}")
PY

python scripts/train_cached.py \
    --cache "$PROJ/$CACHE" --splits "$PROJ/$SPLITS" --out "$PROJ/$OUT" \
    --arms "$ARM" --seeds 0 1 2 3 4 --epochs 40 --num-workers 4 \
    --role confirmatory

echo "mosmed_severity arm '$ARM' complete"
