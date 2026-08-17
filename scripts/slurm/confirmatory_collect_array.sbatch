#!/bin/bash
#SBATCH --job-name=slotmil-conf-collect
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa5000:1
# GPU type pinned in --gres itself: legacygpu12/15 carry both Pascal and Turing
# cards, so a node-level --constraint matches and SLURM still hands out a card
# this torch build has no kernels for.
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-8
#SBATCH --requeue
#SBATCH --output=/fs/nexus-scratch/bthapar/SlotMIL/logs/%x-%A_%a.out
#SBATCH --error=/fs/nexus-scratch/bthapar/SlotMIL/logs/%x-%A_%a.err
#
# Stage 2 of the confirmatory pipeline: cache float32 attention for every trained
# checkpoint. Everything downstream -- axis_gate (H1, H2), template_family
# (H5, H10), the decomposition (H4, H6) -- reads these dumps and nothing else.
#
#   COND=nodule_present    sbatch scripts/slurm/confirmatory_collect_array.sbatch
#   COND=malignancy        sbatch scripts/slurm/confirmatory_collect_array.sbatch
#   COND=balanced_presence sbatch scripts/slurm/confirmatory_collect_array.sbatch
#   COND=mosmed_severity   sbatch scripts/slurm/confirmatory_collect_array.sbatch
#
# Any pre-registered condition, not just the LIDC three: the split, the feature
# cache and the run root all resolve from the config below, so a second dataset
# needs a condition stanza and nothing here.
#
# One array task per arm; the five seeds loop inside, because a dump costs ~2
# minutes (job 7256442 ran 00:01:49 for one checkpoint, val+test) and 45 separate
# jobs would spend more time queueing than collecting.
#
# --role confirmatory does two things that both matter here. It forces float32:
# protocol.dtype is pre-registered because fp16 caching cost seed 2 0.034 AUC,
# and null_collect_attn.py's own default is float16, so a run that forgot the
# flag would produce a complete set of dumps that no estimand may read -- late,
# silently, and expensively. And it writes a .prereg.json sidecar per dump, which
# is what lets prereg_freeze.py --check place the dump on the amendment chain.
#
# Collection runs fp32 with no autocast, so it is the tighter memory budget of
# the two paths; training gets AMP for free. One model at a time, so 48G is ample.

set -euo pipefail
PROJ=/fs/nexus-scratch/bthapar/SlotMIL
cd "$PROJ"

COND=${COND:?set COND to a pre-registered condition name, e.g. nodule_present}

source /usr/share/Modules/init/bash
module load Python3/3.11.11
# shellcheck disable=SC1091
source "$PROJ/.venv/bin/activate"

export TORCH_HOME=/fs/nexus-scratch/bthapar/.cache/torch
export TMPDIR=/fs/nexus-scratch/bthapar/.tmp
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Same nine arms, same declaration order, as both training arrays. mean and
# centre_gaussian are collected even though their hypothesis outcomes are fixed
# by construction: H1's tie_floor_check asserts mean's gap is exactly 0, and both
# appear in every reported table. An arm excluded from a hypothesis is not
# excluded from the paper.
ARMS=(mean abmil gated_abmil mh_abmil "slot:div=0.5" \
      centre_gaussian "normal_guidance:lam=0.1" clam_sb dsmil)
SEEDS=(0 1 2 3 4)

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#ARMS[@]}" ]; then
    echo "array index $SLURM_ARRAY_TASK_ID beyond ${#ARMS[@]} arms -- --array and ARMS have drifted" >&2
    exit 1
fi
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}

# train_cached.py writes the run directory as the spec with ':' replaced by '_',
# and the pooling is the spec up to the first ':'.
ARM_DIR=${ARM//:/_}
POOLING=${ARM%%:*}

# Resolve the split, the cache and the run root from the frozen config, so this
# script holds no second copy of any of them. Refuses an undeclared condition.
#
# This was LIDC-only until MosMed's confirmatory collection needed it. The guard
# that enforced that is gone, not loosened: what it was really protecting was the
# hardcoded `CACHE=data/lidc/...` two lines below, which would have paired MosMed
# checkpoints with LIDC features and produced dumps that looked entirely normal.
# The cache now comes from the same config stanza as the split, so the pairing
# cannot come apart and the guard has nothing left to guard.
read -r SPLITS CACHE RUN_ROOT <<<"$(python - "$COND" <<'PY'
import sys
sys.path.insert(0, "/fs/nexus-scratch/bthapar/SlotMIL")
from slotmil import prereg

name = sys.argv[1]
pre = prereg.load()
cond = next((c for c in pre.get("conditions") if c["name"] == name), None)
if cond is None:
    declared = ", ".join(c["name"] for c in pre.get("conditions"))
    raise SystemExit(f"condition {name!r} is not pre-registered. Declared: {declared}")

ds = cond["dataset"]
sp = pre.split(ds, cond["split"])
cache = pre.get("datasets")[ds]["cache"]
# The family carrier keeps its historical directory name; every other condition
# is named for itself. Spelled as three explicit branches rather than one clever
# expression, because a run root that is subtly wrong reads as "training never
# finished" rather than as a path bug.
if cond.get("carries_hypothesis_family"):
    root = "runs/lidc_confirmatory"
elif ds == "lidc":
    root = f"runs/lidc_{name}_confirmatory"
else:
    root = f"runs/{name}_confirmatory"
print(sp["path"], cache, root)
PY
)"

OUTDIR=$PROJ/runs/nulls_${COND}_confirmatory
CACHE=$PROJ/$CACHE

echo "task $SLURM_ARRAY_TASK_ID | condition '$COND' | arm '$ARM' | pooling '$POOLING'"
echo "  splits   $SPLITS"
echo "  run root $RUN_ROOT"
echo "  outdir   $OUTDIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# The K=8 dumps run ~490 MB per seed for val+test and the scratch filesystem is
# shared. Refuse to start rather than fill it and take out every other running
# job -- a partial dump set is recoverable, a full filesystem is not.
FREE_GB=$(df -BG --output=avail "$PROJ" | tail -1 | tr -dc '0-9')
if [ "$FREE_GB" -lt 20 ]; then
    echo "only ${FREE_GB}G free on $PROJ; refusing to collect (need >= 20G)" >&2
    exit 1
fi
echo "  free     ${FREE_GB}G"

for SEED in "${SEEDS[@]}"; do
    CKPT=$PROJ/$RUN_ROOT/$ARM_DIR/seed$SEED/best.pt
    RESULT=$PROJ/$RUN_ROOT/$ARM_DIR/seed$SEED/result.json
    TAG=${ARM_DIR}_seed${SEED}

    if [ ! -f "$CKPT" ]; then
        echo "[$TAG] no checkpoint at $CKPT -- arm/seed did not finish training" >&2
        exit 1
    fi

    # The checkpoint must have been produced by a confirmatory run on the split
    # we are about to score against. Without this a dump could silently pair a
    # discovery checkpoint with a confirmatory split and look entirely normal.
    python - "$RESULT" "$SPLITS" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, "/fs/nexus-scratch/bthapar/SlotMIL")
from slotmil import prereg

res = json.loads(Path(sys.argv[1]).read_text())
if res.get("analysis_role") != "confirmatory":
    raise SystemExit(f"{sys.argv[1]} has analysis_role={res.get('analysis_role')!r}, "
                     "not 'confirmatory'")
want = json.loads(Path(sys.argv[2]).read_text())["hash"]
if res.get("splits_hash") != want:
    raise SystemExit(f"{sys.argv[1]} was trained on splits_hash "
                     f"{res.get('splits_hash')!r}, not {want!r}")
stamped = res.get("prereg", {}).get("prereg_hash")
current = prereg.load().hash
if stamped != current:
    print(f"  NOTE: checkpoint stamped {stamped}, current is {current} "
          f"(placed on the chain by prereg_freeze.py --check)")
PY

    echo "[$TAG] collecting"
    python scripts/null_collect_attn.py \
        --cache "$CACHE" --splits "$SPLITS" \
        --ckpt "$CKPT" --pooling "$POOLING" --num-slots 8 \
        --dtype float32 --role confirmatory \
        --tag "$TAG" --outdir "$OUTDIR"
done

echo "condition '$COND' arm '$ARM' collected for seeds ${SEEDS[*]}"
