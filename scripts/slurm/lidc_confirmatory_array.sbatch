#!/bin/bash
#SBATCH --job-name=slotmil-conf-arm
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa5000:1
# Pin the GPU type in --gres itself. A node-level --constraint=Turing|Ampere is
# NOT sufficient: legacygpu12/15 carry both Pascal and Turing cards, so the node
# matches the feature and SLURM still hands out a card this torch build has no
# kernels for. torch.cuda.get_arch_list() is sm_75/80/86/90/100/120.
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --requeue
#SBATCH --output=/fs/nexus-scratch/bthapar/SlotMIL/logs/%x-%A_%a.out
#SBATCH --error=/fs/nexus-scratch/bthapar/SlotMIL/logs/%x-%A_%a.err
#
# THE CONFIRMATORY SWEEP. A separate file from lidc_train_array.sbatch on
# purpose: that one trains the discovery split and must stay submittable exactly
# as it was, or the published discovery numbers stop being reproducible.
#
# Four things differ, and the first two fail SILENTLY if you get them wrong,
# which is why they are here rather than as flags on the old script:
#
#   1. --out is runs/lidc_confirmatory, NOT runs/lidc. The discovery root holds
#      18 completed result.json files. train_cached.py's resume now compares
#      splits_hash and refuses to cross splits, but the right answer is a fresh
#      root, not a caught error.
#   2. The split is NOT regenerated. lidc_train_array.sbatch has task 0 run
#      make_splits.py over $SPLITS before training; pointed at the confirmatory
#      path that would overwrite a seed-2027 partition with a seed-0 one and
#      break the hash pinned in configs/prereg/isbi2027.yaml. Here the split is
#      VERIFIED and never written.
#   3. Five seeds, per protocol.seeds in the pre-registration. The discovery
#      sweep ran three.
#   4. --role confirmatory, which stamps provenance into every result.json and
#      suppresses test metrics on stdout. Do not add --report-test to this job.

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
# mh_abmil materialises [4, 45312, 2304] intermediates for V(x) and U(x); with
# tanh, sigmoid and their product that is ~6 GB per forward on a 24 GB card.
# Present in ng_lambda_preflight.sbatch and untrained_floor.sbatch, and missing
# from lidc_train_array.sbatch, which is the arm most likely to need it.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Exactly the arms declared `status: implemented` in
# configs/prereg/isbi2027.yaml, in declaration order.
#
# Two deliberate differences from the discovery sweep's ARMS list:
#   - `abmil` is pre-registered, constructible and blinded, and the discovery
#     array never trained it. It is here.
#   - bare `slot` and `slot:div=0.1` are NOT pre-registered arms. They were in
#     the discovery list; training them here would put undeclared arms in a
#     confirmatory table.
#
# clam_sb and dsmil were promoted to `implemented` by the 2026-08-15 amendment
# and are appended here, with --array raised to match. transmil followed on
# 2026-08-17, taking --array to 0-9 and the arm count to ten. Its pre-committed
# drop date of 2026-09-22 did not fire: it trains at LIDC bag size with headroom
# (job 7266534, 16.22 GiB peak of 23.55). This list and --array must be raised
# together; the bounds check below is what catches it when they are not.
ARMS=(mean abmil gated_abmil mh_abmil "slot:div=0.5" \
      centre_gaussian "normal_guidance:lam=0.1" clam_sb dsmil transmil)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#ARMS[@]}" ]; then
    echo "array index $SLURM_ARRAY_TASK_ID beyond ${#ARMS[@]} arms -- --array and ARMS have drifted" >&2
    exit 1
fi

CACHE=$PROJ/data/lidc/features_dinov2_vitb14.h5
SPLITS=$PROJ/data/lidc/splits_confirmatory.json
OUT=$PROJ/runs/lidc_confirmatory

echo "array task $SLURM_ARRAY_TASK_ID -> arm '$ARM'"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Verify, never generate. Every task checks; a mismatch on any task is fatal for
# that task, and the recorded hash lives in the frozen config so this is the
# same check scripts/prereg_freeze.py --check performs.
python - "$SPLITS" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, "/fs/nexus-scratch/bthapar/SlotMIL")
from slotmil import prereg

path = sys.argv[1]
pre = prereg.load()
declared = pre.split("lidc", "confirmatory")
if str(Path(declared["path"]).name) != Path(path).name:
    raise SystemExit(f"{path} is not the declared confirmatory split {declared['path']}")
actual = json.loads(Path(path).read_text()).get("hash")
if actual != declared["recorded_hash"]:
    raise SystemExit(
        f"confirmatory split hash {actual} != recorded {declared['recorded_hash']}. "
        "The split has been rewritten. Do NOT train on it -- restore it, or the "
        "confirmatory claims rest on a partition the pre-registration does not "
        "describe."
    )
print(f"[splits] {path} verified at {actual} (seed {declared['seed']})")
PY

python scripts/train_cached.py \
    --cache "$CACHE" --splits "$SPLITS" --out "$OUT" \
    --arms "$ARM" --seeds 0 1 2 3 4 --epochs 40 --num-workers 4 \
    --role confirmatory

echo "arm '$ARM' complete"
