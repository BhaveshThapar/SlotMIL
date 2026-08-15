#!/bin/bash
#SBATCH --job-name=slotmil-lidc-arm
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
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-7
#SBATCH --requeue
#SBATCH --output=/fs/nexus-scratch/bthapar/SlotMIL/logs/%x-%A_%a.out
#SBATCH --error=/fs/nexus-scratch/bthapar/SlotMIL/logs/%x-%A_%a.err
#
# One arm per array task, so the six arms train in parallel.
#
# The first attempt trained all six serially in one job and hit the 12 h wall
# clock partway through the third arm. As an array each task runs 3 seeds of one
# arm.
#
# Runs on the SCAVENGER partition, which has its own quota separate from tron.
# The tron partition caps a user at cpu=32 and gres/gpu=4; this array previously
# consumed the entire allocation and blocked the user's other (lg-ppo) jobs.
# Profiling showed the GPUs at 0% utilisation -- the work is I/O bound, not
# compute bound -- so it was holding four GPUs to do no GPU work.
#
# Scavenger is preemptible (PreemptMode=REQUEUE), which is safe here because
# train_cached.py resumes per seed: a requeued task skips seeds whose
# result.json already exists, so preemption costs at most one seed rather than
# the whole arm. --gres=gpu:1 (untyped) so it can land on whatever is free,
# including the idle a100/h100/l40s nodes -- though given 0% GPU utilisation the
# card type is irrelevant to throughput.

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

ARMS=(mean gated_abmil mh_abmil slot "slot:div=0.1" "slot:div=0.5" \
      centre_gaussian "normal_guidance:lam=0.1")
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
    --arms "$ARM" --seeds 0 1 2 --epochs 40 --num-workers 4

echo "arm '$ARM' complete"
