#!/bin/bash
#SBATCH --job-name=slotmil-floor
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa5000:1
# GPU type pinned in --gres, not via --constraint: scavenger nodes like
# legacygpu12/15 carry mixed Pascal and Turing cards, so a node-level feature
# match still hands out a card this torch build has no kernels for.
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/fs/nexus-scratch/bthapar/SlotMIL/logs/%x-%j.out
#SBATCH --error=/fs/nexus-scratch/bthapar/SlotMIL/logs/%x-%j.err
#
# H3: the reference floor, from the pre-registered 30 untrained initialisations
# on the confirmatory split.
#
# Unlike the lung-mask job this one wants a GPU -- it holds 30 models in memory
# and runs every one of them on each batch, which is the whole point: the naive
# loop over null_collect_attn.py would read the 78 GB cache 60 times to run
# forward passes that cost almost nothing. Two passes total here.
#
#   sbatch scripts/slurm/untrained_floor.sbatch
#   sbatch scripts/slurm/untrained_floor.sbatch --pooling mh_abmil \
#       --out runs/untrained_floor_mh.json

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
# Thirty models forwarded in sequence on one batch leave the caching allocator
# heavily fragmented -- the mh_abmil run OOM'd with 4.3 GB reserved-but-unallocated
# against a 6.15 GB request. Expandable segments is the allocator's own remedy.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)"

# n_inits and seed_base come from the pre-registration, not from here. Passing
# them on the command line would let a run quietly disagree with the document it
# claims to follow.
python scripts/untrained_floor.py \
    --splits data/lidc/splits_confirmatory.json \
    --role confirmatory \
    "$@"

echo "=============== untrained floor complete ==============="
