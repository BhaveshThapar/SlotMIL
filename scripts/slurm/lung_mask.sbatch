#!/bin/bash
#SBATCH --job-name=slotmil-lungmask
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/fs/nexus-scratch/bthapar/SlotMIL/logs/%x-%j.out
#SBATCH --error=/fs/nexus-scratch/bthapar/SlotMIL/logs/%x-%j.err
#
# Derive per-patch lung masks for LIDC. Unlike every other job in this
# directory, it asks for NO GPU: there is no DINOv2 pass here, only download,
# pylidc volume load, morphology and an adaptive-avg-pool onto the 16x16 grid.
# Requesting a card would idle it for hours and queue behind GPU demand for no
# reason -- the bottleneck is TCIA throughput.
#
# --requeue is load-bearing on scavenger, where preemption is routine. The
# script resumes from whatever landed: the output HDF5 is written per series and
# scanned on startup, per-batch stats are flushed to JSON, and stale staging
# directories from a killed batch are cleared before anything else runs.
#
# Args are forwarded, so both modes go through this file:
#   sbatch scripts/slurm/lung_mask.sbatch --compare-methods --limit 30 \
#          --sample-seed 2027 --exclude-from data/lidc/splits_confirmatory.json
#   sbatch scripts/slurm/lung_mask.sbatch --method fill_hull

set -euo pipefail
PROJ=/fs/nexus-scratch/bthapar/SlotMIL
cd "$PROJ"

source /usr/share/Modules/init/bash
module load Python3/3.11.11
# shellcheck disable=SC1091
source "$PROJ/.venv/bin/activate"

export TMPDIR=/fs/nexus-scratch/bthapar/.tmp
export PYTHONUNBUFFERED=1

echo "host $(hostname)  free $(df -h /fs/nexus-scratch | tail -1 | awk '{print $4}')"
python scripts/lung_mask_lidc.py "$@"
echo "=============== lung mask job complete ==============="
