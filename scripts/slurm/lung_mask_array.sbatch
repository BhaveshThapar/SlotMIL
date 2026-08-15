#!/bin/bash
#SBATCH --job-name=slotmil-lungmask-arr
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --array=0-7
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/fs/nexus-scratch/bthapar/SlotMIL/logs/%x-%A_%a.out
#SBATCH --error=/fs/nexus-scratch/bthapar/SlotMIL/logs/%x-%A_%a.err
#
# The full 999-series lung-mask pass, sharded eight ways.
#
# One pass measured ~1.5 min/series, so 999 series is ~24h -- exactly the
# walltime ceiling, and SLURM does NOT requeue on TIMEOUT, so a single-task run
# would die at the wall and need manual resubmission. Eight shards bring it to
# ~3h with headroom.
#
# No GPU: there is no DINOv2 pass here, only download, morphology and a pool onto
# the 16x16 grid. The bottleneck is TCIA throughput, which parallelises across
# tasks even though it does not speed up within one.
#
# Shards share one DICOM root on purpose. pylidc reads a single global
# ~/.pylidcrc at import time, so per-shard roots would race on that file; the
# script instead deletes each series' directory immediately after processing it,
# so tasks never touch each other's data. Each shard writes its own HDF5 because
# h5py has no safe concurrent writer.
#
#   sbatch scripts/slurm/lung_mask_array.sbatch
#   # then, once all tasks finish:
#   python scripts/merge_lung_masks.py

set -euo pipefail
PROJ=/fs/nexus-scratch/bthapar/SlotMIL
cd "$PROJ"

source /usr/share/Modules/init/bash
module load Python3/3.11.11
# shellcheck disable=SC1091
source "$PROJ/.venv/bin/activate"

export TMPDIR=/fs/nexus-scratch/bthapar/.tmp
export PYTHONUNBUFFERED=1

I=${SLURM_ARRAY_TASK_ID:-0}
N=${SHARD_COUNT:-8}

echo "shard $I/$N on $(hostname)  free $(df -h /fs/nexus-scratch | tail -1 | awk '{print $4}')"

# Method and threshold are pinned by the pre-registration amendment of
# 2026-08-14 -- fill at lung_thresh 0.0. Do not change them here; change the
# config and log an amendment, or the results stop being confirmatory.
python scripts/lung_mask_lidc.py \
    --method fill \
    --lung-thresh 0.0 \
    --shard-index "$I" --shard-count "$N" \
    --out "data/lidc/lung_masks_shard${I}.h5" \
    --stats-out "runs/lung_mask_shard${I}.json"

echo "=============== shard $I complete ==============="
