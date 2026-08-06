#!/bin/bash
#SBATCH --job-name=slotmil-lidc
#SBATCH --account=nexus
#SBATCH --qos=medium
#SBATCH --partition=tron
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/fs/nexus-scratch/bthapar/SlotMIL/logs/%x-%j.out
#SBATCH --error=/fs/nexus-scratch/bthapar/SlotMIL/logs/%x-%j.err
#
# Staged LIDC download + DINOv2 feature extraction.
#
# 24 GB rtxa5000. DINOv2 ViT-B/14 over 32 slices at 224px needs only a few GB,
# and this job is dominated by DICOM download and pylidc mask construction (CPU)
# rather than by the GPU forward pass. The rtxa6000 nodes are also persistently
# contended, so asking for one just means queueing for capacity we do not use.
#
# --requeue is safe because the script is idempotent: series already in the h5
# cache are skipped, so a preempted job resumes rather than restarting.

set -euo pipefail
PROJ=/fs/nexus-scratch/bthapar/SlotMIL
mkdir -p "$PROJ/logs"

source /usr/share/Modules/init/bash
module load Python3/3.11.11
# shellcheck disable=SC1091
source "$PROJ/.venv/bin/activate"

export TORCH_HOME=/fs/nexus-scratch/bthapar/.cache/torch
export TMPDIR=/fs/nexus-scratch/bthapar/.tmp
export PYTHONUNBUFFERED=1

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /fs/nexus-scratch/bthapar | tail -1

python "$PROJ/scripts/download_lidc_staged.py" "$@"
