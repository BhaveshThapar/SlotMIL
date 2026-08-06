#!/bin/bash
#SBATCH --job-name=slotmil-train
#SBATCH --account=nexus
#SBATCH --qos=medium
#SBATCH --partition=tron
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/fs/nexus-scratch/bthapar/SlotMIL/logs/%x-%j.out
#SBATCH --error=/fs/nexus-scratch/bthapar/SlotMIL/logs/%x-%j.err
#
# Training runs on cached features, so a 24 GB rtxa5000 is ample -- that is the
# whole point of the frozen-feature cache (plan.md line 129).
#
# Usage: sbatch scripts/slurm/train.sbatch scripts/w1_gonogo.py --epochs 30

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

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python "$@"
