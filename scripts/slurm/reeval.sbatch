#!/bin/bash
#SBATCH --job-name=slotmil-reeval
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/fs/nexus-scratch/bthapar/SlotMIL/logs/%x-%j.out
#SBATCH --error=/fs/nexus-scratch/bthapar/SlotMIL/logs/%x-%j.err
set -euo pipefail
PROJ=/fs/nexus-scratch/bthapar/SlotMIL
cd "$PROJ"
source /usr/share/Modules/init/bash
module load Python3/3.11.11
source "$PROJ/.venv/bin/activate"
export TORCH_HOME=/fs/nexus-scratch/bthapar/.cache/torch PYTHONUNBUFFERED=1
python scripts/reeval_all_alignment.py "$@"
