#!/bin/bash
#SBATCH --job-name=slotmil-ng-preflight
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa5000:1
# GPU type pinned in --gres itself, not via --constraint: scavenger nodes such as
# legacygpu12/15 carry both Pascal and Turing cards, so a node-level arch feature
# still hands out a card this torch build has no kernels for. See
# lidc_train_array.sbatch for the full account.
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=/fs/nexus-scratch/bthapar/SlotMIL/logs/%x-%j.out
#SBATCH --error=/fs/nexus-scratch/bthapar/SlotMIL/logs/%x-%j.err
#
# Calibration pre-flight for Normal Guidance's lam, and NOTHING else.
#
# The pre-registered lam=0.1 may be inert. A near-uniform slice marginal pays
# about 0.089 nats, so the term would contribute ~0.009 against a bag loss around
# 0.4. If that is what happens, H6's first clause ("NG raises slice-level AUC over
# its base arm") fails for a reason that is about scale rather than about the
# science -- and re-tuning lam after seeing a confirmatory result would be a
# protocol violation.
#
# So this runs on the DISCOVERY split (data/lidc/splits.json, seed 0), which
# PREREGISTRATION.md:40-46 already labels exploratory by construction and which
# cannot carry confirmatory claims.
#
# WHAT MAY BE READ FROM THIS RUN: loss_kl_prior and loss_bag magnitudes from
# history.json. That is the entire permitted scope.
#
# WHAT MAY NOT: any localisation estimand, any attention dump, any test-split
# number, any comparison against the base arm. If lam is amended on the strength
# of this run, AMENDMENTS.md must record that the results seen were discovery-split
# loss magnitudes only.
#
# centre_gaussian is trained alongside purely as a smoke test -- it has no tunable
# anything, so nothing about it is being calibrated.

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
# Thirty-odd forwards over a 45k-instance bag leave the caching allocator
# fragmented; expandable segments is the allocator's own remedy. Cheap insurance
# here even though neither of these arms carries the mh_abmil memory profile.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CACHE=$PROJ/data/lidc/features_dinov2_vitb14.h5
SPLITS=$PROJ/data/lidc/splits.json   # DISCOVERY, deliberately

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python scripts/train_cached.py \
    --cache "$CACHE" --splits "$SPLITS" --out "$PROJ/runs/ng_preflight" \
    --arms centre_gaussian "normal_guidance:lam=0.1" \
    --seeds 0 --epochs 8 --num-workers 4

echo "pre-flight complete -- read loss_kl_prior and loss_bag ONLY"
