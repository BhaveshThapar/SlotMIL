#!/bin/bash
# Fetch lesion MASKS (never images) for the confound atlas, into data/atlas/.
#
# Disk discipline: scratch is a 200 GB per-user quota, so nothing here lands a
# multi-GB archive on disk. MSD's 28.5 GB tar is streamed through a pipe and
# only labelsTr/ (~10 MB) is extracted; KiTS19's masks are committed in its git
# repository so the imaging download is never invoked; COVID-19-CT-Seg's
# infection masks are a ~5 MB zip.
#
# Anonymous sources only. LiTS (Kaggle credentials) and LNDb (Zenodo access
# request) are user-gated and deliberately absent here.
#
#   bash scripts/atlas_fetch.sh [covid_ct_seg|kits19|msd_task06|all]

set -euo pipefail
ROOT=/fs/nexus-scratch/bthapar/SlotMIL/data/atlas
mkdir -p "$ROOT"
what="${1:-all}"

covid_ct_seg() {
  # Zenodo 3757476: 20 cases, infection masks only.
  local d="$ROOT/covid_ct_seg"
  [ -d "$d/Infection_Mask" ] && { echo "[atlas] covid_ct_seg present"; return; }
  mkdir -p "$d"
  curl -sL "https://zenodo.org/records/3757476/files/Infection_Mask.zip?download=1" \
    -o "$d/Infection_Mask.zip"
  # The zip is flat; the adapter expects an Infection_Mask/ directory.
  unzip -q -o "$d/Infection_Mask.zip" -d "$d/Infection_Mask"
  rm "$d/Infection_Mask.zip"
  echo "[atlas] covid_ct_seg: $(ls "$d/Infection_Mask" | wc -l) masks"
}

kits19() {
  # Masks are committed in the kits19 repo; imaging is a separate download
  # this script never performs.
  local d="$ROOT/kits19"
  [ -d "$d/data" ] && { echo "[atlas] kits19 present"; return; }
  git clone --quiet --depth 1 https://github.com/neheller/kits19 "$d"
  echo "[atlas] kits19: $(ls -d "$d"/data/case_*/ | wc -l) cases," \
       "$(find "$d/data" -name segmentation.nii.gz | wc -l) segmentations"
}

msd_task06() {
  # Stream the 28.5 GB tar; keep only labelsTr. Nothing large touches disk.
  local d="$ROOT/msd_task06"
  [ -d "$d/labelsTr" ] && { echo "[atlas] msd_task06 present"; return; }
  mkdir -p "$d"
  curl -sL "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task06_Lung.tar" \
    | tar -x -C "$d" --strip-components=1 --wildcards "Task06_Lung/labelsTr/lung_*"
  echo "[atlas] msd_task06: $(ls "$d/labelsTr" | grep -c '^lung_') labels"
}

case "$what" in
  covid_ct_seg) covid_ct_seg ;;
  kits19) kits19 ;;
  msd_task06) msd_task06 ;;
  all) covid_ct_seg; kits19; msd_task06 ;;
  *) echo "unknown target $what" >&2; exit 2 ;;
esac
