#!/bin/bash
#SBATCH --job-name=slotmil-conf-score
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --output=/fs/nexus-scratch/bthapar/SlotMIL/logs/%x-%j.out
#SBATCH --error=/fs/nexus-scratch/bthapar/SlotMIL/logs/%x-%j.err
#
# Stage 4: score every LIDC condition once the whole analyse array has drained,
# and verify the chain.
#
# This exists for one reason: H9 compares MosMed's template family against each
# LIDC condition's, and array tasks are unordered. A LIDC task that finishes
# before the MosMed task writes H9 as NOT_RUN -- correctly, for the artefacts
# that existed at the time. Re-scoring once everything is on disk is what turns
# that into a real verdict, and it costs seconds because it reads only stamped
# artefacts and recomputes nothing.
#
# It does NOT unblind. Unblinding is a dated, appending act through
# scripts/prereg_unblind.py, and running it inside an unattended batch job would
# put arm names into AMENDMENTS.md at a moment nobody was reading the output.
# The point of the audit event is that a person performs it.
#
#   sbatch --dependency=afterok:<analyse-array-id> scripts/slurm/confirmatory_score.sbatch

set -euo pipefail
PROJ=/fs/nexus-scratch/bthapar/SlotMIL
cd "$PROJ"

source /usr/share/Modules/init/bash
module load Python3/3.11.11
# shellcheck disable=SC1091
source "$PROJ/.venv/bin/activate"

export TMPDIR=/fs/nexus-scratch/bthapar/.tmp
export PYTHONUNBUFFERED=1

MOSMED_TF=$PROJ/runs/nulls_mosmed_severity_confirmatory/template_family.json
if [ ! -f "$MOSMED_TF" ]; then
    echo "no MosMed template family at $MOSMED_TF -- H9 will score NOT_RUN." >&2
    echo "That is a real outcome, not a crash, so this continues rather than exits." >&2
    MOSMED_ARG=()
else
    MOSMED_ARG=(--mosmed-template-family "$MOSMED_TF")
fi

for COND in nodule_present malignancy balanced_presence; do
    DIR=$PROJ/runs/nulls_${COND}_confirmatory
    echo "=============== scoring $COND ==============="
    python scripts/prereg_verdict.py --dir "$DIR" --out "$DIR/prereg_verdict.json" \
        "${MOSMED_ARG[@]}"
done

echo "=============== chain ==============="
# The gate is 0 UNKNOWN. Anything stamped off the recorded chain is a result
# produced under a config the amendment log does not describe, and no prose
# fixes that. Superseded stamps are expected here -- the pre-amendment roots are
# preserved on disk on purpose.
python scripts/prereg_freeze.py --check | tail -3

echo "=============== dirty stamps ==============="
# The defect that cost the 2026-08-16 re-run, checked rather than remembered.
# train_cached.py samples git state ONCE per task before the seed loop, so a
# task launched from an uncommitted tree brands all five of its seeds
# git_dirty:true and they can carry no confirmatory claim under any hash. It is
# cheap to detect and expensive to discover later.
python - <<'PY'
import json
from pathlib import Path

roots = [p for p in Path("runs").glob("*_confirmatory")
         if "superseded" not in p.name]
dirty, clean = [], 0
for r in roots:
    for f in r.rglob("result.json"):
        stamp = json.loads(f.read_text()).get("prereg", {})
        if stamp.get("git_dirty"):
            dirty.append((str(f), stamp.get("git_commit", "")[:7]))
        else:
            clean += 1
print(f"{clean} clean, {len(dirty)} dirty across {len(roots)} confirmatory roots")
for f, c in dirty[:20]:
    print(f"  DIRTY {c} {f}")
if dirty:
    print("\nThese seeds cannot carry a confirmatory claim. Re-run the affected")
    print("tasks from a committed tree; see AMENDMENTS.md 2026-08-16.")
PY

echo "=============== verdicts ==============="
python - <<'PY'
import json
from pathlib import Path

for cond in ("nodule_present", "malignancy", "balanced_presence"):
    p = Path(f"runs/nulls_{cond}_confirmatory/prereg_verdict.json")
    if not p.exists():
        print(f"{cond}: no verdict written")
        continue
    doc = json.loads(p.read_text())
    row = " ".join(f"{v['id']}={v['outcome']}" for v in doc["verdicts"])
    print(f"{cond} (carries_family={doc.get('carries_hypothesis_family')})")
    print(f"  {row}")
PY

echo "=============== done ==============="
echo "Next, and deliberately NOT automated: scripts/prereg_unblind.py"
