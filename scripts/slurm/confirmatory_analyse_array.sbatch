#!/bin/bash
#SBATCH --job-name=slotmil-conf-analyse
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --array=0-3
#SBATCH --requeue
#SBATCH --output=/fs/nexus-scratch/bthapar/SlotMIL/logs/%x-%A_%a.out
#SBATCH --error=/fs/nexus-scratch/bthapar/SlotMIL/logs/%x-%A_%a.err
#
# Stage 3 of the confirmatory pipeline: turn the cached attention into stamped
# estimands and then into verdicts. One array task per condition.
#
#   sbatch scripts/slurm/confirmatory_analyse_array.sbatch
#
# NO GPU. Every driver here is numpy over the .npz dumps -- `--gres` is deliberately
# absent rather than forgotten, and asking for a card would queue behind GPU demand
# for work that cannot use one.
#
# The drivers run in dependency order inside one task rather than as separate array
# jobs, because two of them read another's output: h4_cross_patient needs
# template_family for the fitted_template link of the reference chain, and
# h7_content_free needs it for the reported_beside_the_gate rows. Expressing that as
# SLURM dependencies across a 3x6 array buys nothing and can deadlock a requeue.
#
# EVERY invocation passes --dir AND --out. The defaults in axis_gate.py,
# template_family.py and h8_in_lung.py all point at runs/nulls/<name>.json -- the
# EXPLORATORY artefacts that set every threshold in PREREGISTRATION.md. A
# confirmatory run that omitted --out would overwrite them, and the thresholds would
# then have no recoverable provenance.
#
# Wall time is dominated by h4_cross_patient: 100 derangements per tag, each a full
# per_bag_axes pass, measured at 2.19s per derangement over 129 discovery bags, so
# ~4 minutes per tag and ~2.5 hours over the 35 learned_attention tags. 16 hours
# leaves room for a scavenger requeue to restart a task from its last completed
# step.

set -euo pipefail
PROJ=/fs/nexus-scratch/bthapar/SlotMIL
cd "$PROJ"

source /usr/share/Modules/init/bash
module load Python3/3.11.11
# shellcheck disable=SC1091
source "$PROJ/.venv/bin/activate"

export TMPDIR=/fs/nexus-scratch/bthapar/.tmp
export PYTHONUNBUFFERED=1

# mosmed_severity joined on 2026-08-17 for H9, taking --array to 0-3. It runs a
# strict SUBSET of the steps below -- template_family and nothing else -- because
# the config's own split note says MosMed "supports the stereotypy contrast (H9)
# and nothing else, and the paper says so". Running the rest on it would produce
# a full table of MosMed numbers for hypotheses MosMed is not declared to carry,
# and a number that exists is a number someone will eventually quote.
CONDS=(nodule_present malignancy balanced_presence mosmed_severity)
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#CONDS[@]}" ]; then
    echo "array index $SLURM_ARRAY_TASK_ID beyond ${#CONDS[@]} conditions -- --array and CONDS have drifted" >&2
    exit 1
fi
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}

DIR=$PROJ/runs/nulls_${COND}_confirmatory
if [ ! -d "$DIR" ]; then
    echo "no dumps at $DIR -- run confirmatory_collect_array.sbatch for COND=$COND first" >&2
    exit 1
fi

# The split, dataset and feature cache for this condition, resolved from the frozen
# config so this file holds no second copy of any of them. Refuses an undeclared
# condition, same as the collect array. The cache is resolved rather than hardcoded
# for the reason step 6 makes concrete: probe_gate reads a cache, and a hardcoded
# LIDC path would have scored a MosMed split against LIDC features.
read -r SPLITS DATASET CACHE <<<"$(python - "$COND" <<'PY'
import sys
sys.path.insert(0, "/fs/nexus-scratch/bthapar/SlotMIL")
from slotmil import prereg

name = sys.argv[1]
pre = prereg.load()
cond = next((c for c in pre.get("conditions") if c["name"] == name), None)
if cond is None:
    declared = ", ".join(c["name"] for c in pre.get("conditions"))
    raise SystemExit(f"condition {name!r} is not pre-registered. Declared: {declared}")
ds = cond["dataset"]
print(pre.split(ds, cond["split"])["path"], ds, pre.get("datasets")[ds]["cache"])
PY
)"

# The uid -> patient map every cluster bootstrap here resamples on. One per
# dataset: LIDC groups several series under a patient, MosMed is one study per
# patient. Rebuilt from the cache if absent -- it is derived, not authored, and a
# missing derived file should not cost a queue slot.
if [ "$DATASET" = "lidc" ]; then
    META=$PROJ/runs/_audit_meta.json
else
    META=$PROJ/runs/_audit_meta_${DATASET}.json
fi
if [ ! -f "$META" ]; then
    echo "[$COND] building $META"
    python scripts/make_audit_meta.py --cache "$PROJ/$CACHE" --out "$META"
fi

CARRIES=$(python - "$COND" <<'PY'
import sys
sys.path.insert(0, "/fs/nexus-scratch/bthapar/SlotMIL")
from slotmil import prereg
name = sys.argv[1]
pre = prereg.load()
print("yes" if name == pre.get("hypothesis_family_condition") else "no")
PY
)

N_DUMPS=$(find "$DIR" -name '*_test.npz' | wc -l)
echo "task $SLURM_ARRAY_TASK_ID | condition '$COND' | $N_DUMPS test dumps"
echo "  dir     $DIR"
echo "  dataset $DATASET"
echo "  splits  $SPLITS"
echo "  cache   $CACHE"
echo "  meta    $META"
echo "  carries the hypothesis family: $CARRIES"

# Skip a step whose artefact already exists AND is stamped at the current config
# hash from a clean tree. A scavenger requeue then resumes rather than recomputing
# hours of bootstrap; a SUPERSEDED artefact is not skipped, because that is exactly
# the file that has to be rewritten.
current() {
    python - "$1" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, "/fs/nexus-scratch/bthapar/SlotMIL")
from slotmil import prereg

p = Path(sys.argv[1])
if not p.exists():
    raise SystemExit(1)
doc = json.loads(p.read_text())
stamp = doc.get("prereg", {})
ok = (stamp.get("prereg_hash") == prereg.load().hash
      and not stamp.get("git_dirty")
      and doc.get("analysis_role") == "confirmatory")
raise SystemExit(0 if ok else 1)
PY
}

step() {          # step <artefact> <command...>
    local out=$1; shift
    if current "$out"; then
        echo "[$COND] SKIP $(basename "$out") -- already current and clean"
        return 0
    fi
    echo "[$COND] RUN  $(basename "$out")"
    "$@"
}

# template_family is the ONLY step MosMed runs: it is H9's sole input. Everything
# else below is guarded on the dataset rather than on the condition name, so a
# third dataset would inherit the same scoping without editing each guard.
if [ "$DATASET" = "lidc" ]; then

# 1. axis_gate      -> H1, H2, and H6's slice column
step "$DIR/axis_gate.json" \
    python scripts/axis_gate.py --dir "$DIR" --role confirmatory \
        --meta "$META" --out "$DIR/axis_gate.json"

fi

# 2. template_family -> H5, H10, H7's beside-the-gate rows, and the skill denominator
#    that steps 3 and 5 read. Must precede both. On MosMed it is H9's input and the
#    only thing produced.
step "$DIR/template_family.json" \
    python scripts/template_family.py --dir "$DIR" --role confirmatory \
        --meta "$META" --out "$DIR/template_family.json"

if [ "$DATASET" != "lidc" ]; then
    echo "[$COND] steps 3-8 skipped: $DATASET carries H9 only, through template_family above"
    echo "[$COND] done"
    exit 0
fi

# 3. h4_cross_patient -> H4, H6's skill column, the cross_patient link of the chain
step "$DIR/h4_cross_patient.json" \
    python scripts/h4_cross_patient.py --dir "$DIR" --role confirmatory \
        --meta "$META" --template-family "$DIR/template_family.json" \
        --out "$DIR/h4_cross_patient.json"

# 4. h8_in_lung -> H8. LIDC only, and the lung masks are keyed by LIDC uid.
step "$DIR/h8_in_lung.json" \
    python scripts/h8_in_lung.py --dir "$DIR" --role confirmatory \
        --meta "$META" --lung "$PROJ/data/lidc/lung_masks.h5" \
        --out "$DIR/h8_in_lung.json"

# 5. h7_content_free -> H7's content-free half. Since the 2026-08-17 amendment its
#    two stochastic members are drawn 30 times per tag (hypotheses.H7.
#    content_free_draws, read by the driver -- not repeated here), so this step is
#    the slowest of the five after h4: ~4.5s per draw per tag on the roll member.
step "$DIR/h7_content_free.json" \
    python scripts/h7_content_free.py --dir "$DIR" --role confirmatory \
        --meta "$META" --template-family "$DIR/template_family.json" \
        --out "$DIR/h7_content_free.json"

# 6. probe_gate -> H7's probe half, per condition. It reads the feature cache and the
#    split rather than the dumps, so it is the one step here that is not pure numpy.
step "$DIR/probe_gate.json" \
    python scripts/probe_gate.py \
        --cache "$PROJ/$CACHE" --splits "$SPLITS" \
        --role confirmatory --out "$DIR/probe_gate.json"

# 7. holm_family -> the declared multiplicity correction. ONLY on the condition
#    carrying the family. hypothesis_family_condition was ruled before any
#    confirmatory number existed, precisely so that correcting across all three --
#    which spends the power of the condition that carries the claim to protect two
#    that cannot -- is not a choice available at analysis time.
if [ "$CARRIES" = "yes" ]; then
    step "$DIR/holm_family.json" \
        python scripts/holm_family.py --dir "$DIR" --role confirmatory \
            --out "$DIR/holm_family.json"
else
    echo "[$COND] holm_family skipped: this condition does not carry the hypothesis family"
fi

# 8. the verdicts. Always re-run -- it is seconds, it reads only stamped artefacts,
#    and a stale verdict table beside fresh estimands is the single worst artefact
#    this pipeline could leave behind.
#
# H9 compares MosMed's template family against this condition's, so it is passed
# only if the MosMed task has already written one. Array tasks are unordered, so
# on a cold run the LIDC tasks will usually finish first and H9 will read NOT_RUN
# with `mosmed_template_family: not supplied`. That is a correct verdict for the
# artefacts that existed, not a failure -- and it is why the verdicts are re-run
# once after the whole array drains. NOT_RUN and VOID exist so that "the input was
# not there" cannot be confused with "the hypothesis was tested and held".
MOSMED_TF=$PROJ/runs/nulls_mosmed_severity_confirmatory/template_family.json
MOSMED_ARG=()
if [ -f "$MOSMED_TF" ]; then
    MOSMED_ARG=(--mosmed-template-family "$MOSMED_TF")
    echo "[$COND] H9 will read $MOSMED_TF"
else
    echo "[$COND] no MosMed template family yet -- H9 will read NOT_RUN; re-run the"
    echo "[$COND] verdict after the array drains to pick it up"
fi

python scripts/prereg_verdict.py --dir "$DIR" --out "$DIR/prereg_verdict.json" \
    "${MOSMED_ARG[@]}"

echo "[$COND] done"
