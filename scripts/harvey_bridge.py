#!/usr/bin/env python
"""The bridge to Harvey et al.: their metric on our arms, and what it cannot see.

Harvey, Loevlie & Hughes (arXiv:2605.27306) diagnose positional attention with
a one-dimensional statistic: an AUROC along the slice index. Our slice-axis
column is that statistic's fitted-protocol counterpart -- the per-bag slice AUC
of ``slotmil.eval.axes.per_bag_axes``, already computed for every arm and seed
by ``scripts/axis_gate.py``. This script does three things, and computes
almost nothing new:

1. **Reproduce their observation on LIDC.** Per-arm slice AUC, read from the
   three confirmatory ``axis_gate.json`` artefacts (never recomputed), with
   the \\armNG{}-versus-gated-ABMIL contrast that is their headline mechanism.
2. **Extend it to MosMed.** ``axis_gate`` never ran there (the analyse array
   deliberately runs only the template family on MosMed); this script runs the
   identical ``analyse``/``analyse_prior`` over the cached MosMed dumps. Those
   rows are exploratory and say so in the artefact -- see the note field.
3. **Show what the decomposition reveals that a 1-D statistic cannot.** For
   each arm the (flat, within-slice, slice) triple: flat coincides with
   within-slice for every arm but one, which is precisely the claim a
   slice-index AUROC can neither state nor check, because it never measures
   the in-plane axis at all.

Their published-table arithmetic is imported from
``scripts/harvey_reanalysis.py`` so their numbers stay typed in exactly one
file; the stale ``OURS`` constants there (discovery-era) are superseded by the
confirmatory values this script reads from artefacts at runtime.

**MosMed discipline.** The pre-registration scopes MosMed to H9 and nothing
else. The MosMed rows here are therefore written *outside* the confirmatory
directory, stamped exploratory, and carry a note saying they must not be
quoted as confirmatory results of the pre-registration. This follows the
``h5_floored_denominator`` precedent: verdict-neutral, no amendment.

    python scripts/harvey_bridge.py --out runs/harvey_bridge/harvey_bridge.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from axis_gate import analyse, analyse_prior  # noqa: E402
from harvey_reanalysis import HARVEY_TABLE1, prior_normalised_skill  # noqa: E402

from slotmil import prereg  # noqa: E402
from slotmil.eval.verdict import arm_tag  # noqa: E402

LIDC_CONDITIONS = ("nodule_present", "malignancy", "balanced_presence")

MOSMED_NOTE = (
    "MosMed is declared to carry H9 and nothing else. The per-arm MosMed rows "
    "here are exploratory, exist only for this companion analysis, and must "
    "not be quoted as confirmatory results of this pre-registration.")


def stored_rows(doc: dict) -> dict[str, dict]:
    return {r["tag"]: r for r in doc.get("results", [])}


def per_arm_axes(pre, rows: dict[str, dict], seeds) -> dict[str, dict]:
    """Mean over seeds of the (flat, within-slice, slice) triple, per arm."""
    out = {}
    for name in pre.arm_set("H1"):
        spec = pre.arm(name)["spec"]
        picked = [rows[t] for t in (arm_tag(spec, s) for s in seeds) if t in rows]
        if not picked:
            continue
        out[name] = {
            k: statistics.fmean(r[k]["mean"] for r in picked)
            for k in ("flat_auc", "slice_auc", "within_slice_auc")
        } | {"n_seeds": len(picked)}
    return out


def lidc_section(pre, runs: Path, seeds) -> dict:
    conds = {}
    for c in LIDC_CONDITIONS:
        path = runs / f"nulls_{c}_confirmatory" / "axis_gate.json"
        doc = json.loads(path.read_text())
        rows = stored_rows(doc)
        arms = per_arm_axes(pre, rows, seeds)
        prior = rows.get("centre_prior")
        conds[c] = {
            "source": str(path),
            "per_arm": arms,
            "centre_prior_slice_auc": prior["slice_auc"]["mean"] if prior else None,
            # Their headline mechanism, on our data: the prior-injected arm
            # against its own base.
            "normal_guidance_vs_base_slice_auc": {
                "normal_guidance": arms.get("normal_guidance", {}).get("slice_auc"),
                "gated_abmil": arms.get("gated_abmil", {}).get("slice_auc"),
            },
        }
    return conds


def mosmed_section(pre, runs: Path, meta: Path, reps: int, seed: int,
                   seeds) -> dict:
    d = runs / "nulls_mosmed_severity_confirmatory"
    uid_to_pat = json.loads(meta.read_text())["pat"]
    rows = {}
    for name in pre.arm_set("H1"):
        spec = pre.arm(name)["spec"]
        for s in seeds:
            tag = arm_tag(spec, s)
            val, test = d / f"{tag}_val.npz", d / f"{tag}_test.npz"
            if not (val.exists() and test.exists()):
                continue
            rows[tag] = analyse(tag, val, test, uid_to_pat, reps, seed)
            print(f"[mosmed {tag}] slice "
                  f"{rows[tag]['slice_auc']['mean']:.4f}", flush=True)
    if not rows:
        raise SystemExit(f"no MosMed dumps found in {d}")
    any_tag = next(iter(rows))
    prior = analyse_prior(d / f"{any_tag}_test.npz", uid_to_pat, reps, seed)
    return {
        "note": MOSMED_NOTE,
        "per_arm": per_arm_axes(pre, rows, seeds),
        "centre_prior_slice_auc": prior["slice_auc"]["mean"],
        "per_tag": rows,
    }


def harvey_section(lidc: dict) -> dict:
    """Their Table 1 arithmetic beside ours, single-sourced."""
    theirs = []
    for name, (n_slices, prior, ng) in HARVEY_TABLE1.items():
        theirs.append({
            "dataset": name, "n_slices": n_slices,
            "auc_content_free_prior": prior, "auc_normal_guidance": ng,
            "prior_normalised_skill": round(prior_normalised_skill(ng, prior), 4),
        })
    fam = lidc["nodule_present"]
    ours = {
        "centre_prior_slice_auc": fam["centre_prior_slice_auc"],
        "normal_guidance_slice_auc":
            fam["per_arm"].get("normal_guidance", {}).get("slice_auc"),
        "note": ("Confirmatory values read from axis_gate.json at runtime. "
                 "These supersede the discovery-era OURS constants in "
                 "scripts/harvey_reanalysis.py."),
    }
    return {"published": theirs, "ours_lidc": ours,
            "source": "scripts/harvey_reanalysis.py HARVEY_TABLE1 "
                      "(arXiv:2605.27306, Table 1)"}


def what_their_metric_cannot_see(lidc: dict) -> dict:
    """Flat == within-slice is invisible to any slice-index statistic."""
    fam = lidc["nodule_present"]["per_arm"]
    gaps = {a: abs(v["flat_auc"] - v["within_slice_auc"]) for a, v in fam.items()}
    return {
        "claim": ("A slice-index AUROC measures the axial axis only. The "
                  "decomposition additionally shows flat AUC coinciding with "
                  "within-slice AUC -- the reported 3D metric is in-plane -- "
                  "which a 1-D statistic can neither state nor check."),
        "abs_flat_minus_within_per_arm": gaps,
        "n_arms_gap_over_0.02": sum(1 for g in gaps.values() if g > 0.02),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--meta-mosmed", default="runs/_audit_meta_mosmed.json")
    ap.add_argument("--out", required=True,
                    help="explicit; deliberately outside any confirmatory "
                         "directory")
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    out = Path(args.out)
    if "confirmatory" in out.parts or any("confirmatory" in p for p in out.parts):
        raise SystemExit(
            f"{out}: this artefact carries exploratory MosMed rows and must "
            "not live in a confirmatory directory")

    pre = prereg.load()
    runs = Path(args.runs)
    lidc = lidc_section(pre, runs, args.seeds)
    mosmed = mosmed_section(pre, runs, Path(args.meta_mosmed), args.reps,
                            args.seed, args.seeds)

    res = {
        "declared_in_prereg": False,
        "analysis_role": "exploratory_outside_confirmatory_family",
        "note": MOSMED_NOTE,
        "their_metric_on_our_arms": lidc,
        "mosmed_exploratory": mosmed,
        "their_published_numbers": harvey_section(lidc),
        "what_their_metric_cannot_see": what_their_metric_cannot_see(lidc),
        "reps": args.reps,
    }

    fam = lidc["nodule_present"]
    print(f"\ncentre prior slice AUC: LIDC "
          f"{fam['centre_prior_slice_auc']:.4f}, MosMed "
          f"{mosmed['centre_prior_slice_auc']:.4f}", flush=True)

    payload = pre.stamp(res)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}  (exploratory, prereg "
          f"{payload['prereg']['prereg_hash']})", flush=True)


if __name__ == "__main__":
    main()
