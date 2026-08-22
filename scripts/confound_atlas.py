#!/usr/bin/env python
"""The confound atlas: where does the positional prior live, per dataset?

Figure~4's LIDC-versus-MosMed contrast showed that which axis carries the
positional confound is a property of the dataset --- and H9 showed our
intuition about it was reversed by measurement. This script generalises that
from an anecdote to a survey: it runs the mask-fitted template family over any
public volumetric dataset with lesion masks and reports the per-dataset
axial / in-plane / separable profile, with patient-clustered intervals.

Nothing here is pre-registered and nothing touches the confirmatory family:
every artefact is stamped exploratory. Nothing here trains, reads an image, or
needs a GPU --- ``fit_family(source="masks")`` and ``score_family`` are
mask-only by construction, and the adapters in ``slotmil/data/atlas.py``
download masks, never images.

Geometry is pinned to the paper's (16x16 patches per slice, 32 depth bins) so
each dataset's row is comparable with the LIDC and MosMed columns already in
the paper. The fit/score split is a deterministic hash of the case id ---
declared here as a convention, chosen before any of these datasets was
downloaded, and refused if it ever assigns a case to both halves.

    python scripts/confound_atlas.py --dataset covid_ct_seg \
        --root data/atlas/covid_ct_seg --out runs/confound_atlas/covid_ct_seg.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from null_decompose import centre_prior_scores  # noqa: E402

from slotmil import prereg  # noqa: E402
from slotmil.data.atlas import ADAPTERS, GRID  # noqa: E402
from slotmil.eval.axes import per_bag_axes  # noqa: E402
from slotmil.eval.estimands import cluster_bootstrap  # noqa: E402
from slotmil.eval.templates import fit_family, score_family  # noqa: E402

N_PATCH = GRID * GRID
MEMBERS = ("inplane", "axial", "separable", "joint")
AXES = (("flat_auc", 1), ("slice_auc", 2), ("within_slice_auc", 3))


def fit_half(case_id: str) -> bool:
    """Deterministic 50/50 assignment by case-id hash. A convention, not a
    pre-registered split; documented in the artefact."""
    return int(hashlib.sha256(case_id.encode()).hexdigest(), 16) % 2 == 0


def analyse(cases, reps: int, seed: int, n_bins: int) -> dict:
    fit_masks, score_masks, score_pats, fit_ids, score_ids = [], [], [], [], []
    for case_id, patient_id, flat in cases:
        if case_id in fit_ids or case_id in score_ids:
            raise SystemExit(f"case {case_id} yielded twice; the fit/score "
                             "halves would overlap")
        if flat.shape[0] % N_PATCH:
            raise SystemExit(
                f"case {case_id}: {flat.shape[0]} patches is not a whole "
                f"number of {N_PATCH}-patch slices; the adapter broke the "
                "layout contract")
        if fit_half(case_id):
            fit_ids.append(case_id)
            fit_masks.append(flat)
        else:
            score_ids.append(case_id)
            score_masks.append(flat)
            score_pats.append(patient_id)
    if not fit_masks or not score_masks:
        raise SystemExit(f"degenerate split: {len(fit_ids)} fit / "
                         f"{len(score_ids)} score cases")

    fam = fit_family(fit_masks, source="masks", n_patch=N_PATCH, n_bins=n_bins)
    scores = score_family(score_masks, fam)

    scorers = {}
    n_scored = None
    for member in MEMBERS:
        rows = per_bag_axes(scores[member], score_masks, 0, n_patch=N_PATCH)
        pats = [score_pats[r[0]] for r in rows]
        scorers[f"masks:{member}"] = {
            name: cluster_bootstrap([r[col] for r in rows], pats, reps, seed)
            for name, col in AXES}
        n_scored = len(rows)
    prior_rows = per_bag_axes(centre_prior_scores(score_masks), score_masks, 0,
                              n_patch=N_PATCH)
    pats = [score_pats[r[0]] for r in prior_rows]
    scorers["centre_prior"] = {
        name: cluster_bootstrap([r[col] for r in prior_rows], pats, reps, seed)
        for name, col in AXES}

    return {
        "n_cases_fit": len(fit_ids),
        "n_cases_score": len(score_ids),
        "n_cases_scored": n_scored,
        "n_cases_dropped_no_lesion": len(score_ids) - n_scored,
        "fit_case_ids": sorted(fit_ids),
        "scorers": scorers,
    }


def report(res: dict, dataset: str) -> None:
    print(f"\n== {dataset} ==  fit {res['n_cases_fit']} / score "
          f"{res['n_cases_score']} cases "
          f"({res['n_cases_dropped_no_lesion']} dropped, no lesion)", flush=True)
    print(f"   {'scorer':18s} {'flat':22s} {'slice':22s} within", flush=True)

    def fmt(s):
        return "   --" if s["mean"] is None else \
            f"{s['mean']:.4f} [{s['lo']:.4f},{s['hi']:.4f}]"

    for name, ax in res["scorers"].items():
        print(f"   {name:18s} {fmt(ax['flat_auc']):22s} "
              f"{fmt(ax['slice_auc']):22s} {fmt(ax['within_slice_auc'])}",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(ADAPTERS))
    ap.add_argument("--root", required=True,
                    help="directory the fetch script populated with masks")
    ap.add_argument("--out", required=True,
                    help="explicit; this script has no default output path on "
                         "purpose, so a forgotten flag cannot overwrite an artefact")
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-bins", type=int, default=32,
                    help="pinned to the paper's depth binning by default")
    args = ap.parse_args()

    cases = ADAPTERS[args.dataset](Path(args.root))
    res = analyse(cases, args.reps, args.seed, args.n_bins)
    res.update({
        "dataset": args.dataset,
        "declared_in_prereg": False,
        "analysis_role": "exploratory_atlas",
        "note": (
            "Confound atlas: mask-fitted template family on a public dataset, "
            "no training, no images, no GPU. Not pre-registered; the split is "
            "a deterministic case-id hash declared as a convention. Geometry "
            "pinned to the paper's (16x16 patches, 32 depth bins) for "
            "comparability with the LIDC/MosMed columns."),
        "grid": GRID,
        "n_bins": args.n_bins,
        "reps": args.reps,
        "lesion_semantics": "lesion-only (organ labels excluded)",
    })
    report(res, args.dataset)

    payload = prereg.load().stamp(res)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}  (exploratory, prereg "
          f"{payload['prereg']['prereg_hash']})", flush=True)


if __name__ == "__main__":
    main()
