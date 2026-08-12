#!/usr/bin/env python
"""Reanalyse Harvey et al.'s published localisation numbers as prior-normalised skill.

Zero compute, entirely from their printed tables -- but it is the paper's opening
move, so it gets a script rather than a hand-typed table: every number carries its
source and the arithmetic is auditable.

Source
------
Harvey, Loevlie & Hughes, "Normal Guidance is what Attention Needs",
arXiv:2605.27306 (26 May 2026), Table 1. Localisation metric is macro-averaged
instance AUROC over positive bags -- the *same* statistic this project reports.
Their content-free baseline is a discretised univariate Normal over slice index,
    a_ij  proportional to  NormPDF(j | S_i / 2, 1),
i.e. axial position only, fixed sigma = 1, never fit to the data.

The point
---------
Normal Guidance improves the metric by regularising attention *toward* that
positional prior. Expressed as skill over the prior it is already claiming a
fraction of what the headline delta suggests:

    prior_normalised_skill = (AUC_method - AUC_prior) / (1 - AUC_prior)

And this is an *upper* bound on their skill, twice over: their prior is fixed
rather than fit (a template fit on validation scores higher, so the denominator
shrinks), and it is axial-only (an in-plane term would raise it further). On our
LIDC data the fitted in-plane template reaches 0.77-0.83 where the model-free
centre prior reaches 0.7752.

    python scripts/harvey_reanalysis.py --out runs/harvey_reanalysis.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slotmil import prereg  # noqa: E402

# arXiv:2605.27306 Table 1 -- localisation AUROC, macro-averaged over positive bags.
#   dataset: (n_slices, centred-Gaussian baseline, Normal Guidance)
HARVEY_TABLE1 = {
    "RSNA 2019 intracranial haemorrhage (head CT)": (752_803, 0.850, 0.871),
    "RSNA 2020 pulmonary embolism (chest CT)": (1_790_594, 0.780, 0.866),
    "RSNA 2023 abdominal trauma (abdomen CT)": (1_500_653, 0.573, 0.663),
}

# This project, LIDC-IDRI nodule_present, 4-radiologist consensus voxel masks.
# scripts/axis_gate.py, patient-level cluster bootstrap over 129 patients.
OURS = {
    "axial (slice AUC), centre prior": 0.6026,
    "axial (slice AUC), trained networks": (0.4822, 0.5565),
    "axial (slice AUC), untrained networks": (0.4950, 0.4975),
}


def prior_normalised_skill(auc, auc_prior):
    """Fraction of the headroom above a content-free prior that a method claims.

    Kummerer, Wallis & Bethge (PNAS 2015) make the same move information-
    theoretically as information gain over a baseline model; this is the AUC-scale
    version, kept in AUC units for legibility against the published tables.
    """
    return (auc - auc_prior) / (1.0 - auc_prior)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/harvey_reanalysis.json")
    args = ap.parse_args()

    rows = []
    print(f"{'dataset':<46} {'prior':>7} {'NG':>7} {'delta':>7} {'skill':>7}")
    print("-" * 78)
    for name, (n_slices, prior, ng) in HARVEY_TABLE1.items():
        skill = prior_normalised_skill(ng, prior)
        rows.append({
            "dataset": name,
            "n_slices": n_slices,
            "auc_content_free_prior": prior,
            "auc_normal_guidance": ng,
            "delta": round(ng - prior, 4),
            "prior_normalised_skill": round(skill, 4),
        })
        print(f"{name:<46} {prior:>7.3f} {ng:>7.3f} {ng - prior:>7.3f} {skill:>7.3f}")

    out = {
        "source": "Harvey, Loevlie & Hughes, arXiv:2605.27306 (2026), Table 1",
        "metric": "macro-averaged instance AUROC over positive bags",
        "baseline": "a_ij proportional to NormPDF(j | S_i/2, 1); axial only, fixed sigma=1, not fit",
        "caveat": (
            "Upper bound on their skill: the prior is fixed rather than fit to "
            "validation, and is axial-only. A fitted template raises the "
            "denominator's floor and lowers the skill further."
        ),
        "rows": rows,
        "ours_lidc_axial_auc": OURS,
        # Their numbers, our arithmetic -- no data of ours enters this table, so
        # the confirmatory/exploratory split does not apply to it.
        "analysis_role": "reanalysis_of_published_numbers",
    }
    out = prereg.load().stamp(out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}  (prereg {out['prereg']['prereg_hash']})")


if __name__ == "__main__":
    main()
