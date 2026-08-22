#!/usr/bin/env python
"""Figures and LaTeX tables for the paper, from stamped artefacts only.

``scripts/make_figures.py`` is the pre-pivot figure generator: it loads a
checkpoint, runs deletion/insertion curves and writes slot overlays, which is the
"slot attention localises lesions" story. That story is what the confirmatory
sweep falsified, so this is a sibling rather than an edit -- the old script stays
runnable and is not what the paper cites.

**This script reads JSON and nothing else.** No checkpoint, no ``.npz``, no CUDA,
no dataset. It runs in seconds on a login node, which is the property that lets a
reader regenerate every figure from a clone plus the artefacts without a GPU
allocation. Numbers are never retyped from ``RESULTS.md``: its upper 546 lines are
discovery-split and superseded, and a figure that agrees with prose but not with
an artefact is the failure mode this paper is about.

Every plotted value is also written to ``figure_data.json`` beside the figures,
keyed by figure and series, with the source path and that source's
``prereg_hash`` and ``git_commit``. The argument is ``template_family.py``'s:
printing ``pre_registered_denominator`` next to the number keeps a fact about a
config file visible instead of implicit.

Arms are **unblinded** through ``runs/prereg/unblind_key.json`` so ``ARM-AEFB92``
never reaches a figure caption. Unblinding is already recorded in ``AMENDMENTS.md``
(2026-08-19).

    python scripts/make_paper_figures.py --out paper/figures --tables paper/tables
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slotmil import prereg  # noqa: E402
from slotmil.prereg import BlindKey  # noqa: E402

FAMILY = "nodule_present"
LIDC_CONDITIONS = ["nodule_present", "malignancy", "balanced_presence"]

# Display names, keyed by the pre-registration's arm *name* -- which is what
# `Prereg.arm_set` returns and what unblinding a verdict artefact yields.
PRETTY = {
    "abmil": "ABMIL", "gated_abmil": "Gated ABMIL", "mh_abmil": "MH-ABMIL",
    "slot_div0.5": "Slot ($\\mathrm{div}{=}0.5$)", "normal_guidance": "Normal Guidance",
    "clam_sb": "CLAM-SB", "dsmil": "DSMIL", "transmil": "TransMIL",
}

# Colour-blind-safe; one hue per semantic class, used identically across figures.
C_TRAINED = "#0072B2"    # blue   -- reads test images
C_FREE = "#D55E00"       # orange -- reads no image features
C_CEILING = "#009E73"    # green  -- supervised ceiling
C_FLOOR = "#666666"      # grey   -- chance / untrained floor
C_NG = "#CC79A7"         # pink   -- the Normal Guidance arm


# ------------------------------------------------------------------ loading
class Sources:
    """Artefact loader that records provenance for everything it hands out."""

    def __init__(self, runs: Path):
        self.runs = runs
        self._cache: dict[str, dict] = {}
        self.provenance: dict[str, dict] = {}

    def load(self, rel: str) -> dict:
        if rel not in self._cache:
            path = self.runs / rel
            doc = json.loads(path.read_text())
            self._cache[rel] = doc
            p = doc.get("prereg", {})
            self.provenance[rel] = {
                "path": str(path),
                "prereg_hash": p.get("prereg_hash"),
                "git_commit": p.get("git_commit"),
                "git_dirty": p.get("git_dirty"),
            }
        return self._cache[rel]

    def cond(self, condition: str, name: str) -> dict:
        return self.load(f"nulls_{condition}_confirmatory/{name}.json")


def unblind_map(runs: Path) -> dict[str, str]:
    key = runs / "prereg" / "unblind_key.json"
    return BlindKey.load(key).inverse() if key.exists() else {}


class ArmSet:
    """The arms a hypothesis scores, and the three spellings each one has.

    An arm reaches the analysis layer as a *name* (``slot_div0.5``), a *spec*
    (``slot:div=0.5``) and a dump *key* (``slot_div=0.5``, the spec with ``:``
    replaced). Artefacts key rows by the dump key; ``Prereg.arm_set`` and an
    unblinded verdict both return the name. Resolving through the config rather
    than by splitting whatever tags are present is also what keeps references out
    of per-arm panels: ``axis_gate.json`` carries a bare ``centre_prior`` row and
    five ``mean``/``centre_gaussian`` rows, and none of them is a learned arm.
    """

    def __init__(self, pre, hid: str = "H1"):
        self.names = list(pre.arm_set(hid))
        self.key_to_name = {}
        for n in self.names:
            self.key_to_name[pre.arm(n)["spec"].replace(":", "_")] = n
            self.key_to_name[n] = n

    def display(self, name: str) -> str:
        return PRETTY.get(name, name)


def arm_of(tag: str) -> str:
    return tag.rsplit("_seed", 1)[0]


def scorer_means(doc: dict, scorer: str, field: str = "flat_auc") -> dict[str, float]:
    """``{tag: value}`` for one scorer's bootstrap mean."""
    out = {}
    for row in doc.get("results", []):
        for s in row.get("scorers", []):
            if s.get("scorer") == scorer and s.get(field, {}).get("mean") is not None:
                out[row["tag"]] = s[field]["mean"]
    return out


def constant_scorer(doc: dict, scorer: str, field: str = "flat_auc") -> float | None:
    """A scorer that does not depend on the arm, asserted to be constant.

    ``masks:*`` and ``centre_prior`` are fit to lesion masks and patch geometry
    and read no attention, so they take one value across all 50 tags. Asserting
    that rather than averaging it is what stops a per-arm quantity being silently
    reported as a reference line.
    """
    vals = list(scorer_means(doc, scorer, field).values())
    if not vals:
        return None
    if max(vals) - min(vals) > 1e-9:
        raise SystemExit(f"{scorer}'s {field} varies across tags "
                         f"({min(vals):.6f}..{max(vals):.6f}); it is not a constant "
                         "reference and must not be drawn as one")
    return vals[0]


def per_arm(values: dict[str, float], arms: ArmSet,
            inv: dict[str, str] | None = None) -> dict[str, float]:
    """Mean over seeds per scored arm, keyed by arm name, descending."""
    acc = defaultdict(list)
    for tag, v in values.items():
        key = arm_of(tag)
        key = (inv or {}).get(key, key)
        name = arms.key_to_name.get(key)
        if name is not None:
            acc[name].append(v)
    out = {a: statistics.fmean(v) for a, v in acc.items()}
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def verdict(doc: dict, hid: str) -> dict:
    for v in doc.get("verdicts", []):
        if v.get("id") == hid:
            return v
    raise SystemExit(f"{hid} not in verdict artefact")


def unblind_keys(d: dict, inv: dict[str, str]) -> dict:
    return {inv.get(k, k): v for k, v in d.items()}


# ------------------------------------------------------------------ figure 1
def fig1(src: Sources, arms: ArmSet, out: Path) -> dict:
    """The reference ladder, and the axis split that explains it."""
    tf = src.cond(FAMILY, "template_family")
    vd = src.cond(FAMILY, "prereg_verdict")
    pg = src.cond(FAMILY, "probe_gate")

    trained = per_arm(scorer_means(tf, "trained"), arms)
    refs = [
        ("chance", 0.5, C_FLOOR, True),
        ("Mask-fitted, axial only", constant_scorer(tf, "masks:axial"), C_FREE, True),
        ("Mask-fitted, joint", constant_scorer(tf, "masks:joint"), C_FREE, True),
        ("Untrained-init 95th pct", verdict(vd, "H3")["p95"], C_FLOOR, False),
        ("Centre prior (no model, no fit)",
         constant_scorer(tf, "centre_prior"), C_FREE, True),
        ("Mask-fitted, in-plane", constant_scorer(tf, "masks:inplane"), C_FREE, True),
        ("Mask-fitted, separable",
         constant_scorer(tf, "masks:separable"), C_FREE, True),
        ("Supervised patch probe (ceiling)",
         pg["prior_normalised_skill"]["auc_probe"], C_CEILING, False),
    ]
    refs = [(n, v, c, f) for n, v, c, f in refs if v is not None]
    refs.sort(key=lambda r: r[1])

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(11.4, 4.5), gridspec_kw={"width_ratios": [1.12, 1.0]})

    # -- Panel A: the ladder, with the trained arms as a swarm on one row
    ys = list(range(len(refs)))
    axA.barh(ys, [r[1] for r in refs], color=[r[2] for r in refs],
             height=0.62, alpha=.92, zorder=2)
    for y, r in zip(ys, refs):
        axA.text(r[1] + .008, y, f"{r[1]:.3f}", va="center", fontsize=7.6, zorder=3)

    # The eight trained arms get their own labelled row at the top of the ladder,
    # as a swarm rather than a bar: the spread is the point.
    ty = len(refs) + 0.35
    axA.scatter(list(trained.values()), [ty] * len(trained), s=36, zorder=4,
                color=C_TRAINED, edgecolor="white", linewidth=.6)
    best_arm = max(trained, key=lambda a: trained[a])
    best = trained[best_arm]
    axA.annotate(f"best: {arms.display(best_arm)}, {best:.3f}",
                 xy=(best, ty), xytext=(0.045, ty + .52), fontsize=7.6,
                 color=C_TRAINED,
                 arrowprops=dict(arrowstyle="->", color=C_TRAINED, lw=.9))
    axA.set_yticks([*ys, ty])
    axA.set_yticklabels(
        [f"{r[0]}$^\\ast$" if r[3] else r[0] for r in refs]
        + [f"trained arms ($n={len(trained)}$)"], fontsize=8.5)
    axA.get_yticklabels()[-1].set_color(C_TRAINED)
    axA.get_yticklabels()[-1].set_fontweight("bold")

    axA.axvline(0.5, color=C_FLOOR, ls=":", lw=.9, zorder=1)
    axA.set_ylim(-0.7, ty + 1.0)
    axA.set_xlim(0, 1.06)
    axA.set_xlabel("flat instance AUC (mean over 5 seeds)")
    axA.set_title("A. The reference ladder\n"
                  "$^\\ast$ reads no image features", fontsize=9.5, loc="left")
    axA.spines[["top", "right"]].set_visible(False)

    # -- Panel B: flat / within-slice / slice AUC per arm (H1 and H2 in one image)
    ag = src.cond(FAMILY, "axis_gate")
    rows = {r["tag"]: r for r in ag.get("results", [])}
    flat = per_arm({t: r["flat_auc"]["mean"] for t, r in rows.items()}, arms)
    within = per_arm({t: r["within_slice_auc"]["mean"] for t, r in rows.items()}, arms)
    slice_ = per_arm({t: r["slice_auc"]["mean"] for t, r in rows.items()}, arms)
    order = list(flat)
    ys = list(range(len(order)))

    axB.scatter([flat[a] for a in order], ys, s=40, color=C_TRAINED,
                label="flat (3D) AUC", zorder=3)
    axB.scatter([within[a] for a in order], ys, s=40, marker="x",
                color="#111111", label="within-slice AUC", zorder=4)
    axB.scatter([slice_[a] for a in order], ys, s=34, marker="s", color="#B0B0B0",
                label="slice AUC", zorder=3)
    for y, a in zip(ys, order):
        axB.plot([min(flat[a], within[a]), max(flat[a], within[a])], [y, y],
                 color=C_TRAINED, lw=1.1, alpha=.45, zorder=2)

    cp_slice = verdict(src.cond(FAMILY, "prereg_verdict"), "H2")["centre_prior_slice_auc"]
    axB.axvline(cp_slice, color=C_FREE, ls="--", lw=1.2, zorder=1)
    axB.text(cp_slice + .008, len(order) - .35,
             f"centre prior,\nslice AUC {cp_slice:.3f}", color=C_FREE, fontsize=7.4,
             va="top")
    axB.axvline(0.5, color=C_FLOOR, ls=":", lw=.9, zorder=1)
    axB.set_yticks(ys)
    axB.set_yticklabels([arms.display(a) for a in order], fontsize=8.5)
    for y, a in zip(ys, order):
        if a == "normal_guidance":
            axB.get_yticklabels()[y].set_color(C_NG)
            axB.get_yticklabels()[y].set_fontweight("bold")
    h1 = verdict(vd, "H1")
    axB.set_xlabel("instance AUC")
    axB.set_xlim(0.15, 0.95)
    axB.set_title("B. Flat AUC is within-slice AUC\n"
                  f"gap $\\geq {h1['threshold']}$ for {len(h1['arms_over_threshold'])} "
                  f"of {h1['n_arms']} arms", fontsize=9.5, loc="left")
    axB.legend(loc="center right", fontsize=7.5, frameon=False)
    axB.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    save(fig, out / "fig1_reference_ladder")
    return {
        "panelA_references": {r[0]: r[1] for r in refs},
        "panelA_trained_per_arm": trained,
        "panelB_flat": flat, "panelB_within_slice": within, "panelB_slice": slice_,
        "panelB_centre_prior_slice_auc": cp_slice,
    }


# ------------------------------------------------------------------ figure 2
def fig2(src: Sources, arms: ArmSet, inv: dict, out: Path) -> dict:
    """Normal Guidance -- the co-headline. Four axes, one grouped bar chart."""
    vd = src.cond(FAMILY, "prereg_verdict")
    h6, h1, h8 = verdict(vd, "H6"), verdict(vd, "H1"), verdict(vd, "H8")
    gaps = unblind_keys(h1["gaps"], inv)
    lung = unblind_keys(h8["per_arm"], inv)

    # `is_auc` marks the two columns measured on the AUC scale; the other two are
    # differences. The chance line is drawn only across the former, because a
    # dotted 0.5 under a difference would be meaningless.
    axes = [
        ("Slice AUC", h6["slice_base"], h6["slice_arm"], None, True),
        ("Patient-specific skill", h6["skill_base"], h6["skill_arm"],
         h6["threshold"], False),
        ("Axis gap $|$flat $-$ within$|$", gaps["gated_abmil"],
         gaps["normal_guidance"], h1["threshold"], False),
        ("In-lung template AUC", lung["gated_abmil"], lung["normal_guidance"],
         None, True),
    ]

    fig, ax = plt.subplots(figsize=(7.8, 4.3))
    x = range(len(axes))
    w = 0.36
    delta_y = 0.78  # one constant row for every delta, so none collides with a bar
    ax.bar([i - w / 2 for i in x], [a[1] for a in axes], w,
           color="#BBBBBB", label="Gated ABMIL (base arm)", zorder=2)
    ax.bar([i + w / 2 for i in x], [a[2] for a in axes], w,
           color=C_NG, label="Normal Guidance ($\\lambda{=}0.1$)", zorder=2)
    for i, (_, b, a_, thr, is_auc) in enumerate(axes):
        ax.text(i - w / 2, b + (.014 if b >= 0 else -.034), f"{b:.3f}",
                ha="center", fontsize=7.4)
        ax.text(i + w / 2, a_ + (.014 if a_ >= 0 else -.034), f"{a_:.3f}",
                ha="center", fontsize=7.4, fontweight="bold")
        ax.text(i, delta_y, f"$\\Delta$ {a_ - b:+.4f}", ha="center", fontsize=8.2,
                color=C_NG, fontweight="bold")
        if thr is not None:
            ax.hlines(thr, i - .46, i + .46, color="#111111", ls="--", lw=1.0,
                      zorder=3)
        if is_auc:
            ax.hlines(0.5, i - .46, i + .46, color=C_FLOOR, ls=":", lw=1.0, zorder=1)
            ax.text(i - .46, .516, "chance", fontsize=6.6, color=C_FLOOR)
    ax.axhline(0, color="#333333", lw=.8, zorder=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels([a[0] for a in axes], fontsize=8.4)
    ax.set_ylabel("AUC (columns 1, 4) / difference (columns 2, 3)", fontsize=8.4)
    ax.set_ylim(-0.24, 0.90)
    ax.set_title("Explicit prior injection is the one intervention that moves the "
                 "measurement", fontsize=9.5, loc="left")
    # The falsifier gets a legend entry rather than four inline labels: both
    # falsifiers on this figure are 0.02, and inline text collides with the
    # 0.015 bar it sits directly above.
    thrs = sorted({a[3] for a in axes if a[3] is not None})
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color="#111111", ls="--", lw=1.0))
    labels.append("pre-registered falsifier "
                  f"({', '.join(str(t) for t in thrs)})")
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, -.13),
              ncol=3, fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, out / "fig2_normal_guidance")
    return {
        "axes": {a[0]: {"gated_abmil": a[1], "normal_guidance": a[2],
                        "delta": a[2] - a[1], "falsifier": a[3],
                        "auc_scale": a[4]} for a in axes},
        "h6_outcome": h6["outcome"], "h1_arms_over_threshold":
            [inv.get(a, a) for a in h1["arms_over_threshold"]],
    }


# ------------------------------------- figure 4 (dataset contrast, renders 4th)
def fig_contrast(src: Sources, out: Path) -> dict:
    """Which axis carries the prior is a property of the dataset."""
    lidc = src.cond(FAMILY, "template_family")
    mos = src.cond("mosmed_severity", "template_family")
    members = ["masks:inplane", "masks:axial", "masks:separable"]
    label = {"masks:inplane": "in-plane", "masks:axial": "axial",
             "masks:separable": "separable"}

    data = {}
    for name, doc in (("LIDC (nodules)", lidc), ("MosMed (COVID GGO)", mos)):
        d = {}
        for m in members:
            d[label[m]] = {"flat": constant_scorer(doc, m, "flat_auc"),
                           "slice": constant_scorer(doc, m, "slice_auc")}
        d["centre prior"] = {
            "flat": constant_scorer(doc, "centre_prior", "flat_auc"),
            "slice": constant_scorer(doc, "centre_prior", "slice_auc")}
        data[name] = d

    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    for ax, (name, d) in zip(axs, data.items()):
        keys = list(d)
        x = range(len(keys))
        w = .37
        ax.bar([i - w / 2 for i in x], [d[k]["flat"] for k in keys], w,
               color=C_FREE, label="flat (3D) AUC", zorder=2)
        ax.bar([i + w / 2 for i in x], [d[k]["slice"] for k in keys], w,
               color="#7F3F00", label="slice (axial) AUC", zorder=2)
        for i, k in enumerate(keys):
            ax.text(i - w / 2, d[k]["flat"] + .012, f"{d[k]['flat']:.3f}",
                    ha="center", fontsize=7.2)
            ax.text(i + w / 2, d[k]["slice"] + .012, f"{d[k]['slice']:.3f}",
                    ha="center", fontsize=7.2)
        ax.axhline(.5, color=C_FLOOR, ls=":", lw=.9, zorder=1)
        ax.set_xticks(list(x))
        ax.set_xticklabels(keys, fontsize=8.4)
        ax.set_title(name, fontsize=9.5)
        ax.set_ylim(0, 1.02)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].set_ylabel("AUC of a mask-fitted reference\n(fit to lesion masks)")
    axs[0].legend(loc="upper left", fontsize=7.6, frameon=False)
    fig.suptitle("The axial axis is empty on LIDC and dominant on MosMed",
                 fontsize=10, x=.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, .94))
    save(fig, out / "fig4_dataset_contrast")
    return data


# ------------------------------------------- figure 3 (in-lung, renders 3rd)
def fig_in_lung(src: Sources, arms: ArmSet, inv: dict, out: Path) -> dict:
    """H8 -- what survives inside the organ.

    The quantity H8 scores is the **attention-fitted** in-plane template, not the
    trained arm's own attention and not the mask-fitted oracle. That matters for
    how this reads: for most arms the unrestricted template already sits at or
    below chance, so there is no prior for the lung restriction to remove, and a
    "raw minus in-organ" gap would be a difference between two null quantities.
    The finding is the column on the left -- **no arm's fitted template beats
    chance inside the lung** -- and the gap is informative only for the arms whose
    template carried in-plane structure to begin with.
    """
    doc = src.cond(FAMILY, "h8_in_lung")
    rows = doc.get("results", [])
    unres = per_arm({r["tag"]: r["estimands"]["auc"]["unrestricted"]["mean"]
                     for r in rows}, arms)
    inl = per_arm({r["tag"]: r["estimands"]["auc"]["in_lung"]["mean"] for r in rows},
                  arms)
    frac = statistics.fmean(r["mean_in_lung_fraction"] for r in rows)
    h8 = verdict(src.cond(FAMILY, "prereg_verdict"), "H8")
    order = list(unres)
    ys = list(range(len(order)))

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for y, a in zip(ys, order):
        lost = unres[a] - inl[a]
        ax.annotate("", xy=(inl[a], y), xytext=(unres[a], y),
                    arrowprops=dict(arrowstyle="->", lw=1.3,
                                    color="#999999" if lost > .02 else "#DDDDDD"),
                    zorder=2)
    ax.scatter([unres[a] for a in order], ys, s=48, color=C_FREE,
               label="unrestricted (whole bag)", zorder=3)
    ax.scatter([inl[a] for a in order], ys, s=48, color=C_TRAINED,
               label="restricted to lung", zorder=3)
    for y, a in zip(ys, order):
        lost = unres[a] - inl[a]
        if lost > .02:
            ax.text(unres[a] + .012, y, f"$-${lost:.3f}", fontsize=7.4,
                    va="center", color="#444444")
    lo = min(min(unres.values()), min(inl.values()))
    hi = max(max(unres.values()), max(inl.values()))
    ax.axvline(.5, color=C_FLOOR, ls=":", lw=1.0, zorder=1)
    ax.axvline(h8["threshold"], color="#111111", ls="--", lw=1.0, zorder=1)
    ax.text(h8["threshold"] + .006, len(order) - .4,
            f"H8 bar {h8['threshold']}", fontsize=7, rotation=90, va="top")
    ax.text(.5 + .006, -.55, "chance", fontsize=7, color=C_FLOOR)
    ax.set_yticks(ys)
    ax.set_yticklabels([arms.display(a) for a in order], fontsize=8.5)
    ax.set_xlabel("instance AUC of the attention-fitted denominator "
                  "(fit to the arm's own attention)")
    ax.set_xlim(lo - .06, hi + .09)
    ax.set_ylim(-.9, len(order) - .3)
    ax.set_title("Inside the lung, no arm's attention-fitted denominator "
                 "beats chance\n"
                 f"aggregate in-lung AUC {h8['value']:.4f} vs a bar of "
                 f"{h8['threshold']}; the lung is {frac:.1%} of a bag",
                 fontsize=9.5, loc="left")
    ax.legend(loc="upper right", fontsize=7.8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, out / "fig3_in_lung")
    return {"unrestricted": unres, "in_lung": inl,
            "lost_to_lung_restriction": {a: unres[a] - inl[a] for a in order},
            "scored_quantity": "attention-fitted in-plane template, not the "
                               "trained arm's own attention",
            "mean_in_lung_fraction": frac,
            "h8_value": verdict(src.cond(FAMILY, "prereg_verdict"), "H8")["value"]}


def save(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"   wrote {stem}.pdf / .png", flush=True)


# ------------------------------------------------------------------- tables
_TEX_ESCAPES = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
                "<": r"\textless{}", ">": r"\textgreater{}"}


def _tex(s, limit: int | None = None) -> str:
    """Escape for LaTeX text mode, truncating *before* escaping.

    The pre-registration's statements are prose written for a YAML file and carry
    ``<``, ``>``, ``_`` and braces; unescaped they abort the build inside a
    tabular, which is a failure mode that only shows up once the table is real.
    Truncation happens first so a cut can never land inside an escape sequence.
    """
    text = str(s)
    if limit is not None and len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return "".join(_TEX_ESCAPES.get(c, c) for c in text)


def _threshold(thr) -> str:
    """A hypothesis bound, which is a scalar for most and a mapping for H4."""
    if thr is None:
        return "--"
    if isinstance(thr, dict):
        return ", ".join(f"{_tex(k)} {v}" for k, v in thr.items())
    return str(thr)


def table_verdicts(src: Sources) -> str:
    hs = [f"H{i}" for i in range(1, 11)]
    lines = [r"\begin{tabular}{l" + "c" * len(hs) + "}", r"\toprule",
             "Condition & " + " & ".join(hs) + r" \\", r"\midrule"]
    for c in LIDC_CONDITIONS:
        vd = src.cond(c, "prereg_verdict")
        cells = []
        for h in hs:
            o = verdict(vd, h)["outcome"]
            o = {"REPORTED": "RPT"}.get(o, o)
            cells.append(rf"\textbf{{{o}}}" if o == "FAIL" else o)
        name = _tex(c) + (r"$^\dagger$" if c == FAMILY else "")
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_falsifiers(src: Sources, pre) -> str:
    vd = src.cond(FAMILY, "prereg_verdict")
    rows = []
    for i in range(1, 11):
        hid = f"H{i}"
        v = verdict(vd, hid)
        h = pre.hypothesis(hid)
        rows.append((hid, _tex(h.get("statement", ""), 150),
                     _threshold(h.get("threshold")),
                     _tex(v.get("reason", ""), 150), v["outcome"]))
    lines = [r"\begin{longtable}{@{}p{0.5cm}p{5.4cm}p{1.9cm}p{5.4cm}p{1.5cm}@{}}",
             r"\toprule",
             r"& Pre-registered statement & Bar & Observed & Verdict \\",
             r"\midrule", r"\endhead"]
    for hid, stmt, thr, obs, out in rows:
        out = rf"\textbf{{{out}}}" if out == "FAIL" else out
        lines.append(f"{hid} & {stmt} & {thr} & {obs} & {out} " + r"\\")
        lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{longtable}"]
    return "\n".join(lines)


def table_arm_in_lung(src: Sources, arms: ArmSet, inv: dict) -> str | None:
    """Recommendation 1 applied to the paper's own arms.

    Reads ``arm_in_lung.json`` -- the arms' own frozen-slot attention, raw and
    lung-restricted -- not ``h8_in_lung.json``, which scores the
    attention-fitted denominator. Returns ``None`` when the artefact has not
    been generated, so the table build fails loudly at \\input time rather
    than silently emitting a stale file.
    """
    try:
        doc = src.cond(FAMILY, "arm_in_lung")
    except FileNotFoundError:
        return None
    per = unblind_keys(doc["per_arm"], inv)
    order = sorted(per, key=lambda a: -per[a]["raw"])
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"Arm & raw AUC & in-lung AUC & $\Delta$ \\", r"\midrule"]
    for a in order:
        v = per[a]
        name = arms.display(arms.key_to_name.get(a, a))
        lines.append(f"{name} & {v['raw']:.4f} & {v['in_lung']:.4f} & "
                     f"{v['raw'] - v['in_lung']:+.4f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


ATLAS_DATASETS = ("covid_ct_seg", "kits19", "msd_task06", "lits")


def table_confound_atlas(src: Sources) -> str | None:
    """Per-dataset positional-confound profile, mask-fitted members only.

    LIDC and MosMed rows come from the confirmatory ``template_family.json``
    artefacts; the public-dataset rows from ``runs/confound_atlas/*.json``
    (exploratory, and their rows say so in their own artefacts). Returns
    ``None`` when no atlas artefact exists yet.
    """
    rows = []
    for cond, label in ((FAMILY, "LIDC (nodules)"),
                        ("mosmed_severity", "MosMed (COVID GGO)")):
        doc = src.cond(cond, "template_family")
        rows.append((label, {
            f"masks:{m}": {a: constant_scorer(doc, f"masks:{m}", a)
                           for a in ("flat_auc", "slice_auc")}
            for m in ("inplane", "axial", "separable")}
            | {"centre_prior": {a: constant_scorer(doc, "centre_prior", a)
                                for a in ("flat_auc", "slice_auc")}}))
    n_atlas = 0
    for name in ATLAS_DATASETS:
        try:
            doc = src.load(f"confound_atlas/{name}.json")
        except FileNotFoundError:
            continue
        n_atlas += 1
        rows.append((f"{name} ({doc['n_cases_score']} cases)", {
            k: {a: doc["scorers"][k][a]["mean"] for a in
                ("flat_auc", "slice_auc")}
            for k in ("masks:inplane", "masks:axial", "masks:separable",
                      "centre_prior")}))
    if not n_atlas:
        return None
    lines = [r"\begin{tabular}{lcccccccc}", r"\toprule",
             r"& \multicolumn{2}{c}{in-plane} & \multicolumn{2}{c}{axial} & "
             r"\multicolumn{2}{c}{separable} & \multicolumn{2}{c}{centre prior} \\",
             r"Dataset & flat & slice & flat & slice & flat & slice & "
             r"flat & slice \\", r"\midrule"]
    for label, d in rows:
        cells = []
        for k in ("masks:inplane", "masks:axial", "masks:separable",
                  "centre_prior"):
            for a in ("flat_auc", "slice_auc"):
                v = d[k][a]
                cells.append("--" if v is None else f"{v:.3f}")
        lines.append(_tex(label) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_h5(src: Sources) -> str:
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"Condition & max skill, as published & denominator floored at 0.5 "
             r"& bar \\", r"\midrule"]
    for c in LIDC_CONDITIONS:
        rel = f"nulls_{c}_confirmatory/h5_floored_denominator.json"
        try:
            d = src.load(rel)
        except FileNotFoundError:
            continue
        lines.append(f"{_tex(c)} & {d['max_published']:+.4f} & "
                     f"{d['max_floored']:+.4f} & "
                     f"{d['threshold_read_from_prereg']} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", required=True,
                    help="explicit; no default, so a forgotten flag cannot "
                         "overwrite an artefact directory")
    ap.add_argument("--tables", default=None,
                    help="also emit LaTeX tables into this directory")
    args = ap.parse_args()

    runs = Path(args.runs)
    out = Path(args.out)
    src = Sources(runs)
    inv = unblind_map(runs)
    if not inv:
        print("[warn] no unblind key; figures will carry ARM- codes", flush=True)

    print("building figures", flush=True)
    pre = prereg.load()
    arms = ArmSet(pre)
    data = {
        "fig1_reference_ladder": fig1(src, arms, out),
        "fig2_normal_guidance": fig2(src, arms, inv, out),
        "fig3_in_lung": fig_in_lung(src, arms, inv, out),
        "fig4_dataset_contrast": fig_contrast(src, out),
    }

    if args.tables:
        tdir = Path(args.tables)
        tdir.mkdir(parents=True, exist_ok=True)
        for name, body in (("verdicts", table_verdicts(src)),
                           ("falsifiers", table_falsifiers(src, pre)),
                           ("h5_sensitivity", table_h5(src)),
                           ("arm_in_lung", table_arm_in_lung(src, arms, inv)),
                           ("confound_atlas", table_confound_atlas(src))):
            if body is None:
                print(f"   [warn] {name}: artefact missing, table not written",
                      flush=True)
                continue
            (tdir / f"{name}.tex").write_text(body + "\n")
            print(f"   wrote {tdir / name}.tex", flush=True)

    sidecar = prereg.load().stamp({
        "generated_by": "scripts/make_paper_figures.py",
        "note": ("Every number plotted in the paper's figures, with the artefact "
                 "it came from and that artefact's config hash and commit. A "
                 "figure that disagrees with this file is a bug in the figure."),
        "sources": src.provenance,
        "figures": data,
    })
    (out / "figure_data.json").write_text(json.dumps(sidecar, indent=2))
    print(f"\nwrote {out / 'figure_data.json'}  "
          f"(prereg {sidecar['prereg']['prereg_hash']})", flush=True)

    hashes = {v["prereg_hash"] for v in src.provenance.values()}
    if len(hashes) > 1:
        print(f"[warn] figures mix config hashes: {sorted(hashes)}", flush=True)
    else:
        print(f"all sources at one config hash: {hashes.pop()}", flush=True)


if __name__ == "__main__":
    main()
