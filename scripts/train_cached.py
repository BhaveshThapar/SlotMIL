#!/usr/bin/env python
"""Train MIL arms on a cached feature bag dataset (LIDC / MosMed).

The counterpart to w1_gonogo.py for real CT: instead of a trainable CNN encoder
over MedMNIST slices, bags come from the frozen-DINOv2 HDF5 cache.

Memory note: at patch level a LIDC bag is n_slices x 256 tokens -- around 28k
instances for a typical scan, or ~86 MB in fp32. Training therefore subsamples
slices per epoch (``--max-slices``), which plan.md line 88 prescribes for long
volumes anyway; evaluation always uses the full bag, so no test-time information
is discarded.

Checkpoints are written per arm/seed because scripts/eval_alignment.py consumes
them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slotmil import prereg  # noqa: E402
from slotmil.data.feature_cache import FeatureBagDataset, collate_bags  # noqa: E402
from slotmil.eval.classification import aggregate_seeds  # noqa: E402
from slotmil.losses import (  # noqa: E402
    CLAM_BAG_WEIGHT,
    CLAM_INSTANCE_WEIGHT,
    CLAM_TOPK_B,
    DSMIL_BAG_WEIGHT,
    DSMIL_MAX_WEIGHT,
    NG_VAR_FLOOR_SLICES2,
    SlotMILLoss,
)
from slotmil.models.baselines import (  # noqa: E402
    CENTRE_GAUSSIAN_SIGMA_Z,
    DEFAULT_PATCHES_PER_SLICE,
)
from slotmil.models.mil import build_model, slot_pooling_param_count  # noqa: E402
from slotmil.train import TrainConfig, fit  # noqa: E402


def parse_arm(spec: str):
    """``"slot:div=0.1,K=8"`` -> (label, pooling, overrides)."""
    pooling, _, rest = spec.partition(":")
    overrides = {}
    for kv in filter(None, rest.split(",")):
        k, _, v = kv.partition("=")
        overrides[k.strip()] = float(v)
    return spec, pooling, overrides


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", default="runs/lidc")
    ap.add_argument("--arms", nargs="+",
                    default=["mean", "gated_abmil", "mh_abmil", "slot", "slot:div=0.1"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--num-slots", type=int, default=8)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--readout", default="gated")
    ap.add_argument("--instance-level", default="patch", choices=["patch", "slice"])
    ap.add_argument("--max-slices", type=int, default=48,
                    help="stochastic slice subsampling during training only")
    ap.add_argument("--patches-per-slice", type=int, default=DEFAULT_PATCHES_PER_SLICE,
                    help="instance -> slice divisor for centre_gaussian and the "
                         "normal_guidance KL. Forced to 1 at --instance-level slice.")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--num-classes", type=int, default=2)
    ap.add_argument("--role", choices=["exploratory", "confirmatory"],
                    default="exploratory",
                    help="stamped into every result.json. 'confirmatory' also "
                         "suppresses test metrics on stdout unless --report-test "
                         "is passed explicitly.")
    ap.add_argument("--report-test", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="print test AUC/ACC to stdout. Defaults to on for "
                         "exploratory runs and OFF for confirmatory ones.")
    args = ap.parse_args()

    splits = json.loads(Path(args.splits).read_text())
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # A declared scope that the tooling does not enforce is a comment, not a
    # control. The ng_lambda_preflight sbatch declared test metrics out of scope
    # and this driver printed two of them to the job log anyway, which had to be
    # recorded as a deviation (AMENDMENTS.md 2026-08-15). Suppression now happens
    # at the source. The numbers are still written to result.json -- the analysis
    # layer reads them under blinding; a human reading a job log does not.
    show_test = args.report_test if args.report_test is not None else (
        args.role != "confirmatory")

    # Provenance. PREREGISTRATION.md's "Verifying the chain" makes this the difference between a
    # result that can carry a confirmatory claim and one that cannot, and until
    # now training wrote none of it: the only record of the split was
    # summary.json["args"]["splits"], a bare path, which merge_results.py then
    # dropped on rewrite.
    provenance = prereg.load().stamp({
        "analysis_role": args.role,
        "splits": str(args.splits),
        "splits_hash": splits.get("hash"),
        "splits_file_sha256": prereg.file_sha256(args.splits),
    })

    # A split file carrying a label map (from --merge-classes) overrides the
    # labels stored in the cache. Without this the merge would apply to the
    # partitioning but not to the labels actually served.
    label_map = splits.get("labels")
    if label_map:
        n_cls = splits.get("num_classes", len(set(label_map.values())))
        if n_cls != args.num_classes:
            print(f"[train] splits define {n_cls} classes "
                  f"(merge_classes={splits.get('merge_classes')}); "
                  f"overriding --num-classes {args.num_classes} -> {n_cls}")
            args.num_classes = n_cls

    # `seed` reaches the dataset because it drives slice subsampling, which is a
    # per-seed source of variance the study reports. It was never passed, so
    # every arm at every seed trained on the same subsampling stream and the
    # reported seed-to-seed std covered init and shuffle order but not the data
    # view. Expect the confirmatory std to be wider than the discovery std for
    # that reason alone; that is the bug being removed, not a change in method.
    mk = lambda keys, train, seed=0: FeatureBagDataset(  # noqa: E731
        args.cache, keys=keys, labels=label_map,
        instance_level=args.instance_level,
        max_slices=args.max_slices if train else None,
        train=train, return_mask=False, seed=seed,
    )
    train_ds, val_ds, test_ds = mk(splits["train"], True), mk(splits["val"], False), mk(splits["test"], False)
    input_dim = train_ds.dim
    print(f"[train] cache dim={input_dim} grid={train_ds.grid_h}x{train_ds.grid_w} "
          f"| train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}", flush=True)

    # Verify every served label is in range before touching the GPU. Out-of-range
    # labels surface as `nll_loss ... Assertion t >= 0 && t < n_classes failed`
    # deep in a CUDA kernel, with no indication of which dataset or which label.
    for name, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        labels = {
            (label_map[uid] if label_map else None) for uid in ds.keys
        } if label_map else None
        if labels is None:
            import h5py
            with h5py.File(args.cache, "r") as f:
                labels = {int(f[uid].attrs["label"]) for uid in ds.keys}
        bad = sorted(l for l in labels if not (0 <= l < args.num_classes))
        if bad:
            raise SystemExit(
                f"[train] split '{name}' contains label(s) {bad} outside "
                f"[0, {args.num_classes}). Either --num-classes is wrong, or the "
                f"split file needs a label map (make_splits.py --merge-classes "
                f"writes one)."
            )
    print(f"[train] labels validated against num_classes={args.num_classes}", flush=True)

    # instance_level="slice" means N == S, so there is no patch grid to divide
    # by. Both position-aware arms would otherwise compute their axial geometry
    # against a grid that is not there.
    pps = 1 if args.instance_level == "slice" else args.patches_per_slice

    results = {}
    for spec in args.arms:
        label, pooling, ov = parse_arm(spec)

        # Normal Guidance is plain gated_abmil plus a loss term. If lam never
        # reaches SlotMILLoss the arm trains as its own base arm, writes a
        # perfectly valid result.json, and gets reported as NG -- no crash, and a
        # fabricated H6. That is not hypothetical: before this guard, `lam` was
        # parsed by parse_arm and then silently dropped, because the criterion
        # below only ever wired `div` and `ent`.
        lam = float(ov.get("lam", 0.0))
        if pooling == "normal_guidance" and lam <= 0:
            raise SystemExit(
                f"[train] arm {label!r} is Normal Guidance but lam is absent or "
                "zero. It would train as plain gated_abmil and be written out as "
                "NG. The pre-registered spec is 'normal_guidance:lam=0.1'."
            )
        if pooling != "normal_guidance" and lam > 0:
            raise SystemExit(
                f"[train] arm {label!r} sets lam={lam} but is not normal_guidance; "
                "the KL term would be applied to an arm that does not declare it."
            )

        # Same failure mode as lam, twice over. CLAM-SB without its clustering
        # term is plain gated_abmil; DSMIL without its max term is a
        # single-stream non-local pooling. Either would train, write a valid
        # result.json and be reported under a published method's name. The
        # weights are pre-registered constants rather than arm overrides, so
        # there is nothing to parse and nothing to drop -- but the lookup is
        # explicit here so that adding an auxiliary-stream arm and forgetting to
        # wire it is a KeyError-shaped mistake, not a silent one.
        AUX_WEIGHTS = {  # pooling -> (w_bag, w_clam_inst, w_dsmil_max)
            "clam_sb": (CLAM_BAG_WEIGHT, CLAM_INSTANCE_WEIGHT, 0.0),
            "dsmil": (DSMIL_BAG_WEIGHT, 0.0, DSMIL_MAX_WEIGHT),
        }
        w_bag, w_clam_inst, w_dsmil_max = AUX_WEIGHTS.get(pooling, (1.0, 0.0, 0.0))
        if pooling in AUX_WEIGHTS and w_clam_inst + w_dsmil_max <= 0:
            raise SystemExit(
                f"[train] arm {label!r} declares an auxiliary stream but its "
                "weight is zero; it would train as its base pooling and be "
                "written out under the published name."
            )

        runs = []
        for seed in args.seeds:
            # Per-seed resume. The scavenger partition preempts with REQUEUE, so a
            # job can restart at any time; without this it would redo every
            # completed seed. Presence of result.json means that seed finished.
            seed_dir = out_root / label.replace(":", "_") / f"seed{seed}"
            done_file = seed_dir / "result.json"
            if done_file.exists():
                try:
                    prev = json.loads(done_file.read_text())
                    # A completed run from a DIFFERENT split must never be
                    # resumed into this one. The check used to be "file exists,
                    # parses, has a 'test' key" and nothing else, so a
                    # confirmatory job pointed at runs/lidc would have printed
                    # "already complete, skipping" for all 18 discovery runs and
                    # reported discovery numbers under a confirmatory banner --
                    # silently, which is the only kind of failure that matters
                    # here. Runs predating provenance stamping have no
                    # splits_hash and are refused rather than assumed.
                    prev_hash = prev.get("splits_hash")
                    if "test" in prev and prev_hash != splits.get("hash"):
                        raise SystemExit(
                            f"[train] {done_file} was produced on splits_hash="
                            f"{prev_hash!r} but this run uses "
                            f"{splits.get('hash')!r}. Refusing to resume across "
                            "splits. Point --out at a fresh run root (the "
                            "confirmatory sweep must not share one with "
                            "discovery), or delete the stale directory."
                        )
                    if "test" in prev:
                        runs.append({
                            "seed": seed,
                            # Stored via TrainConfig.extra, which fit() persists
                            # under "config" -- without it a resumed seed would
                            # report NaN params and corrupt the summary table.
                            "n_params": prev.get("config", {}).get("extra", {}).get(
                                "n_params", float("nan")),
                            **prev["test"]})
                        print(f"[train] {label} seed={seed} already complete, skipping",
                              flush=True)
                        continue
                except (json.JSONDecodeError, KeyError):
                    print(f"[train] {label} seed={seed} result.json unreadable, redoing",
                          flush=True)

            match_to = (
                slot_pooling_param_count(input_dim, args.dim,
                                         int(ov.get("K", args.num_slots)))
                if pooling == "mh_abmil" else None
            )
            model = build_model(
                pooling=pooling, input_dim=input_dim, dim=args.dim,
                num_classes=args.num_classes,
                num_slots=int(ov.get("K", args.num_slots)),
                readout=args.readout, iters=int(ov.get("iters", args.iters)),
                match_params_to=match_to, patches_per_slice=pps,
            )
            n_params = sum(p.numel() for p in model.parameters())
            print(f"[train] {label} seed={seed} params={n_params/1e3:.0f}k", flush=True)

            # Evaluation sees the FULL bag -- protocol.max_slices is train-only --
            # so an arm can train comfortably and then run out of memory at the
            # first validation pass. transmil did.
            #
            # The cost this buys back is `collate_bags` padding, not the model:
            # a batch is padded to its deepest bag, and the deepest LIDC series is
            # 162,560 instances, so batching one of those with three shallow bags
            # materialises a [4, 162560, 768] feature tensor -- ~2 GB in fp32 --
            # of which three quarters is padding that TransMIL then indexes away.
            # (The model itself is already per-bag: TransMIL.forward loops so the
            # squaring cannot depend on batch composition, which caps its own
            # resident grid at one bag regardless of B.)
            #
            # The value lives here rather than in the four sbatch files that
            # enumerate arms, for the same reason AUX_WEIGHTS does: an arm-specific
            # constant spread across launchers is one an arm is eventually launched
            # without. Batching moves no number -- the model is batch-invariant
            # (tests/test_transmil.py), every metric is computed over all bags, and
            # eval runs with grad disabled -- so this is a memory budget and not a
            # pre-registered parameter.
            EVAL_BATCH = {"transmil": 1}
            eval_bs = EVAL_BATCH.get(pooling)
            if eval_bs:
                print(f"[train] {label}: eval batch {eval_bs} "
                      f"(train batch {args.batch_size}) -- full-bag eval memory",
                      flush=True)

            cfg = TrainConfig(
                epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                eval_batch_size=eval_bs,
                num_workers=args.num_workers, seed=seed, select_metric="auc",
                # Every knob that defines the arm goes in extra, so result.json
                # records what actually ran rather than what the label implies.
                extra={"n_params": n_params, "arm": label, "lam": lam,
                       "eval_batch_size": eval_bs or args.batch_size,
                       "patches_per_slice": pps,
                       "kl_var_floor": NG_VAR_FLOOR_SLICES2,
                       "sigma_z": CENTRE_GAUSSIAN_SIGMA_Z,
                       "w_bag": w_bag, "w_clam_inst": w_clam_inst,
                       "w_dsmil_max": w_dsmil_max, "clam_topk_b": CLAM_TOPK_B},
            )
            res = fit(
                model, mk(splits["train"], True, seed), val_ds, cfg,
                collate_fn=collate_bags,
                criterion=SlotMILLoss(w_diversity=ov.get("div", 0.0),
                                      w_entropy=ov.get("ent", 0.0),
                                      w_kl=lam, kl_patches_per_slice=pps,
                                      kl_var_floor=NG_VAR_FLOOR_SLICES2,
                                      w_bag=w_bag, w_clam_inst=w_clam_inst,
                                      w_dsmil_max=w_dsmil_max,
                                      clam_topk_b=CLAM_TOPK_B),
                test_ds=test_ds,
                out_dir=out_root / label.replace(":", "_") / f"seed{seed}",
                provenance=provenance,
            )
            row = {"seed": seed, "n_params": n_params, **res["test"]}
            runs.append(row)
            if show_test:
                msg = f"[train]   -> auc={row['auc']:.4f} acc={row['acc']:.4f}"
                if "active_slots" in row:
                    msg += (f" active={row['active_slots']:.2f}"
                            f" maxcos={row['max_off_diag_cos']:.3f}")
            else:
                # Everything in `row` is computed on test, slot health included,
                # so the suppressed line reports validation only. Nothing is
                # lost: result.json holds the full test dict either way.
                msg = (f"[train]   -> done, best_val_auc="
                       f"{res['best_val_auc']:.4f}, test metrics suppressed "
                       f"(--role {args.role})")
            print(msg, flush=True)
        results[label] = runs

    summary = {k: aggregate_seeds(v) for k, v in results.items()}

    print("\n" + "=" * 80)
    if not show_test:
        print(f"test metrics suppressed (--role {args.role}); "
              f"{len(summary)} arm(s) written to disk. Run the analysis layer "
              "to read them.")
    else:
        print(f"{'arm':<18}{'AUC':>18}{'ACC':>18}{'params':>10}{'active':>8}{'maxcos':>8}")
        for k, s in summary.items():
            act = f"{s['active_slots']['mean']:>8.2f}" if "active_slots" in s else " " * 8
            cos = f"{s['max_off_diag_cos']['mean']:>8.3f}" if "max_off_diag_cos" in s else " " * 8
            print(f"{k:<18}{s['auc']['mean']:>10.4f} +/-{s['auc']['std']:.4f}"
                  f"{s['acc']['mean']:>10.4f} +/-{s['acc']['std']:.4f}"
                  f"{s['n_params']['mean']/1e3:>9.0f}k{act}{cos}")

    # Head-to-head with significance, same gate as W1: a bare mean comparison
    # over a few seeds is not evidence. Computed and written either way -- it is
    # the *printing* that leaks, and the pre-registered family-wise correction
    # lives in the analysis layer, not here.
    slot_arms = {k: v for k, v in summary.items() if k.split(":")[0] == "slot"}
    comparisons = {}
    if slot_arms:
        best_label = max(slot_arms, key=lambda k: slot_arms[k]["auc"]["mean"])
        a = slot_arms[best_label]["auc"]["values"]
        if show_test:
            print(f"\nbest slot arm: {best_label}")
        for other in summary:
            if other == best_label:
                continue
            b = summary[other]["auc"]["values"]
            if len(a) > 1 and len(b) > 1:
                p = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
                d = float(np.mean(a) - np.mean(b))
                comparisons[other] = {"delta": d, "p": p, "significant": bool(d > 0 and p < 0.05)}
                verdict = "SIGNIFICANT" if comparisons[other]["significant"] else "not significant"
                if show_test:
                    print(f"  vs {other:<16} delta={d:+.4f}  p={p:.3f}  {verdict}")

    (out_root / "summary.json").write_text(json.dumps(
        {**provenance, "args": vars(args), "summary": summary,
         "comparisons": comparisons}, indent=2, default=str))
    print(f"\nwrote {out_root/'summary.json'}")


if __name__ == "__main__":
    main()
