#!/usr/bin/env python
"""The reference floor: the frozen-slot protocol run on N untrained networks.

Two untrained initialisations, scoring 0.6433 and 0.7858, cannot support a
statement about where chance lies -- that spread is wider than most of the
effects the paper discusses. H3 asks for at least 30, reported as a distribution
rather than a point.

The obvious implementation runs ``null_collect_attn.py`` thirty times, which
reads the 78 GB feature cache sixty times (val and test per init) to run a model
whose forward pass is trivially cheap. This reads it **twice** instead, holding
all N untrained models in memory and evaluating every one on each batch as it
arrives. The cost is N small models on the GPU; the saving is ~30x the wall
clock, and the cache read is what dominates -- profiling on this project has
consistently found these jobs HDF5-bound with the GPU idle.

Nothing large is stored. The protocol needs only a slot-to-finding affinity
accumulated over validation, then a per-bag AUC accumulated over test, so both
passes are streamed and only scalars survive. Thirty raw attention dumps would be
7.5 GB for numbers that fit in a JSON file.

    python scripts/untrained_floor.py --splits data/lidc/splits_confirmatory.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slotmil import prereg  # noqa: E402
from slotmil.data.feature_cache import FeatureBagDataset, collate_bags  # noqa: E402
from slotmil.eval.alignment import fit_slot_assignment  # noqa: E402
from slotmil.eval.estimands import cluster_bootstrap  # noqa: E402
from slotmil.models.mil import build_model, slot_pooling_param_count  # noqa: E402


def build_fleet(seeds, input_dim, num_classes, pooling, num_slots, dim):
    """One untrained model per seed.

    The seed is set immediately before each construction rather than once up
    front, so init i is reproducible on its own: chunking the fleet, or extending
    it later, cannot change a model that has already been reported.
    """
    fleet = []
    for s in seeds:
        torch.manual_seed(s)
        match_to = (slot_pooling_param_count(input_dim, dim, num_slots)
                    if pooling == "mh_abmil" else None)
        fleet.append(build_model(
            pooling=pooling, input_dim=input_dim, dim=dim, num_classes=num_classes,
            num_slots=num_slots, readout="gated", match_params_to=match_to).eval())
    return fleet


@torch.no_grad()
def accumulate_affinity(fleet, loader, device, n_slots):
    """Slot-to-finding affinity per model, summed over validation bags.

    Mirrors ``alignment.slot_finding_affinity``: attention is normalised over
    instances, then projected onto the two complementary finding channels
    (lesion, background). Accumulated as a running sum so nothing per-bag is kept.
    """
    total = [np.zeros((n_slots, 2), dtype=np.float64) for _ in fleet]
    count = 0
    for bi, b in enumerate(loader):
        feats, pad = b["features"].to(device), b["pad_mask"].to(device)
        targets = b["patch_target"]
        for mi, model in enumerate(fleet):
            attn = model(feats, pad)["attn"].float().cpu().numpy()
            for i, n in enumerate(b["lengths"].tolist()):
                t = (targets[i, :n].numpy() > 0).astype(np.float32)
                if t.sum() == 0:
                    continue
                a = attn[i, :, :n]
                a = a / np.clip(a.sum(axis=-1, keepdims=True), 1e-8, None)
                total[mi] += a @ np.stack([t, 1.0 - t]).T
        count += sum(1 for i, n in enumerate(b["lengths"].tolist())
                     if (targets[i, :n].numpy() > 0).any())
        if bi % 10 == 0:
            print(f"  [val] batch {bi}", flush=True)
    return [t / max(count, 1) for t in total]


@torch.no_grad()
def score_test(fleet, loader, device, slots, uid_to_pat):
    """Per-bag instance AUC on test, using each model's frozen slot."""
    aucs = [[] for _ in fleet]
    pats = [[] for _ in fleet]
    for bi, b in enumerate(loader):
        feats, pad = b["features"].to(device), b["pad_mask"].to(device)
        targets, uids = b["patch_target"], b["uid"]
        for mi, model in enumerate(fleet):
            attn = model(feats, pad)["attn"].float().cpu().numpy()
            for i, n in enumerate(b["lengths"].tolist()):
                t = (targets[i, :n].numpy() > 0).astype(np.int8)
                if t.sum() == 0 or t.sum() == n:
                    continue
                aucs[mi].append(roc_auc_score(t, attn[i, slots[mi], :n]))
                pats[mi].append(uid_to_pat.get(str(uids[i]), str(uids[i])))
        if bi % 10 == 0:
            print(f"  [test] batch {bi}", flush=True)
    return aucs, pats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/lidc/features_dinov2_vitb14.h5")
    ap.add_argument("--splits", default="data/lidc/splits_confirmatory.json")
    ap.add_argument("--meta", default="runs/_audit_meta.json")
    # centre_gaussian is accepted but degenerate here: its attention has no
    # learned parameters, so 30 untrained inits produce 30 identical numbers.
    # Useful as a smoke check, never reportable as a floor.
    # Every pooling reachable from an `implemented` arm spec in
    # configs/prereg/isbi2027.yaml, pinned by
    # tests/test_prereg.py::test_every_implemented_arm_is_collectible. mean, abmil and
    # gated_abmil were missing until 2026-08-15: H1 is stated over every arm, H2 over
    # every trained arm, and H6 compares normal_guidance against gated_abmil
    # specifically -- so three hypotheses were uncomputable because of an argparse list.
    # mean is here for H1's tie-floor check, which asserts its gap is exactly 0.
    ap.add_argument("--pooling", default="slot",
                    choices=["mean", "abmil", "gated_abmil", "slot", "mh_abmil",
                             "centre_gaussian", "normal_guidance", "clam_sb", "dsmil",
                             "transmil"])
    ap.add_argument("--num-slots", type=int, default=8)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-inits", type=int, default=None,
                    help="default: the pre-registered untrained_fleet.n_inits")
    ap.add_argument("--seed-base", type=int, default=None)
    ap.add_argument("--model-chunk", type=int, default=None,
                    help="models resident at once. Default: all of them, which "
                         "reads the cache twice. Lower it when a single forward "
                         "is large -- matched MH-ABMIL needs ~6 GB per batch of "
                         "4 and OOMs a 24 GB card at 30 resident.")
    ap.add_argument("--out", default="runs/untrained_floor.json")
    ap.add_argument("--role", choices=["exploratory", "confirmatory"],
                    default="confirmatory")
    args = ap.parse_args()

    pre = prereg.load()
    fleet_cfg = next(r for r in pre.get("reference_baselines")
                     if r["name"] == "untrained_fleet")
    n_inits = args.n_inits or fleet_cfg["n_inits"]
    seed_base = args.seed_base if args.seed_base is not None else fleet_cfg["init_seed_base"]
    if args.role == "confirmatory" and n_inits < fleet_cfg["n_inits"]:
        # Gate the claim, not the smoke test. A two-init run is a useful way to
        # check the plumbing; what must never happen is a two-init run being
        # written out as the confirmatory floor.
        print(f"[floor] refusing: {n_inits} inits is below the pre-registered "
              f"{fleet_cfg['n_inits']} for a confirmatory floor. Pass "
              f"--role exploratory if this is a smoke test.", file=sys.stderr)
        return 1

    sp = json.loads(Path(args.splits).read_text())
    lm = sp.get("labels")
    uid_to_pat = json.loads(Path(args.meta).read_text())["pat"]

    def sub(keys):
        p = FeatureBagDataset(args.cache, keys=keys, labels=lm, return_mask=True)
        return FeatureBagDataset(args.cache, keys=p.masked_keys(), labels=lm,
                                 return_mask=True)

    val_ds, test_ds = sub(sp["val"]), sub(sp["test"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[floor] {n_inits} inits from seed {seed_base}, pooling {args.pooling}\n"
          f"[floor] val {len(val_ds)} bags, test {len(test_ds)} bags, device {device}",
          flush=True)

    def mk(ds):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.workers, collate_fn=collate_bags)

    # Holding all N models resident is the point -- it is what turns 60 reads of
    # a 78 GB cache into 2. But a parameter-matched MH-ABMIL bisects its hidden
    # width up to ~8.8k units, so one forward over a 43.8k-instance bag allocates
    # ~6 GB, and thirty of those in sequence fragment a 24 GB card until it OOMs.
    # Chunking bounds resident memory at the cost of re-reading once per chunk:
    # 2 x n_chunks passes, still far short of 60.
    chunk = args.model_chunk or n_inits
    slots, aucs, pats = [], [], []
    for start in range(0, n_inits, chunk):
        seeds = [seed_base + i for i in range(start, min(start + chunk, n_inits))]
        tag = f"inits {seeds[0]}-{seeds[-1]}"
        fleet = [m.to(device) for m in build_fleet(
            seeds, val_ds.dim, sp.get("num_classes", 2),
            args.pooling, args.num_slots, args.dim)]

        print(f"[floor] {tag}: pass 1/2, fitting the lesion slot on val", flush=True)
        chunk_slots = []
        for aff in accumulate_affinity(fleet, mk(val_ds), device, args.num_slots):
            assign = fit_slot_assignment(aff)
            chunk_slots.append(next((k for k, v in assign.items() if v == 0), 0))
        print(f"[floor] {tag}: frozen slots {chunk_slots}", flush=True)

        print(f"[floor] {tag}: pass 2/2, scoring test", flush=True)
        a, p = score_test(fleet, mk(test_ds), device, chunk_slots, uid_to_pat)
        slots += chunk_slots
        aucs += a
        pats += p

        del fleet
        if device == "cuda":
            torch.cuda.empty_cache()

    per_init = []
    for i, (a, p) in enumerate(zip(aucs, pats)):
        ci = cluster_bootstrap(a, p, reps=pre.get("statistics.bootstrap.reps"),
                               seed=pre.get("statistics.bootstrap.rng_seed"))
        per_init.append({"init_seed": seed_base + i, "frozen_slot": int(slots[i]),
                         "instance_auc": ci["mean"], "lo": ci["lo"], "hi": ci["hi"],
                         "n_bags": ci["n"], "n_patients": ci["n_clusters"]})

    vals = np.array([r["instance_auc"] for r in per_init], float)
    floor = {
        "n_inits": int(len(vals)),
        "median": float(np.median(vals)),
        "mean": float(vals.mean()),
        "p5": float(np.percentile(vals, 5)),
        "p95": float(np.percentile(vals, 95)),
        "min": float(vals.min()),
        "max": float(vals.max()),
        # H3: the floor is not 0.5, and the point of reporting a distribution is
        # that its upper tail is what a trained model has to beat to mean anything.
        "H3_p95_above_0.70": bool(np.percentile(vals, 95) > 0.70),
    }
    print("\n[floor] untrained instance AUC over "
          f"{floor['n_inits']} inits:\n"
          f"        median {floor['median']:.4f}  "
          f"5-95pct [{floor['p5']:.4f}, {floor['p95']:.4f}]  "
          f"range [{floor['min']:.4f}, {floor['max']:.4f}]")
    print(f"[floor] H3 (95th pct > 0.70): "
          f"{'SUPPORTED' if floor['H3_p95_above_0.70'] else 'FALSIFIED'}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(prereg.load().stamp({
        "analysis_role": args.role,
        "splits": args.splits,
        "splits_hash": sp.get("hash"),
        "pooling": args.pooling,
        "seed_base": seed_base,
        "floor": floor,
        "per_init": per_init,
    }), indent=2))
    print(f"[floor] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
