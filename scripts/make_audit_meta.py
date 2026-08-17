#!/usr/bin/env python
"""Write the ``_audit_meta.json`` map that seven analysis drivers read.

``runs/_audit_meta.json`` is the uid -> patient map every patient-clustered
bootstrap resamples on. It is read by ``axis_gate.py``, ``template_family.py``,
``h4_cross_patient.py``, ``h7_content_free.py``, ``h8_in_lung.py``,
``probe_gate.py`` and ``untrained_floor.py`` -- and until this script, **written
by nothing committed**. A file that every reported interval depends on and that
no committed code can regenerate is a hole in the chain: if it were lost or
edited, nothing would say so and every CI in the paper would silently change
width.

The schema is four flat uid-keyed maps, not one record per uid, because that is
what the existing file has and what ``["pat"]`` at ``template_family.py:238``
indexes:

    {"pat": {uid: patient}, "lab": {uid: label},
     "mask": {uid: bool},   "nsl": {uid: n_slices}}

**Patient identity differs by dataset and the difference matters.** LIDC carries
``patient_id`` on the group because one patient can contribute more than one
series, so resampling series instead of patients would understate every
interval. MosMed has no such attribute: one study is one patient, so the uid is
the patient and the map is the identity. That is not a fallback for missing
data -- it is the correct map -- so it is stated here rather than left to
``uid_to_pat.get(uid, uid)`` to do silently downstream.

**Mask presence is read from the group, not from an attribute.** MosMed sets
``has_mask``; LIDC does not set it at all and carries a ``mask`` dataset
instead. Trusting the attribute would mark all 999 LIDC series unmasked.

The output is byte-compatible with the committed LIDC file, which is the test:
``--check`` regenerates and diffs rather than overwriting.

    python scripts/make_audit_meta.py --cache data/lidc/features_dinov2_vitb14.h5 --check
    python scripts/make_audit_meta.py --cache data/mosmed/features_dinov2_vitb14.h5 \
        --out runs/_audit_meta_mosmed.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py

from slotmil import prereg


def build(cache: Path) -> dict[str, dict]:
    """The four uid-keyed maps, in the committed file's key order."""
    pat: dict[str, str] = {}
    lab: dict[str, int] = {}
    mask: dict[str, bool] = {}
    nsl: dict[str, int] = {}
    with h5py.File(cache, "r") as f:
        for uid in f:
            g = f[uid]
            # LIDC groups one patient's several series under `patient_id`; MosMed
            # has one study per patient, so the uid IS the patient.
            pat[uid] = str(g.attrs.get("patient_id", uid))
            lab[uid] = int(g.attrs["label"])
            mask[uid] = "mask" in g
            nsl[uid] = int(g.attrs["n_slices"])
    return {"pat": pat, "lab": lab, "mask": mask, "nsl": nsl}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default="runs/_audit_meta.json")
    ap.add_argument("--check", action="store_true",
                    help="regenerate and diff against --out instead of writing; "
                         "exits non-zero on any difference")
    args = ap.parse_args()

    cache = Path(args.cache)
    out = Path(args.out)
    meta = build(cache)
    n = len(meta["pat"])
    n_pat = len(set(meta["pat"].values()))
    n_mask = sum(meta["mask"].values())
    print(f"[meta] {cache}: {n} series, {n_pat} patients, {n_mask} with masks")

    if args.check:
        if not out.exists():
            raise SystemExit(f"{out} does not exist; nothing to check against")
        have = json.loads(out.read_text())
        # Compare the four maps by content. Key order is not load-bearing -- every
        # reader indexes by name -- so a reordering must not be reported as drift.
        diffs = [k for k in ("pat", "lab", "mask", "nsl") if have.get(k) != meta[k]]
        extra = sorted(set(have) - {"pat", "lab", "mask", "nsl"})
        if extra:
            print(f"[meta] {out} carries extra top-level keys: {extra}")
        if diffs:
            for k in diffs:
                a, b = have.get(k) or {}, meta[k]
                only_disk = sorted(set(a) - set(b))[:3]
                only_built = sorted(set(b) - set(a))[:3]
                changed = sorted(u for u in set(a) & set(b) if a[u] != b[u])[:3]
                print(f"  {k}: on disk {len(a)}, rebuilt {len(b)}; "
                      f"only-on-disk {only_disk}, only-rebuilt {only_built}, "
                      f"changed {changed}")
            raise SystemExit(f"{out} does not match a rebuild from {cache}: {diffs}")
        print(f"[meta] MATCH -- {out} is reproducible from {cache}")
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta))
    # Provenance rides in a sidecar rather than in the file, so the map stays
    # byte-compatible with the committed LIDC one and no reader has to learn a
    # new top-level key. Same convention as null_collect_attn.py's .prereg.json.
    side = out.with_suffix(".provenance.json")
    pre = prereg.load()
    side.write_text(json.dumps(pre.stamp({
        "cache": str(cache),
        "cache_sha256_note": "the cache is too large to hash per run; identity is "
                             "the path plus the counts below",
        "n_series": n, "n_patients": n_pat, "n_with_mask": n_mask,
        "identity_patient_map": n_pat == n,
    }), indent=2))
    print(f"[meta] wrote {out} and {side}")


if __name__ == "__main__":
    main()
