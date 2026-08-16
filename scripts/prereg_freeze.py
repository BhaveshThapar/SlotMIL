#!/usr/bin/env python
"""Freeze the pre-registration: resolve split hashes, stamp the doc, make the key.

Run this once, after the confirmatory split exists and before any confirmatory
analysis. It is the step that turns a config file into a commitment:

    1. fills in any pending split hash (there must be exactly one)
    2. verifies every declared split file exists and still hashes as recorded
    3. computes the config hash and writes it into PREREGISTRATION.md
    4. creates the blinding key if it is not already there

Re-running with a changed config refuses to overwrite an existing hash unless
``--amend`` is passed, and ``--amend`` reminds you to log the change. That is the
whole enforcement story: the hash in the committed document is the promise, and
changing it has to be a deliberate, recorded act.

    .venv/bin/python scripts/prereg_freeze.py --update-splits
    .venv/bin/python scripts/prereg_freeze.py --check      # CI-style, no writes
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slotmil.prereg import (  # noqa: E402
    PreregViolation,
    amendment_chain,
    classify_hash,
    file_sha256,
    load,
)

DOC_PATTERN = re.compile(r"(\*\*Frozen config hash:\*\* `)([0-9a-f]{16}|PENDING_FREEZE)(`)")
PENDING = re.compile(r"^(\s*recorded_hash:\s*)null(\s*(?:#.*)?)$", re.M)


def fill_pending_split_hash(cfg_path: Path) -> str | None:
    """Replace the single pending ``recorded_hash: null`` with the real hash.

    Deliberately refuses when there is more than one pending entry rather than
    guessing which split a bare ``null`` belongs to -- a wrong guess here would
    silently pin the wrong file.
    """
    text = cfg_path.read_text()
    hits = PENDING.findall(text)
    if not hits:
        return None
    if len(hits) > 1:
        raise PreregViolation(
            f"{len(hits)} pending recorded_hash entries in {cfg_path}; this script "
            "fills exactly one so it cannot mis-assign. Fill the others by hand."
        )

    pre = load(cfg_path)
    pending = [k for k, v in pre.verify_splits(must_exist=False).items()
               if v in ("unrecorded", "not-yet-generated")]
    if len(pending) != 1:
        raise PreregViolation(f"expected one unresolved split, found {pending}")
    dataset, role = pending[0].split(".")
    scfg = pre.split(dataset, role)
    path = Path(scfg["path"])
    if not path.exists():
        raise PreregViolation(
            f"split {pending[0]} declared at {path} does not exist yet. Generate it "
            "first:\n  .venv/bin/python scripts/make_splits.py --cache "
            f"{pre.get(f'datasets.{dataset}.cache')} --out {path} --seed {scfg['seed']}"
        )
    recorded = json.loads(path.read_text())["hash"]
    cfg_path.write_text(PENDING.sub(rf"\g<1>{recorded}\g<2>", text, count=1))
    print(f"[freeze] {pending[0]} -> recorded_hash {recorded}  ({path})")
    return recorded


def report_stamp_lineage(runs: Path, current: str, amendments: Path) -> int:
    """Place every stamped result under ``runs`` on the amendment chain.

    Returns the number of stamps that could not be placed, which is what should
    fail the check. Superseded is *not* a failure: an amendment is expected to
    leave ancestors behind, and PREREGISTRATION.md:189-191 read mechanically
    ("hash does not match the frozen config -> exploratory") would demote
    runs/untrained_floor.json -- the only confirmatory results on disk, carrying
    H3 -- over two amendments that never touched what it computes. UNKNOWN is a
    failure, because a stamp that is on no recorded config state means a result
    was produced under a plan nobody wrote down.
    """
    chain = amendment_chain(amendments)
    print(f"[freeze] lineage {len(chain)} recorded transition(s): "
          f"{chain[0].before} -> {chain[-1].after}")
    for a in chain:
        note = "" if a.changed_the_config else "  (no config change)"
        print(f"[freeze]   amend {a.date}  {a.before} -> {a.after}{note}")

    if not runs.exists():
        print(f"[freeze]   no {runs}/ directory; no stamped results to place")
        return 0

    counts = {"current": 0, "superseded": 0, "unknown": 0}
    unreadable = 0
    for p in sorted(runs.rglob("*.json")):
        try:
            obj = json.loads(p.read_text())
        except (OSError, ValueError):
            unreadable += 1
            continue
        stamp = obj.get("prereg") if isinstance(obj, dict) else None
        if not isinstance(stamp, dict) or "prereg_hash" not in stamp:
            continue
        lin = classify_hash(stamp["prereg_hash"], current, chain)
        counts[lin.status] += 1
        print(f"[freeze]   stamp {str(p):44s} {stamp['prereg_hash']}  {lin.label}")

    total = sum(counts.values())
    print(f"[freeze] stamps {total} placed: {counts['current']} current, "
          f"{counts['superseded']} superseded, {counts['unknown']} UNKNOWN")
    if unreadable:
        print(f"[freeze] {unreadable} unreadable json under {runs} (skipped, "
              "not treated as stamped)", file=sys.stderr)
    if counts["unknown"]:
        print(f"[freeze] {counts['unknown']} stamp(s) carry a hash on no recorded "
              f"config state; record the missing transition in {amendments} or "
              "re-run them", file=sys.stderr)
    return counts["unknown"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/prereg/isbi2027.yaml")
    ap.add_argument("--doc", default="PREREGISTRATION.md")
    ap.add_argument("--amendments", default="AMENDMENTS.md")
    ap.add_argument("--runs", default="runs",
                    help="tree of stamped result files to place on the chain")
    ap.add_argument("--update-splits", action="store_true",
                    help="resolve a pending recorded_hash from the split file on disk")
    ap.add_argument("--amend", action="store_true",
                    help="permit overwriting a hash already recorded in the doc")
    ap.add_argument("--check", action="store_true",
                    help="verify only; make no writes. Exit 1 on any mismatch.")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if args.update_splits and not args.check:
        fill_pending_split_hash(cfg_path)

    pre = load(cfg_path)
    print(f"[freeze] config {cfg_path}  version {pre.version}  hash {pre.hash}")

    report = pre.verify_splits(must_exist=False)
    bad = {k: v for k, v in report.items() if v != "ok"}
    for k, v in sorted(report.items()):
        print(f"[freeze]   split {k:32s} {v}")
    if bad:
        print(f"[freeze] {len(bad)} split(s) not verified: {sorted(bad)}", file=sys.stderr)
        if args.check:
            return 1

    # Independent byte hash -- make_splits' own hash covers an absolute path, not
    # the cache contents, so it is not on its own a pin.
    for ds, dcfg in pre.get("datasets").items():
        for role, scfg in dcfg.get("splits", {}).items():
            p = Path(scfg["path"])
            if p.exists():
                print(f"[freeze]   bytes {ds}.{role:22s} sha256 {file_sha256(p)}")

    doc = Path(args.doc)
    if not doc.exists():
        print(f"[freeze] {doc} not found", file=sys.stderr)
        return 1
    text = doc.read_text()
    m = DOC_PATTERN.search(text)
    if not m:
        print(f"[freeze] no '**Frozen config hash:** `...`' marker in {doc}",
              file=sys.stderr)
        return 1

    existing = m.group(2)
    if args.check:
        ok = existing == pre.hash
        print(f"[freeze] doc hash {existing} vs config {pre.hash}: "
              f"{'MATCH' if ok else 'MISMATCH'}")
        # Lineage is reported only under --check. A half-written amendment
        # (`before -> PENDING`) makes the chain unreadable by design, and that
        # is exactly the state --amend exists to leave, so the write path must
        # not depend on it.
        try:
            lineage_ok = report_stamp_lineage(
                Path(args.runs), pre.hash, Path(args.amendments)) == 0
        except PreregViolation as exc:
            print(f"[freeze] amendment chain unusable: {exc}", file=sys.stderr)
            lineage_ok = False
        return 0 if ok and not bad and lineage_ok else 1

    if existing not in ("PENDING_FREEZE", pre.hash) and not args.amend:
        print(
            f"[freeze] {doc} already records {existing}, config now hashes to "
            f"{pre.hash}.\n"
            "         The pre-registration has changed since it was frozen. Pass "
            "--amend if that\n"
            "         is intended, and add a dated entry to AMENDMENTS.md saying "
            "what changed,\n"
            "         why, and whether you had seen the affected results.",
            file=sys.stderr,
        )
        return 1

    doc.write_text(DOC_PATTERN.sub(rf"\g<1>{pre.hash}\g<3>", text, count=1))
    print(f"[freeze] wrote hash {pre.hash} into {doc}")

    key_path = Path(pre.get("blinding.key_path"))
    if key_path.exists():
        print(f"[freeze] blinding key already exists at {key_path} (left alone)")
    else:
        key = pre.blind_key(create_if_missing=True)
        print(f"[freeze] created blinding key at {key_path} "
              f"({len(key.codes)} arms; gitignored, do not commit)")

    print("\n[freeze] Next: run the tests, then commit. That commit is the "
          "pre-registration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
