#!/usr/bin/env python
"""Reveal the arm -> code mapping, and record that it happened.

Unblinding is not forbidden -- the numbers have to be written up eventually. It
is meant to be *deliberate and dated*, so that "we looked at which method was
which, then changed the analysis" is visible in the record rather than
reconstructable only from memory.

Anyone with filesystem access can read runs/prereg/unblind_key.json directly and
skip this script entirely. That is understood and stated in PREREGISTRATION.md;
the blinding is procedural, not cryptographic. This script exists so that the
honest path is also the easy one.

    .venv/bin/python scripts/prereg_unblind.py \\
        --reason "writing Table 1" --results runs/nulls/axis_gate.json
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slotmil.prereg import git_state, load  # noqa: E402

ENTRY = """
## {when} — unblinding

- **Config hash at unblinding:** `{hash}`
- **Git commit:** `{commit}`{dirty}
- **Results in scope:** {results}
- **Reason:** {reason}
- **Arms revealed:** {n} ({codes})
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/prereg/isbi2027.yaml")
    ap.add_argument("--log", default="AMENDMENTS.md")
    ap.add_argument("--reason", required=True,
                    help="why you are unblinding now, in one line")
    ap.add_argument("--results", default="(unspecified)",
                    help="which results files this unblinding covers")
    ap.add_argument("--no-log", action="store_true",
                    help="print the mapping without recording it (discouraged)")
    args = ap.parse_args()

    pre = load(args.config)
    key = pre.blind_key()

    print(f"{'arm':<24} code")
    print("-" * 40)
    for name, code in sorted(key.codes.items()):
        print(f"{name:<24} {code}")

    if args.no_log:
        print("\n[unblind] NOT recorded (--no-log).", file=sys.stderr)
        return 0

    g = git_state()
    entry = ENTRY.format(
        when=date.today().isoformat(),
        hash=pre.hash,
        commit=g["commit"] or "unknown",
        dirty=" *(working tree dirty)*" if g["dirty"] else "",
        results=args.results,
        reason=args.reason,
        n=len(key.codes),
        codes=", ".join(sorted(key.codes.values())),
    )
    log = Path(args.log)
    log.write_text(log.read_text().rstrip() + "\n" + entry if log.exists() else entry)
    print(f"\n[unblind] recorded in {log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
