#!/usr/bin/env python
"""Download COV19-CT-DB (PHAROS-AIF-MIH competition distribution).

Access granted 2026-02-12 to team "Terps" (UMD), PoC Mohammad Nayeem Teli, under
the EULA already on file. Links come from the organisers' distribution email.

Two datasets ship together:

**Multi-Source Covid-19 Detection** -- the one plan.md line 70 wants. Volume-level
COVID / non-COVID labels plus a per-scan CSV giving the source data centre (0-3),
which is exactly what the cross-centre generalisation experiment needs.

**Fair Disease Diagnosis** -- a different task: 4 classes (Adenocarcinoma,
Squamous Cell Carcinoma, normal, covid), each split male/female. Not required by
the plan; downloaded only with --include-fair.

Citation obligation: using this data requires citing 12 papers plus a
forthcoming white paper. See docs/cov19d_access_request.md -- that is a real
constraint on a 4-page ISBI submission.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Multi-Source Covid-19 Detection Challenge
COVID_DETECTION = {
    "train_covid_part1":     "1g26d6-QoPWpG-SIjkqwKCKRekgATeFCE",
    "train_covid_part2":     "1JkbqK9bxKyuIhj-ZB9joRkYo-CuJCCAG",
    "train_covid_centres":   "11zjdJztL8DATNsO21JAX9QafxKeYUvbP",   # CSV
    "train_noncovid_part1":  "1e7Kv2VC0xMTtgiafyst7S-bPQ2lZ8fp_",
    "train_noncovid_part2":  "1ab6fzG5_96SLjA6ETwi_F9z6z8FgPiHe",
    "train_noncovid_part3":  "1gLbkKDmW5YGz73f23zf8iYMQ2iIiwmcd",
    "train_noncovid_centres":"1nrjof8Qu55WEazgGtx2oSM_cCYO1J-Tx",   # CSV
    "val_all":               "1PqQB41L-7rcFRfdpes5iaF6IWVMxX-pN",
    "val_covid_centres":     "155W8e4h0t1odKspjxCiK6Cl4-81PK2zH",   # CSV
    "val_noncovid_centres":  "1dhVi_0Sldyj4hrBpfRGkLV4PObyzDsj5",   # CSV
}

# Fair Disease Diagnosis Challenge (optional)
FAIR_DIAGNOSIS = {
    "fair_train_part1": "1ESQtA_vkaqdSYjk8152jUy45sGorI1je",
    "fair_train_part2": "1vexIaYCm5r_dgklmQUtSj73O_zrtpd-U",
    "fair_val":         "17QNyqvzX4dkOe4gyCwv8AwRKLzH2tCqv",
}

CSV_KEYS = {k for k in COVID_DETECTION if k.endswith("centres")}


def free_gb(path: Path) -> float:
    import os
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1024**3


def download(gdown_bin: str, file_id: str, dest: Path, quiet: bool) -> bool:
    """Fetch one Drive file. Returns True on success.

    Uses gdown rather than curl: large Drive files return an HTML confirmation
    interstitial instead of bytes, and gdown handles that handshake.
    """
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  {dest.name}: already present ({dest.stat().st_size/1e9:.2f} GB), skipping")
        return True
    cmd = [gdown_bin, "--no-cookies", "-O", str(dest),
           f"https://drive.google.com/uc?id={file_id}"]
    if quiet:
        cmd.insert(1, "--quiet")
    r = subprocess.run(cmd)
    ok = r.returncode == 0 and dest.exists() and dest.stat().st_size > 0
    if not ok:
        # A Drive quota block writes a small HTML body rather than failing loudly.
        if dest.exists() and dest.stat().st_size < 100_000:
            head = dest.read_bytes()[:400].decode("utf-8", "ignore")
            if "<html" in head.lower():
                print(f"  {dest.name}: got HTML, not data -- likely a Drive download "
                      f"quota block. Retry later or use a browser.", file=sys.stderr)
            dest.unlink()
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/fs/nexus-scratch/bthapar/SlotMIL/data/cov19d")
    ap.add_argument("--include-fair", action="store_true",
                    help="also fetch the Fair Disease Diagnosis dataset (not needed by the plan)")
    ap.add_argument("--csv-only", action="store_true",
                    help="fetch only the small centre CSVs, to inspect the split first")
    ap.add_argument("--min-free-gb", type=float, default=15.0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    gdown_bin = str(Path(__file__).resolve().parents[1] / ".venv" / "bin" / "gdown")
    if not Path(gdown_bin).exists():
        gdown_bin = "gdown"

    targets = dict(COVID_DETECTION)
    if args.include_fair:
        targets.update(FAIR_DIAGNOSIS)
    if args.csv_only:
        targets = {k: v for k, v in targets.items() if k in CSV_KEYS}

    print(f"[cov19d] {len(targets)} files -> {out}")
    print(f"[cov19d] {free_gb(out):.1f} GB free")

    ok = failed = 0
    for name, fid in targets.items():
        if free_gb(out) < args.min_free_gb:
            print(f"[cov19d] ABORT: below --min-free-gb {args.min_free_gb}", file=sys.stderr)
            break
        suffix = ".csv" if name in CSV_KEYS else ".zip"
        print(f"[cov19d] {name}")
        if download(gdown_bin, fid, out / f"{name}{suffix}", quiet=False):
            ok += 1
        else:
            failed += 1
            print(f"[cov19d]   FAILED {name}", file=sys.stderr)

    print(f"\n[cov19d] {ok} ok, {failed} failed. {free_gb(out):.1f} GB free")
    if failed:
        print("[cov19d] Google Drive rate-limits large public files. Re-running is "
              "safe -- completed files are skipped.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
