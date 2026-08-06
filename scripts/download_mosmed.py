#!/usr/bin/env python
"""Fetch MosMedData via a Kaggle mirror.

There is no unauthenticated route to this dataset. mosmed.ai requires
registration, and no HuggingFace or Zenodo mirror of the full release exists --
that was checked directly, not assumed:

    huggingface  api/datasets?search=mosmed        -> 0 results
    zenodo       api/records?q=mosmed              -> only "2 samples from MosMed"
    mosmed.ai    known direct archive URLs         -> 404

Kaggle mirrors do exist and are what this script uses. It needs a Kaggle API
token at ~/.kaggle/kaggle.json (Kaggle -> Settings -> API -> Create New Token).

Licence: CC BY-NC-ND 3.0 -- non-commercial, no-derivatives. Fine for academic
evaluation; note it in the paper.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

MIRRORS = {
    "full": ("mathurinache/mosmeddata-chest-ct-scans-with-covid19", "~11.9 GB, all 1110 volumes"),
    "masks": ("andrewmvd/mosmed-covid19-ct-scans", "~1.8 GB, includes the 50 expert masks"),
}

TOKEN_PATHS = [
    Path.home() / ".kaggle" / "kaggle.json",
    Path.home() / ".config" / "kaggle" / "kaggle.json",
]


def have_token() -> bool:
    return any(p.exists() for p in TOKEN_PATHS) or (
        "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/fs/nexus-scratch/bthapar/SlotMIL/data/mosmed")
    ap.add_argument("--which", nargs="+", default=["masks", "full"], choices=list(MIRRORS))
    args = ap.parse_args()

    if not have_token():
        print(
            "No Kaggle credentials found.\n\n"
            "  Looked in:\n"
            + "".join(f"    {p}\n" for p in TOKEN_PATHS)
            + "    $KAGGLE_USERNAME / $KAGGLE_KEY\n\n"
            "  To fix: Kaggle -> Settings -> API -> 'Create New Token', then\n"
            "    mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/\n"
            "    chmod 600 ~/.kaggle/kaggle.json\n\n"
            "  Alternative: register at mosmed.ai, download COVID19_1110 manually,\n"
            f"  and unpack it to {args.out}\n",
            file=sys.stderr,
        )
        return 1

    try:
        import kaggle  # noqa: F401  (authenticates at import)
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("pip install kaggle", file=sys.stderr)
        return 1

    api = KaggleApi()
    api.authenticate()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for key in args.which:
        slug, desc = MIRRORS[key]
        dest = out / key
        dest.mkdir(parents=True, exist_ok=True)
        print(f"[mosmed] downloading {slug} ({desc}) -> {dest}", flush=True)
        api.dataset_download_files(slug, path=str(dest), unzip=True, quiet=False)

    from slotmil.data.mosmed import MosMedIndex

    print("\n[mosmed] index:", MosMedIndex(out).summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
