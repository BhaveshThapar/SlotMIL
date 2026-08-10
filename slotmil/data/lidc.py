"""LIDC-IDRI: TCIA download, pylidc consensus masks, and label construction.

LIDC is the key localisation dataset (plan.md line 63): 1,018 thoracic CT series
with nodules outlined by four radiologists, so slot attention can be scored
against real expert masks rather than against a proxy.

The hard constraint that shapes this whole module: **pylidc needs the DICOM on
disk to build a mask**, because placing an annotation's contours into the volume
requires the per-slice z positions, which live in the DICOM headers and not in
pylidc's bundled annotation database. In the staged pipeline that means masks
must be extracted *before* the DICOM batch is deleted. Getting that order wrong
means re-downloading 124 GB.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import numpy as np
import requests

NBIA_BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v1"


def _restore_numpy_aliases() -> None:
    """Restore the ``np.int`` / ``np.float`` / ``np.bool`` aliases pylidc needs.

    pylidc 0.2.3 (the current release, from 2020) still uses ``np.int`` in
    ``Annotation.py`` and ``Contour.py``, ``np.float`` in ``Annotation.py`` and
    ``utils.py``, and ``np.bool`` in ``Annotation.py``. NumPy deprecated those
    aliases in 1.20 and removed them in 1.24; we run 1.26.

    Downgrading NumPy below 1.24 to satisfy a 2020 library would drag the whole
    stack (torch, scikit-learn, scipy) backwards, so the aliases are restored
    here instead. They are exact synonyms for the builtins, which is what NumPy
    objected to -- restoring them is safe, it is the ambiguity NumPy wanted gone,
    not the behaviour.

    Must run before pylidc is imported.
    """
    for name, builtin in (("int", int), ("float", float), ("bool", bool),
                          ("object", object), ("str", str)):
        if not hasattr(np, name):
            setattr(np, name, builtin)

# Malignancy is rated 1-5 by each radiologist. The standard binary setup takes
# the median across readers and drops median==3 ("indeterminate"), which is what
# the 656-nodule / 352-malignant split in the literature refers to.
MALIGNANCY_INDETERMINATE = 3


def list_ct_series(timeout: int = 120) -> list[dict]:
    """All LIDC-IDRI CT series from the TCIA NBIA API.

    Public collection -- no authentication required.
    """
    r = requests.get(
        f"{NBIA_BASE}/getSeries",
        params={"Collection": "LIDC-IDRI", "Modality": "CT"},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def download_series(
    series_uid: str, dest_dir: str | Path, timeout: int = 600, chunk: int = 1 << 20
) -> Path:
    """Fetch one series as DICOM and unpack it into ``dest_dir``.

    Files are laid out as ``<dest_dir>/<patient>/<study>/<series>/`` because that
    is the hierarchy pylidc walks; the NBIA zip itself is flat.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp_zip = dest_dir / f"{series_uid}.zip"

    with requests.get(
        f"{NBIA_BASE}/getImage",
        params={"SeriesInstanceUID": series_uid},
        stream=True,
        timeout=timeout,
    ) as r:
        r.raise_for_status()
        with open(tmp_zip, "wb") as f:
            for block in r.iter_content(chunk_size=chunk):
                f.write(block)

    with zipfile.ZipFile(tmp_zip) as z:
        z.extractall(dest_dir)
    tmp_zip.unlink()
    return dest_dir


def arrange_for_pylidc(
    flat_dir: str | Path, patient_id: str, study_uid: str, series_uid: str
) -> Path:
    """Move flat DICOM files into the TCIA hierarchy pylidc expects."""
    flat_dir = Path(flat_dir)
    target = flat_dir.parent / patient_id / study_uid / series_uid
    target.mkdir(parents=True, exist_ok=True)
    for f in flat_dir.glob("*.dcm"):
        f.rename(target / f.name)
    return target


def configure_pylidc(dicom_root: str | Path) -> None:
    """Write ``~/.pylidcrc`` pointing at ``dicom_root``.

    pylidc reads this at import time, so it must be written before the library is
    imported in the worker process.
    """
    cfg = Path.home() / ".pylidcrc"
    cfg.write_text(f"[dicom]\npath = {Path(dicom_root).resolve()}\nwarn = False\n")


def load_scan_and_masks(
    series_uid: str, consensus_level: float = 0.5, verbose: bool = False
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return ``(volume_hu [S, H, W], lesion_mask [S, H, W], meta)``.

    ``consensus_level`` 0.5 keeps voxels marked by at least half the radiologists
    who annotated a given nodule -- the usual choice, and it exploits the
    four-reader structure that makes LIDC worth using.
    """
    _restore_numpy_aliases()
    import pylidc as pl
    from pylidc.utils import consensus

    scan = pl.query(pl.Scan).filter(pl.Scan.series_instance_uid == series_uid).first()
    if scan is None:
        raise KeyError(f"series {series_uid} not in the pylidc annotation database")

    volume = scan.to_volume(verbose=verbose)  # H, W, S in pylidc's axis order
    volume = np.transpose(volume, (2, 0, 1))  # -> S, H, W

    mask = np.zeros_like(volume, dtype=np.uint8)
    clusters = scan.cluster_annotations()
    nodules = []

    for anns in clusters:
        try:
            cmask, cbbox, _ = consensus(anns, clevel=consensus_level)
        except Exception:
            continue
        # cbbox indexes (H, W, S); transpose to match the volume above.
        sub = mask[cbbox[2], cbbox[0], cbbox[1]]
        mask[cbbox[2], cbbox[0], cbbox[1]] = np.maximum(
            sub, np.transpose(cmask, (2, 0, 1)).astype(np.uint8)
        )
        mals = [a.malignancy for a in anns]
        nodules.append(
            {
                "n_annotations": len(anns),
                "malignancy_median": float(np.median(mals)),
                "diameter": float(np.mean([a.diameter for a in anns])),
            }
        )

    meta = {
        "patient_id": scan.patient_id,
        "slice_thickness": float(scan.slice_thickness),
        "pixel_spacing": float(scan.pixel_spacing),
        "n_nodules": len(nodules),
        "nodules": nodules,
    }
    return volume, mask, meta


def scan_label(meta: dict, mode: str = "nodule_present") -> int | None:
    """Bag-level label.

    ``nodule_present``  -- 1 if the scan has any consensus nodule. The weakest and
                           most honest weak label; every scan gets one.
    ``malignancy``      -- 1 if any nodule has median malignancy > 3, 0 if all are
                           < 3. Returns ``None`` when every nodule is
                           indeterminate (median == 3), so the caller can drop the
                           scan rather than guess.
    """
    if mode == "nodule_present":
        return int(meta["n_nodules"] > 0)

    if mode == "malignancy":
        meds = [n["malignancy_median"] for n in meta["nodules"]]
        decisive = [m for m in meds if m != MALIGNANCY_INDETERMINATE]
        if not decisive:
            return None
        return int(any(m > MALIGNANCY_INDETERMINATE for m in decisive))

    raise ValueError(f"unknown label mode {mode!r}")


def free_gb(path: str | Path = "/fs/nexus-scratch/bthapar") -> float:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1024**3
