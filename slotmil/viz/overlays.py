"""Slot-coloured attention overlays -- the paper's main qualitative figure.

plan.md line 138 lists "slot overlays + alignment table" as the two ISBI figures.
Each slot gets a fixed colour so the same slot index is visually identifiable
across scans, which is what makes the specialisation claim legible at a glance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Colour-blind-safe qualitative palette (Okabe-Ito). Slot k always gets colour k,
# across every figure in the paper -- if slot 2 is green in one scan and blue in
# another, the reader cannot verify consistency visually.
SLOT_COLORS = np.array(
    [
        [0.90, 0.62, 0.00], [0.34, 0.71, 0.91], [0.00, 0.62, 0.45],
        [0.94, 0.89, 0.26], [0.00, 0.45, 0.70], [0.84, 0.37, 0.00],
        [0.80, 0.47, 0.65], [0.35, 0.35, 0.35],
    ]
)


def colorize_slots(attn_grid: np.ndarray, power: float = 1.0) -> np.ndarray:
    """``[K, H, W]`` slot attention -> ``[H, W, 3]`` RGB, plus an alpha map.

    Each pixel takes the colour of the slot that wins it, with opacity set by how
    decisively it was won. Rendering the argmax rather than a blend keeps the
    competition visible -- blending would hide exactly the duplication that the
    diversity regulariser exists to prevent.
    """
    k = attn_grid.shape[0]
    if k > len(SLOT_COLORS):
        raise ValueError(f"{k} slots exceeds the {len(SLOT_COLORS)}-colour palette")

    total = attn_grid.sum(axis=0, keepdims=True)
    share = attn_grid / np.clip(total, 1e-8, None)
    winner = share.argmax(axis=0)
    confidence = share.max(axis=0) ** power

    rgb = SLOT_COLORS[winner]
    return rgb, confidence


def overlay_slice(
    ct_slice: np.ndarray, attn_grid: np.ndarray, alpha: float = 0.55
) -> np.ndarray:
    """Blend a slot colour map over one CT slice. ``ct_slice`` in [0, 1]."""
    import torch
    import torch.nn.functional as F

    h, w = ct_slice.shape
    a = torch.from_numpy(attn_grid).float().unsqueeze(1)
    a = F.interpolate(a, size=(h, w), mode="bilinear", align_corners=False)
    a = a.squeeze(1).numpy()

    rgb, conf = colorize_slots(a)
    base = np.repeat(ct_slice[..., None], 3, axis=-1)
    weight = (alpha * conf)[..., None]
    return np.clip(base * (1 - weight) + rgb * weight, 0, 1)


def save_slot_figure(
    ct_volume: np.ndarray,
    attn: np.ndarray,
    out_path: str | Path,
    slice_indices: list[int] | None = None,
    lesion_mask: np.ndarray | None = None,
    n_cols: int = 4,
    title: str | None = None,
):
    """Grid figure: CT with slot overlay, optionally contoured against truth.

    Args:
        ct_volume: ``[S, H, W]`` in [0, 1].
        attn: ``[K, S, gh, gw]`` slot attention on the patch grid.
        slice_indices: which slices to render. Defaults to the slices carrying the
            most attention mass, since empty slices make an uninformative figure.
        lesion_mask: ``[S, H, W]`` ground truth, drawn as a contour so the reader
            can judge alignment rather than take the Dice number on faith.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if slice_indices is None:
        mass = attn.sum(axis=(0, 2, 3))
        n = min(n_cols * 2, len(mass))
        slice_indices = sorted(np.argsort(mass)[::-1][:n].tolist())

    n = len(slice_indices)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), squeeze=False)

    for ax in axes.ravel():
        ax.axis("off")

    for i, z in enumerate(slice_indices):
        ax = axes[i // n_cols][i % n_cols]
        ax.imshow(overlay_slice(ct_volume[z], attn[:, z]))
        if lesion_mask is not None and lesion_mask[z].any():
            ax.contour(lesion_mask[z], levels=[0.5], colors="white", linewidths=1.2)
        ax.set_title(f"z={z}", fontsize=9)

    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=9,
                   markerfacecolor=SLOT_COLORS[k], label=f"slot {k}")
        for k in range(attn.shape[0])
    ]
    fig.legend(handles=handles, loc="lower center", ncol=min(attn.shape[0], 8), frameon=False)
    if title:
        fig.suptitle(title)

    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
