"""Training / evaluation loop shared by SlotMIL and every baseline.

Deliberately model-agnostic: the pooling module is the only thing that varies
between the SlotMIL run and the matched multi-head-ABMIL control, so the loop
cannot become an accidental confound.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .eval.classification import classification_metrics
from .losses import SlotMILLoss
from .utils.seed import seed_worker, set_seed


@dataclass
class TrainConfig:
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 8
    num_workers: int = 4
    device: str = "cuda"
    grad_clip: float | None = None  # implicit diff should make this unnecessary
    amp: bool = True
    early_stop_patience: int | None = 10
    seed: int = 0
    log_every: int = 20
    select_metric: str = "auc"
    extra: dict = field(default_factory=dict)


def _to_device(batch: dict, device: str) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def run_epoch(
    model,
    loader: DataLoader,
    criterion: SlotMILLoss,
    optimizer=None,
    device: str = "cuda",
    amp: bool = True,
    grad_clip: float | None = None,
    multilabel: bool = False,
) -> dict:
    train = optimizer is not None
    model.train(train)

    losses, comps_acc = [], {}
    all_scores, all_targets, health_acc = [], [], {}

    scaler = torch.amp.GradScaler("cuda", enabled=amp and "cuda" in device)

    for step, batch in enumerate(loader):
        batch = _to_device(batch, device)
        feats, pad_mask, labels = batch["features"], batch["pad_mask"], batch["label"]

        with torch.set_grad_enabled(train):
            with torch.autocast("cuda", enabled=amp and "cuda" in device):
                out = model(feats, pad_mask)
                # slice_index reaches the criterion, never the model: only
                # losses.normal_guidance_loss needs it, and routing it through
                # the loss leaves MILModel.forward's signature -- and every
                # eval-time caller of it -- untouched. Absent (older collates,
                # synthetic fixtures) it falls back to arange(S), which is what
                # the true index equals at eval time anyway.
                loss, comps = criterion(
                    out, labels, pad_mask=pad_mask,
                    slice_index=batch.get("slice_index"),
                )

        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

        losses.append(loss.item())
        for k, v in comps.items():
            comps_acc.setdefault(k, []).append(float(v))
        if "health" in out:
            for k, v in out["health"].items():
                health_acc.setdefault(k, []).append(v)

        logits = out["logits"].float()
        scores = (
            torch.sigmoid(logits) if multilabel else torch.softmax(logits, dim=-1)
        )
        all_scores.append(scores.detach().cpu().numpy())
        all_targets.append(labels.detach().cpu().numpy())

    y_score = np.concatenate(all_scores)
    y_true = np.concatenate(all_targets)

    metrics = classification_metrics(y_true, y_score, multilabel=multilabel)
    metrics["loss"] = float(np.mean(losses))
    for k, v in comps_acc.items():
        metrics[f"loss_{k}"] = float(np.mean(v))
    for k, v in health_acc.items():
        metrics[k] = float(np.mean(v))
    return metrics


def fit(
    model,
    train_ds,
    val_ds,
    cfg: TrainConfig,
    collate_fn,
    criterion: SlotMILLoss | None = None,
    test_ds=None,
    out_dir: str | Path | None = None,
    multilabel: bool = False,
) -> dict:
    set_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    criterion = criterion or SlotMILLoss(multilabel=multilabel)

    g = torch.Generator()
    g.manual_seed(cfg.seed)
    mk = lambda ds, shuffle: DataLoader(  # noqa: E731
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory="cuda" in device,
        worker_init_fn=seed_worker,
        generator=g if shuffle else None,
        drop_last=False,
    )
    train_loader, val_loader = mk(train_ds, True), mk(val_ds, False)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best = {"score": -np.inf, "epoch": -1, "state": None}
    history, patience = [], 0
    t0 = time.time()

    for epoch in range(cfg.epochs):
        tr = run_epoch(
            model, train_loader, criterion, optimizer, device,
            cfg.amp, cfg.grad_clip, multilabel,
        )
        va = run_epoch(model, val_loader, criterion, None, device, cfg.amp, None, multilabel)
        scheduler.step()

        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in tr.items()},
            **{f"val_{k}": v for k, v in va.items()},
        }
        history.append(row)

        score = va[cfg.select_metric]
        if score > best["score"]:
            best = {
                "score": score,
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
            patience = 0
        else:
            patience += 1
            if cfg.early_stop_patience and patience >= cfg.early_stop_patience:
                break

    result = {
        "best_epoch": best["epoch"],
        f"best_val_{cfg.select_metric}": best["score"],
        "history": history,
        "seconds": time.time() - t0,
        "config": asdict(cfg),
    }

    # Model selection happens on validation only; test is read once, at the end,
    # with the selected weights (plan.md line 58 -- "K was tuned on test" is the
    # objection this guards against).
    if test_ds is not None:
        if best["state"] is not None:
            model.load_state_dict(best["state"])
        te = run_epoch(model, mk(test_ds, False), criterion, None, device, cfg.amp, None, multilabel)
        result["test"] = te

    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "result.json", "w") as f:
            json.dump({k: v for k, v in result.items() if k != "history"}, f, indent=2)
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        if best["state"] is not None:
            torch.save(best["state"], out_dir / "best.pt")

    return result
