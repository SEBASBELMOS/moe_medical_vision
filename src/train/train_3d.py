from __future__ import annotations

import math
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.amp import GradScaler, autocast


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    warnings.filterwarnings("ignore", category=FutureWarning, message=r"`torch\.cuda\.amp\..*")
    warnings.filterwarnings("ignore", category=FutureWarning, message=r"`torch\.cpu\.amp\..*")




def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation for 3D volumes."""
    if alpha > 0:
        lam = float(torch.distributions.Beta(alpha, alpha).sample())
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


@torch.no_grad()
def validate_3d_epoch(model, loader, criterion, device: str) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    preds_all, labels_all = [], []

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        losses.append(loss.item())

        preds = logits.argmax(dim=1)
        preds_all.extend(preds.cpu().numpy().tolist())
        labels_all.extend(y.cpu().numpy().tolist())

    return {
        "loss": float(np.mean(losses)) if losses else math.nan,
        "f1_macro": f1_score(labels_all, preds_all, average="macro") if labels_all else 0.0,
        "accuracy": accuracy_score(labels_all, preds_all) if labels_all else 0.0,
    }


def train_3d_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device: str,
    scaler: GradScaler,
    accum_steps: int = 1,
    mixed_precision: bool = True,
    max_grad_norm: Optional[float] = 1.0,
    use_mixup: bool = False,
    mixup_alpha: float = 0.2,
) -> Dict[str, float]:
    model.train()
    losses: List[float] = []
    preds_all, labels_all = [], []

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, start=1):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        if use_mixup and x.size(0) > 1:
            x, y_a, y_b, lam = mixup_data(x, y, alpha=mixup_alpha)
        else:
            y_a, y_b, lam = y, y, 1.0

        with autocast(device_type='cuda', dtype=torch.float16, enabled=mixed_precision and device.startswith('cuda')):
            logits = model(x)
            if use_mixup and x.size(0) > 1 and lam < 1.0:
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam) / accum_steps
            else:
                loss = criterion(logits, y) / accum_steps

        scaler.scale(loss).backward()

        if step % accum_steps == 0:
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        losses.append(loss.item() * accum_steps)
        preds = logits.argmax(dim=1)
        preds_all.extend(preds.detach().cpu().numpy().tolist())
        labels_all.extend(y.detach().cpu().numpy().tolist())

    # flush final grads si quedó residuo
    if len(loader) % accum_steps != 0:
        if max_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return {
        "loss": float(np.mean(losses)) if losses else math.nan,
        "f1_macro": f1_score(labels_all, preds_all, average="macro") if labels_all else 0.0,
        "accuracy": accuracy_score(labels_all, preds_all) if labels_all else 0.0,
    }


@torch.no_grad()
def sanity_check_single_batch(model, loader, criterion, device: str) -> Dict[str, object]:
    model.eval()
    batch = next(iter(loader))
    x = batch["image"].to(device)
    y = batch["label"].to(device)
    logits = model(x)
    loss = criterion(logits, y)
    return {
        "input_shape": tuple(x.shape),
        "logits_shape": tuple(logits.shape),
        "loss": float(loss.item()),
        "labels": y[:8].detach().cpu().tolist(),
    }


def fit_3d_expert(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device: str,
    epochs: int,
    checkpoint_path: str | Path,
    scheduler=None,
    accum_steps: int = 1,
    mixed_precision: bool = True,
    patience: int = 5,
    max_grad_norm: Optional[float] = 1.0,
    use_mixup: bool = False,
    mixup_alpha: float = 0.2,
) -> Dict[str, object]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    scaler = GradScaler('cuda', enabled=mixed_precision and device.startswith('cuda'))
    best_f1 = -1.0
    best_epoch = 0
    wait = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        train_metrics = train_3d_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            accum_steps=accum_steps, mixed_precision=mixed_precision,
            max_grad_norm=max_grad_norm,
            use_mixup=use_mixup, mixup_alpha=mixup_alpha,
        )
        val_metrics = validate_3d_epoch(model, val_loader, criterion, device)

        if scheduler is not None:
            if scheduler.__class__.__name__ == "ReduceLROnPlateau":
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_metrics["loss"],
            "train_f1": train_metrics["f1_macro"],
            "train_acc": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_f1": val_metrics["f1_macro"],
            "val_acc": val_metrics["accuracy"],
        }
        history.append(row)
        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={row['train_loss']:.4f} train_f1={row['train_f1']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_f1={row['val_f1']:.4f} lr={lr:.2e}"
        )

        if row["val_f1"] > best_f1:
            best_f1 = row["val_f1"]
            best_epoch = epoch
            wait = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_f1": best_f1,
                    "history": history,
                },
                checkpoint_path,
            )
            print(f"  -> nuevo mejor checkpoint: {checkpoint_path.name} (val_f1={best_f1:.4f})")
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping activado en epoch {epoch}. Mejor epoch: {best_epoch}")
                break

    return {
        "best_val_f1": best_f1,
        "best_epoch": best_epoch,
        "history": history,
        "checkpoint_path": str(checkpoint_path),
    }
