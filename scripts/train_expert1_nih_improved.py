from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path("/workspace/moe_medical_vision")
sys.path.insert(0, str(ROOT / "src"))

from data.datasets import NIHChestXray14Dataset  # noqa: E402
from nih_preprocessing_v2 import NIHXrayTransform  # noqa: E402
import timm  # noqa: E402


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR = CHECKPOINT_DIR / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 16
FREEZE_EPOCHS = 2
BATCH_SIZE = 24
ACCUM_STEPS = 2
NUM_WORKERS = 8
SEED = 42
PATIENCE = 5
USE_FP16 = torch.cuda.is_available()

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class AsymmetricLoss(nn.Module):
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        xs_pos = prob
        xs_neg = 1.0 - prob

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        pt = xs_pos * targets + xs_neg * (1 - targets)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
        one_sided_w = torch.pow(1 - pt, gamma)
        return -(loss * one_sided_w).mean()


class NIHDenseNetHead(nn.Module):
    def __init__(self, num_classes: int = 14, dropout: float = 0.2):
        super().__init__()
        self.backbone = timm.create_model("densenet121", pretrained=True, num_classes=0)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(self.backbone.num_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.head(feat)


def build_weighted_sampler(dataset: NIHChestXray14Dataset) -> WeightedRandomSampler:
    labels = np.array(dataset.df["labels_list"].tolist(), dtype=np.float32)
    freq = labels.mean(axis=0)
    inv = 1.0 / np.clip(freq, 1e-4, None)
    sample_weights = (labels * inv[None, :]).sum(axis=1)
    sample_weights = np.where(
        sample_weights > 0,
        sample_weights,
        np.median(sample_weights[sample_weights > 0]),
    )
    sample_weights = np.clip(sample_weights, None, np.percentile(sample_weights, 95))
    return WeightedRandomSampler(
        sample_weights.tolist(), num_samples=len(sample_weights), replacement=True
    )


def tune_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    grid = np.arange(0.10, 0.91, 0.05)
    thresholds = np.full(y_true.shape[1], 0.5, dtype=np.float32)
    for i in range(y_true.shape[1]):
        best_t = 0.5
        best_f1 = -1.0
        yt = y_true[:, i]
        yp = y_prob[:, i]
        for t in grid:
            pred = (yp >= t).astype(np.int64)
            f1 = f1_score(yt, pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        thresholds[i] = best_t
    return thresholds


def evaluate(model: nn.Module, loader: DataLoader) -> dict:
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(DEVICE, non_blocking=True)
            y = batch["label"].cpu().numpy()
            with autocast(device_type="cuda", enabled=USE_FP16):
                logits = model(x)
            p = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(p)
            all_labels.append(y)

    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    thresholds = tune_thresholds(y_true, y_prob)
    y_pred = (y_prob >= thresholds[None, :]).astype(np.int64)

    aucs = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) > 1:
            aucs.append(roc_auc_score(y_true[:, i], y_prob[:, i]))
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_samples": float(
            f1_score(y_true, y_pred, average="samples", zero_division=0)
        ),
        "auc_macro": float(np.mean(aucs)) if aucs else None,
        "thresholds": thresholds.tolist(),
        "y_true": y_true,
        "y_prob": y_prob,
    }


def main():
    train_ds = NIHChestXray14Dataset(
        root=ROOT / "data/raw/nih",
        split="train",
        transform=NIHXrayTransform(split="train"),
        val_frac=0.15,
        seed=42,
    )
    val_ds = NIHChestXray14Dataset(
        root=ROOT / "data/raw/nih",
        split="val",
        transform=NIHXrayTransform(split="val"),
        val_frac=0.15,
        seed=42,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=build_weighted_sampler(train_ds),
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    model = NIHDenseNetHead(num_classes=14, dropout=0.2).to(DEVICE)
    warm_ckpt = CHECKPOINT_DIR / "expert1_nih_enriched_best.pth"
    if warm_ckpt.exists():
        ck = torch.load(warm_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state_dict"], strict=True)
        print(f"Loaded warm checkpoint: {warm_ckpt.name} | best_f1={ck.get('best_f1')}")

    for p in model.backbone.parameters():
        p.requires_grad = False

    criterion = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, clip=0.05)
    scaler = GradScaler("cuda", enabled=USE_FP16)

    head_params = list(model.head.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": 1e-5},
            {"params": head_params, "lr": 3e-4},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    best_f1 = -1.0
    no_improve = 0
    out_ckpt = CHECKPOINT_DIR / "expert1_nih_improved_best.pth"
    out_metrics = METRICS_DIR / "expert1_nih_improved_history.jsonl"

    for epoch in range(1, EPOCHS + 1):
        if epoch == FREEZE_EPOCHS + 1:
            for p in model.backbone.parameters():
                p.requires_grad = True
            print("Unfroze backbone")

        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            x = batch["image"].to(DEVICE, non_blocking=True)
            y = batch["label"].to(DEVICE, non_blocking=True)
            with autocast(device_type="cuda", enabled=USE_FP16):
                logits = model(x)
                loss = criterion(logits, y) / ACCUM_STEPS

            scaler.scale(loss).backward()
            if step % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_loss += loss.item() * ACCUM_STEPS

        scheduler.step()
        val = evaluate(model, val_loader)
        row = {
            "epoch": epoch,
            "train_loss": running_loss / len(train_loader),
            "val_f1_macro": val["f1_macro"],
            "val_f1_samples": val["f1_samples"],
            "val_auc_macro": val["auc_macro"],
            "lr_backbone": optimizer.param_groups[0]["lr"],
            "lr_head": optimizer.param_groups[1]["lr"],
        }
        print(json.dumps(row))
        with out_metrics.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

        if val["f1_macro"] > best_f1 + 1e-4:
            best_f1 = val["f1_macro"]
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_f1": best_f1,
                    "best_auc": val["auc_macro"],
                    "thresholds": val["thresholds"],
                    "expert_name": "expert1_nih_improved",
                    "architecture": "NIHDenseNetHead",
                    "preprocess": "NIHXrayTransform(CLAHE+gamma+unsharp)",
                    "loss": "AsymmetricLoss(gamma_neg=4.0,gamma_pos=1.0,clip=0.05)",
                },
                out_ckpt,
            )
            print(f"NEW_BEST epoch={epoch} f1={best_f1:.4f} auc={val['auc_macro']:.4f}")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"DONE best_f1={best_f1:.4f} ckpt={out_ckpt}")


if __name__ == "__main__":
    main()
