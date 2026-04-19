from __future__ import annotations

import argparse
import json
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
USE_FP16 = torch.cuda.is_available()


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
        if self.clip and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        pt = xs_pos * targets + xs_neg * (1 - targets)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
        one_sided_w = torch.pow(1 - pt, gamma)
        return -(loss * one_sided_w).mean()


class BCEPosLoss(nn.Module):
    def __init__(self, pos_weight: torch.Tensor):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss(logits, targets)


class NIHDenseNetHead(nn.Module):
    def __init__(
        self, num_classes: int = 14, dropout: float = 0.2, pretrained: bool = True
    ):
        super().__init__()
        self.backbone = timm.create_model(
            "densenet121", pretrained=pretrained, num_classes=0
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(self.backbone.num_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def build_weighted_sampler(
    dataset: NIHChestXray14Dataset, cap_percentile: float = 95.0
) -> WeightedRandomSampler:
    labels = np.array(dataset.df["labels_list"].tolist(), dtype=np.float32)
    freq = labels.mean(axis=0)
    inv = 1.0 / np.clip(freq, 1e-4, None)
    sample_weights = (labels * inv[None, :]).sum(axis=1)
    fallback = (
        np.median(sample_weights[sample_weights > 0])
        if np.any(sample_weights > 0)
        else 1.0
    )
    sample_weights = np.where(sample_weights > 0, sample_weights, fallback)
    sample_weights = np.clip(
        sample_weights, None, np.percentile(sample_weights, cap_percentile)
    )
    return WeightedRandomSampler(
        sample_weights.tolist(), num_samples=len(sample_weights), replacement=True
    )


def tune_thresholds(
    y_true: np.ndarray, y_prob: np.ndarray, start: float, end: float, step: float
) -> np.ndarray:
    grid = np.arange(start, end + 1e-9, step)
    thresholds = np.full(y_true.shape[1], 0.5, dtype=np.float32)
    for i in range(y_true.shape[1]):
        best_t, best_f1 = 0.5, -1.0
        yt, yp = y_true[:, i], y_prob[:, i]
        for t in grid:
            pred = (yp >= t).astype(np.int64)
            f1 = f1_score(yt, pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thresholds[i] = best_t
    return thresholds


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    thr_start: float,
    thr_end: float,
    thr_step: float,
) -> dict:
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
    thresholds = tune_thresholds(y_true, y_prob, thr_start, thr_end, thr_step)
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
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--epochs", type=int, default=18)
    p.add_argument("--freeze-epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--accum-steps", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr-backbone", type=float, default=1e-5)
    p.add_argument("--lr-head", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--loss", choices=["asl", "bce"], default="asl")
    p.add_argument("--gamma-neg", type=float, default=4.0)
    p.add_argument("--gamma-pos", type=float, default=1.0)
    p.add_argument("--clip", type=float, default=0.05)
    p.add_argument("--gamma-min", type=float, default=0.7)
    p.add_argument("--gamma-max", type=float, default=1.5)
    p.add_argument("--clahe-clip", type=float, default=2.0)
    p.add_argument("--thr-start", type=float, default=0.10)
    p.add_argument("--thr-end", type=float, default=0.90)
    p.add_argument("--thr-step", type=float, default=0.05)
    p.add_argument("--warm-ckpt", default="expert1_nih_enriched_best.pth")
    p.add_argument("--no-sampler", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_ds = NIHChestXray14Dataset(
        root=ROOT / "data/raw/nih",
        split="train",
        transform=NIHXrayTransform(
            split="train",
            clahe_clip=args.clahe_clip,
            gamma_range=(args.gamma_min, args.gamma_max),
        ),
        val_frac=0.15,
        seed=args.seed,
    )
    val_ds = NIHChestXray14Dataset(
        root=ROOT / "data/raw/nih",
        split="val",
        transform=NIHXrayTransform(
            split="val", clahe_clip=args.clahe_clip, gamma_range=(1.0, 1.0)
        ),
        val_frac=0.15,
        seed=args.seed,
    )

    sampler = None if args.no_sampler else build_weighted_sampler(train_ds)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    model = NIHDenseNetHead(num_classes=14, dropout=args.dropout, pretrained=True).to(
        DEVICE
    )
    warm_ckpt = CHECKPOINT_DIR / args.warm_ckpt
    if warm_ckpt.exists():
        ck = torch.load(warm_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state_dict"], strict=False)
        print(f"Loaded warm checkpoint: {warm_ckpt.name} | best_f1={ck.get('best_f1')}")

    for p in model.backbone.parameters():
        p.requires_grad = False

    if args.loss == "asl":
        criterion = AsymmetricLoss(
            gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos, clip=args.clip
        )
        loss_name = f"ASL(gn={args.gamma_neg},gp={args.gamma_pos},clip={args.clip})"
    else:
        criterion = BCEPosLoss(train_ds.get_pos_weight().to(DEVICE))
        loss_name = "BCEWithLogits(pos_weight)"

    scaler = GradScaler("cuda", enabled=USE_FP16)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.lr_backbone},
            {"params": model.head.parameters(), "lr": args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    best_f1 = -1.0
    no_improve = 0
    out_ckpt = CHECKPOINT_DIR / f"{args.tag}.pth"
    out_metrics = METRICS_DIR / f"{args.tag}.jsonl"

    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1:
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
                loss = criterion(logits, y) / args.accum_steps
            scaler.scale(loss).backward()
            if step % args.accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_loss += loss.item() * args.accum_steps

        scheduler.step()
        val = evaluate(model, val_loader, args.thr_start, args.thr_end, args.thr_step)
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(len(train_loader), 1),
            "val_f1_macro": val["f1_macro"],
            "val_f1_samples": val["f1_samples"],
            "val_auc_macro": val["auc_macro"],
            "lr_backbone": optimizer.param_groups[0]["lr"],
            "lr_head": optimizer.param_groups[1]["lr"],
            "tag": args.tag,
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
                    "expert_name": args.tag,
                    "architecture": "NIHDenseNetHead",
                    "preprocess": f"NIHXrayTransform(CLAHE={args.clahe_clip},gamma=({args.gamma_min},{args.gamma_max}))",
                    "loss": loss_name,
                    "training": vars(args),
                },
                out_ckpt,
            )
            print(f"NEW_BEST epoch={epoch} f1={best_f1:.4f} auc={val['auc_macro']:.4f}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"DONE tag={args.tag} best_f1={best_f1:.4f} ckpt={out_ckpt}")


if __name__ == "__main__":
    main()
