from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models.video import mc3_18, MC3_18_Weights

ROOT = Path("/workspace/moe_medical_vision")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR = CHECKPOINT_DIR / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


class LunaHighResDataset(Dataset):
    def __init__(self, root: str, split: str = "train"):
        self.files = sorted(Path(root).glob(f"{split}_*.npz"))
        self.split = split
        self.labels = np.array(
            [int(np.load(f)["label"]) for f in self.files], dtype=np.int64
        )
        counts = np.bincount(self.labels, minlength=2)
        self.class_weights = counts.sum() / (2 * counts + 1e-6)
        self.mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
        self.std = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        vol = torch.from_numpy(d["volume"][0].copy()).float()
        if self.split == "train":
            if torch.rand(1).item() < 0.5:
                vol = torch.flip(vol, dims=(0,))
            if torch.rand(1).item() < 0.5:
                vol = torch.flip(vol, dims=(1,))
            if torch.rand(1).item() < 0.5:
                vol = torch.flip(vol, dims=(2,))
            if torch.rand(1).item() < 0.5:
                vol = torch.rot90(vol, k=1, dims=(1, 2))
        vol_3c = vol.unsqueeze(0).repeat(3, 1, 1, 1)
        vol_norm = (vol_3c - self.mean) / self.std
        return {
            "image": vol_norm,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }

    def sampler(self):
        w = self.class_weights[self.labels]
        return WeightedRandomSampler(w.tolist(), len(self.files), replacement=True)


def eval_binary(y_true, y_pred, y_prob_pos):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / max(tn + fp, 1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob_pos))
        if len(np.unique(y_true)) > 1
        else None,
        "confusion_matrix": cm.tolist(),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--warm-ckpt", default="")
    return p.parse_args()


def main():
    args = parse_args()
    fast_root = "/workspace/moe_medical_vision/data/processed/luna16_highres"
    train_ds = LunaHighResDataset(fast_root, "train")
    val_ds = LunaHighResDataset(fast_root, "val")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_ds.sampler(),
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = mc3_18(weights=MC3_18_Weights.DEFAULT)
    for p in model.parameters():
        p.requires_grad = True
    model.fc = nn.Sequential(
        nn.Dropout(args.dropout), nn.Linear(model.fc.in_features, 2)
    )
    model = model.to(DEVICE)

    if args.warm_ckpt:
        warm = CHECKPOINT_DIR / args.warm_ckpt
        if warm.exists():
            ck = torch.load(warm, map_location="cpu", weights_only=False)
            model.load_state_dict(ck["model_state_dict"], strict=False)
            print(f"Loaded warm checkpoint: {warm.name}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    best_f1 = -1.0
    out_ckpt = CHECKPOINT_DIR / f"{args.tag}.pth"
    out_metrics = METRICS_DIR / f"{args.tag}.jsonl"

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss, tr_p, tr_y = [], [], []
        start = time.time()
        for batch in train_loader:
            x = batch["image"].to(DEVICE, non_blocking=True)
            y = batch["label"].to(DEVICE, non_blocking=True)
            out = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tr_loss.append(loss.item())
            tr_p.extend(out.argmax(1).detach().cpu().tolist())
            tr_y.extend(y.cpu().tolist())
        scheduler.step()

        model.eval()
        va_p, va_y, va_prob = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(DEVICE, non_blocking=True)
                y = batch["label"].to(DEVICE, non_blocking=True)
                out = model(x)
                p = torch.softmax(out, dim=1).cpu().numpy()
                va_prob.extend(p[:, 1].tolist())
                va_p.extend(out.argmax(1).cpu().tolist())
                va_y.extend(y.cpu().tolist())

        val = eval_binary(np.array(va_y), np.array(va_p), np.array(va_prob))
        row = {
            "epoch": epoch,
            "seconds": round(time.time() - start, 2),
            "train_loss": float(np.mean(tr_loss)),
            "train_f1": float(f1_score(tr_y, tr_p, average="macro", zero_division=0)),
            "val_f1": val["f1_macro"],
            "val_auc": val["auc"],
            "val_acc": val["accuracy"],
            "tag": args.tag,
        }
        print(json.dumps(row))
        with out_metrics.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        if val["f1_macro"] > best_f1:
            best_f1 = val["f1_macro"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_f1": best_f1,
                    "best_auc": val["auc"],
                    "history_last": row,
                    "training": vars(args),
                },
                out_ckpt,
            )
            print(f"NEW_BEST epoch={epoch} f1={best_f1:.4f} auc={val['auc']:.4f}")
    print(f"DONE tag={args.tag} best_f1={best_f1:.4f} ckpt={out_ckpt}")


if __name__ == "__main__":
    main()
