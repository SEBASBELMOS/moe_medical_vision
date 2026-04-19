import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from sklearn.metrics import f1_score

sys.path.insert(0, "/workspace/moe_medical_vision/src")
from data.datasets import get_dataloader
from train.train_3d import seed_everything

seed_everything(42)
device = "cuda" if torch.cuda.is_available() else "cpu"


class FocalLossMultiClass(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


ROOT = "/workspace/moe_medical_vision/data/raw/isic"
train_loader_raw, train_ds = get_dataloader(
    "isic2019", ROOT, split="train", batch_size=64, num_workers=0
)
val_loader, val_ds = get_dataloader(
    "isic2019", ROOT, split="val", batch_size=64, num_workers=0
)

labels_train = train_ds.df["label_idx"].values
counts = np.bincount(labels_train, minlength=9)
total = counts.sum()

safe_counts = np.maximum(counts, 1)
raw_weights = total / (9.0 * safe_counts)
clamped_weights = np.clip(raw_weights, 0.1, 20.0)
clamped_weights[counts == 0] = 0.0

sample_weights = clamped_weights[labels_train]
sampler = WeightedRandomSampler(
    sample_weights.tolist(), len(train_ds), replacement=True
)
train_loader = DataLoader(
    train_ds, batch_size=64, sampler=sampler, num_workers=4, pin_memory=True
)

model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features, 9))
model = model.to(device)

criterion = FocalLossMultiClass(
    alpha=torch.tensor(clamped_weights, dtype=torch.float32).to(device), gamma=2.0
)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2
)

best_f1 = -1.0
CKPT_OUT = Path("/workspace/moe_medical_vision/checkpoints/expert2_isic_best_final.pth")

for epoch in range(1, 31):
    model.train()
    tr_loss, tr_p, tr_y = [], [], []
    for batch in train_loader:
        x, y = (
            batch["image"].to(device, non_blocking=True),
            batch["label"].to(device, non_blocking=True),
        )
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
    va_loss, va_p, va_y = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            x, y = (
                batch["image"].to(device, non_blocking=True),
                batch["label"].to(device, non_blocking=True),
            )
            out = model(x)
            va_loss.append(criterion(out, y).item())
            va_p.extend(out.argmax(1).cpu().tolist())
            va_y.extend(y.cpu().tolist())

    val_f1 = f1_score(va_y, va_p, average="macro")
    print(
        f"[ISIC Epoch {epoch:02d}] Train F1: {f1_score(tr_y, tr_p, average='macro'):.4f} | Val F1: {val_f1:.4f}",
        flush=True,
    )

    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "best_val_f1": best_f1,
                "epoch": epoch,
            },
            CKPT_OUT,
        )

print(f"ISIC FINALIZADO. BEST F1: {best_f1:.4f}")
