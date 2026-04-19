import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from sklearn.metrics import f1_score

sys.path.insert(0, "/workspace/moe_medical_vision/src")
from data.datasets import get_dataloader
from train.train_3d import seed_everything

seed_everything(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

print("============================================================")
print("ISIC 2019 - EFFICIENTNET-B3 - 30 EPOCHS (SIN ATAJOS)")
print("============================================================")

ROOT = "/workspace/moe_medical_vision/data/raw/isic"
train_loader_raw, train_ds = get_dataloader(
    "isic2019", ROOT, split="train", batch_size=32, num_workers=0
)
val_loader, val_ds = get_dataloader(
    "isic2019", ROOT, split="val", batch_size=32, num_workers=0
)

labels_train = train_ds.df["label_idx"].values
counts = np.bincount(labels_train, minlength=9)
class_weights = 1.0 / (counts + 1e-6)
sample_weights = class_weights[labels_train]

sampler = WeightedRandomSampler(
    sample_weights.tolist(), len(train_ds), replacement=True
)
train_loader = DataLoader(
    train_ds, batch_size=32, sampler=sampler, num_workers=4, pin_memory=True
)

model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_features, 9))
model = model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=30, eta_min=1e-6
)

best_f1 = -1.0
CKPT_OUT = Path("/workspace/moe_medical_vision/checkpoints/expert2_isic_best_fixed.pth")

for epoch in range(1, 31):
    model.train()
    tr_loss, tr_p, tr_y = [], [], []
    start_time = time.time()

    for i, batch in enumerate(train_loader):
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
    epoch_time = time.time() - start_time

    print(
        f"[Epoch {epoch:02d} | {epoch_time:.0f}s] Train Loss: {np.mean(tr_loss):.4f} | Train F1: {f1_score(tr_y, tr_p, average='macro'):.4f} | Val F1: {val_f1:.4f}",
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

print(f"ISIC FINALIZADO. MEJOR F1: {best_f1:.4f}")
