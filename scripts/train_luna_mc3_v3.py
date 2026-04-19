import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models.video import mc3_18, MC3_18_Weights
from sklearn.metrics import f1_score

sys.path.insert(0, "/workspace/moe_medical_vision/src")
from train.train_3d import seed_everything

seed_everything(42)
device = "cuda" if torch.cuda.is_available() else "cpu"


class LunaHighResDataset(Dataset):
    def __init__(self, root, split="train"):
        self.files = sorted(Path(root).glob(f"{split}_*.npz"))
        self.split = split
        self.labels = np.array([int(np.load(f)["label"]) for f in self.files])
        counts = np.bincount(self.labels, minlength=2)
        self.class_weights = counts.sum() / (2 * counts + 1e-6)
        print(
            f"[LUNA16 {split}] {len(self.files)} vols | neg={counts[0]} pos={counts[1]}"
        )

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


FAST_ROOT = "/workspace/moe_medical_vision/data/processed/luna16_highres"
train_ds = LunaHighResDataset(FAST_ROOT, "train")
val_ds = LunaHighResDataset(FAST_ROOT, "val")

train_loader = DataLoader(
    train_ds, batch_size=8, sampler=train_ds.sampler(), num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True
)

print("============================================================")
print("LUNA16 - MC3-18 (FORMULA ISIC) - 40 EPOCHS")
print("============================================================")

model = mc3_18(weights=MC3_18_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = True

model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.fc.in_features, 2))
model = model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=40, eta_min=1e-6
)

best_f1 = -1.0
CKPT_OUT = Path("/workspace/moe_medical_vision/checkpoints/expert4_luna_fixed.pth")

for epoch in range(1, 41):
    model.train()
    tr_loss, tr_p, tr_y = [], [], []
    start_time = time.time()
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
        f"[Epoch {epoch:02d} | {time.time() - start_time:.0f}s] Train Loss: {np.mean(tr_loss):.4f} | Train F1: {f1_score(tr_y, tr_p, average='macro'):.4f} | Val F1: {val_f1:.4f}",
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

print(f"LUNA FINALIZADO. BEST F1 MACRO: {best_f1:.4f}")
