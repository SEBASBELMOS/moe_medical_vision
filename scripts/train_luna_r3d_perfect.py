import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models.video import r3d_18, R3D_18_Weights
from sklearn.metrics import f1_score

sys.path.insert(0, "/workspace/moe_medical_vision/src")
from train.train_3d import seed_everything

seed_everything(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

print("============================================================")
print("LUNA16 - R3D-18 - 40 EPOCHS (NORMALIZACION KINETICS-400)")
print("============================================================")


class LunaR3DDataset(Dataset):
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
            if torch.rand(1).item() < 0.3:
                vol = vol + torch.randn_like(vol) * 0.05

        vol_3c = vol.unsqueeze(0).repeat(3, 1, 1, 1)
        vol_norm = (vol_3c - self.mean) / self.std

        return {
            "image": vol_norm,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }

    def sampler(self):
        w = self.class_weights[self.labels]
        return WeightedRandomSampler(w.tolist(), len(self.files), replacement=True)


FAST_ROOT = "/workspace/moe_medical_vision/data/processed/luna16_fast"
train_ds = LunaR3DDataset(FAST_ROOT, "train")
val_ds = LunaR3DDataset(FAST_ROOT, "val")
train_loader = DataLoader(
    train_ds, batch_size=8, sampler=train_ds.sampler(), num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True
)

model = r3d_18(weights=R3D_18_Weights.DEFAULT)
model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, 2))
model = model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

backbone_params = [p for n, p in model.named_parameters() if "fc" not in n]
fc_params = [p for n, p in model.named_parameters() if "fc" in n]

optimizer = torch.optim.AdamW(
    [{"params": backbone_params, "lr": 1e-5}, {"params": fc_params, "lr": 1e-3}],
    weight_decay=1e-2,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2
)

best_f1 = -1.0
CKPT_OUT = Path(
    "/workspace/moe_medical_vision/checkpoints/expert4_luna16_r3d18_perfect_best.pth"
)

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
    va_p, va_y = [], []
    with torch.no_grad():
        for batch in val_loader:
            x, y = (
                batch["image"].to(device, non_blocking=True),
                batch["label"].to(device, non_blocking=True),
            )
            out = model(x)
            va_p.extend(out.argmax(1).cpu().tolist())
            va_y.extend(y.cpu().tolist())

    val_f1 = f1_score(va_y, va_p, average="macro")
    print(
        f"[Epoch {epoch:02d} | {time.time() - start_time:.0f}s] Train F1: {f1_score(tr_y, tr_p, average='macro'):.4f} | Val F1: {val_f1:.4f}",
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

print(f"LUNA R3D-18 FINALIZADO. BEST F1 MACRO: {best_f1:.4f}")
