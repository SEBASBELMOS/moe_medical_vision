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


class FocalLossBinary(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-bce)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_loss = (alpha_t * (1 - pt) ** self.gamma * bce).mean()
        return focal_loss


class LunaFast3DDataset(Dataset):
    def __init__(self, root, split="train"):
        self.files = sorted(Path(root).glob(f"{split}_*.npz"))
        self.split = split
        labels = [int(np.load(f)["label"]) for f in self.files]
        self.labels = np.array(labels)
        counts = np.bincount(self.labels, minlength=2)
        self.class_weights = torch.tensor(
            counts.sum() / (2 * counts + 1e-6), dtype=torch.float32
        )
        self.alpha = float(counts[0] / counts.sum())

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        vol = torch.from_numpy(d["volume"][0].copy()).float()
        if self.split == "train":
            if torch.rand(1).item() < 0.5:
                vol = torch.flip(vol, dims=(1,))
            if torch.rand(1).item() < 0.5:
                vol = torch.flip(vol, dims=(2,))
        return {
            "image": vol.unsqueeze(0),
            "label": torch.tensor(int(d["label"]), dtype=torch.long),
        }

    def sampler(self):
        w = self.class_weights[self.labels]
        return WeightedRandomSampler(w.tolist(), len(self.files), replacement=True)


FAST_ROOT = "/workspace/moe_medical_vision/data/processed/luna16_fast"
train_ds = LunaFast3DDataset(FAST_ROOT, "train")
val_ds = LunaFast3DDataset(FAST_ROOT, "val")
train_loader = DataLoader(
    train_ds, batch_size=8, sampler=train_ds.sampler(), num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True
)

model = mc3_18(weights=MC3_18_Weights.DEFAULT)
old_conv = model.stem[0]
new_conv = nn.Conv3d(
    1,
    old_conv.out_channels,
    kernel_size=old_conv.kernel_size,
    stride=old_conv.stride,
    padding=old_conv.padding,
    bias=False,
)
with torch.no_grad():
    new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
model.stem[0] = new_conv

model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, 2))
model = model.to(device)

criterion = FocalLossBinary(alpha=train_ds.alpha, gamma=2.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2
)

best_f1 = -1.0
CKPT_OUT = Path(
    "/workspace/moe_medical_vision/checkpoints/expert4_luna16_mc3_best_final.pth"
)

for epoch in range(1, 41):
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
        f"[LUNA Epoch {epoch:02d}] Train F1: {f1_score(tr_y, tr_p, average='macro'):.4f} | Val F1: {val_f1:.4f}",
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

print(f"LUNA FINALIZADO. BEST F1: {best_f1:.4f}")
