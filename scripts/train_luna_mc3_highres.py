import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models.video import mc3_18, MC3_18_Weights
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, "/workspace/moe_medical_vision/src")
from train.train_3d import seed_everything

seed_everything(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

print("============================================================")
print("ENTRENAMIENTO LUNA16 SOTA: MC3-18 (HIGH-RES CROPS & KINETICS)")
print("============================================================")


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


class LunaHighResDataset(Dataset):
    def __init__(self, root, split="train"):
        self.files = sorted(Path(root).glob(f"{split}_*.npz"))
        self.split = split
        self.labels = np.array([int(np.load(f)["label"]) for f in self.files])
        counts = np.bincount(self.labels, minlength=2)

        # Pesos para Focal Loss y Sampler
        self.class_weights = counts.sum() / (2 * counts + 1e-6)
        self.alpha = float(counts[0] / counts.sum())
        print(
            f"[LUNA16 {split}] {len(self.files)} patches | neg={counts[0]} pos={counts[1]}"
        )

        # Normalización estricta de Kinetics-400 para modelos de Video de PyTorch
        self.mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
        self.std = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        vol = torch.from_numpy(d["volume"][0].copy()).float()  # [64, 64, 64]

        if self.split == "train":
            # Data Augmentation 3D Severo para evitar Overfitting en Kinetics
            if torch.rand(1).item() < 0.5:
                vol = torch.flip(vol, dims=(0,))  # Profundidad
            if torch.rand(1).item() < 0.5:
                vol = torch.flip(vol, dims=(1,))  # Altura
            if torch.rand(1).item() < 0.5:
                vol = torch.flip(vol, dims=(2,))  # Ancho
            if torch.rand(1).item() < 0.5:
                vol = torch.rot90(vol, k=1, dims=(1, 2))  # Rotacion H/W
            if torch.rand(1).item() < 0.3:
                vol = vol + torch.randn_like(vol) * 0.05  # Ruido Gaussiano

        # Replicamos el canal Gris a 3 Canales (Video RGB de Kinetics-400)
        vol_3c = vol.unsqueeze(0).repeat(3, 1, 1, 1)  # [3, 64, 64, 64]

        # LA MAGIA DEL TRANSFER LEARNING: Aplicamos Normalización de Kinetics
        vol_norm = (vol_3c - self.mean) / self.std

        return {
            "image": vol_norm,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }

    def sampler(self):
        w = self.class_weights[self.labels]
        return WeightedRandomSampler(w.tolist(), len(self.files), replacement=True)


HIGHRES_ROOT = "/workspace/moe_medical_vision/data/processed/luna16_highres"
train_ds = LunaHighResDataset(HIGHRES_ROOT, "train")
val_ds = LunaHighResDataset(HIGHRES_ROOT, "val")

# Usamos batch size 16 porque ahora son recortes de 64^3, caben perfecto en VRAM
train_loader = DataLoader(
    train_ds, batch_size=16, sampler=train_ds.sampler(), num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True
)

print("Instanciando MC3-18 con Pesos Originales (3 Canales)...")
model = mc3_18(weights=MC3_18_Weights.DEFAULT)

# Freezing estricto de Stem y Capas 1/2. Dejamos Capas 3, 4 y FC libres.
for param in model.stem.parameters():
    param.requires_grad = False
for param in model.layer1.parameters():
    param.requires_grad = False
for param in model.layer2.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, 2))
model = model.to(device)

criterion = FocalLossBinary(alpha=train_ds.alpha, gamma=2.0)

# Optimizador con Weight Decay
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2
)

best_f1 = -1.0
CKPT_OUT = Path(
    "/workspace/moe_medical_vision/checkpoints/expert4_luna16_mc3_highres_best.pth"
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
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
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
    val_acc = accuracy_score(va_y, va_p)
    epoch_time = time.time() - start_time

    print(
        f"[Epoch {epoch:02d} | {epoch_time:.0f}s] Train Loss: {np.mean(tr_loss):.4f} | Train F1: {f1_score(tr_y, tr_p, average='macro'):.4f} | Val Loss: {np.mean(va_loss):.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc * 100:.2f}%",
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

print(f"ENTRENAMIENTO LUNA HIGH-RES FINALIZADO. BEST F1 MACRO: {best_f1:.4f}")
