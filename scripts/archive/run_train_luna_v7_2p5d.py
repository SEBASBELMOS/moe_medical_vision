import sys
from pathlib import Path
import random
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import f1_score, accuracy_score

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT = Path('/workspace/moe_medical_vision/data/raw/luna16')
CKPT = Path('/workspace/moe_medical_vision/checkpoints/expert4_luna16_2p5d_v7_best.pth')
LOG_PREFIX = '[LUNA-v7]'
print('device=', DEVICE)

cand = pd.read_csv(ROOT / 'candidates.csv')
pos = cand[cand['class']==1].copy()
neg = cand[cand['class']==0].sample(n=len(pos), random_state=SEED)
df = pd.concat([pos,neg], ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

class LUNASliceStackDataset(Dataset):
    def __init__(self, df, split='train', val_frac=0.2, seed=42):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(df))
        n_val = int(len(df)*val_frac)
        sel = idx[:n_val] if split=='val' else idx[n_val:]
        self.df = df.iloc[sel].reset_index(drop=True)
        self.split = split
        mhd_dir = ROOT / 'seg-lungs-LUNA16' / 'seg-lungs-LUNA16'
        self.series_map = {p.stem: p for p in mhd_dir.glob('*.mhd')}
        counts = np.bincount(self.df['class'].values, minlength=2)
        total = counts.sum()
        self.class_weights = torch.tensor(total / (2 * counts + 1e-6), dtype=torch.float32)
        print(f'{LOG_PREFIX} {split}: {len(self.df)} samples neg={counts[0]} pos={counts[1]}')

    def __len__(self): return len(self.df)

    def _load_volume(self, seriesuid):
        img = sitk.ReadImage(str(self.series_map[seriesuid]))
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        arr = np.clip(arr, -1000.0, 400.0)
        arr = (arr + 1000.0) / 1400.0
        return img, arr

    def _augment(self, x):
        if self.split == 'train':
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=(2,))
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=(1,))
            if torch.rand(1).item() < 0.3:
                x = (x + torch.randn_like(x)*0.02).clamp(0,1)
            if torch.rand(1).item() < 0.3:
                x = (x * (1.0 + float(torch.empty(1).uniform_(-0.08,0.08))) + float(torch.empty(1).uniform_(-0.05,0.05))).clamp(0,1)
        return x

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img, vol = self._load_volume(row['seriesuid'])
        idx_xyz = img.TransformPhysicalPointToIndex((float(row['coordX']), float(row['coordY']), float(row['coordZ'])))
        z = int(idx_xyz[2])
        z_ids = [max(0, min(vol.shape[0]-1, z+d)) for d in (-1,0,1)]
        stack = torch.from_numpy(vol[z_ids]).float()  # [3,H,W]
        stack = F.interpolate(stack.unsqueeze(0), size=(224,224), mode='bilinear', align_corners=False).squeeze(0)
        stack = self._augment(stack)
        stack = (stack - 0.5) / 0.25
        y = torch.tensor(int(row['class']), dtype=torch.long)
        return {'image': stack, 'label': y}

    def sampler(self):
        w = self.class_weights[self.df['class'].values]
        return WeightedRandomSampler(w.tolist(), len(self.df), replacement=True)

train_ds = LUNASliceStackDataset(df, 'train')
val_ds = LUNASliceStackDataset(df, 'val')
train_loader = DataLoader(train_ds, batch_size=16, sampler=train_ds.sampler(), num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, 2))
model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=train_ds.class_weights.to(DEVICE), label_smoothing=0.02)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=12, eta_min=1e-6)

batch = next(iter(train_loader))
print(f'{LOG_PREFIX} sanity', batch['image'].shape)

best = -1.0; best_epoch = 0; history=[]; wait=0
for epoch in range(1, 13):
    model.train(); tr_preds=[]; tr_y=[]; tr_losses=[]
    for batch in train_loader:
        x = batch['image'].to(DEVICE, non_blocking=True)
        y = batch['label'].to(DEVICE, non_blocking=True)
        out = model(x)
        loss = criterion(out,y)
        opt.zero_grad(set_to_none=True)
        loss.backward(); opt.step()
        tr_losses.append(loss.item())
        tr_preds.extend(out.argmax(1).detach().cpu().tolist())
        tr_y.extend(y.cpu().tolist())
    sched.step()

    model.eval(); va_preds=[]; va_y=[]; va_losses=[]
    with torch.no_grad():
        for batch in val_loader:
            x = batch['image'].to(DEVICE, non_blocking=True)
            y = batch['label'].to(DEVICE, non_blocking=True)
            out = model(x)
            loss = criterion(out,y)
            va_losses.append(loss.item())
            va_preds.extend(out.argmax(1).cpu().tolist())
            va_y.extend(y.cpu().tolist())
    row = {
        'epoch': epoch,
        'lr': opt.param_groups[0]['lr'],
        'train_loss': float(np.mean(tr_losses)),
        'train_f1': f1_score(tr_y,tr_preds,average='macro'),
        'train_acc': accuracy_score(tr_y,tr_preds),
        'val_loss': float(np.mean(va_losses)),
        'val_f1': f1_score(va_y,va_preds,average='macro'),
        'val_acc': accuracy_score(va_y,va_preds),
    }
    history.append(row)
    print(f"[Epoch {epoch:02d}] train_loss={row['train_loss']:.4f} train_f1={row['train_f1']:.4f} val_loss={row['val_loss']:.4f} val_f1={row['val_f1']:.4f} lr={row['lr']:.2e}")
    if row['val_f1'] > best:
        best = row['val_f1']; best_epoch = epoch; wait = 0
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'best_val_f1': best, 'history': history}, CKPT)
        print(f'  -> nuevo mejor checkpoint: {CKPT.name} (val_f1={best:.4f})')
    else:
        wait += 1
        if wait >= 5:
            print(f'Early stopping en epoch {epoch}. Mejor epoch: {best_epoch}')
            break
print('DONE', best, best_epoch)
