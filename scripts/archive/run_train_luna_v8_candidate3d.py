import sys
from pathlib import Path
import random
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, '/workspace/moe_medical_vision/src')
from data.datasets import get_transform_3d
from models.experts_3d import build_luna_patch_expert_mc3
from train.train_3d import fit_3d_expert, sanity_check_single_batch, seed_everything

seed_everything(42)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT = Path('/workspace/moe_medical_vision/data/raw/luna16')
CKPT = Path('/workspace/moe_medical_vision/checkpoints/expert4_luna16_mc3_candidate_v8_best.pth')
print('device=', DEVICE)

cand = pd.read_csv(ROOT / 'candidates.csv')
pos = cand[cand['class']==1].copy()
neg = cand[cand['class']==0].sample(n=len(pos), random_state=42)
df = pd.concat([pos,neg], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

class LUNACandidate3DDataset(Dataset):
    def __init__(self, df, split='train', val_frac=0.2, seed=42):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(df))
        n_val = int(len(df)*val_frac)
        sel = idx[:n_val] if split=='val' else idx[n_val:]
        self.df = df.iloc[sel].reset_index(drop=True)
        self.transform = get_transform_3d(split=split, augment=(split=='train'))
        mhd_dir = ROOT / 'seg-lungs-LUNA16' / 'seg-lungs-LUNA16'
        self.series_map = {p.stem: p for p in mhd_dir.glob('*.mhd')}
        counts = np.bincount(self.df['class'].values, minlength=2)
        total = counts.sum()
        self.class_weights = torch.tensor(total / (2 * counts + 1e-6), dtype=torch.float32)
        print(f'[LUNA-v8 {split}] {len(self.df)} patches neg={counts[0]} pos={counts[1]}')

    def __len__(self): return len(self.df)

    def _extract_patch(self, vol, center_idx, patch_size=64):
        half = patch_size // 2
        z,y,x = [int(v) for v in center_idx]
        z0,z1=z-half,z+half; y0,y1=y-half,y+half; x0,x1=x-half,x+half
        patch = np.zeros((patch_size,patch_size,patch_size), dtype=np.float32)
        sz0,sz1=max(0,z0),min(vol.shape[0],z1); sy0,sy1=max(0,y0),min(vol.shape[1],y1); sx0,sx1=max(0,x0),min(vol.shape[2],x1)
        dz0,dy0,dx0 = sz0-z0, sy0-y0, sx0-x0
        dz1,dy1,dx1 = dz0+(sz1-sz0), dy0+(sy1-sy0), dx0+(sx1-sx0)
        patch[dz0:dz1,dy0:dy1,dx0:dx1] = vol[sz0:sz1,sy0:sy1,sx0:sx1]
        return patch

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = sitk.ReadImage(str(self.series_map[row['seriesuid']]))
        vol = sitk.GetArrayFromImage(img).astype(np.float32)
        vol = np.clip(vol, -1000.0, 400.0)
        vol = (vol + 1000.0) / 1400.0
        idx_xyz = img.TransformPhysicalPointToIndex((float(row['coordX']), float(row['coordY']), float(row['coordZ'])))
        idx_zyx = (idx_xyz[2], idx_xyz[1], idx_xyz[0])
        patch = self._extract_patch(vol, idx_zyx, patch_size=64)
        x = torch.from_numpy(patch).unsqueeze(0)
        x = self.transform(x)
        y = torch.tensor(int(row['class']), dtype=torch.long)
        return {'image': x, 'label': y}

    def sampler(self):
        w = self.class_weights[self.df['class'].values]
        return WeightedRandomSampler(w.tolist(), len(self.df), replacement=True)

    def get_class_weights(self):
        return self.class_weights

train_ds = LUNACandidate3DDataset(df, 'train')
val_ds = LUNACandidate3DDataset(df, 'val')
train_loader = DataLoader(train_ds, batch_size=2, sampler=train_ds.sampler(), num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=2, pin_memory=True)
model = build_luna_patch_expert_mc3(pretrained=True, use_gradient_checkpointing=True).to(DEVICE)
criterion = torch.nn.CrossEntropyLoss(weight=train_ds.get_class_weights().to(DEVICE), label_smoothing=0.02)
print('sanity:', sanity_check_single_batch(model, train_loader, criterion, DEVICE))
opt = torch.optim.AdamW(model.parameters(), lr=7e-5, weight_decay=3e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=18, eta_min=1e-6)
result = fit_3d_expert(model, train_loader, val_loader, criterion, opt, DEVICE, epochs=18, checkpoint_path=CKPT, scheduler=sched, accum_steps=4, mixed_precision=True, patience=6)
print('DONE', result['best_val_f1'], result['best_epoch'])
