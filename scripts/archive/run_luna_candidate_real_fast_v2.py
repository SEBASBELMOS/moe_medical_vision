import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, '/workspace/moe_medical_vision/src')
from data.datasets import LUNA16Dataset
from models.experts_3d import build_luna_patch_expert_mc3
from train.train_3d import seed_everything

seed_everything(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device={device}')

RAW_ROOT = Path('/workspace/moe_medical_vision/data/raw/luna16')
FAST_ROOT = Path('/workspace/moe_medical_vision/data/processed/luna16_fast')
CKPT_OUT = Path('/workspace/moe_medical_vision/checkpoints/expert4_luna16_candidate_REAL_v2_best.pth')
PATCH_SIZE = 32
TARGET_SIZE = 64
VAL_NEG_POS_RATIO = 8
TRAIN_NEG_POS_RATIO = 1


def build_split_maps():
    out = {}
    for split in ['train', 'val']:
        ds = LUNA16Dataset(RAW_ROOT, split=split, transform=None)
        mapping = {}
        for i, (path, study_label) in enumerate(ds.samples):
            seriesuid = path.stem
            npz_path = FAST_ROOT / f'{split}_{i:05d}.npz'
            mapping[seriesuid] = {
                'npz': npz_path,
                'mhd': path,
                'study_label': int(study_label),
            }
        out[split] = mapping
    return out

SPLIT_MAPS = build_split_maps()
CAND = pd.read_csv(RAW_ROOT / 'candidates.csv')

class LUNACandidateRealDataset(Dataset):
    def __init__(self, split='train', seed=42):
        self.split = split
        split_map = SPLIT_MAPS[split]
        df = CAND[CAND['seriesuid'].isin(split_map.keys())].copy()
        pos = df[df['class'] == 1].copy()
        neg = df[df['class'] == 0].copy()
        ratio = TRAIN_NEG_POS_RATIO if split == 'train' else VAL_NEG_POS_RATIO
        neg_n = min(len(neg), max(len(pos) * ratio, len(pos)))
        neg = neg.sample(n=neg_n, random_state=seed)
        df = pd.concat([pos, neg], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        self.df = df
        self.split_map = split_map
        self.meta = {}
        for seriesuid, info in split_map.items():
            img = sitk.ReadImage(str(info['mhd']))
            self.meta[seriesuid] = {'size_xyz': img.GetSize()}
        counts = np.bincount(self.df['class'].values, minlength=2)
        total = counts.sum()
        self.class_weights = torch.tensor(total / (2 * counts + 1e-6), dtype=torch.float32)
        print(f'[LUNA CAND REAL V2 {split}] {len(self.df)} candidates | neg={counts[0]} pos={counts[1]}')

    def __len__(self):
        return len(self.df)

    def _crop_centered(self, vol, center_zyx, patch_size=PATCH_SIZE):
        z, y, x = center_zyx
        half = patch_size // 2
        z0, z1 = z-half, z+half
        y0, y1 = y-half, y+half
        x0, x1 = x-half, x+half
        out = np.zeros((patch_size, patch_size, patch_size), dtype=np.float32)
        sz0, sz1 = max(0,z0), min(vol.shape[0], z1)
        sy0, sy1 = max(0,y0), min(vol.shape[1], y1)
        sx0, sx1 = max(0,x0), min(vol.shape[2], x1)
        dz0, dy0, dx0 = sz0-z0, sy0-y0, sx0-x0
        dz1, dy1, dx1 = dz0+(sz1-sz0), dy0+(sy1-sy0), dx0+(sx1-sx0)
        out[dz0:dz1, dy0:dy1, dx0:dx1] = vol[sz0:sz1, sy0:sy1, sx0:sx1]
        return out

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seriesuid = row['seriesuid']
        info = self.split_map[seriesuid]
        meta = self.meta[seriesuid]
        d = np.load(info['npz'])
        vol = d['volume'][0]
        size_x, size_y, size_z = meta['size_xyz']
        img = sitk.ReadImage(str(info['mhd']))
        idx_xyz = img.TransformPhysicalPointToIndex((float(row['coordX']), float(row['coordY']), float(row['coordZ'])))
        sx = (TARGET_SIZE - 1) / max(size_x - 1, 1)
        sy = (TARGET_SIZE - 1) / max(size_y - 1, 1)
        sz = (TARGET_SIZE - 1) / max(size_z - 1, 1)
        x = int(round(idx_xyz[0] * sx)); y = int(round(idx_xyz[1] * sy)); z = int(round(idx_xyz[2] * sz))
        x = max(0, min(TARGET_SIZE - 1, x)); y = max(0, min(TARGET_SIZE - 1, y)); z = max(0, min(TARGET_SIZE - 1, z))
        patch = self._crop_centered(vol, (z, y, x), patch_size=PATCH_SIZE)
        x_t = torch.from_numpy(patch.copy()).float().unsqueeze(0)
        x_t = F.interpolate(x_t.unsqueeze(0), size=(TARGET_SIZE, TARGET_SIZE, TARGET_SIZE), mode='trilinear', align_corners=False).squeeze(0)
        if self.split == 'train':
            if torch.rand(1).item() < 0.5:
                x_t = torch.flip(x_t, dims=(2,))
            if torch.rand(1).item() < 0.5:
                x_t = torch.flip(x_t, dims=(3,))
            if torch.rand(1).item() < 0.3:
                x_t = (x_t + torch.randn_like(x_t) * 0.02).clamp(0,1)
        return {
            'image': x_t,
            'label': torch.tensor(int(row['class']), dtype=torch.long),
            'seriesuid': seriesuid,
            'study_label': torch.tensor(int(info['study_label']), dtype=torch.long),
        }

    def sampler(self):
        weights = self.class_weights[self.df['class'].values]
        return WeightedRandomSampler(weights.tolist(), len(self.df), replacement=True)

    def get_class_weights(self):
        return self.class_weights


def eval_study_level(model, loader):
    model.eval()
    by_series = defaultdict(list)
    by_study = {}
    with torch.no_grad():
        for batch in loader:
            x = batch['image'].to(device, non_blocking=True)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
            for i, sid in enumerate(batch['seriesuid']):
                by_series[sid].append(float(probs[i]))
                by_study[sid] = int(batch['study_label'][i].item())
    y_true=[]; y_pred=[]
    for sid, plist in by_series.items():
        y_true.append(by_study[sid])
        y_pred.append(1 if max(plist) >= 0.5 else 0)
    return {
        'study_f1': f1_score(y_true, y_pred, average='macro'),
        'study_acc': accuracy_score(y_true, y_pred),
    }

print('='*80)
print('LUNA CANDIDATE REAL FAST V2 — balanced val, study-level eval')
print('='*80)
train_ds = LUNACandidateRealDataset('train')
val_ds = LUNACandidateRealDataset('val')
train_loader = DataLoader(train_ds, batch_size=8, sampler=train_ds.sampler(), num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

model = build_luna_patch_expert_mc3(pretrained=True, use_gradient_checkpointing=True).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=train_ds.get_class_weights().to(device), label_smoothing=0.02)

sanity = next(iter(train_loader))
print('sanity:', sanity['image'].shape)
optimizer = torch.optim.AdamW(model.parameters(), lr=8e-5, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=18, eta_min=1e-6)

best = -1.0; best_epoch = 0; history=[]; wait=0
for epoch in range(1,19):
    model.train(); tr_loss=[]; tr_p=[]; tr_y=[]
    for batch in train_loader:
        x=batch['image'].to(device, non_blocking=True)
        y=batch['label'].to(device, non_blocking=True)
        out=model(x)
        loss=criterion(out,y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward(); optimizer.step()
        tr_loss.append(loss.item())
        tr_p.extend(out.argmax(1).detach().cpu().tolist())
        tr_y.extend(y.cpu().tolist())
    scheduler.step()
    study_metrics = eval_study_level(model, val_loader)
    row={
        'epoch': epoch,
        'lr': optimizer.param_groups[0]['lr'],
        'train_loss': float(np.mean(tr_loss)),
        'train_f1': f1_score(tr_y,tr_p,average='macro'),
        'val_f1': study_metrics['study_f1'],
        'val_acc': study_metrics['study_acc'],
    }
    history.append(row)
    print(f"[Epoch {epoch:02d}] train_loss={row['train_loss']:.4f} train_f1={row['train_f1']:.4f} study_val_f1={row['val_f1']:.4f} lr={row['lr']:.2e}")
    if row['val_f1'] > best:
        best = row['val_f1']; best_epoch = epoch; wait = 0
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'best_val_f1': best, 'history': history}, CKPT_OUT)
        print(f'  -> nuevo mejor checkpoint: {CKPT_OUT.name} (study_f1={best:.4f})')
    else:
        wait += 1
        if wait >= 8:
            print(f'Early stopping activado en epoch {epoch}. Mejor epoch: {best_epoch}')
            break
print(f"LUNA CANDIDATE REAL FAST V2: best_val_f1={best:.4f} best_epoch={best_epoch}")
print('DONE')
