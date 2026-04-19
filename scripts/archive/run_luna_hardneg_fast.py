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
CHECKPOINT_DIR = Path('/workspace/moe_medical_vision/checkpoints')
CKPT_BASE = CHECKPOINT_DIR / 'expert4_luna16_candidate_REAL_v2_best.pth'
CKPT_OUT = CHECKPOINT_DIR / 'expert4_luna16_hardneg_best.pth'
PATCH_SIZE = 32
TARGET_SIZE = 64
TRAIN_NEG_POS_RATIO = 2
VAL_NEG_POS_RATIO = 8


def build_split_maps():
    out = {}
    for split in ['train', 'val']:
        ds = LUNA16Dataset(RAW_ROOT, split=split, transform=None)
        mapping = {}
        for i, (path, study_label) in enumerate(ds.samples):
            mapping[path.stem] = {
                'npz': FAST_ROOT / f'{split}_{i:05d}.npz',
                'mhd': path,
                'study_label': int(study_label),
            }
        out[split] = mapping
    return out

SPLIT_MAPS = build_split_maps()
CAND = pd.read_csv(RAW_ROOT / 'candidates.csv')

class CandidateDataset(Dataset):
    def __init__(self, df, split='train'):
        self.df = df.reset_index(drop=True)
        self.split = split
        self.split_map = SPLIT_MAPS[split]
        self.meta = {}
        for sid, info in self.split_map.items():
            img = sitk.ReadImage(str(info['mhd']))
            self.meta[sid] = {'size_xyz': img.GetSize()}
        counts = np.bincount(self.df['class'].values, minlength=2)
        total = counts.sum()
        self.class_weights = torch.tensor(total / (2 * counts + 1e-6), dtype=torch.float32)
        print(f'[HARDNEG {split}] {len(self.df)} candidates | neg={counts[0]} pos={counts[1]}')

    def __len__(self): return len(self.df)

    def _crop(self, vol, c, size=PATCH_SIZE):
        z, y, x = c; h = size // 2
        z0, z1 = z - h, z + h; y0, y1 = y - h, y + h; x0, x1 = x - h, x + h
        out = np.zeros((size, size, size), dtype=np.float32)
        sz0, sz1 = max(0, z0), min(vol.shape[0], z1)
        sy0, sy1 = max(0, y0), min(vol.shape[1], y1)
        sx0, sx1 = max(0, x0), min(vol.shape[2], x1)
        dz0, dy0, dx0 = sz0 - z0, sy0 - y0, sx0 - x0
        dz1, dy1, dx1 = dz0 + (sz1 - sz0), dy0 + (sy1 - sy0), dx0 + (sx1 - sx0)
        out[dz0:dz1, dy0:dy1, dx0:dx1] = vol[sz0:sz1, sy0:sy1, sx0:sx1]
        return out

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sid = row['seriesuid']
        info = self.split_map[sid]
        meta = self.meta[sid]
        d = np.load(info['npz'])
        vol = d['volume'][0]
        img = sitk.ReadImage(str(info['mhd']))
        idx_xyz = img.TransformPhysicalPointToIndex((float(row['coordX']), float(row['coordY']), float(row['coordZ'])))
        sx = 63 / max(meta['size_xyz'][0] - 1, 1)
        sy = 63 / max(meta['size_xyz'][1] - 1, 1)
        sz = 63 / max(meta['size_xyz'][2] - 1, 1)
        x = int(round(idx_xyz[0] * sx)); y = int(round(idx_xyz[1] * sy)); z = int(round(idx_xyz[2] * sz))
        x = max(0, min(63, x)); y = max(0, min(63, y)); z = max(0, min(63, z))
        patch = self._crop(vol, (z, y, x))
        x_t = torch.from_numpy(patch.copy()).float().unsqueeze(0)
        x_t = F.interpolate(x_t.unsqueeze(0), size=(TARGET_SIZE, TARGET_SIZE, TARGET_SIZE), mode='trilinear', align_corners=False).squeeze(0)
        if self.split == 'train':
            if torch.rand(1).item() < 0.5: x_t = torch.flip(x_t, dims=(2,))
            if torch.rand(1).item() < 0.5: x_t = torch.flip(x_t, dims=(3,))
            if torch.rand(1).item() < 0.3: x_t = (x_t + torch.randn_like(x_t) * 0.02).clamp(0,1)
        return {'image': x_t, 'label': torch.tensor(int(row['class']), dtype=torch.long), 'seriesuid': sid, 'study_label': torch.tensor(int(info['study_label']), dtype=torch.long)}

    def sampler(self):
        weights = self.class_weights[self.df['class'].values]
        return WeightedRandomSampler(weights.tolist(), len(self.df), replacement=True)

    def get_class_weights(self): return self.class_weights


def make_balanced_candidate_df(split='train', neg_pos_ratio=1, seed=42):
    split_map = SPLIT_MAPS[split]
    df = CAND[CAND['seriesuid'].isin(split_map.keys())].copy()
    pos = df[df['class'] == 1].copy()
    neg = df[df['class'] == 0].copy()
    neg_n = min(len(neg), len(pos) * neg_pos_ratio)
    neg = neg.sample(n=neg_n, random_state=seed)
    return pd.concat([pos, neg], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def make_val_df(seed=42):
    split_map = SPLIT_MAPS['val']
    df = CAND[CAND['seriesuid'].isin(split_map.keys())].copy()
    pos = df[df['class'] == 1].copy()
    neg = df[df['class'] == 0].copy()
    neg_n = min(len(neg), len(pos) * VAL_NEG_POS_RATIO)
    neg = neg.sample(n=neg_n, random_state=seed)
    return pd.concat([pos, neg], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def eval_study(model, loader):
    model.eval()
    by_series = defaultdict(list)
    by_label = {}
    with torch.no_grad():
        for batch in loader:
            probs = torch.softmax(model(batch['image'].to(device, non_blocking=True)), dim=1)[:,1].cpu().numpy()
            for i, sid in enumerate(batch['seriesuid']):
                by_series[sid].append(float(probs[i]))
                by_label[sid] = int(batch['study_label'][i].item())
    yt, yp = [], []
    for sid, vals in by_series.items():
        yt.append(by_label[sid])
        yp.append(1 if max(vals) >= 0.5 else 0)
    return f1_score(yt, yp, average='macro'), accuracy_score(yt, yp)


def collect_hard_negative_scores(model, df_train):
    neg_df = df_train[df_train['class'] == 0].copy().reset_index(drop=True)
    ds = CandidateDataset(neg_df, split='train')
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    scores = []
    model.eval()
    with torch.no_grad():
        idx = 0
        for batch in loader:
            probs = torch.softmax(model(batch['image'].to(device, non_blocking=True)), dim=1)[:,1].cpu().numpy()
            for p in probs:
                scores.append(float(p))
                idx += 1
    neg_df['hard_score'] = scores
    return neg_df.sort_values('hard_score', ascending=False).reset_index(drop=True)

print('='*80)
print('LUNA HARD-NEG FAST')
print('='*80)
base_train_df = make_balanced_candidate_df('train', neg_pos_ratio=1)
val_df = make_val_df()
base_train_ds = CandidateDataset(base_train_df, split='train')
val_ds = CandidateDataset(val_df, split='val')
base_train_loader = DataLoader(base_train_ds, batch_size=8, sampler=base_train_ds.sampler(), num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

model = build_luna_patch_expert_mc3(pretrained=True, use_gradient_checkpointing=True).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=base_train_ds.get_class_weights().to(device), label_smoothing=0.02)
print('sanity:', next(iter(base_train_loader))['image'].shape)
optimizer = torch.optim.AdamW(model.parameters(), lr=8e-5, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12, eta_min=1e-6)

# Phase 1: warmup on balanced candidates
best = -1.0; best_epoch = 0; history=[]; wait=0
for epoch in range(1,7):
    model.train(); tr_loss=[]; tr_p=[]; tr_y=[]
    for batch in base_train_loader:
        x=batch['image'].to(device, non_blocking=True); y=batch['label'].to(device, non_blocking=True)
        out=model(x); loss=criterion(out,y)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        tr_loss.append(loss.item()); tr_p.extend(out.argmax(1).detach().cpu().tolist()); tr_y.extend(y.cpu().tolist())
    scheduler.step()
    vf1,vacc = eval_study(model, val_loader)
    row={'epoch':epoch,'lr':optimizer.param_groups[0]['lr'],'train_loss':float(np.mean(tr_loss)),'train_f1':f1_score(tr_y,tr_p,average='macro'),'val_f1':vf1,'val_acc':vacc}
    history.append(row)
    print(f"[Warmup {epoch:02d}] train_loss={row['train_loss']:.4f} train_f1={row['train_f1']:.4f} study_val_f1={row['val_f1']:.4f}")
    if vf1>best:
        best=vf1; best_epoch=epoch; wait=0
        torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),'best_val_f1':best,'history':history}, CKPT_OUT)
        print(f'  -> nuevo mejor checkpoint: {CKPT_OUT.name} (study_f1={best:.4f})')
    else:
        wait+=1

# Build hard negatives from current model
full_train_df = CAND[CAND['seriesuid'].isin(SPLIT_MAPS['train'].keys())].copy()
pos_df = full_train_df[full_train_df['class']==1].copy().reset_index(drop=True)
hard_neg_df = collect_hard_negative_scores(model, full_train_df)
hard_neg_df = hard_neg_df.head(min(len(hard_neg_df), len(pos_df)*TRAIN_NEG_POS_RATIO)).copy()
train_hard_df = pd.concat([pos_df, hard_neg_df], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
print(f'HARDNEG dataset: pos={len(pos_df)} neg={len(hard_neg_df)}')
train_hard_ds = CandidateDataset(train_hard_df, split='train')
train_hard_loader = DataLoader(train_hard_ds, batch_size=8, sampler=train_hard_ds.sampler(), num_workers=4, pin_memory=True)
criterion = torch.nn.CrossEntropyLoss(weight=train_hard_ds.get_class_weights().to(device), label_smoothing=0.02)
optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12, eta_min=1e-6)
wait=0
for epoch in range(7,19):
    model.train(); tr_loss=[]; tr_p=[]; tr_y=[]
    for batch in train_hard_loader:
        x=batch['image'].to(device, non_blocking=True); y=batch['label'].to(device, non_blocking=True)
        out=model(x); loss=criterion(out,y)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        tr_loss.append(loss.item()); tr_p.extend(out.argmax(1).detach().cpu().tolist()); tr_y.extend(y.cpu().tolist())
    scheduler.step()
    vf1,vacc = eval_study(model, val_loader)
    row={'epoch':epoch,'lr':optimizer.param_groups[0]['lr'],'train_loss':float(np.mean(tr_loss)),'train_f1':f1_score(tr_y,tr_p,average='macro'),'val_f1':vf1,'val_acc':vacc}
    history.append(row)
    print(f"[HardNeg {epoch:02d}] train_loss={row['train_loss']:.4f} train_f1={row['train_f1']:.4f} study_val_f1={row['val_f1']:.4f}")
    if vf1>best:
        best=vf1; best_epoch=epoch; wait=0
        torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),'best_val_f1':best,'history':history}, CKPT_OUT)
        print(f'  -> nuevo mejor checkpoint: {CKPT_OUT.name} (study_f1={best:.4f})')
    else:
        wait+=1
        if wait>=8:
            print(f'Early stopping activado en epoch {epoch}. Mejor epoch: {best_epoch}')
            break
print(f"LUNA HARDNEG FAST: best_val_f1={best:.4f} best_epoch={best_epoch}")
print('DONE')
