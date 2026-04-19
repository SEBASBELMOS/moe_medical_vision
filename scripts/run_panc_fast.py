import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

sys.path.insert(0, '/workspace/moe_medical_vision/src')
from losses import FocalLoss
from models.experts_3d import build_pancreatic_expert
from train.train_3d import seed_everything, fit_3d_expert, sanity_check_single_batch

seed_everything(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device={device}')

FAST_ROOT = Path('/workspace/moe_medical_vision/data/processed/pancreatic_fast')
CHECKPOINT_DIR = Path('/workspace/moe_medical_vision/checkpoints')
CKPT_OUT = CHECKPOINT_DIR / 'expert5_pancreatic_FAST_best.pth'

class PancFastDataset(Dataset):
    def __init__(self, root, split='train'):
        self.files = sorted(root.glob(f'{split}_*.npz'))
        labels=[]
        for f in self.files:
            d=np.load(f)
            labels.append(int(d['label']))
        self.labels=np.array(labels)
        counts=np.bincount(self.labels, minlength=2)
        total=counts.sum()
        self.class_counts=counts
        self.class_weights=torch.tensor(total/(2*counts+1e-6), dtype=torch.float32)
        self.focal_alpha=float(counts[0]/total) if total>0 else 0.5
        print(f'[PANC FAST {split}] {len(self.files)} | neg={counts[0]} pos={counts[1]} alpha={self.focal_alpha:.4f}')
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        d=np.load(self.files[idx])
        x=torch.from_numpy(d['volume'].copy()).float()
        y=torch.tensor(int(d['label']), dtype=torch.long)
        return {'image':x,'label':y}
    def sampler(self):
        w=self.class_weights[self.labels]
        return WeightedRandomSampler(w.tolist(), len(self.files), replacement=True)
    def get_focal_alpha(self): return self.focal_alpha

print('='*80)
print('PANCREATIC FAST — fine-tune from best v5 on precomputed npz')
print('='*80)
train_ds=PancFastDataset(FAST_ROOT,'train')
val_ds=PancFastDataset(FAST_ROOT,'val')
train_loader=DataLoader(train_ds,batch_size=4,sampler=train_ds.sampler(),num_workers=4,pin_memory=True)
val_loader=DataLoader(val_ds,batch_size=4,shuffle=False,num_workers=4,pin_memory=True)

model=build_pancreatic_expert(pretrained=True,use_gradient_checkpointing=True).to(device)
v5=torch.load(CHECKPOINT_DIR/'expert5_pancreatic_r3d18_v5_best.pth', map_location=device, weights_only=False)
model.load_state_dict(v5['model_state_dict'])
print(f"Loaded v5: val_f1={v5['best_val_f1']:.4f}")

criterion=FocalLoss(gamma=2.0, alpha=train_ds.get_focal_alpha())
print('sanity:', sanity_check_single_batch(model, train_loader, criterion, device))
print('START_TRAIN_LOOP', flush=True)

optimizer=torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-3)
scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=8, T_mult=2, eta_min=1e-7)
result=fit_3d_expert(model, train_loader, val_loader, criterion, optimizer, device, epochs=20, checkpoint_path=CKPT_OUT, scheduler=scheduler, accum_steps=2, mixed_precision=True, patience=10)
print(f"PANC FAST: best_val_f1={result['best_val_f1']:.4f} best_epoch={result['best_epoch']}")
print('DONE')
