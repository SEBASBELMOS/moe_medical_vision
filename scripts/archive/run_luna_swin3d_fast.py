import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models.video import swin3d_t, Swin3D_T_Weights

sys.path.insert(0, '/workspace/moe_medical_vision/src')
from train.train_3d import seed_everything, fit_3d_expert, sanity_check_single_batch

seed_everything(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device={device}')
FAST_ROOT = Path('/workspace/moe_medical_vision/data/processed/luna16_fast')
CKPT_OUT = Path('/workspace/moe_medical_vision/checkpoints/expert4_luna16_swin3d_fast_best.pth')

class LunaFast3Channel(Dataset):
    def __init__(self, root, split='train'):
        self.files = sorted(root.glob(f'{split}_*.npz'))
        labels=[]
        for f in self.files:
            d=np.load(f)
            labels.append(int(d['label']))
        self.labels=np.array(labels)
        counts=np.bincount(self.labels, minlength=2)
        total=counts.sum()
        self.class_weights=torch.tensor(total/(2*counts+1e-6), dtype=torch.float32)
        self.split=split
        print(f'[LUNA Swin3D FAST {split}] {len(self.files)} | neg={counts[0]} pos={counts[1]}')
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        d=np.load(self.files[idx])
        x=torch.from_numpy(d['volume'].copy()).float()  # [1,D,H,W]
        x=x.repeat(3,1,1,1)  # [3,D,H,W]
        if self.split=='train':
            if torch.rand(1).item()<0.5:
                x=torch.flip(x,dims=(2,))
            if torch.rand(1).item()<0.5:
                x=torch.flip(x,dims=(3,))
            if torch.rand(1).item()<0.3:
                x=(x+torch.randn_like(x)*0.02).clamp(0,1)
        y=torch.tensor(int(d['label']), dtype=torch.long)
        return {'image':x,'label':y}
    def sampler(self):
        w=self.class_weights[self.labels]
        return WeightedRandomSampler(w.tolist(), len(self.files), replacement=True)
    def get_class_weights(self): return self.class_weights

print('='*80)
print('LUNA SWIN3D FAST')
print('='*80)
train_ds=LunaFast3Channel(FAST_ROOT,'train')
val_ds=LunaFast3Channel(FAST_ROOT,'val')
train_loader=DataLoader(train_ds,batch_size=2,sampler=train_ds.sampler(),num_workers=4,pin_memory=True)
val_loader=DataLoader(val_ds,batch_size=2,shuffle=False,num_workers=4,pin_memory=True)

model=swin3d_t(weights=Swin3D_T_Weights.DEFAULT)
in_features=model.head.in_features
model.head=nn.Linear(in_features,2)
model=model.to(device)
criterion=nn.CrossEntropyLoss(weight=train_ds.get_class_weights().to(device), label_smoothing=0.02)
print('sanity:', sanity_check_single_batch(model, train_loader, criterion, device))
print('START_TRAIN_LOOP', flush=True)
opt=torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20, eta_min=1e-6)
result=fit_3d_expert(model, train_loader, val_loader, criterion, opt, device, epochs=20, checkpoint_path=CKPT_OUT, scheduler=sched, accum_steps=2, mixed_precision=True, patience=10)
print(f"LUNA SWIN3D FAST: best_val_f1={result['best_val_f1']:.4f} best_epoch={result['best_epoch']}")
print('DONE')
