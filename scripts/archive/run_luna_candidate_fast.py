import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn.functional as F

sys.path.insert(0, '/workspace/moe_medical_vision/src')
from models.experts_3d import build_luna_patch_expert_mc3
from train.train_3d import seed_everything, fit_3d_expert, sanity_check_single_batch

seed_everything(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device={device}')

FAST_ROOT = Path('/workspace/moe_medical_vision/data/processed/luna16_fast')
CKPT_OUT = Path('/workspace/moe_medical_vision/checkpoints/expert4_luna16_candidate_FAST_best.pth')

class LunaCandidateFastDataset(Dataset):
    def __init__(self, root, split='train', patch_size=48):
        self.files = sorted(root.glob(f'{split}_*.npz'))
        self.patch_size = patch_size
        labels=[]
        for f in self.files:
            d=np.load(f)
            labels.append(int(d['label']))
        self.labels=np.array(labels)
        counts=np.bincount(self.labels, minlength=2)
        total=counts.sum()
        self.class_weights=torch.tensor(total/(2*counts+1e-6), dtype=torch.float32)
        self.split=split
        print(f'[LUNA CAND FAST {split}] {len(self.files)} | neg={counts[0]} pos={counts[1]}')

    def __len__(self): return len(self.files)

    def _crop_center(self, vol, size):
        d,h,w = vol.shape
        z0=max(0,(d-size)//2); y0=max(0,(h-size)//2); x0=max(0,(w-size)//2)
        crop = vol[z0:z0+size, y0:y0+size, x0:x0+size]
        out = np.zeros((size,size,size), dtype=np.float32)
        out[:crop.shape[0], :crop.shape[1], :crop.shape[2]] = crop
        return out

    def __getitem__(self, idx):
        d=np.load(self.files[idx])
        vol=d['volume'][0]
        crop=self._crop_center(vol, self.patch_size)
        x=torch.from_numpy(crop.copy()).float().unsqueeze(0)
        x=F.interpolate(x.unsqueeze(0), size=(64,64,64), mode='trilinear', align_corners=False).squeeze(0)
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
print('LUNA CANDIDATE FAST — centered crop + MC3')
print('='*80)
train_ds=LunaCandidateFastDataset(FAST_ROOT,'train',patch_size=48)
val_ds=LunaCandidateFastDataset(FAST_ROOT,'val',patch_size=48)
train_loader=DataLoader(train_ds,batch_size=4,sampler=train_ds.sampler(),num_workers=4,pin_memory=True)
val_loader=DataLoader(val_ds,batch_size=4,shuffle=False,num_workers=4,pin_memory=True)

model=build_luna_patch_expert_mc3(pretrained=True,use_gradient_checkpointing=True).to(device)
criterion=torch.nn.CrossEntropyLoss(weight=train_ds.get_class_weights().to(device), label_smoothing=0.02)
print('sanity:', sanity_check_single_batch(model, train_loader, criterion, device))
print('START_TRAIN_LOOP', flush=True)
optimizer=torch.optim.AdamW(model.parameters(), lr=8e-5, weight_decay=1e-3)
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=18, eta_min=1e-6)
result=fit_3d_expert(model, train_loader, val_loader, criterion, optimizer, device, epochs=18, checkpoint_path=CKPT_OUT, scheduler=scheduler, accum_steps=2, mixed_precision=True, patience=8)
print(f"LUNA CAND FAST: best_val_f1={result['best_val_f1']:.4f} best_epoch={result['best_epoch']}")
print('DONE')
