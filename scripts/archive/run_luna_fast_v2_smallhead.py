import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

sys.path.insert(0, '/workspace/moe_medical_vision/src')
from models.experts_3d import build_luna_expert
from train.train_3d import seed_everything, fit_3d_expert, sanity_check_single_batch

seed_everything(42)
device='cuda' if torch.cuda.is_available() else 'cpu'
FAST_ROOT = Path('/workspace/moe_medical_vision/data/processed/luna16_fast')
CHECKPOINT_DIR = Path('/workspace/moe_medical_vision/checkpoints')
CKPT_OUT = CHECKPOINT_DIR / 'expert4_luna16_FAST_v2_smallhead_best.pth'

class LunaFastDataset(Dataset):
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
        print(f'[FAST {split}] {len(self.files)} | neg={counts[0]} pos={counts[1]}')
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        d=np.load(self.files[idx])
        x=torch.from_numpy(d['volume'].copy()).float()
        y=torch.tensor(int(d['label']), dtype=torch.long)
        return {'image':x,'label':y}
    def sampler(self):
        w=self.class_weights[self.labels]
        return WeightedRandomSampler(w.tolist(), len(self.files), replacement=True)
    def get_class_weights(self): return self.class_weights

train_ds=LunaFastDataset(FAST_ROOT,'train')
val_ds=LunaFastDataset(FAST_ROOT,'val')
train_loader=DataLoader(train_ds,batch_size=4,sampler=train_ds.sampler(),num_workers=4,pin_memory=True)
val_loader=DataLoader(val_ds,batch_size=4,shuffle=False,num_workers=4,pin_memory=True)

model=build_luna_expert(pretrained=True,use_gradient_checkpointing=True).to(device)
v1=torch.load(CHECKPOINT_DIR/'expert4_luna16_r3d18_best.pth', map_location=device, weights_only=False)
model.load_state_dict(v1['model_state_dict'])
print(f"Loaded v1: {v1['best_val_f1']:.4f}")

in_features=model.backbone.fc[1].in_features
model.backbone.fc=nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(in_features,32),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(32,2),
).to(device)

criterion=torch.nn.CrossEntropyLoss(weight=train_ds.get_class_weights().to(device), label_smoothing=0.01)
print('sanity:', sanity_check_single_batch(model, train_loader, criterion, device))
print('START_TRAIN_LOOP', flush=True)
opt=torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=2e-3)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=15,eta_min=1e-7)
res=fit_3d_expert(model, train_loader, val_loader, criterion, opt, device, epochs=15, checkpoint_path=CKPT_OUT, scheduler=sched, accum_steps=2, mixed_precision=True, patience=8)
print('DONE', res['best_val_f1'], res['best_epoch'])
