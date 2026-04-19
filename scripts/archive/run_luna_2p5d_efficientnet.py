import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import f1_score

seed=42
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
device='cuda' if torch.cuda.is_available() else 'cpu'
print(f'device={device}')
FAST_ROOT = Path('/workspace/moe_medical_vision/data/processed/luna16_fast')
CKPT_OUT = Path('/workspace/moe_medical_vision/checkpoints/expert4_luna16_2p5d_efficientnet_best.pth')

class Luna2p5DFast(Dataset):
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
        print(f'[LUNA 2P5D EfficientNet {split}] {len(self.files)} | neg={counts[0]} pos={counts[1]}')
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        d=np.load(self.files[idx])
        vol=d['volume'][0]
        zmid=vol.shape[0]//2
        z_ids=[max(0,min(vol.shape[0]-1,zmid+off)) for off in (-8,0,8)]
        stack=torch.from_numpy(vol[z_ids].copy()).float()
        stack=F.interpolate(stack.unsqueeze(0), size=(224,224), mode='bilinear', align_corners=False).squeeze(0)
        if self.split=='train':
            if torch.rand(1).item()<0.5:
                stack=torch.flip(stack,dims=(2,))
            if torch.rand(1).item()<0.3:
                stack=(stack+torch.randn_like(stack)*0.02).clamp(0,1)
        stack=(stack-0.5)/0.25
        y=torch.tensor(int(d['label']), dtype=torch.long)
        return {'image':stack,'label':y}
    def sampler(self):
        w=self.class_weights[self.labels]
        return WeightedRandomSampler(w.tolist(), len(self.files), replacement=True)

train_ds=Luna2p5DFast(FAST_ROOT,'train')
val_ds=Luna2p5DFast(FAST_ROOT,'val')
train_loader=DataLoader(train_ds,batch_size=16,sampler=train_ds.sampler(),num_workers=4,pin_memory=True)
val_loader=DataLoader(val_ds,batch_size=16,shuffle=False,num_workers=4,pin_memory=True)

model=efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
in_features=model.classifier[1].in_features
model.classifier[1]=nn.Linear(in_features,2)
model=model.to(device)
criterion=nn.CrossEntropyLoss(weight=train_ds.class_weights.to(device), label_smoothing=0.02)

batch=next(iter(train_loader))
print('sanity:', batch['image'].shape)
opt=torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=15, eta_min=1e-6)

best=-1.0; best_epoch=0; history=[]; wait=0
for epoch in range(1,16):
    model.train(); tr_loss=[]; tr_p=[]; tr_y=[]
    for batch in train_loader:
        x=batch['image'].to(device, non_blocking=True)
        y=batch['label'].to(device, non_blocking=True)
        out=model(x)
        loss=criterion(out,y)
        opt.zero_grad(set_to_none=True)
        loss.backward(); opt.step()
        tr_loss.append(loss.item())
        tr_p.extend(out.argmax(1).detach().cpu().tolist())
        tr_y.extend(y.cpu().tolist())
    sched.step()
    model.eval(); va_loss=[]; va_p=[]; va_y=[]
    with torch.no_grad():
        for batch in val_loader:
            x=batch['image'].to(device, non_blocking=True)
            y=batch['label'].to(device, non_blocking=True)
            out=model(x)
            loss=criterion(out,y)
            va_loss.append(loss.item())
            va_p.extend(out.argmax(1).cpu().tolist())
            va_y.extend(y.cpu().tolist())
    row={'epoch':epoch,'lr':opt.param_groups[0]['lr'],'train_loss':float(np.mean(tr_loss)),'train_f1':f1_score(tr_y,tr_p,average='macro'),'val_loss':float(np.mean(va_loss)),'val_f1':f1_score(va_y,va_p,average='macro')}
    history.append(row)
    print(f"[Epoch {epoch:02d}] train_loss={row['train_loss']:.4f} train_f1={row['train_f1']:.4f} val_loss={row['val_loss']:.4f} val_f1={row['val_f1']:.4f} lr={row['lr']:.2e}")
    if row['val_f1']>best:
        best=row['val_f1']; best_epoch=epoch; wait=0
        torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),'best_val_f1':best,'history':history}, CKPT_OUT)
        print(f'  -> nuevo mejor checkpoint: {CKPT_OUT.name} (val_f1={best:.4f})')
    else:
        wait+=1
        if wait>=6:
            print(f'Early stopping activado en epoch {epoch}. Mejor epoch: {best_epoch}')
            break
print(f"LUNA 2P5D EfficientNet: best_val_f1={best:.4f} best_epoch={best_epoch}")
print('DONE')
