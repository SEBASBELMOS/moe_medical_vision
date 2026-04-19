import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

seed=42
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
device='cuda' if torch.cuda.is_available() else 'cpu'
print(f'device={device}')
FAST_ROOT = Path('/workspace/moe_medical_vision/data/processed/luna16_fast')
OUT_PATH = Path('/workspace/moe_medical_vision/checkpoints/luna_cv_resnet_summary.txt')

class Luna2p5DAll(Dataset):
    def __init__(self, root, files, labels, train=True):
        self.files = files
        self.labels = labels
        self.train = train
        counts=np.bincount(self.labels, minlength=2)
        total=counts.sum()
        self.class_weights=torch.tensor(total/(2*counts+1e-6), dtype=torch.float32)
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        d=np.load(self.files[idx])
        vol=d['volume'][0]
        zmid=vol.shape[0]//2
        z_ids=[max(0,min(vol.shape[0]-1,zmid+off)) for off in (-8,0,8)]
        stack=torch.from_numpy(vol[z_ids].copy()).float()
        stack=F.interpolate(stack.unsqueeze(0), size=(224,224), mode='bilinear', align_corners=False).squeeze(0)
        if self.train:
            if torch.rand(1).item()<0.5:
                stack=torch.flip(stack,dims=(2,))
            if torch.rand(1).item()<0.3:
                stack=(stack+torch.randn_like(stack)*0.02).clamp(0,1)
        stack=(stack-0.5)/0.25
        y=torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return {'image':stack,'label':y}
    def sampler(self):
        w=self.class_weights[self.labels]
        return WeightedRandomSampler(w.tolist(), len(self.files), replacement=True)

files = sorted((FAST_ROOT.glob('train_*.npz')))
labels=[]
for f in files:
    d=np.load(f)
    labels.append(int(d['label']))
labels=np.array(labels)

skf=StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
results=[]
for fold,(tr_idx,va_idx) in enumerate(skf.split(np.arange(len(files)), labels), start=1):
    print(f'=== FOLD {fold} ===')
    tr_files=[files[i] for i in tr_idx]; va_files=[files[i] for i in va_idx]
    tr_labels=labels[tr_idx]; va_labels=labels[va_idx]
    train_ds=Luna2p5DAll(FAST_ROOT,tr_files,tr_labels,train=True)
    val_ds=Luna2p5DAll(FAST_ROOT,va_files,va_labels,train=False)
    train_loader=DataLoader(train_ds,batch_size=32,sampler=train_ds.sampler(),num_workers=4,pin_memory=True)
    val_loader=DataLoader(val_ds,batch_size=32,shuffle=False,num_workers=4,pin_memory=True)
    model=resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc=nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features,2))
    model=model.to(device)
    criterion=nn.CrossEntropyLoss(weight=train_ds.class_weights.to(device), label_smoothing=0.02)
    opt=torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=6, eta_min=1e-6)
    best=-1.0; wait=0
    for epoch in range(1,7):
        model.train(); tr_p=[]; tr_y=[]; tr_l=[]
        for batch in train_loader:
            x=batch['image'].to(device); y=batch['label'].to(device)
            out=model(x); loss=criterion(out,y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tr_l.append(loss.item()); tr_p.extend(out.argmax(1).detach().cpu().tolist()); tr_y.extend(y.cpu().tolist())
        sched.step()
        model.eval(); va_p=[]; va_y=[]
        with torch.no_grad():
            for batch in val_loader:
                x=batch['image'].to(device); y=batch['label'].to(device)
                out=model(x); va_p.extend(out.argmax(1).cpu().tolist()); va_y.extend(y.cpu().tolist())
        f1=f1_score(va_y,va_p,average='macro')
        print(f'fold={fold} epoch={epoch} val_f1={f1:.4f}')
        if f1>best:
            best=f1; wait=0
        else:
            wait+=1
            if wait>=3: break
    results.append(best)

mean_f1=float(np.mean(results)); std_f1=float(np.std(results))
print(f'LUNA CV mean_f1={mean_f1:.4f} std={std_f1:.4f} folds={results}')
OUT_PATH.write_text(f'mean_f1={mean_f1:.4f}\nstd={std_f1:.4f}\nfolds={results}\n')
