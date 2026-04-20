
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset

ROOT = Path('/workspace/router_moe_rebuild')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class LinearHead(nn.Module):
    def __init__(self, d_model=192, n_classes=2):
        super().__init__()
        self.gate = nn.Linear(d_model, n_classes)
    def forward(self, z):
        return self.gate(z)

class Runtime3DFeatureDataset(Dataset):
    def __init__(self, split='train', limit_per_class=160):
        import sys
        sys.path.insert(0, '/workspace/moe_medical_vision/src')
        from data.datasets import get_dataloader
        # build lists from dataset-native preprocessing
        self.samples=[]
        # LUNA -> label 0
        _, ds_luna = get_dataloader('luna16','/workspace/moe_medical_vision/data/raw/luna16', split=split, batch_size=1, num_workers=0)
        neg=[]; pos=[]
        for i in range(len(ds_luna)):
            item=ds_luna[i]
            y=int(item['label'])
            (pos if y==1 else neg).append(item['image'])
        luna_samples = (neg[:limit_per_class//2] + pos[:limit_per_class//2]) if split=='train' else (neg[:40] + pos[:40])
        for x in luna_samples:
            self.samples.append((x,0))
        # Pancreatic -> label 1
        _, ds_pan = get_dataloader('pancreatic','/workspace/moe_medical_vision/data/raw/pancreatic', split=split, batch_size=1, num_workers=0)
        neg=[]; pos=[]
        for i in range(len(ds_pan)):
            item=ds_pan[i]
            y=int(item['label'])
            (pos if y==1 else neg).append(item['image'])
        pan_samples = (neg[:limit_per_class//2] + pos[:limit_per_class//2]) if split=='train' else (neg[:40] + pos[:40])
        for x in pan_samples:
            self.samples.append((x,1))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return {'image': self.samples[idx][0], 'label': torch.tensor(self.samples[idx][1], dtype=torch.long)}


def prep_router_input(x3d):
    # match runtime final: 3 orthogonal MIPs
    x = x3d
    if x.ndim == 5 and x.shape[1] == 1:
        x = x.squeeze(1)
    mip = torch.stack([x.max(dim=1)[0], x.max(dim=2)[0], x.max(dim=3)[0]], dim=1)
    x = F.interpolate(mip, size=(224,224), mode='bilinear', align_corners=False)
    return x


def main():
    train_ds = Runtime3DFeatureDataset(split='train', limit_per_class=160)
    val_ds = Runtime3DFeatureDataset(split='val', limit_per_class=80)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    backbone = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0).to(DEVICE)
    backbone.eval()
    for p in backbone.parameters(): p.requires_grad = False
    router = LinearHead(192,2).to(DEVICE)
    opt = torch.optim.AdamW(router.parameters(), lr=5e-4, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    best={'acc':-1,'state':None,'epoch':-1,'cm':None}
    hist=[]
    for epoch in range(1,31):
        router.train(); losses=[]
        for batch in train_loader:
            x3d=batch['image'].to(DEVICE); y=batch['label'].to(DEVICE)
            x=prep_router_input(x3d)
            with torch.no_grad():
                z=backbone.forward_features(x)[:,0,:]
            logits=router(z)
            loss=crit(logits,y)
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        router.eval(); preds=[]; ytrue=[]
        with torch.no_grad():
            for batch in val_loader:
                x3d=batch['image'].to(DEVICE); y=batch['label'].cpu().numpy()
                x=prep_router_input(x3d)
                z=backbone.forward_features(x)[:,0,:]
                pred=router(z).argmax(dim=-1).cpu().numpy()
                preds.extend(pred.tolist()); ytrue.extend(y.tolist())
        acc=accuracy_score(ytrue,preds)
        cm=confusion_matrix(ytrue,preds,labels=[0,1]).tolist()
        row={'epoch':epoch,'loss':float(np.mean(losses)),'acc_bal':float(acc),'cm':cm}
        hist.append(row)
        print(json.dumps(row))
        if acc>best['acc']:
            best={'acc':acc,'state':{k:v.clone().cpu() for k,v in router.state_dict().items()},'epoch':epoch,'cm':cm}
            print(f'NEW_BEST epoch={epoch} acc={acc:.4f}')
    router.load_state_dict(best['state'])
    ckpt=ROOT/'checkpoints'/'router_3d_linear_v2.pth'
    torch.save({'model_state_dict':router.state_dict(),'best_acc':best['acc'],'epoch':best['epoch'],'class_ids':[3,4],'confusion_matrix':best['cm']}, ckpt)
    (ROOT/'metrics'/'router_3d_linear_v2.json').write_text(json.dumps(hist, indent=2))
    print(json.dumps({'best_acc':best['acc'],'epoch':best['epoch'],'ckpt':str(ckpt),'cm':best['cm']}, indent=2))

if __name__=='__main__':
    main()
