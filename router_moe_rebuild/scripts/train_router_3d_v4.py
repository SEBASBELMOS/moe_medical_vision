
from __future__ import annotations

import json
from pathlib import Path
import random
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
import timm

ROOT = Path('/workspace/router_moe_rebuild')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class LinearHead(nn.Module):
    def __init__(self, d_model=192, n_classes=2):
        super().__init__()
        self.gate = nn.Linear(d_model, n_classes)
    def forward(self, z):
        return self.gate(z)


def load_luna_mip(path: Path):
    d=np.load(path)
    vol=torch.from_numpy(d['volume'][0].copy()).float()  # [D,H,W]
    mip=torch.stack([vol.max(dim=0)[0], vol.max(dim=1)[0], vol.max(dim=2)[0]], dim=0)
    mip=F.interpolate(mip.unsqueeze(0), size=(224,224), mode='bilinear', align_corners=False).squeeze(0)
    return mip, 0


def load_pancreatic_mip(path: Path):
    vol=nib.load(str(path)).get_fdata(dtype=np.float32)
    vol=np.clip(vol,-1000.0,400.0)
    vol=(vol+1000.0)/1400.0
    vol=torch.from_numpy(vol).float()  # [D,H,W]
    vol=F.interpolate(vol.unsqueeze(0).unsqueeze(0), size=(64,64,64), mode='trilinear', align_corners=False).squeeze(0).squeeze(0)
    mip=torch.stack([vol.max(dim=0)[0], vol.max(dim=1)[0], vol.max(dim=2)[0]], dim=0)
    mip=F.interpolate(mip.unsqueeze(0), size=(224,224), mode='bilinear', align_corners=False).squeeze(0)
    return mip, 1


def build_tensors():
    random.seed(42)
    luna_train = random.sample(sorted(Path('/workspace/moe_medical_vision/data/processed/luna16_highres').glob('train_*.npz')), 160)
    luna_val = random.sample(sorted(Path('/workspace/moe_medical_vision/data/processed/luna16_highres').glob('val_*.npz')), 40)
    pan_files = sorted(Path('/workspace/moe_medical_vision/data/raw/pancreatic').glob('*.nii.gz'))
    random.shuffle(pan_files)
    pan_train = pan_files[:80]
    pan_val = pan_files[80:100]

    def build(file_list, loader):
        xs=[]; ys=[]
        for p in file_list:
            x,y=loader(p)
            xs.append(x)
            ys.append(y)
        return torch.stack(xs), torch.tensor(ys, dtype=torch.long)

    X_luna_tr, y_luna_tr = build(luna_train, load_luna_mip)
    X_pan_tr, y_pan_tr = build(pan_train, load_pancreatic_mip)
    X_luna_va, y_luna_va = build(luna_val, load_luna_mip)
    X_pan_va, y_pan_va = build(pan_val, load_pancreatic_mip)
    X_train = torch.cat([X_luna_tr, X_pan_tr], dim=0)
    y_train = torch.cat([y_luna_tr, y_pan_tr], dim=0)
    X_val = torch.cat([X_luna_va, X_pan_va], dim=0)
    y_val = torch.cat([y_luna_va, y_pan_va], dim=0)
    return X_train, y_train, X_val, y_val


def main():
    X_train, y_train, X_val, y_val = build_tensors()
    print(json.dumps({'train_shape': list(X_train.shape), 'val_shape': list(X_val.shape), 'train_counts': torch.bincount(y_train, minlength=2).tolist(), 'val_counts': torch.bincount(y_val, minlength=2).tolist()}))
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True, num_workers=0)
    backbone = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0).to(DEVICE)
    backbone.eval()
    for p in backbone.parameters(): p.requires_grad=False
    router = LinearHead(192,2).to(DEVICE)
    opt = torch.optim.AdamW(router.parameters(), lr=5e-4, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    best={'acc':-1,'state':None,'epoch':-1,'cm':None}
    hist=[]
    for epoch in range(1,21):
        router.train(); losses=[]
        for xb,yb in train_loader:
            xb,yb=xb.to(DEVICE), yb.to(DEVICE)
            with torch.no_grad():
                z=backbone.forward_features(xb)[:,0,:]
            logits=router(z)
            loss=crit(logits,yb)
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        router.eval()
        with torch.no_grad():
            z=backbone.forward_features(X_val.to(DEVICE))[:,0,:]
            pred=router(z).argmax(dim=-1).cpu().numpy()
        acc=accuracy_score(y_val.numpy(), pred)
        cm=confusion_matrix(y_val.numpy(), pred, labels=[0,1]).tolist()
        row={'epoch':epoch,'loss':float(np.mean(losses)),'acc_bal':float(acc),'cm':cm}
        hist.append(row)
        print(json.dumps(row))
        if acc>best['acc']:
            best={'acc':acc,'state':{k:v.clone().cpu() for k,v in router.state_dict().items()},'epoch':epoch,'cm':cm}
            ckpt=ROOT/'checkpoints'/'router_3d_linear_v4.pth'
            torch.save({'model_state_dict':router.state_dict(),'best_acc':best['acc'],'epoch':best['epoch'],'class_ids':[3,4],'confusion_matrix':best['cm']}, ckpt)
            print(f'NEW_BEST epoch={epoch} acc={acc:.4f}')
            print(f'SAVED_BEST {ckpt}')
    router.load_state_dict(best['state'])
    ckpt=ROOT/'checkpoints'/'router_3d_linear_v4.pth'
    torch.save({'model_state_dict':router.state_dict(),'best_acc':best['acc'],'epoch':best['epoch'],'class_ids':[3,4],'confusion_matrix':best['cm']}, ckpt)
    (ROOT/'metrics'/'router_3d_v4.json').write_text(json.dumps(hist, indent=2))
    print(json.dumps({'best_acc':best['acc'],'epoch':best['epoch'],'ckpt':str(ckpt),'cm':best['cm']}, indent=2))

if __name__=='__main__':
    main()
