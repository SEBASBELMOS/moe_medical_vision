
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
from torch.utils.data import DataLoader, Dataset
import timm

ROOT = Path('/workspace/router_moe_rebuild')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class LinearHead(nn.Module):
    def __init__(self, d_model=192, n_classes=2):
        super().__init__()
        self.gate = nn.Linear(d_model, n_classes)
    def forward(self, z):
        return self.gate(z)

class Fast3DRouterDataset(Dataset):
    def __init__(self, split='train', n_luna=120, n_pan=120, seed=42):
        self.items=[]
        random.seed(seed)
        # LUNA highres npz
        luna_files = sorted(Path('/workspace/moe_medical_vision/data/processed/luna16_highres').glob(f'{split}_*.npz'))
        luna_sel = random.sample(luna_files, min(n_luna, len(luna_files)))
        for f in luna_sel:
            self.items.append(('luna', str(f), 0))
        # pancreatic raw nii.gz (fast enough on subset)
        pan_files = sorted(Path('/workspace/moe_medical_vision/data/raw/pancreatic').glob('*.nii.gz'))
        pan_sel = random.sample(pan_files, min(n_pan, len(pan_files)))
        split_cut = int(len(pan_sel)*0.8)
        pan_sel = pan_sel[:split_cut] if split=='train' else pan_sel[split_cut:]
        for f in pan_sel:
            self.items.append(('pancreatic', str(f), 1))
        random.shuffle(self.items)

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        kind, path, y = self.items[idx]
        if kind == 'luna':
            d=np.load(path)
            vol=torch.from_numpy(d['volume'][0].copy()).float().unsqueeze(0)
        else:
            vol = nib.load(path).get_fdata(dtype=np.float32)
            vol = np.clip(vol, -1000.0, 400.0)
            vol = (vol + 1000.0) / 1400.0
            vol = torch.from_numpy(vol).float().unsqueeze(0)
        # resize to 64^3 and create same MIP multiview as runtime
        vol = F.interpolate(vol.unsqueeze(0), size=(64,64,64), mode='trilinear', align_corners=False).squeeze(0)  # [1,D,H,W]
        x = vol.squeeze(0)  # [D,H,W]
        mip = torch.stack([x.max(dim=0)[0], x.max(dim=1)[0], x.max(dim=2)[0]], dim=0)
        mip = F.interpolate(mip.unsqueeze(0), size=(224,224), mode='bilinear', align_corners=False).squeeze(0)
        return {'image': mip, 'label': torch.tensor(y, dtype=torch.long)}


def main():
    train_ds = Fast3DRouterDataset(split='train', n_luna=160, n_pan=80, seed=42)
    val_ds = Fast3DRouterDataset(split='val', n_luna=40, n_pan=20, seed=43)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    backbone = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0).to(DEVICE)
    backbone.eval()
    for p in backbone.parameters(): p.requires_grad = False
    router = LinearHead(192,2).to(DEVICE)
    opt = torch.optim.AdamW(router.parameters(), lr=5e-4, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    best={'acc':-1,'state':None,'epoch':-1,'cm':None}
    hist=[]
    print(json.dumps({'train_n': len(train_ds), 'val_n': len(val_ds)}))
    for epoch in range(1,21):
        router.train(); losses=[]
        for batch in train_loader:
            x=batch['image'].to(DEVICE); y=batch['label'].to(DEVICE)
            with torch.no_grad():
                z=backbone.forward_features(x)[:,0,:]
            logits=router(z)
            loss=crit(logits,y)
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        router.eval(); preds=[]; ytrue=[]
        with torch.no_grad():
            for batch in val_loader:
                x=batch['image'].to(DEVICE); y=batch['label'].cpu().numpy()
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
            ckpt=ROOT/'checkpoints'/'router_3d_linear_v3.pth'
            torch.save({'model_state_dict':router.state_dict(),'best_acc':best['acc'],'epoch':best['epoch'],'class_ids':[3,4],'confusion_matrix':best['cm']}, ckpt)
            print(f'SAVED_BEST {ckpt}')
    router.load_state_dict(best['state'])
    ckpt=ROOT/'checkpoints'/'router_3d_linear_v3.pth'
    torch.save({'model_state_dict':router.state_dict(),'best_acc':best['acc'],'epoch':best['epoch'],'class_ids':[3,4],'confusion_matrix':best['cm']}, ckpt)
    (ROOT/'metrics'/'router_3d_v3.json').write_text(json.dumps(hist, indent=2))
    print(json.dumps({'best_acc':best['acc'],'epoch':best['epoch'],'ckpt':str(ckpt),'cm':best['cm']}, indent=2))

if __name__=='__main__':
    main()
