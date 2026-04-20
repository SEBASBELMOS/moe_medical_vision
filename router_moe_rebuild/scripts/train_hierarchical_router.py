
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
ROOT=Path('/workspace/router_moe_rebuild')
SRC=Path('/workspace/moe_medical_vision/data/processed/router_embeddings')
(ROOT/'checkpoints').mkdir(parents=True, exist_ok=True)
(ROOT/'metrics').mkdir(parents=True, exist_ok=True)

class LinearHead(nn.Module):
    def __init__(self, d_model, n_classes):
        super().__init__()
        self.gate = nn.Linear(d_model, n_classes)
    def forward(self, z):
        return self.gate(z)

def build_balanced_indices(y, classes, per_class=1000, seed=42):
    g = torch.Generator().manual_seed(seed)
    idx_all=[]
    for c in classes:
        idx = torch.where(y == c)[0]
        if len(idx) >= per_class:
            sel = idx[torch.randperm(len(idx), generator=g)[:per_class]]
        else:
            sel = idx[torch.randint(0, len(idx), (per_class,), generator=g)]
        idx_all.append(sel)
    idx_all = torch.cat(idx_all)
    idx_all = idx_all[torch.randperm(len(idx_all), generator=g)]
    return idx_all

def train_router(name, class_ids, per_class_train, per_class_val, lr=1e-3, epochs=20):
    tr=np.load(SRC/'Z_train.npz'); va=np.load(SRC/'Z_val.npz')
    X_train=torch.tensor(tr['z']).float(); y_train=torch.tensor(tr['y_expert']).long()
    X_val=torch.tensor(va['z']).float(); y_val=torch.tensor(va['y_expert']).long()

    idx_tr=build_balanced_indices(y_train, class_ids, per_class_train, seed=42)
    idx_va=build_balanced_indices(y_val, class_ids, per_class_val, seed=43)
    X_tr, y_tr = X_train[idx_tr], y_train[idx_tr]
    X_va, y_va = X_val[idx_va], y_val[idx_va]

    # remap labels to 0..n-1
    id2local={c:i for i,c in enumerate(class_ids)}
    y_tr_local=torch.tensor([id2local[int(v)] for v in y_tr.numpy()]).long()
    y_va_local=torch.tensor([id2local[int(v)] for v in y_va.numpy()]).long()

    model=LinearHead(X_tr.shape[1], len(class_ids)).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit=nn.CrossEntropyLoss()
    loader=DataLoader(TensorDataset(X_tr, y_tr_local), batch_size=256, shuffle=True)
    best={'acc':-1,'state':None,'epoch':-1,'cm':None}
    hist=[]
    for epoch in range(1,epochs+1):
        model.train()
        losses=[]
        for xb,yb in loader:
            xb,yb=xb.to(DEVICE), yb.to(DEVICE)
            logits=model(xb)
            loss=crit(logits,yb)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            pred=model(X_va.to(DEVICE)).argmax(dim=-1).cpu().numpy()
        acc=accuracy_score(y_va_local.numpy(), pred)
        cm=confusion_matrix(y_va_local.numpy(), pred, labels=list(range(len(class_ids)))).tolist()
        row={'epoch':epoch,'loss':float(np.mean(losses)),'acc_bal':float(acc)}
        hist.append(row)
        print(name, json.dumps(row))
        if acc>best['acc']:
            best={'acc':acc,'state':{k:v.clone().cpu() for k,v in model.state_dict().items()},'epoch':epoch,'cm':cm}
            print(f'NEW_BEST {name} epoch={epoch} acc={acc:.4f}')
    model.load_state_dict(best['state'])
    ckpt=ROOT/'checkpoints'/f'{name}.pth'
    torch.save({'model_state_dict':model.state_dict(),'best_acc':best['acc'],'epoch':best['epoch'],'class_ids':class_ids,'confusion_matrix':best['cm']}, ckpt)
    (ROOT/'metrics'/f'{name}.json').write_text(json.dumps(hist, indent=2))
    return {'name':name,'best_acc':best['acc'],'epoch':best['epoch'],'ckpt':str(ckpt),'class_ids':class_ids,'cm':best['cm']}


def main():
    out={
        'router_2d': train_router('router_2d_linear', [0,1,2], per_class_train=1500, per_class_val=200, lr=1e-3, epochs=25),
        'router_3d': train_router('router_3d_linear', [3,4], per_class_train=160, per_class_val=40, lr=5e-4, epochs=30),
    }
    out_path=ROOT/'metrics'/'hierarchical_router_meta.json'
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print('saved', out_path)

if __name__=='__main__':
    main()
