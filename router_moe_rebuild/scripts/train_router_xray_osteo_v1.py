
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
ROOT = Path('/workspace/router_moe_rebuild')
SRC = Path('/workspace/moe_medical_vision/data/processed/router_embeddings')
(ROOT/'checkpoints').mkdir(parents=True, exist_ok=True)
(ROOT/'metrics').mkdir(parents=True, exist_ok=True)

class LinearHead(nn.Module):
    def __init__(self, d_model, n_classes):
        super().__init__()
        self.gate = nn.Linear(d_model, n_classes)
    def forward(self, z):
        return self.gate(z)


def build_balanced_indices(y, classes, per_class=2000, seed=42):
    g = torch.Generator().manual_seed(seed)
    idx_all=[]
    for c in classes:
        idx=torch.where(y==c)[0]
        if len(idx)>=per_class:
            sel=idx[torch.randperm(len(idx), generator=g)[:per_class]]
        else:
            sel=idx[torch.randint(0,len(idx),(per_class,), generator=g)]
        idx_all.append(sel)
    idx_all=torch.cat(idx_all)
    idx_all=idx_all[torch.randperm(len(idx_all), generator=g)]
    return idx_all


def main():
    tr=np.load(SRC/'Z_train.npz'); va=np.load(SRC/'Z_val.npz')
    X_train=torch.tensor(tr['z']).float(); y_train=torch.tensor(tr['y_expert']).long()
    X_val=torch.tensor(va['z']).float(); y_val=torch.tensor(va['y_expert']).long()
    class_ids=[0,2]
    idx_tr=build_balanced_indices(y_train, class_ids, per_class=2500, seed=42)
    idx_va=build_balanced_indices(y_val, class_ids, per_class=300, seed=43)
    X_tr, y_tr = X_train[idx_tr], y_train[idx_tr]
    X_va, y_va = X_val[idx_va], y_val[idx_va]
    id2local={0:0,2:1}
    y_tr_local=torch.tensor([id2local[int(v)] for v in y_tr.numpy()]).long()
    y_va_local=torch.tensor([id2local[int(v)] for v in y_va.numpy()]).long()
    model=LinearHead(X_tr.shape[1], 2).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    crit=nn.CrossEntropyLoss(label_smoothing=0.05)
    loader=DataLoader(TensorDataset(X_tr, y_tr_local), batch_size=256, shuffle=True)
    best={'acc':-1,'state':None,'epoch':-1,'cm':None}
    hist=[]
    for epoch in range(1,41):
        model.train(); losses=[]
        for xb,yb in loader:
            xb,yb=xb.to(DEVICE), yb.to(DEVICE)
            logits=model(xb)
            loss=crit(logits,yb)
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            pred=model(X_va.to(DEVICE)).argmax(dim=-1).cpu().numpy()
        acc=accuracy_score(y_va_local.numpy(), pred)
        cm=confusion_matrix(y_va_local.numpy(), pred, labels=[0,1]).tolist()
        row={'epoch':epoch,'loss':float(np.mean(losses)),'acc_bal':float(acc),'cm':cm}
        hist.append(row)
        print(json.dumps(row))
        if acc>best['acc']:
            best={'acc':acc,'state':{k:v.clone().cpu() for k,v in model.state_dict().items()},'epoch':epoch,'cm':cm}
            print(f'NEW_BEST epoch={epoch} acc={acc:.4f}')
    model.load_state_dict(best['state'])
    ckpt=ROOT/'checkpoints'/'router_xray_osteo_linear_v1.pth'
    torch.save({'model_state_dict':model.state_dict(),'best_acc':best['acc'],'epoch':best['epoch'],'class_ids':[0,2],'confusion_matrix':best['cm']}, ckpt)
    (ROOT/'metrics'/'router_xray_osteo_linear_v1.json').write_text(json.dumps(hist, indent=2))
    print(json.dumps({'best_acc':best['acc'],'epoch':best['epoch'],'ckpt':str(ckpt),'cm':best['cm']}, indent=2))

if __name__=='__main__':
    main()
