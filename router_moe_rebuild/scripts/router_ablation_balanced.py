
from __future__ import annotations

import json
import time
from pathlib import Path

import faiss
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path('/workspace/router_moe_rebuild')
SRC = Path('/workspace/moe_medical_vision/data/processed/router_embeddings')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT.mkdir(parents=True, exist_ok=True)
(ROOT/'metrics').mkdir(parents=True, exist_ok=True)
(ROOT/'checkpoints').mkdir(parents=True, exist_ok=True)

class LinearGatingHead(nn.Module):
    def __init__(self, d_model, n_experts):
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts)
    def forward(self, z):
        return self.gate(z)


def build_balanced_indices(y, per_class=1000, seed=42):
    g = torch.Generator().manual_seed(seed)
    indices = []
    for i in range(5):
        idx = torch.where(y == i)[0]
        if len(idx) == 0:
            continue
        if len(idx) >= per_class:
            perm = idx[torch.randperm(len(idx), generator=g)[:per_class]]
        else:
            perm = idx[torch.randint(0, len(idx), (per_class,), generator=g)]
        indices.append(perm)
    out = torch.cat(indices)
    out = out[torch.randperm(len(out), generator=g)]
    return out


def latency_ms(fn, n=5):
    vals=[]
    for _ in range(n):
        t0=time.time(); fn(); vals.append((time.time()-t0)*1000)
    return float(np.mean(vals))


def main():
    tr = np.load(SRC/'Z_train.npz')
    va = np.load(SRC/'Z_val.npz')
    X_train = torch.tensor(tr['z']).float()
    y_train = torch.tensor(tr['y_expert']).long()
    X_val = torch.tensor(va['z']).float()
    y_val = torch.tensor(va['y_expert']).long()

    idx_tr = build_balanced_indices(y_train, per_class=1000, seed=42)
    idx_va = build_balanced_indices(y_val, per_class=200, seed=43)
    X_trb, y_trb = X_train[idx_tr], y_train[idx_tr]
    X_vab, y_vab = X_val[idx_va], y_val[idx_va]

    results=[]
    d_model = X_train.shape[1]
    n_exp = 5

    # Router A: Linear
    router = LinearGatingHead(d_model, n_exp).to(DEVICE)
    opt = torch.optim.AdamW(router.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(X_trb, y_trb), batch_size=256, shuffle=True)
    best={'acc':-1,'state':None,'epoch':-1}
    for epoch in range(1,31):
        router.train()
        for xb,yb in loader:
            xb,yb=xb.to(DEVICE), yb.to(DEVICE)
            logits=router(xb)
            loss=crit(logits,yb)
            opt.zero_grad(); loss.backward(); opt.step()
        router.eval()
        with torch.no_grad():
            pred = router(X_vab.to(DEVICE)).argmax(dim=-1).cpu().numpy()
        acc = accuracy_score(y_vab.numpy(), pred)
        if acc>best['acc']:
            best={'acc':acc,'state':{k:v.clone().cpu() for k,v in router.state_dict().items()},'epoch':epoch}
    router.load_state_dict(best['state'])
    torch.save({'model_state_dict': router.state_dict(), 'best_acc_bal': best['acc'], 'epoch': best['epoch']}, ROOT/'checkpoints/router_linear_ablation_best.pth')
    with torch.no_grad():
        pred_bal = router(X_vab.to(DEVICE)).argmax(dim=-1).cpu().numpy()
        pred_full = router(X_val.to(DEVICE)).argmax(dim=-1).cpu().numpy()
    acc_bal = accuracy_score(y_vab.numpy(), pred_bal)
    acc_full = accuracy_score(y_val.numpy(), pred_full)
    lat = latency_ms(lambda: router(X_vab[:256].to(DEVICE))) / len(X_vab[:256])
    results.append({'Router':'ViT + Linear','Routing Acc (bal)':acc_bal,'Routing Acc (full)':acc_full,'Latencia ms':lat,'Params':sum(p.numel() for p in router.parameters()),'VRAM':'~1.5GB'})

    # normalize embeddings
    Z_train = X_trb.numpy(); Z_val_bal = X_vab.numpy(); Z_val_full = X_val.numpy()
    Z_train_norm = Z_train / np.linalg.norm(Z_train, axis=1, keepdims=True)
    Z_val_bal_norm = Z_val_bal / np.linalg.norm(Z_val_bal, axis=1, keepdims=True)
    Z_val_full_norm = Z_val_full / np.linalg.norm(Z_val_full, axis=1, keepdims=True)

    # GMM
    gmms=[]
    for i in range(n_exp):
        zc = Z_train_norm[y_trb.numpy()==i]
        g=GaussianMixture(n_components=1, covariance_type='diag', random_state=42)
        g.fit(zc)
        gmms.append(g)
    def gmm_pred(Z):
        scores=np.stack([g.score_samples(Z) for g in gmms], axis=1)
        return scores.argmax(axis=1)
    acc_bal=accuracy_score(y_vab.numpy(), gmm_pred(Z_val_bal_norm))
    acc_full=accuracy_score(y_val.numpy(), gmm_pred(Z_val_full_norm))
    lat = latency_ms(lambda: gmm_pred(Z_val_bal_norm[:256])) / len(Z_val_bal_norm[:256])
    results.append({'Router':'ViT + GMM','Routing Acc (bal)':acc_bal,'Routing Acc (full)':acc_full,'Latencia ms':lat,'Params':n_exp*d_model*2,'VRAM':'CPU'})

    # NB
    nb=GaussianNB(); nb.fit(Z_train_norm, y_trb.numpy())
    acc_bal=accuracy_score(y_vab.numpy(), nb.predict(Z_val_bal_norm))
    acc_full=accuracy_score(y_val.numpy(), nb.predict(Z_val_full_norm))
    lat = latency_ms(lambda: nb.predict(Z_val_bal_norm[:256])) / len(Z_val_bal_norm[:256])
    results.append({'Router':'ViT + Naive Bayes','Routing Acc (bal)':acc_bal,'Routing Acc (full)':acc_full,'Latencia ms':lat,'Params':n_exp*d_model*2,'VRAM':'CPU'})

    # kNN + PCA
    pca = PCA(n_components=32, random_state=42)
    Ztr_pca = pca.fit_transform(Z_train_norm).astype(np.float32)
    Zvb_pca = pca.transform(Z_val_bal_norm).astype(np.float32)
    Zvf_pca = pca.transform(Z_val_full_norm).astype(np.float32)
    faiss.normalize_L2(Ztr_pca); faiss.normalize_L2(Zvb_pca); faiss.normalize_L2(Zvf_pca)
    index = faiss.IndexFlatIP(32); index.add(Ztr_pca)
    labels_tr = y_trb.numpy().astype(np.int32)
    def knn_predict(Z):
        D,I=index.search(Z,5)
        neigh=labels_tr[I]
        return np.apply_along_axis(lambda x: np.bincount(x,minlength=5).argmax(),1,neigh)
    acc_bal=accuracy_score(y_vab.numpy(), knn_predict(Zvb_pca))
    acc_full=accuracy_score(y_val.numpy(), knn_predict(Zvf_pca))
    lat = latency_ms(lambda: knn_predict(Zvb_pca[:256])) / len(Zvb_pca[:256])
    joblib.dump(pca, ROOT/'checkpoints/router_pca_balanced.pkl')
    faiss.write_index(index, str(ROOT/'checkpoints/router_knn_balanced.index'))
    joblib.dump(labels_tr, ROOT/'checkpoints/router_knn_labels_balanced.pkl')
    results.append({'Router':'ViT + k-NN (FAISS)','Routing Acc (bal)':acc_bal,'Routing Acc (full)':acc_full,'Latencia ms':lat,'Params':len(Ztr_pca)*32,'VRAM':'CPU'})

    df=pd.DataFrame(results).sort_values('Routing Acc (bal)', ascending=False)
    df.to_csv(ROOT/'metrics/ablation_balanced.csv', index=False)
    print(df.to_string(index=False))
    print('saved', ROOT/'metrics/ablation_balanced.csv')

if __name__=='__main__':
    main()
