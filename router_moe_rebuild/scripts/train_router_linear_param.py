
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT = Path('/workspace/moe_medical_vision')
EMB = ROOT / 'data' / 'processed' / 'router_embeddings'
CKPT = ROOT / 'checkpoints'
MET = CKPT / 'metrics'
MET.mkdir(parents=True, exist_ok=True)

class LinearGatingHead(nn.Module):
    def __init__(self, d_model, n_experts):
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts)
    def forward(self, z):
        return self.gate(z)

def auxiliary_load_balancing_loss(expert_probs, alpha=0.2):
    n_experts = expert_probs.size(1)
    P_i = expert_probs.mean(dim=0)
    expert_assignments = torch.argmax(expert_probs, dim=1)
    f_i = torch.bincount(expert_assignments, minlength=n_experts).float() / expert_probs.size(0)
    loss_aux = alpha * n_experts * torch.sum(f_i * P_i)
    min_nonzero = f_i[f_i > 0].min() if (f_i > 0).any() else torch.tensor(float('inf'), device=f_i.device)
    ratio = f_i.max() / min_nonzero if torch.isfinite(min_nonzero) else torch.tensor(float('inf'), device=f_i.device)
    return loss_aux, f_i, P_i, ratio

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
            reps = idx[torch.randint(0, len(idx), (per_class,), generator=g)]
            perm = reps
        indices.append(perm)
    out = torch.cat(indices)
    out = out[torch.randperm(len(out), generator=g)]
    return out



def build_balanced_subset(X, y, per_class=200, seed=42):
    idx = build_balanced_indices(y, per_class=per_class, seed=seed)
    return X[idx], y[idx]


def evaluate(router, X, y, alpha):
    router.eval()
    with torch.no_grad():
        logits = router(X)
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        acc = accuracy_score(y.cpu().numpy(), preds.cpu().numpy())
        _, f_i, P_i, ratio = auxiliary_load_balancing_loss(probs, alpha=alpha)
        cm = confusion_matrix(y.cpu().numpy(), preds.cpu().numpy(), labels=[0,1,2,3,4]).tolist()
    return {
        'routing_acc': float(acc),
        'f_i': f_i.cpu().numpy().tolist(),
        'P_i': P_i.cpu().numpy().tolist(),
        'ratio': float(ratio.item()),
        'confusion_matrix': cm,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha', type=float, default=0.2)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--per-class', type=int, default=1000)
    ap.add_argument('--val-per-class', type=int, default=200)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--selection', choices=['acc','balance'], default='acc')
    args = ap.parse_args()

    train = np.load(EMB/'Z_train.npz')
    val = np.load(EMB/'Z_val.npz')
    X_train = torch.tensor(train['z']).float().to(DEVICE)
    y_train = torch.tensor(train['y_expert']).long().to(DEVICE)
    X_val = torch.tensor(val['z']).float().to(DEVICE)
    y_val = torch.tensor(val['y_expert']).long().to(DEVICE)

    router = LinearGatingHead(X_train.shape[1], 5).to(DEVICE)
    opt = torch.optim.AdamW(router.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss()

    idx = build_balanced_indices(y_train, per_class=args.per_class)
    X_bal, y_bal = X_train[idx], y_train[idx]
    X_val_bal, y_val_bal = build_balanced_subset(X_val, y_val, per_class=args.val_per_class, seed=43)
    loader = DataLoader(TensorDataset(X_bal, y_bal), batch_size=args.batch_size, shuffle=True)

    best = {'acc': -1.0, 'ratio': float('inf'), 'state': None, 'epoch': -1, 'eval': None}
    history=[]
    print(json.dumps({'tag': args.tag, 'alpha': args.alpha, 'epochs': args.epochs, 'balanced_counts': np.bincount(y_bal.cpu().numpy(), minlength=5).tolist()}))
    for epoch in range(1, args.epochs+1):
        router.train()
        l_task_ep=[]; l_aux_ep=[]
        for xb, yb in loader:
            logits = router(xb)
            probs = F.softmax(logits, dim=-1)
            l_task = crit(logits, yb)
            l_aux, _, _, _ = auxiliary_load_balancing_loss(probs, alpha=args.alpha)
            loss = l_task + l_aux
            opt.zero_grad(); loss.backward(); opt.step()
            l_task_ep.append(l_task.item()); l_aux_ep.append(l_aux.item())
        ev_full = evaluate(router, X_val, y_val, args.alpha)
        ev_bal = evaluate(router, X_val_bal, y_val_bal, args.alpha)
        row = {
            'epoch': epoch,
            'L_task': float(np.mean(l_task_ep)),
            'L_aux': float(np.mean(l_aux_ep)),
            'routing_acc_val_full': ev_full['routing_acc'],
            'ratio_val_full': ev_full['ratio'],
            'routing_acc_val_bal': ev_bal['routing_acc'],
            'ratio_val_bal': ev_bal['ratio'],
            'f_i_val_bal': ev_bal['f_i'],
            'P_i_val_bal': ev_bal['P_i'],
            'tag': args.tag,
        }
        history.append(row)
        print(json.dumps(row))
        if args.selection == 'acc':
            keep = (ev_full['routing_acc'] > best['acc']) or (abs(ev_full['routing_acc']-best['acc']) < 1e-6 and ev_bal['ratio'] < best['ratio'])
        else:
            best_valid = best['ratio'] < 1.30
            cur_valid = ev_bal['ratio'] < 1.30
            if cur_valid and not best_valid:
                keep = True
            elif cur_valid and best_valid:
                keep = (ev_bal['ratio'] < best['ratio']) or (abs(ev_bal['ratio']-best['ratio']) < 1e-6 and ev_bal['routing_acc'] > best['acc'])
            elif (not cur_valid) and (not best_valid):
                keep = (ev_bal['ratio'] < best['ratio']) or (abs(ev_bal['ratio']-best['ratio']) < 1e-6 and ev_bal['routing_acc'] > best['acc'])
            else:
                keep = False
        if keep:
            best = {'acc': ev_bal['routing_acc'], 'ratio': ev_bal['ratio'], 'state': {k:v.clone().cpu() for k,v in router.state_dict().items()}, 'epoch': epoch, 'eval': {'full': ev_full, 'bal': ev_bal}}
            print(f"NEW_BEST epoch={epoch} acc_bal={ev_bal['routing_acc']:.4f} ratio_bal={ev_bal['ratio']:.3f} acc_full={ev_full['routing_acc']:.4f}")

    if best['state'] is not None:
        router.load_state_dict(best['state'])
    ckpt_path = CKPT / f'{args.tag}.pth'
    torch.save({'model_state_dict': router.state_dict(), 'best_acc': best['acc'], 'best_ratio': best['ratio'], 'epoch': best['epoch'], 'alpha': args.alpha, 'eval': best['eval']}, ckpt_path)
    hist_path = MET / f'{args.tag}.json'
    hist_path.write_text(json.dumps(history, indent=2))
    print(json.dumps({'done': True, 'tag': args.tag, 'best_acc': best['acc'], 'best_ratio': best['ratio'], 'epoch': best['epoch'], 'ckpt': str(ckpt_path), 'history': str(hist_path)}, indent=2))

if __name__ == '__main__':
    main()
