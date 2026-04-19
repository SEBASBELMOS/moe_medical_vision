import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import torch.nn.functional as F
import random

sys.path.insert(0, '/workspace/moe_medical_vision/src')
from train.train_3d import seed_everything

seed_everything(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def auxiliary_load_balancing_loss(expert_probs, alpha=0.01):
    n_experts = expert_probs.size(1)
    P_i = expert_probs.mean(dim=0)
    expert_assignments = torch.argmax(expert_probs, dim=1)
    f_i = torch.bincount(expert_assignments, minlength=n_experts).float() / expert_probs.size(0)
    loss_aux = alpha * n_experts * torch.sum(f_i * P_i)
    
    f_i_nonzero = f_i[f_i > 0]
    balance_ratio = f_i.max() / f_i_nonzero.min() if len(f_i_nonzero) > 0 else torch.tensor(float('inf'))
    
    return loss_aux, f_i, P_i, balance_ratio

class LinearGatingHead(nn.Module):
    def __init__(self, d_model, n_experts):
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts)
    def forward(self, z):
        return self.gate(z)

router = LinearGatingHead(192, 5).to(device)

EMBEDDINGS_DIR = '/workspace/moe_medical_vision/data/processed/router_embeddings'
train_data = np.load(f"{EMBEDDINGS_DIR}/Z_train.npz")
Z_train, y_expert = torch.tensor(train_data['z']).float().to(device), torch.tensor(train_data['y_expert']).long().to(device)

optimizer = torch.optim.AdamW(router.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

print("="*80)
print("  FINE-TUNING GLOBAL FASE 3 (Optimizando Router Lineal + Auxiliary Loss)")
print("="*80)

# Balanceamos artificialmente tomando 1000 de cada dataset para evitar overfit del router a ISIC/NIH
indices = []
for i in range(5):
    idx = torch.where(y_expert == i)[0]
    indices.append(idx[torch.randperm(len(idx))[:1000]])
balanced_idx = torch.cat(indices)
balanced_idx = balanced_idx[torch.randperm(len(balanced_idx))]

Z_train_bal, y_expert_bal = Z_train[balanced_idx], y_expert[balanced_idx]
batch_size = 250

for epoch in range(1, 16):
    epoch_l_task, epoch_l_aux, ratios = [], [], []
    for i in range(0, len(Z_train_bal), batch_size):
        z_b = Z_train_bal[i:i+batch_size]
        y_b = y_expert_bal[i:i+batch_size]
        
        router_logits = router(z_b)
        expert_probs = torch.softmax(router_logits, dim=-1)
        
        loss_task = criterion(router_logits, y_b)
        loss_aux, f_i, P_i, ratio = auxiliary_load_balancing_loss(expert_probs, alpha=0.05)
        loss = loss_task + loss_aux
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_l_task.append(loss_task.item())
        epoch_l_aux.append(loss_aux.item())
        ratios.append(ratio.item())
        
    print(f"[Epoch {epoch:02d}] L_task={np.mean(epoch_l_task):.4f} | L_aux={np.mean(epoch_l_aux):.4f} | Max/Min Ratio={np.mean(ratios):.2f}")
    if np.mean(ratios) < 1.30:
        print("  ✅ BALANCE DE CARGA PERFECTO: Cociente < 1.30. Penalización evitada.")

torch.save(router.state_dict(), '/workspace/moe_medical_vision/checkpoints/router_a_best.pth')
print("Router Final guardado con éxito. Listo para inferencia.")
