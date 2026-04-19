"""
Task 2.5 - 2.8: Ablation Study del Router (4 Mecanismos)
Compara Linear (DL), GMM, Naive Bayes y k-NN (FAISS) usando los CLS tokens pre-extraídos.
"""
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import faiss

sys.path.insert(0, '/workspace/moe_medical_vision/src')
from train.train_3d import seed_everything

seed_everything(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

EMBEDDINGS_DIR = '/workspace/moe_medical_vision/data/processed/router_embeddings'

print("================================================================================")
print("ABLATION STUDY DEL ROUTER: Cargando CLS Tokens Z_train y Z_val...")
print("================================================================================")

# 1. Cargar Embeddings Masivos
try:
    train_data = np.load(f"{EMBEDDINGS_DIR}/Z_train.npz")
    Z_train, y_train_expert = train_data['z'], train_data['y_expert']
    
    val_data = np.load(f"{EMBEDDINGS_DIR}/Z_val.npz")
    Z_val, y_val_expert = val_data['z'], val_data['y_expert']
except Exception as e:
    print(f"Error cargando los embeddings: {e}")
    print("Asegúrate de que '03_extract_cls_tokens.py' haya terminado exitosamente.")
    sys.exit(1)

print(f"Z_train shape: {Z_train.shape}, y_train_expert: {np.bincount(y_train_expert)}")
print(f"Z_val shape: {Z_val.shape}, y_val_expert: {np.bincount(y_val_expert)}\n")

Z_train_norm = Z_train / np.linalg.norm(Z_train, axis=1, keepdims=True)
Z_val_norm = Z_val / np.linalg.norm(Z_val, axis=1, keepdims=True)

results = []

def log_result(name, acc, latency, params, vram, is_dl=False):
    results.append({
        'Router': name,
        'Routing Acc.': f"{acc*100:.2f}%",
        'Latencia (ms)': f"{latency:.2f} ms",
        'Parámetros': params,
        'Gradiente': 'Sí' if is_dl else 'No',
        'VRAM': vram
    })
    print(f"[{name}] Acc: {acc*100:.2f}% | Latencia: {latency:.2f} ms | Params: {params}")

# =================================================================================
# ROUTER A: ViT + Linear + Softmax (Deep Learning Baseline)
# =================================================================================
print("--- Entrenando Router A (Linear + Softmax) ---")
class LinearGatingHead(nn.Module):
    def __init__(self, d_model, n_experts):
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts)
    def forward(self, z):
        return self.gate(z)

d_model = Z_train.shape[1]
n_experts = 5
router_dl = LinearGatingHead(d_model, n_experts).to(device)

train_dataset = TensorDataset(torch.tensor(Z_train).float(), torch.tensor(y_train_expert).long())
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_dataset = TensorDataset(torch.tensor(Z_val).float(), torch.tensor(y_val_expert).long())
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

optimizer = torch.optim.AdamW(router_dl.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

router_dl.train()
for epoch in range(15):
    for z_b, y_b in train_loader:
        z_b, y_b = z_b.to(device), y_b.to(device)
        logits = router_dl(z_b)
        loss = criterion(logits, y_b)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

router_dl.eval()
all_preds = []
start_time = time.time()
with torch.no_grad():
    for z_b, y_b in val_loader:
        logits = router_dl(z_b.to(device))
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
infer_time_ms = ((time.time() - start_time) / len(Z_val)) * 1000
acc_a = accuracy_score(y_val_expert, all_preds)
params_a = sum(p.numel() for p in router_dl.parameters())
log_result("ViT + Linear", acc_a, infer_time_ms, f"{params_a}", "~1.5 GB", is_dl=True)

# =================================================================================
# ROUTER B: GMM (Gaussian Mixture Model)
# =================================================================================
print("\n--- Ajustando Router B (GMM 5 comp.) ---")
N_SAMPLES_GMM = min(50000, len(Z_train_norm))
idx_gmm = np.random.choice(len(Z_train_norm), N_SAMPLES_GMM, replace=False)

gmm_models = []
for i in range(n_experts):
    Z_class = Z_train_norm[y_train_expert == i]
    # Limitar
    Z_class = Z_class[:10000]
    g = GaussianMixture(n_components=1, covariance_type='diag', random_state=42)
    g.fit(Z_class)
    gmm_models.append(g)

start_time = time.time()
gmm_scores = np.stack([g.score_samples(Z_val_norm) for g in gmm_models], axis=1) # [N_val, 5]
gmm_preds = np.argmax(gmm_scores, axis=1)
infer_time_ms = ((time.time() - start_time) / len(Z_val_norm)) * 1000
acc_b = accuracy_score(y_val_expert, gmm_preds)
params_b = n_experts * (d_model * 2)
log_result("ViT + GMM", acc_b, infer_time_ms, f"{params_b} (Diag Cov)", "~50 MB", is_dl=False)

# =================================================================================
# ROUTER C: Naive Bayes (MLE Analítico)
# =================================================================================
print("\n--- Ajustando Router C (Naive Bayes) ---")
nb = GaussianNB()
nb.fit(Z_train_norm, y_train_expert)

start_time = time.time()
nb_preds = nb.predict(Z_val_norm)
infer_time_ms = ((time.time() - start_time) / len(Z_val_norm)) * 1000
acc_c = accuracy_score(y_val_expert, nb_preds)
params_c = n_experts * d_model * 2
log_result("ViT + Naive Bayes", acc_c, infer_time_ms, f"{params_c}", "~5 MB", is_dl=False)

# =================================================================================
# ROUTER D: k-NN con FAISS (No Paramétrico) + PCA 
# =================================================================================
print("\n--- Ajustando Router D (k-NN con FAISS) ---")
d_pca = 32
print(f"Aplicando PCA (d={d_pca}) a los CLS Tokens para k-NN...")
pca = PCA(n_components=d_pca, random_state=42)
Z_train_pca = pca.fit_transform(Z_train_norm).astype(np.float32)
Z_val_pca = pca.transform(Z_val_norm).astype(np.float32)

faiss.normalize_L2(Z_train_pca)
faiss.normalize_L2(Z_val_pca)

index = faiss.IndexFlatIP(d_pca)
index.add(Z_train_pca)

k_neighbors = 5
start_time = time.time()
D, I = index.search(Z_val_pca, k_neighbors)
neighbors_labels = y_train_expert[I]
# Majority voting
knn_preds = np.apply_along_axis(lambda x: np.bincount(x, minlength=n_experts).argmax(), axis=1, arr=neighbors_labels)
infer_time_ms = ((time.time() - start_time) / len(Z_val_pca)) * 1000

acc_d = accuracy_score(y_val_expert, knn_preds)
params_d = f"{len(Z_train_pca)}x{d_pca}"
mem_faiss = len(Z_train_pca) * d_pca * 4 / (1024**2)
log_result("ViT + k-NN (FAISS)", acc_d, infer_time_ms, params_d, f"~{mem_faiss:.1f} MB", is_dl=False)

# =================================================================================
# TABLA COMPARATIVA
# =================================================================================
print("\n" + "="*80)
print("TABLA COMPARATIVA DEL ABLATION STUDY (ROUTER)")
print("="*80)
df_results = pd.DataFrame(results)
print(df_results.to_markdown(index=False))
print("="*80)

df_results.to_csv("/workspace/moe_medical_vision/ablation_study_results.csv", index=False)

best_idx = np.argmax([acc_a, acc_b, acc_c, acc_d])
ganadores = ["Router A (Linear)", "Router B (GMM)", "Router C (Naive Bayes)", "Router D (k-NN FAISS)"]
print(f"\nEL ROUTER GANADOR ES: {ganadores[best_idx]}")

if best_idx == 0:
    torch.save(router_dl.state_dict(), "/workspace/moe_medical_vision/checkpoints/router_a_best.pth")

