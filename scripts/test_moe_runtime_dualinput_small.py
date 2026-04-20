
import sys
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
import nibabel as nib
from PIL import Image

sys.path.insert(0, "/workspace/moe_medical_vision/src")
from models.moe_system import MoE_System
from data.datasets import get_dataloader

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Cargando MoE_System y los 5 Expertos Definitivos...")
model = MoE_System(device=device)
model.load_all_weights("/workspace/moe_medical_vision/checkpoints")
model.eval()


def process_router_2d_image(path: str):
    img = Image.open(path).convert('RGB').resize((224,224))
    arr = np.array(img).astype(np.float32)/255.0
    arr = (arr - [0.485,0.456,0.406]) / [0.229,0.224,0.225]
    return torch.from_numpy(arr).float().permute(2,0,1).unsqueeze(0)


def process_luna_npz(file_path):
    d = np.load(file_path)
    vol = torch.from_numpy(d["volume"][0].copy()).float()
    y = int(d["label"])
    x_3d = vol.unsqueeze(0).unsqueeze(0)
    mip = torch.stack([vol.max(dim=0)[0], vol.mean(dim=0), vol.std(dim=0)], dim=0)
    x_router = F.interpolate(mip.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False)
    return x_router, x_3d, y


def process_pancreatic_nii(file_path: Path):
    vol = nib.load(str(file_path)).get_fdata(dtype=np.float32)
    vol = np.clip(vol, -1000.0, 400.0)
    vol = (vol + 1000.0) / 1400.0
    x_3d = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)
    x_3d = F.interpolate(x_3d, size=(64,64,64), mode='trilinear', align_corners=False)
    y = 1 if '_1' in file_path.stem or 'cancer' in file_path.stem.lower() else 0
    x_router = torch.zeros((1,3,224,224), dtype=torch.float32)
    return x_router, x_3d, y


def route_only(model, x_router, x_3d=None):
    x_for_router = model._prepare_for_router(x_router.to(device), x_3d.to(device) if x_3d is not None else None)
    with torch.no_grad():
        z = model.backbone.forward_features(x_for_router)[:, 0, :]
        if model.router_linear_ready:
            logits = model.router_linear(z)
            probs = torch.softmax(logits, dim=-1)
            best_idx = int(torch.argmax(probs, dim=-1).item())
        else:
            z_np = z.cpu().numpy()
            z_norm = z_np / np.linalg.norm(z_np, axis=1, keepdims=True)
            z_pca = model.pca.transform(z_norm).astype(np.float32)
            import faiss
            faiss.normalize_L2(z_pca)
            _, I = model.knn_index.search(z_pca, 5)
            neighbors_labels = model.knn_labels[I[0]]
            best_idx = int(np.bincount(neighbors_labels, minlength=5).argmax())
    return best_idx


def run_expert(model, best_idx, x_expert, x_3d=None):
    expert_input = x_expert.to(device)
    if best_idx in [3,4]:
        expert_input = x_3d.to(device)
        if best_idx == 3 and expert_input.shape[1] == 1:
            expert_input = expert_input.repeat(1,3,1,1,1)
            mean = torch.tensor([0.43216, 0.394666, 0.37645], device=device).view(1,3,1,1,1)
            std = torch.tensor([0.22803, 0.22145, 0.216989], device=device).view(1,3,1,1,1)
            expert_input = (expert_input - mean) / std
    with torch.no_grad():
        return model.experts[best_idx](expert_input)

results = {i:{'router_correct':0,'total':0,'y_true':[],'y_pred':[],'classes_pred':set()} for i in range(5)}
print("\nIniciando Test Masivo de 500 Imágenes (100 por Experto)... Esto tomará un minuto.\n")

# 3D experts
files = list(Path('/workspace/moe_medical_vision/data/processed/luna16_highres').glob('val_*.npz'))
selected = random.sample(files, min(20, len(files)))
for f in selected:
    x_router, x_3d, y_true = process_luna_npz(f)
    best_idx = route_only(model, x_router, x_3d=x_3d)
    out = run_expert(model, best_idx, x_router, x_3d=x_3d)
    exp_id=3
    results[exp_id]['total'] += 1
    if best_idx == exp_id: results[exp_id]['router_correct'] += 1
    pred = torch.argmax(out, dim=-1).item()
    results[exp_id]['y_true'].append(y_true)
    results[exp_id]['y_pred'].append(pred)
    results[exp_id]['classes_pred'].add(pred)

files = list(Path('/workspace/moe_medical_vision/data/raw/pancreatic').glob('*.nii.gz'))
selected = random.sample(files, min(20, len(files)))
for f in selected:
    x_router, x_3d, y_true = process_pancreatic_nii(f)
    best_idx = route_only(model, x_router, x_3d=x_3d)
    out = run_expert(model, best_idx, x_router, x_3d=x_3d)
    exp_id=4
    results[exp_id]['total'] += 1
    if best_idx == exp_id: results[exp_id]['router_correct'] += 1
    pred = torch.argmax(out, dim=-1).item()
    results[exp_id]['y_true'].append(y_true)
    results[exp_id]['y_pred'].append(pred)
    results[exp_id]['classes_pred'].add(pred)

# 2D experts
raw_resolvers = {
    0: lambda item: process_router_2d_image(item['image_path']),
    1: lambda item: process_router_2d_image(item['image_path']),
    2: lambda item: process_router_2d_image(item['image_path']),
}

ds_names = {
    0: ("nih_chestxray", "/workspace/moe_medical_vision/data/raw/nih"),
    1: ("isic2019", "/workspace/moe_medical_vision/data/raw/isic"),
    2: ("osteoarthritis", "/workspace/moe_medical_vision/data/raw/osteoporosis/KLGrade/KLGrade"),
}
for exp_id, (name, root) in ds_names.items():
    try:
        loader, ds = get_dataloader(name, root, split='val', batch_size=1, num_workers=0)
    except Exception as e:
        print(f'[WARN] Could not load {name}: {e}')
        continue
    indices = random.sample(range(len(ds)), min(20, len(ds)))
    for idx in indices:
        item = ds[idx]
        x_expert = item['image'].unsqueeze(0)
        if x_expert.shape[1] == 1:
            x_expert = x_expert.repeat(1,3,1,1)
        if x_expert.shape[-1] != 224:
            x_expert = F.interpolate(x_expert, size=(224,224), mode='bilinear', align_corners=False)
        # recover raw path for router-preprocessing-consistent input
        raw_path = item.get('image_path') if isinstance(item, dict) else None
        if raw_path is None:
            # best effort fallback from dataset internals
            if hasattr(ds, 'df') and 'image_path' in ds.df.columns:
                raw_path = ds.df.iloc[idx]['image_path']
            elif hasattr(ds, 'samples'):
                raw_path = ds.samples[idx][0]
        x_router = process_router_2d_image(str(raw_path))
        y_true = item['label'].numpy() if exp_id==0 else item['label']
        best_idx = route_only(model, x_router)
        out = run_expert(model, best_idx, x_expert)
        results[exp_id]['total'] += 1
        if best_idx == exp_id: results[exp_id]['router_correct'] += 1
        pred = torch.argmax(out, dim=-1).item()
        results[exp_id]['y_true'].append(y_true)
        results[exp_id]['y_pred'].append(pred)
        results[exp_id]['classes_pred'].add(pred)

NAMES={0:'NIH ChestX-ray14',1:'ISIC 2019 (Piel)',2:'Osteoarthritis',3:'LUNA16 (Pulmón)',4:'Pancreatic Cancer'}
print('='*80)
print('  RESULTADOS FINALES DEL PIPELINE END-TO-END (500 IMAGENES)')
print('='*80)

total_router=sum(r['router_correct'] for r in results.values())
total_imgs=sum(r['total'] for r in results.values())
for exp_id in range(5):
    r=results[exp_id]; n=r['total']
    if n==0: continue
    print(f"\n--- Experto {exp_id}: {NAMES[exp_id]} ---")
    print(f"➜ Router Accuracy: {r['router_correct']}/{n} ({(r['router_correct']/n)*100:.1f}%)")
    if exp_id == 0:
        correct_preds=sum(1 for yt, yp in zip(r['y_true'], r['y_pred']) if yt[yp] == 1)
        print(f"➜ Expert Accuracy (Hit Multi-label): {correct_preds}/{n} ({(correct_preds/n)*100:.1f}%)")
    else:
        acc=accuracy_score(r['y_true'], r['y_pred'])
        f1=f1_score(r['y_true'], r['y_pred'], average='macro')
        print(f"➜ Expert Accuracy: {acc*100:.1f}% | Expert F1 Macro: {f1:.4f}")
    if len(r['classes_pred']) == 1:
        print(f"  ⚠️ ¡ALERTA! El modelo predijo la misma clase ({list(r['classes_pred'])[0]}) para las {n} imágenes.")
    else:
        print(f"  ✅ Modelo sano. Predijo {len(r['classes_pred'])} clases diferentes.")
print('\n'+'='*80)
print(f"ROUTING GLOBAL DEL SISTEMA: {total_router}/{total_imgs} ({(total_router/total_imgs)*100:.1f}%)")
print('='*80)
