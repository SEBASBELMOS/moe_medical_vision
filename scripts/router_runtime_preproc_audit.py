
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path('/workspace/moe_medical_vision')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_2d_raw(path: str):
    img = Image.open(path).convert('RGB')
    arr = np.array(img).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)
    return x


def load_luna_npz_raw(path: str):
    d = np.load(path)
    vol = torch.from_numpy(d['volume'][0].copy()).float().unsqueeze(0).unsqueeze(0)
    return vol


def load_pancreatic_raw(path: str):
    vol = nib.load(path).get_fdata(dtype=np.float32)
    vol = np.clip(vol, -1000.0, 400.0)
    vol = (vol + 1000.0) / 1400.0
    return torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)


def sample_files():
    out = {}
    out[0] = [str(x) for x in list((ROOT/'data/raw/nih/images_001/images').glob('*.png'))[:10]]
    out[1] = [str(x) for x in list((ROOT/'data/raw/isic/ISIC_2019_Training_Input/ISIC_2019_Training_Input').glob('*.jpg'))[:10]]
    out[2] = [str(x) for x in list((ROOT/'data/raw/osteoporosis/KLGrade/KLGrade').rglob('*.png'))[:10]]
    out[3] = [str(x) for x in list((ROOT/'data/processed/luna16_highres').glob('val_*.npz'))[:10]]
    out[4] = [str(x) for x in list((ROOT/'data/raw/pancreatic').glob('*.nii.gz'))[:10]]
    return out


class RouterLinear(nn.Module):
    def __init__(self, in_dim=192, n_experts=5):
        super().__init__()
        self.gate = nn.Linear(in_dim, n_experts)
    def forward(self, z):
        return self.gate(z)


def route_knn(moe, x_router):
    with torch.no_grad():
        z = moe.backbone.forward_features(x_router)[:,0,:].cpu().numpy()
    z_norm = z / np.linalg.norm(z, axis=1, keepdims=True)
    z_pca = moe.pca.transform(z_norm).astype(np.float32)
    import faiss
    faiss.normalize_L2(z_pca)
    D, I = moe.knn_index.search(z_pca, 5)
    labels = moe.knn_labels[I[0]]
    pred = int(np.bincount(labels, minlength=5).argmax())
    probs = np.zeros(5)
    for v in labels: probs[v] += 0.2
    return pred, probs.tolist()


def route_linear(backbone, router, x_router):
    with torch.no_grad():
        z = backbone.forward_features(x_router)[:,0,:]
        probs = torch.softmax(router(z), dim=-1)
        pred = int(torch.argmax(probs, dim=-1).item())
    return pred, probs[0].cpu().tolist()


def summarize(results):
    matrix = [[0]*5 for _ in range(5)]
    correct = 0
    for r in results:
        matrix[r['true_expert']][r['pred_expert']] += 1
        correct += int(r['true_expert']==r['pred_expert'])
    return {'n': len(results), 'routing_accuracy': correct/len(results), 'confusion_matrix': matrix, 'results': results}


def main():
    import sys
    sys.path.insert(0, str(ROOT/'src'))
    from models.moe_system import MoE_System
    from data.adaptive_preprocessor import AdaptivePreprocessor
    import timm

    pre = AdaptivePreprocessor().to(DEVICE)
    moe = MoE_System(device=DEVICE)
    moe.load_all_weights(ROOT/'checkpoints')
    moe.eval()

    backbone = timm.create_model('vit_tiny_patch16_224', pretrained=True).to(DEVICE)
    backbone.eval()
    for p in backbone.parameters(): p.requires_grad=False
    router = RouterLinear(192,5).to(DEVICE)
    ck = torch.load(ROOT/'checkpoints/router_a_finetuned.pth', map_location='cpu', weights_only=False)
    sd = ck.get('model_state_dict', ck)
    clean = {}
    for k,v in sd.items():
        if k.startswith('router.'):
            clean[k.replace('router.','')] = v
        elif k.startswith('gate.'):
            clean[k] = v
        elif k in ['weight','bias']:
            clean['gate.'+k] = v
    if not clean:
        clean = sd
    router.load_state_dict(clean, strict=False)
    router.eval()

    samples = sample_files()
    knn_results=[]; lin_results=[]
    for true_idx, paths in samples.items():
        for p in paths:
            if true_idx in [0,1,2]:
                raw = load_2d_raw(p).to(DEVICE)
                x_for_router = pre(raw)
            elif true_idx == 3:
                raw3d = load_luna_npz_raw(p).to(DEVICE)
                proc3d = pre(raw3d)
                D = proc3d.shape[2]
                x_for_router = proc3d[:, :, D//2, :, :]
                if x_for_router.shape[1] == 1:
                    x_for_router = x_for_router.repeat(1,3,1,1)
                x_for_router = F.interpolate(x_for_router, size=(224,224), mode='bilinear', align_corners=False)
            else:
                raw3d = load_pancreatic_raw(p).to(DEVICE)
                proc3d = pre(raw3d)
                D = proc3d.shape[2]
                x_for_router = proc3d[:, :, D//2, :, :]
                if x_for_router.shape[1] == 1:
                    x_for_router = x_for_router.repeat(1,3,1,1)
                x_for_router = F.interpolate(x_for_router, size=(224,224), mode='bilinear', align_corners=False)
            pred, probs = route_knn(moe, x_for_router)
            knn_results.append({'true_expert': true_idx, 'pred_expert': pred, 'path': p, 'probs': probs})
            pred2, probs2 = route_linear(backbone, router, x_for_router)
            lin_results.append({'true_expert': true_idx, 'pred_expert': pred2, 'path': p, 'probs': probs2})

    out = {'knn': summarize(knn_results), 'linear': summarize(lin_results)}
    out_path = ROOT/'checkpoints/metrics/router_runtime_preproc_audit.json'
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print('saved', out_path)

if __name__ == '__main__':
    main()
