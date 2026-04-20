
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path('/workspace/moe_medical_vision')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def process_2d_image(path: str):
    img = Image.open(path).convert('RGB').resize((224,224))
    arr = np.array(img).astype(np.float32)/255.0
    arr = (arr - [0.485,0.456,0.406]) / [0.229,0.224,0.225]
    return torch.from_numpy(arr).float().permute(2,0,1).unsqueeze(0)


def process_luna_npz(path: str):
    d = np.load(path)
    vol = torch.from_numpy(d['volume'][0].copy()).float()
    x_3d = vol.unsqueeze(0).unsqueeze(0)
    mip = torch.stack([vol.max(dim=0)[0], vol.mean(dim=0), vol.std(dim=0)], dim=0)
    x_224 = F.interpolate(mip.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
    return x_224, x_3d


def process_pancreatic_nii(path: str):
    vol = nib.load(str(path)).get_fdata(dtype=np.float32)
    vol = np.clip(vol, -1000.0, 400.0)
    vol = (vol + 1000.0) / 1400.0
    x_3d = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)
    x_3d = F.interpolate(x_3d, size=(64,64,64), mode='trilinear', align_corners=False)
    x_dummy = torch.zeros((1,3,224,224), dtype=torch.float32)
    return x_dummy, x_3d


def describe(t):
    return {
        'shape': list(t.shape),
        'min': float(t.min().item()),
        'max': float(t.max().item()),
        'mean': float(t.mean().item()),
        'std': float(t.std().item()),
    }


def main():
    import sys
    sys.path.insert(0, str(ROOT/'src'))
    from models.moe_system import MoE_System
    model = MoE_System(device=DEVICE)
    model.load_all_weights(ROOT/'checkpoints')
    model.eval()

    samples = {
        'nih': process_2d_image('/workspace/moe_medical_vision/data/raw/nih/images_001/images/00000001_000.png'),
        'isic': process_2d_image('/workspace/moe_medical_vision/data/raw/isic/ISIC_2019_Training_Input/ISIC_2019_Training_Input/ISIC_0000000.jpg'),
        'osteo': process_2d_image('/workspace/moe_medical_vision/data/raw/osteoporosis/KLGrade/KLGrade/0/NormalG0 (1).png'),
        'luna': process_luna_npz('/workspace/moe_medical_vision/data/processed/luna16_highres/val_00000.npz'),
        'pancreatic': process_pancreatic_nii('/workspace/moe_medical_vision/data/raw/pancreatic/100000_00001_0000.nii.gz'),
    }

    out = {}
    with torch.no_grad():
        for name, tup in samples.items():
            x_224 = tup[0].to(DEVICE)
            x_3d = tup[1].to(DEVICE) if len(tup) > 1 and tup[1] is not None else None
            x_for_router = model._prepare_for_router(x_224, x_3d)
            z = model.backbone.forward_features(x_for_router)[:,0,:]
            if model.router_linear_ready:
                logits = model.router_linear(z)
                probs = torch.softmax(logits, dim=-1)
                pred = int(torch.argmax(probs, dim=-1).item())
            else:
                pred = None
                probs = None
            out[name] = {
                'x_224': describe(x_224),
                'x_3d': describe(x_3d) if x_3d is not None else None,
                'x_for_router': describe(x_for_router),
                'pred_expert': pred,
                'probs': probs[0].cpu().tolist() if probs is not None else None,
            }
    out_path = ROOT/'checkpoints/metrics/router_runtime_input_debug.json'
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print('saved', out_path)

if __name__ == '__main__':
    main()
