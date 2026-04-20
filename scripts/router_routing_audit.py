
from __future__ import annotations

import json
from pathlib import Path
import random

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


def process_fast_npz(path: str):
    d=np.load(path)
    vol=d['volume'][0]
    tensor_3d=torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)
    mip=torch.stack([torch.tensor(vol.max(axis=0)), torch.tensor(vol.mean(axis=0)), torch.tensor(vol.std(axis=0))], dim=0)
    tensor_224=F.interpolate(mip.unsqueeze(0), size=(224,224), mode='bilinear', align_corners=False)
    return tensor_224, tensor_3d


def process_3d_volume(path: str):
    if path.endswith('.mhd'):
        import SimpleITK as sitk
        img = sitk.ReadImage(path)
        vol = sitk.GetArrayFromImage(img).astype(np.float32)
    else:
        vol = nib.load(path).get_fdata(dtype=np.float32)
    vol = np.clip(vol, -1000.0, 400.0)
    vol = (vol + 1000.0) / 1400.0
    tensor_3d = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)
    tensor_3d = F.interpolate(tensor_3d, size=(64,64,64), mode='trilinear', align_corners=False)
    x=tensor_3d.squeeze(1)
    mip=torch.stack([x.max(dim=1)[0], x.max(dim=2)[0], x.max(dim=3)[0]], dim=1)
    tensor_224=F.interpolate(mip, size=(224,224), mode='bilinear', align_corners=False)
    return tensor_224, tensor_3d


def sample_files():
    out = {}
    # NIH
    nih = list((ROOT/'data/raw/nih/images_001/images').glob('*.png'))[:10]
    out[0] = [str(x) for x in nih]
    # ISIC
    isic = list((ROOT/'data/raw/isic/ISIC_2019_Training_Input/ISIC_2019_Training_Input').glob('*.jpg'))[:10]
    out[1] = [str(x) for x in isic]
    # Osteo
    osteo = list((ROOT/'data/raw/osteoporosis/KLGrade/KLGrade').rglob('*.png'))[:10]
    out[2] = [str(x) for x in osteo]
    # LUNA
    luna = list((ROOT/'data/processed/luna16_highres').glob('val_*.npz'))[:10]
    out[3] = [str(x) for x in luna]
    # Pancreas
    panc = list((ROOT/'data/raw/pancreatic').glob('*.nii.gz'))[:10]
    out[4] = [str(x) for x in panc]
    return out


def main():
    import sys
    sys.path.insert(0, str(ROOT/'src'))
    from models.moe_system import MoE_System
    model = MoE_System(device=DEVICE)
    model.load_all_weights(ROOT/'checkpoints')
    model.eval()
    samples = sample_files()
    results=[]
    for true_idx, paths in samples.items():
        for p in paths:
            if true_idx in [0,1,2]:
                t224 = process_2d_image(p)
                t3d = None
            elif true_idx == 3:
                t224, t3d = process_fast_npz(p)
            else:
                t224, t3d = process_3d_volume(p)
            with torch.no_grad():
                out, pred_idx, probs = model(t224.to(DEVICE), x_3d=t3d.to(DEVICE) if t3d is not None else None)
            results.append({'true_expert': true_idx, 'pred_expert': int(pred_idx), 'path': p, 'probs': probs[0].cpu().tolist()})
    # summarize
    import collections
    matrix = [[0]*5 for _ in range(5)]
    correct=0
    for r in results:
        matrix[r['true_expert']][r['pred_expert']] += 1
        correct += int(r['true_expert']==r['pred_expert'])
    summary = {
        'n': len(results),
        'routing_accuracy': correct/len(results) if results else None,
        'confusion_matrix': matrix,
        'results': results,
    }
    out_path = ROOT/'checkpoints/metrics/router_routing_audit.json'
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print('saved', out_path)

if __name__ == '__main__':
    main()
