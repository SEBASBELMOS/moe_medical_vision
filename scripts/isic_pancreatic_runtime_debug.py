
import sys, random, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib

sys.path.insert(0, '/workspace/moe_medical_vision/src')
from models.moe_system import MoE_System
from data.datasets import get_dataloader

device='cuda' if torch.cuda.is_available() else 'cpu'
model=MoE_System(device=device)
model.load_all_weights('/workspace/moe_medical_vision/checkpoints')
model.eval()

# ISIC diagnostics
_, ds_isic = get_dataloader('isic2019','/workspace/moe_medical_vision/data/raw/isic', split='val', batch_size=1, num_workers=0)
idxs = random.sample(range(len(ds_isic)), 10)
res={'isic':[],'pancreatic':[]}
for idx in idxs:
    item = ds_isic[idx]
    x_exp = item['image'].unsqueeze(0)
    if x_exp.shape[1]==1: x_exp = x_exp.repeat(1,3,1,1)
    if x_exp.shape[-1]!=224: x_exp = F.interpolate(x_exp, size=(224,224), mode='bilinear', align_corners=False)
    # router input: use same tensor for now
    with torch.no_grad():
        out, best_idx, probs = model(x_exp.to(device))
    pred = int(torch.argmax(out, dim=-1).item())
    res['isic'].append({'idx': idx, 'true_label': int(item['label']), 'router_pred': int(best_idx), 'expert_pred': pred, 'router_probs': probs[0].cpu().tolist()})

# Pancreatic diagnostics
_, ds_pan = get_dataloader('pancreatic','/workspace/moe_medical_vision/data/raw/pancreatic', split='val', batch_size=1, num_workers=0)
idxs = random.sample(range(len(ds_pan)), 10)
for idx in idxs:
    item = ds_pan[idx]
    x_3d = item['image'].unsqueeze(0)
    x_dummy = torch.zeros((1,3,224,224), dtype=torch.float32)
    with torch.no_grad():
        out, best_idx, probs = model(x_dummy.to(device), x_3d=x_3d.to(device))
    pred = int(torch.argmax(out, dim=-1).item())
    y_true = int(item['label'])
    res['pancreatic'].append({'idx': idx, 'true_label': y_true, 'router_pred': int(best_idx), 'expert_pred': pred, 'router_probs': probs[0].cpu().tolist()})

out_path = Path('/workspace/moe_medical_vision/checkpoints/metrics/isic_pancreatic_runtime_debug.json')
out_path.write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
print('saved', out_path)
