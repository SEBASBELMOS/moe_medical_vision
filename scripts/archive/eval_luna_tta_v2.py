"""
Evaluate LUNA16 v1 with expanded TTA (16 views + threshold calibration).
"""
import sys
from pathlib import Path
sys.path.insert(0, '/workspace/moe_medical_vision/src')

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, classification_report
from data.datasets import get_dataloader, get_transform_3d
from models.experts_3d import build_luna_expert
from train.train_3d import seed_everything

seed_everything(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

CHECKPOINT_DIR = Path('/workspace/moe_medical_vision/checkpoints')
ROOT = '/workspace/moe_medical_vision/data/raw/luna16'

model = build_luna_expert(pretrained=False, use_gradient_checkpointing=False).to(device)
ckpt = torch.load(CHECKPOINT_DIR / 'expert4_luna16_r3d18_best.pth', map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f'Loaded v1 checkpoint: val_f1={ckpt["best_val_f1"]:.4f}')

val_loader, val_ds = get_dataloader(
    'luna16', ROOT, split='val', batch_size=1, num_workers=2,
    transform=get_transform_3d(split='val', size=(64, 64, 64))
)

def tta_predict_expanded(model, volume, device):
    """Run expanded TTA with 16 views."""
    model.eval()
    all_logits = []
    
    with torch.no_grad():
        v = volume.to(device)
        # Original
        all_logits.append(model(v))
        
        # All single-axis flips
        for dim in [2, 3, 4]:
            all_logits.append(model(torch.flip(v, dims=[dim])))
        
        # All double-axis flips
        for d1, d2 in [(2,3), (2,4), (3,4)]:
            all_logits.append(model(torch.flip(v, dims=[d1, d2])))
        
        # Triple flip
        all_logits.append(model(torch.flip(v, dims=[2,3,4])))
        
        # Intensity variations
        for factor in [0.92, 0.96, 1.04, 1.08]:
            v_mod = (v * factor).clamp(0, 1)
            all_logits.append(model(v_mod))
        
        # Gaussian noise (multiple seeds)
        for std in [0.015, 0.025, 0.035]:
            v_noisy = (v + torch.randn_like(v) * std).clamp(0, 1)
            all_logits.append(model(v_noisy))
        
        # Gamma correction
        for gamma in [0.9, 1.1]:
            v_gamma = v.clamp(0.001, 1.0).pow(gamma)
            all_logits.append(model(v_gamma))

    # Average probabilities (softmax then average)
    all_probs = torch.stack([F.softmax(l, dim=1) for l in all_logits])
    avg_probs = all_probs.mean(dim=0)
    return avg_probs

print(f'\n=== Expanded TTA evaluation ({len([0]*17)} views) ===')
all_probs_list = []
labels_all = []
for batch in val_loader:
    x = batch['image']
    y = batch['label']
    avg_probs = tta_predict_expanded(model, x, device)
    all_probs_list.append(avg_probs.cpu())
    labels_all.extend(y.numpy().tolist())

all_probs_tensor = torch.cat(all_probs_list, dim=0)
labels_arr = np.array(labels_all)

# Standard threshold (argmax)
preds_std = all_probs_tensor.argmax(dim=1).numpy()
f1_std = f1_score(labels_arr, preds_std, average='macro')
print(f'TTA (argmax): F1_macro={f1_std:.4f}')
print(classification_report(labels_arr, preds_std, target_names=['No_Nodule', 'Nodule']))

# Threshold calibration: sweep threshold for class 1
print('=== Threshold calibration ===')
best_f1 = 0
best_thresh = 0.5
for thresh in np.arange(0.30, 0.70, 0.01):
    preds_t = (all_probs_tensor[:, 1] >= thresh).long().numpy()
    f1_t = f1_score(labels_arr, preds_t, average='macro')
    if f1_t > best_f1:
        best_f1 = f1_t
        best_thresh = thresh

print(f'Best threshold: {best_thresh:.2f} -> F1_macro={best_f1:.4f}')
preds_best = (all_probs_tensor[:, 1] >= best_thresh).long().numpy()
print(classification_report(labels_arr, preds_best, target_names=['No_Nodule', 'Nodule']))
print('DONE')
