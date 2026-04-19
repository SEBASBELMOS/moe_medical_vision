"""
PANC-v5: Fine-tune from v0 best checkpoint (R3D-18, 0.513)
Strategy: Lower LR + mixup + stronger augmentation + higher weight decay + 
weighted sampler (v0 didn't use one)
Hypothesis: v0 overfit at epoch 1. Mixup + strong aug + careful LR prevents overfitting.
"""
import sys
from pathlib import Path
sys.path.insert(0, '/workspace/moe_medical_vision/src')

import torch
from data.datasets import get_dataloader, get_transform_3d_strong
from losses import FocalLoss
from models.experts_3d import build_pancreatic_expert
from train.train_3d import seed_everything, fit_3d_expert, sanity_check_single_batch

seed_everything(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device={device}')

CHECKPOINT_DIR = Path('/workspace/moe_medical_vision/checkpoints')
ROOT = '/workspace/moe_medical_vision/data/raw/pancreatic'

print('\n' + '='*80)
print('TRAINING EXPERT 5 - PANCREATIC V5 (R3D-18 fine-tune from v0 + mixup + strong aug)')
print('='*80)

# Use STRONG augmentation 
train_loader, train_ds = get_dataloader(
    'pancreatic', ROOT, split='train', batch_size=2, num_workers=2,
    transform=get_transform_3d_strong(split='train', size=(64, 64, 64))
)
val_loader, val_ds = get_dataloader(
    'pancreatic', ROOT, split='val', batch_size=2, num_workers=2,
    transform=get_transform_3d_strong(split='val', size=(64, 64, 64))
)

print('train_counts=', train_ds.class_counts.tolist())
print('val_counts=', val_ds.class_counts.tolist())

# Build model and load v0 best weights
model = build_pancreatic_expert(pretrained=True, use_gradient_checkpointing=True).to(device)
ckpt = torch.load(CHECKPOINT_DIR / 'expert5_pancreatic_r3d18_best.pth', map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
print(f'Loaded v0 checkpoint: val_f1={ckpt["best_val_f1"]:.4f} epoch={ckpt["epoch"]}')

alpha_pan = train_ds.get_focal_alpha()
print('alpha_pan=', alpha_pan)

# FocalLoss (proven better for this dataset)
criterion = FocalLoss(gamma=2.0, alpha=alpha_pan)

print('PANC-v5 sanity:', sanity_check_single_batch(model, train_loader, criterion, device))

# Very low LR for fine-tuning
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-7)

result = fit_3d_expert(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    epochs=25,
    accum_steps=4,
    mixed_precision=True,
    patience=12,
    checkpoint_path=CHECKPOINT_DIR / 'expert5_pancreatic_r3d18_v5_best.pth',
    use_mixup=True,
    mixup_alpha=0.4,
)

print(f'panc_v5_result= best_val_f1={result["best_val_f1"]:.4f} best_epoch={result["best_epoch"]}')
print('DONE')
