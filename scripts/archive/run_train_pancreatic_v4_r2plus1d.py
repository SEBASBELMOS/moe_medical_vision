"""
PANC-v4: R(2+1)D-18 at 96^3 resolution with FocalLoss + weighted sampler + stronger aug
Hypothesis: Higher resolution + better backbone + stronger regularization on small dataset.
"""
import sys
from pathlib import Path
sys.path.insert(0, '/workspace/moe_medical_vision/src')

import torch
from data.datasets import get_dataloader, get_transform_3d
from losses import FocalLoss
from models.experts_3d import build_pancreatic_expert_r2plus1d
from train.train_3d import seed_everything, fit_3d_expert, sanity_check_single_batch

seed_everything(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device={device}')
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'gpu={props.name} vram={props.total_memory/1024**3:.2f}GB')

CHECKPOINT_DIR = Path('/workspace/moe_medical_vision/checkpoints')
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
ROOT = '/workspace/moe_medical_vision/data/raw/pancreatic'

SIZE = (96, 96, 96)

print('\n' + '='*80)
print('TRAINING EXPERT 5 - PANCREATIC V4 (R2Plus1D-18 @ 96^3)')
print('='*80)

train_loader, train_ds = get_dataloader(
    'pancreatic', ROOT, split='train', batch_size=1, num_workers=2,
    transform=get_transform_3d(split='train', augment=True, size=SIZE)
)
val_loader, val_ds = get_dataloader(
    'pancreatic', ROOT, split='val', batch_size=1, num_workers=2,
    transform=get_transform_3d(split='val', augment=False, size=SIZE)
)

print('train_counts=', train_ds.class_counts.tolist())
print('val_counts=', val_ds.class_counts.tolist())

alpha_pan = train_ds.get_focal_alpha()
print('alpha_pan=', alpha_pan)

model = build_pancreatic_expert_r2plus1d(pretrained=True, use_gradient_checkpointing=True).to(device)

# FocalLoss with dataset-derived alpha
criterion = FocalLoss(gamma=2.0, alpha=alpha_pan)

print('PANC-v4 sanity:', sanity_check_single_batch(model, train_loader, criterion, device))

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=2e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

result = fit_3d_expert(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    epochs=30,
    accum_steps=8,
    mixed_precision=True,
    patience=12,
    checkpoint_path=CHECKPOINT_DIR / 'expert5_pancreatic_r2plus1d_v4_best.pth',
)

print(f'panc_v4_result= best_val_f1={result["best_val_f1"]:.4f} best_epoch={result["best_epoch"]}')
print('DONE')
