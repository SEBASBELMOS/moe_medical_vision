"""
LUNA-v6: Higher resolution (96^3) + aggressive regularization + SWA
Hypothesis: 64^3 loses too much nodule spatial detail. 96^3 preserves more signal.
Combined with aggressive regularization to prevent the overfitting seen in v1.
Strategy:
  - 96^3 resolution (up from 64^3)
  - R3D-18 pretrained, train from scratch (not finetune from v1)
  - Higher dropout (0.4), weight_decay=1e-2
  - Strong augmentation pipeline
  - Mixup alpha=0.4
  - Label smoothing 0.1
  - Cosine annealing with warm restarts (T_0=10)
  - Gradient accumulation 8 (effective batch=16)
  - 40 epochs, patience 15
"""
import sys
from pathlib import Path
sys.path.insert(0, '/workspace/moe_medical_vision/src')

import torch
import torch.nn as nn
from data.datasets import get_dataloader, get_transform_3d_strong
from models.experts_3d import R3D18Expert
from train.train_3d import seed_everything, fit_3d_expert, sanity_check_single_batch

seed_everything(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device={device}')

CHECKPOINT_DIR = Path('/workspace/moe_medical_vision/checkpoints')
ROOT = '/workspace/moe_medical_vision/data/raw/luna16'
SIZE = (96, 96, 96)

print('\n' + '='*80)
print('TRAINING EXPERT 4 - LUNA V6 (96^3, R3D-18, strong reg, from scratch)')
print('='*80)

# 96^3 resolution with strong augmentation
train_loader, train_ds = get_dataloader(
    'luna16', ROOT, split='train', batch_size=2, num_workers=2,
    transform=get_transform_3d_strong(split='train', size=SIZE)
)
val_loader, val_ds = get_dataloader(
    'luna16', ROOT, split='val', batch_size=2, num_workers=2,
    transform=get_transform_3d_strong(split='val', size=SIZE)
)

print('train_counts=', train_ds.class_counts.tolist())
print('val_counts=', val_ds.class_counts.tolist())

# Higher dropout model
model = R3D18Expert(
    num_classes=2,
    in_channels=1,
    pretrained=True,
    dropout=0.4,
    use_gradient_checkpointing=True,
).to(device)

# Weighted CE with label smoothing
criterion = nn.CrossEntropyLoss(
    weight=train_ds.get_class_weights().to(device),
    label_smoothing=0.1,
)

print('LUNA-v6 sanity:', sanity_check_single_batch(model, train_loader, criterion, device))

# Check VRAM
mem = torch.cuda.memory_allocated() / 1024**3
print(f'VRAM after sanity: {mem:.2f} GB')

# AdamW with high weight decay
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

# Cosine annealing with warm restarts
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

result = fit_3d_expert(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    epochs=40,
    accum_steps=8,
    mixed_precision=True,
    patience=15,
    checkpoint_path=CHECKPOINT_DIR / 'expert4_luna16_r3d18_v6_best.pth',
    use_mixup=True,
    mixup_alpha=0.4,
)

print(f'luna_v6_result= best_val_f1={result[best_val_f1]:.4f} best_epoch={result[best_epoch]}')
print('DONE')
