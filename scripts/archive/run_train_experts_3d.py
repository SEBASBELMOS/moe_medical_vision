import sys
from pathlib import Path

sys.path.insert(0, '/workspace/moe_medical_vision/src')

import torch
from data.datasets import get_dataloader
from losses import FocalLoss
from models.experts_3d import build_luna_expert, build_pancreatic_expert
from train.train_3d import seed_everything, fit_3d_expert, sanity_check_single_batch

seed_everything(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device={device}')
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'gpu={props.name} vram={props.total_memory/1024**3:.2f}GB')

CHECKPOINT_DIR = Path('/workspace/moe_medical_vision/checkpoints')
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ---- LUNA ----
print('\n' + '='*80)
print('TRAINING EXPERT 4 - LUNA16')
print('='*80)
train_loader_luna, train_ds_luna = get_dataloader('luna16', '/workspace/moe_medical_vision/data/raw/luna16', split='train', batch_size=1, num_workers=2)
val_loader_luna, val_ds_luna = get_dataloader('luna16', '/workspace/moe_medical_vision/data/raw/luna16', split='val', batch_size=1, num_workers=2)
model_luna = build_luna_expert(pretrained=True, use_gradient_checkpointing=True).to(device)
criterion_luna = torch.nn.CrossEntropyLoss(weight=train_ds_luna.get_class_weights().to(device))
print('luna sanity:', sanity_check_single_batch(model_luna, train_loader_luna, criterion_luna, device))
optimizer_luna = torch.optim.AdamW(model_luna.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler_luna = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_luna, mode='min', factor=0.5, patience=2)
luna_result = fit_3d_expert(
    model=model_luna,
    train_loader=train_loader_luna,
    val_loader=val_loader_luna,
    criterion=criterion_luna,
    optimizer=optimizer_luna,
    scheduler=scheduler_luna,
    device=device,
    epochs=16,
    accum_steps=4,
    mixed_precision=True,
    patience=6,
    checkpoint_path=CHECKPOINT_DIR / 'expert4_luna16_r3d18_best.pth',
)
print('luna_result=', luna_result['best_val_f1'], luna_result['best_epoch'])

del model_luna, train_loader_luna, val_loader_luna, train_ds_luna, val_ds_luna, optimizer_luna, scheduler_luna
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ---- PANCREATIC ----
print('\n' + '='*80)
print('TRAINING EXPERT 5 - PANCREATIC')
print('='*80)
train_loader_pan, train_ds_pan = get_dataloader('pancreatic', '/workspace/moe_medical_vision/data/raw/pancreatic', split='train', batch_size=1, num_workers=2)
val_loader_pan, val_ds_pan = get_dataloader('pancreatic', '/workspace/moe_medical_vision/data/raw/pancreatic', split='val', batch_size=1, num_workers=2)
model_pan = build_pancreatic_expert(pretrained=True, use_gradient_checkpointing=True).to(device)
criterion_pan = FocalLoss(gamma=2.0, alpha=train_ds_pan.get_focal_alpha())
print('pancreatic alpha=', train_ds_pan.get_focal_alpha())
print('pancreatic sanity:', sanity_check_single_batch(model_pan, train_loader_pan, criterion_pan, device))
optimizer_pan = torch.optim.AdamW(model_pan.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler_pan = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_pan, mode='min', factor=0.5, patience=2)
pan_result = fit_3d_expert(
    model=model_pan,
    train_loader=train_loader_pan,
    val_loader=val_loader_pan,
    criterion=criterion_pan,
    optimizer=optimizer_pan,
    scheduler=scheduler_pan,
    device=device,
    epochs=15,
    accum_steps=8,
    mixed_precision=True,
    patience=5,
    checkpoint_path=CHECKPOINT_DIR / 'expert5_pancreatic_r3d18_best.pth',
)
print('pan_result=', pan_result['best_val_f1'], pan_result['best_epoch'])

print('\nDONE')
for f in sorted(CHECKPOINT_DIR.glob('expert*_best.pth')):
    print(f.name, round(f.stat().st_size / 1e6, 1), 'MB')
