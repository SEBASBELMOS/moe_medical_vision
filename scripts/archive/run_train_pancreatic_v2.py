import sys
from pathlib import Path
sys.path.insert(0, '/workspace/moe_medical_vision/src')

import torch
from data.datasets import get_dataloader, get_transform_3d
from losses import FocalLoss
from models.experts_3d import build_pancreatic_expert
from train.train_3d import seed_everything, fit_3d_expert, sanity_check_single_batch

seed_everything(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device={device}')
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'gpu={props.name} vram={props.total_memory/1024**3:.2f}GB')

CHECKPOINT_DIR = Path('/workspace/moe_medical_vision/checkpoints')
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_ROOT = '/workspace/moe_medical_vision/data/processed/pancreatic_roi_v2'

print('\n' + '='*80)
print('TRAINING EXPERT 5 - PANCREATIC V2 (ROI)')
print('='*80)
train_loader_pan, train_ds_pan = get_dataloader(
    'pancreatic', PROCESSED_ROOT, split='train', batch_size=1, num_workers=2,
    transform=get_transform_3d(split='train', augment=True)
)
val_loader_pan, val_ds_pan = get_dataloader(
    'pancreatic', PROCESSED_ROOT, split='val', batch_size=1, num_workers=2,
    transform=get_transform_3d(split='val', augment=False)
)
print('train_counts=', train_ds_pan.class_counts.tolist())
print('val_counts=', val_ds_pan.class_counts.tolist())
alpha_pan = train_ds_pan.get_focal_alpha()
print('alpha_pan=', alpha_pan)
model_pan = build_pancreatic_expert(pretrained=True, use_gradient_checkpointing=True).to(device)
criterion_pan = FocalLoss(gamma=2.0, alpha=alpha_pan)
print('pancreatic_v2 sanity:', sanity_check_single_batch(model_pan, train_loader_pan, criterion_pan, device))
optimizer_pan = torch.optim.AdamW(model_pan.parameters(), lr=7e-5, weight_decay=3e-4)
scheduler_pan = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_pan, mode='min', factor=0.5, patience=3)
pan_result = fit_3d_expert(
    model=model_pan,
    train_loader=train_loader_pan,
    val_loader=val_loader_pan,
    criterion=criterion_pan,
    optimizer=optimizer_pan,
    scheduler=scheduler_pan,
    device=device,
    epochs=20,
    accum_steps=8,
    mixed_precision=True,
    patience=7,
    checkpoint_path=CHECKPOINT_DIR / 'expert5_pancreatic_r3d18_roi_v2_best.pth',
)
print('pan_v2_result=', pan_result['best_val_f1'], pan_result['best_epoch'])
print('DONE')
