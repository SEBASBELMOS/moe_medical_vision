"""
PANCREATIC FINAL — R3D-18 fine-tune from v0 best (0.513)
Strategy: load best checkpoint, FocalLoss, balanced sampler,
augment, long patience.
num_workers=0 to avoid killed workers.
"""
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

CHECKPOINT_DIR = Path('/workspace/moe_medical_vision/checkpoints')
ROOT = '/workspace/moe_medical_vision/data/raw/pancreatic'
CKPT_OUT = CHECKPOINT_DIR / 'expert5_pancreatic_FINAL_best.pth'

print('='*80)
print('PANCREATIC FINAL — R3D-18 fine-tune from v0 best')
print('='*80)

train_loader, train_ds = get_dataloader(
    'pancreatic', ROOT, split='train', batch_size=2, num_workers=4,
    transform=get_transform_3d(split='train', size=(64,64,64), augment=True)
)
val_loader, val_ds = get_dataloader(
    'pancreatic', ROOT, split='val', batch_size=2, num_workers=4,
    transform=get_transform_3d(split='val', size=(64,64,64), augment=False)
)
print('train_counts=', train_ds.class_counts.tolist())
print('val_counts=', val_ds.class_counts.tolist())

# Load best v0 checkpoint
model = build_pancreatic_expert(pretrained=True, use_gradient_checkpointing=True).to(device)
ckpt_v0 = torch.load(CHECKPOINT_DIR / 'expert5_pancreatic_r3d18_best.pth', map_location=device, weights_only=False)
model.load_state_dict(ckpt_v0['model_state_dict'])
print(f"Loaded v0: val_f1={ckpt_v0['best_val_f1']:.4f}")

alpha = train_ds.get_focal_alpha()
print('alpha=', alpha)
criterion = FocalLoss(gamma=2.0, alpha=alpha)

print('sanity:', sanity_check_single_batch(model, train_loader, criterion, device), flush=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=5e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)

print('START_TRAIN_LOOP', flush=True)
result = fit_3d_expert(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    epochs=30,
    accum_steps=4,
    mixed_precision=True,
    patience=12,
    checkpoint_path=CKPT_OUT,
)

print(f"PANC FINAL: best_val_f1={result['best_val_f1']:.4f} best_epoch={result['best_epoch']}")
print('DONE')
