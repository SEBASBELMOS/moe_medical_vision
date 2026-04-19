"""
LUNA FINAL — R3D-18 fine-tune from v1 best (0.578)
Strategy: load best checkpoint, custom 2-layer head, heavy regularization,
weighted CE, balanced sampler, augment, long patience.
num_workers=0 to avoid killed workers.
"""
import sys
from pathlib import Path
sys.path.insert(0, '/workspace/moe_medical_vision/src')

import torch
import torch.nn as nn
from data.datasets import get_dataloader, get_transform_3d
from models.experts_3d import build_luna_expert
from train.train_3d import seed_everything, fit_3d_expert, sanity_check_single_batch

seed_everything(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device={device}')

CHECKPOINT_DIR = Path('/workspace/moe_medical_vision/checkpoints')
ROOT = '/workspace/moe_medical_vision/data/raw/luna16'
CKPT_OUT = CHECKPOINT_DIR / 'expert4_luna16_FINAL_best.pth'

print('='*80)
print('LUNA FINAL — R3D-18 fine-tune from v1 best')
print('='*80)

train_loader, train_ds = get_dataloader(
    'luna16', ROOT, split='train', batch_size=2, num_workers=4,
    transform=get_transform_3d(split='train', size=(64,64,64), augment=True)
)
val_loader, val_ds = get_dataloader(
    'luna16', ROOT, split='val', batch_size=2, num_workers=4,
    transform=get_transform_3d(split='val', size=(64,64,64), augment=False)
)
print('train_counts=', train_ds.class_counts.tolist())
print('val_counts=', val_ds.class_counts.tolist())

# Load best v1 checkpoint
model = build_luna_expert(pretrained=True, use_gradient_checkpointing=True).to(device)
ckpt_v1 = torch.load(CHECKPOINT_DIR / 'expert4_luna16_r3d18_best.pth', map_location=device, weights_only=False)
model.load_state_dict(ckpt_v1['model_state_dict'])
print(f"Loaded v1: val_f1={ckpt_v1['best_val_f1']:.4f}")

# Replace head with stronger regularized head
in_features = model.backbone.fc[1].in_features
model.backbone.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 2),
).to(device)

criterion = nn.CrossEntropyLoss(
    weight=train_ds.get_class_weights().to(device),
    label_smoothing=0.03,
)

print('sanity:', sanity_check_single_batch(model, train_loader, criterion, device), flush=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=5e-3)
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
    epochs=40,
    accum_steps=4,
    mixed_precision=True,
    patience=15,
    checkpoint_path=CKPT_OUT,
)

print(f"LUNA FINAL: best_val_f1={result['best_val_f1']:.4f} best_epoch={result['best_epoch']}")
print('DONE')
