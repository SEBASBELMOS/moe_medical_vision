"""
LUNA-v4: R(2+1)D-18 at 96^3 resolution with weighted sampler + weighted CE
Hypothesis: Higher resolution preserves nodule signal better; R(2+1)D factored
convolutions capture spatiotemporal patterns more efficiently than R3D.
"""
import sys
from pathlib import Path
sys.path.insert(0, '/workspace/moe_medical_vision/src')

import torch
from data.datasets import get_dataloader, get_transform_3d
from models.experts_3d import build_luna_expert_r2plus1d
from train.train_3d import seed_everything, fit_3d_expert, sanity_check_single_batch

seed_everything(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device={device}')
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'gpu={props.name} vram={props.total_memory/1024**3:.2f}GB')

CHECKPOINT_DIR = Path('/workspace/moe_medical_vision/checkpoints')
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
ROOT = '/workspace/moe_medical_vision/data/raw/luna16'

# Key change: 96^3 resolution instead of 64^3
SIZE = (96, 96, 96)

print('\n' + '='*80)
print('TRAINING EXPERT 4 - LUNA V4 (R2Plus1D-18 @ 96^3)')
print('='*80)

train_loader, train_ds = get_dataloader(
    'luna16', ROOT, split='train', batch_size=1, num_workers=2,
    transform=get_transform_3d(split='train', augment=True, size=SIZE)
)
val_loader, val_ds = get_dataloader(
    'luna16', ROOT, split='val', batch_size=1, num_workers=2,
    transform=get_transform_3d(split='val', augment=False, size=SIZE)
)

print('train_counts=', train_ds.class_counts.tolist())
print('val_counts=', val_ds.class_counts.tolist())
print('class_weights=', train_ds.get_class_weights().tolist())

model = build_luna_expert_r2plus1d(pretrained=True, use_gradient_checkpointing=True).to(device)

# Use weighted CE like v1 (which was best at 0.578)
criterion = torch.nn.CrossEntropyLoss(
    weight=train_ds.get_class_weights().to(device),
    label_smoothing=0.01
)

print('LUNA-v4 sanity:', sanity_check_single_batch(model, train_loader, criterion, device))

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=8, T_mult=2, eta_min=1e-6)

result = fit_3d_expert(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    epochs=25,
    accum_steps=8,
    mixed_precision=True,
    patience=10,
    checkpoint_path=CHECKPOINT_DIR / 'expert4_luna16_r2plus1d_v4_best.pth',
)

print(f'luna_v4_result= best_val_f1={result["best_val_f1"]:.4f} best_epoch={result["best_epoch"]}')
print('DONE')
