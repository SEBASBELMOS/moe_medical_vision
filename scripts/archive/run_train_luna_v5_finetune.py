"""
LUNA-v5: Fine-tune from v1 best checkpoint (R3D-18, 0.578)
Strategy: Lower LR + mixup + stronger augmentation + higher weight decay
Hypothesis: v1 was overfitting (train_f1=0.98, val_f1=0.58). Fine-tuning from
that checkpoint with strong regularization should push val_f1 higher.
"""
import sys
from pathlib import Path
sys.path.insert(0, '/workspace/moe_medical_vision/src')

import torch
from data.datasets import get_dataloader, get_transform_3d_strong
from models.experts_3d import build_luna_expert
from train.train_3d import seed_everything, fit_3d_expert, sanity_check_single_batch

seed_everything(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device={device}')

CHECKPOINT_DIR = Path('/workspace/moe_medical_vision/checkpoints')
ROOT = '/workspace/moe_medical_vision/data/raw/luna16'

print('\n' + '='*80)
print('TRAINING EXPERT 4 - LUNA V5 (R3D-18 fine-tune from v1 + mixup + strong aug)')
print('='*80)

# Use STRONG augmentation (the key difference)
train_loader, train_ds = get_dataloader(
    'luna16', ROOT, split='train', batch_size=2, num_workers=2,
    transform=get_transform_3d_strong(split='train', size=(64, 64, 64))
)
val_loader, val_ds = get_dataloader(
    'luna16', ROOT, split='val', batch_size=2, num_workers=2,
    transform=get_transform_3d_strong(split='val', size=(64, 64, 64))
)

print('train_counts=', train_ds.class_counts.tolist())
print('val_counts=', val_ds.class_counts.tolist())

# Build model and load v1 best weights
model = build_luna_expert(pretrained=True, use_gradient_checkpointing=True).to(device)
ckpt = torch.load(CHECKPOINT_DIR / 'expert4_luna16_r3d18_best.pth', map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
print(f'Loaded v1 checkpoint: val_f1={ckpt["best_val_f1"]:.4f} epoch={ckpt["epoch"]}')

# Weighted CE with slightly more label smoothing
criterion = torch.nn.CrossEntropyLoss(
    weight=train_ds.get_class_weights().to(device),
    label_smoothing=0.05
)

print('LUNA-v5 sanity:', sanity_check_single_batch(model, train_loader, criterion, device))

# Much lower LR since we're fine-tuning from a pretrained checkpoint
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=5e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-7)

result = fit_3d_expert(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    epochs=20,
    accum_steps=4,
    mixed_precision=True,
    patience=10,
    checkpoint_path=CHECKPOINT_DIR / 'expert4_luna16_r3d18_v5_best.pth',
    use_mixup=True,
    mixup_alpha=0.3,
)

print(f'luna_v5_result= best_val_f1={result["best_val_f1"]:.4f} best_epoch={result["best_epoch"]}')
print('DONE')
