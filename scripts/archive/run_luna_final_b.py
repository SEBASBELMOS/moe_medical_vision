"""
LUNA FINAL B: MC3-18 from scratch, full volume 64^3.
Balanced sampling + weighted CE + 30 epochs + patience=15.
num_workers=0 to avoid worker kill issues.
"""
import sys
from pathlib import Path
sys.path.insert(0, '/workspace/moe_medical_vision/src')

import torch
from data.datasets import get_dataloader, get_transform_3d
from models.experts_3d import build_luna_patch_expert_mc3
from train.train_3d import seed_everything, fit_3d_expert, sanity_check_single_batch

seed_everything(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device={device}')

CHECKPOINT_DIR = Path('/workspace/moe_medical_vision/checkpoints')
ROOT = '/workspace/moe_medical_vision/data/raw/luna16'
CKPT = CHECKPOINT_DIR / 'expert4_luna16_FINAL_B_best.pth'

print('='*80)
print('LUNA FINAL B: MC3-18 from scratch, 64^3, 30ep')
print('='*80)

train_loader, train_ds = get_dataloader(
    'luna16', ROOT, split='train', batch_size=2, num_workers=0,
    transform=get_transform_3d(split='train', size=(64,64,64), augment=True)
)
val_loader, val_ds = get_dataloader(
    'luna16', ROOT, split='val', batch_size=2, num_workers=0,
    transform=get_transform_3d(split='val', size=(64,64,64))
)
print('train_counts=', train_ds.class_counts.tolist())
print('val_counts=', val_ds.class_counts.tolist())

model = build_luna_patch_expert_mc3(pretrained=True, use_gradient_checkpointing=True).to(device)

criterion = torch.nn.CrossEntropyLoss(
    weight=train_ds.get_class_weights().to(device),
    label_smoothing=0.03,
)

print('sanity:', sanity_check_single_batch(model, train_loader, criterion, device))

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

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
    patience=15,
    checkpoint_path=CKPT,
)
print(f'FINAL_B: best_val_f1={result["best_val_f1"]:.4f} best_epoch={result["best_epoch"]}')
print('DONE_B')
