import sys
from pathlib import Path
sys.path.insert(0, '/workspace/moe_medical_vision/src')

import torch
from data.datasets import get_dataloader
from models.experts_3d import build_luna_expert
from train.train_3d import seed_everything, fit_3d_expert, sanity_check_single_batch

seed_everything(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device={device}')
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'gpu={props.name} vram={props.total_memory/1024**3:.2f}GB')

CHECKPOINT_DIR = Path('/workspace/moe_medical_vision/checkpoints')
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

print('\n' + '='*80)
print('TRAINING EXPERT 4 - LUNA16 (BALANCED)')
print('='*80)
train_loader_luna, train_ds_luna = get_dataloader('luna16', '/workspace/moe_medical_vision/data/raw/luna16', split='train', batch_size=1, num_workers=2)
val_loader_luna, val_ds_luna = get_dataloader('luna16', '/workspace/moe_medical_vision/data/raw/luna16', split='val', batch_size=1, num_workers=2)
print('class_counts=', train_ds_luna.class_counts.tolist())
print('class_weights=', train_ds_luna.get_class_weights().tolist())
model_luna = build_luna_expert(pretrained=True, use_gradient_checkpointing=True).to(device)
criterion_luna = torch.nn.CrossEntropyLoss(weight=train_ds_luna.get_class_weights().to(device))
print('luna sanity:', sanity_check_single_batch(model_luna, train_loader_luna, criterion_luna, device))
optimizer_luna = torch.optim.AdamW(model_luna.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler_luna = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_luna, mode='min', factor=0.5, patience=3)
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
print('DONE')
