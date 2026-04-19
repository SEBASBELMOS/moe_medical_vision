"""
LUNA-v9: Best features from v1 (0.578) + heavy regularization + moderate resolution.
Hypothesis: v1 works because it's the best baseline. v9 should improve on v1 by:
  1. Heavy dropout (0.5) in a custom 2-layer head 
  2. Label smoothing 0.05
  3. Weighted sampler + weighted CE
  4. NO mixup (mixup hurt LUNA in v5)
  5. Light augmentation only (strong aug hurt in v2)
  6. Longer training with early stopping (40 epochs, patience 12)
  7. Cosine LR schedule with warm restarts
  8. Higher weight decay (5e-3)
"""
import sys
from pathlib import Path
sys.path.insert(0, "/workspace/moe_medical_vision/src")

import torch
import torch.nn as nn
from data.datasets import get_dataloader, get_transform_3d
from models.experts_3d import build_luna_expert
from train.train_3d import seed_everything, fit_3d_expert, sanity_check_single_batch

seed_everything(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device=" + device)

CHECKPOINT_DIR = Path("/workspace/moe_medical_vision/checkpoints")
ROOT = "/workspace/moe_medical_vision/data/raw/luna16"
CKPT_PATH = CHECKPOINT_DIR / "expert4_luna16_r3d18_v9_best.pth"

print("=" * 80)
print("TRAINING EXPERT 4 - LUNA V9 (from v1, heavy reg, custom head)")
print("=" * 80)

train_loader, train_ds = get_dataloader(
    "luna16", ROOT, split="train", batch_size=2, num_workers=2,
    transform=get_transform_3d(split="train", size=(64, 64, 64), augment=True)
)
val_loader, val_ds = get_dataloader(
    "luna16", ROOT, split="val", batch_size=2, num_workers=2,
    transform=get_transform_3d(split="val", size=(64, 64, 64))
)

print("train_counts=", train_ds.class_counts.tolist())
print("val_counts=", val_ds.class_counts.tolist())

# Build model 
model = build_luna_expert(pretrained=True, use_gradient_checkpointing=True).to(device)

# Load v1 checkpoint (best baseline at 0.578)
ckpt_v1 = torch.load(CHECKPOINT_DIR / "expert4_luna16_r3d18_best.pth", map_location=device, weights_only=False)
model.load_state_dict(ckpt_v1["model_state_dict"])
print("Loaded v1 checkpoint: val_f1=%.4f" % ckpt_v1["best_val_f1"])

# Replace FC head with higher dropout (0.5) + hidden layer
in_features = model.backbone.fc[1].in_features
model.backbone.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 2),
).to(device)

criterion = nn.CrossEntropyLoss(
    weight=train_ds.get_class_weights().to(device),
    label_smoothing=0.05,
)

print("LUNA-v9 sanity:", sanity_check_single_batch(model, train_loader, criterion, device))

# Moderate LR, high weight decay
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-6)

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
    patience=12,
    checkpoint_path=CKPT_PATH,
    use_mixup=False,
)

print("FINAL: luna_v9_result= best_val_f1=%.4f best_epoch=%d" % (result["best_val_f1"], result["best_epoch"]))
print("DONE")
