"""
PANC-v6: Continue from v5 (0.6422) + better training recipe.
Hypothesis: v5 was at 0.6422 at epoch 1. Continue training with:
  1. Continue from v5 checkpoint
  2. NO mixup (or very mild alpha=0.2)
  3. Strong augmentation  
  4. Balanced sampler (v0/v5 didn't use one, v5 only used focal loss)
  5. Lower LR (fine-tuning from v0 which was not optimal)
  6. 25 epochs total with patience 12
  7. SWA in last 5 epochs
"""
import sys
from pathlib import Path
sys.path.insert(0, "/workspace/moe_medical_vision/src")

import torch
import torch.nn as nn
from data.datasets import get_dataloader, get_transform_3d_strong
from losses import FocalLoss
from models.experts_3d import build_pancreatic_expert
from train.train_3d import seed_everything, fit_3d_expert, sanity_check_single_batch

seed_everything(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device=" + device)

CHECKPOINT_DIR = Path("/workspace/moe_medical_vision/checkpoints")
ROOT = "/workspace/moe_medical_vision/data/raw/pancreatic"
CKPT_PATH = CHECKPOINT_DIR / "expert5_pancreatic_r3d18_v6_best.pth"

print("=" * 80)
print("TRAINING EXPERT 5 - PANC V6 (continue from v5, stronger reg)")
print("=" * 80)

# Strong augmentation
train_loader, train_ds = get_dataloader(
    "pancreatic", ROOT, split="train", batch_size=2, num_workers=2,
    transform=get_transform_3d_strong(split="train", size=(64, 64, 64))
)
val_loader, val_ds = get_dataloader(
    "pancreatic", ROOT, split="val", batch_size=2, num_workers=2,
    transform=get_transform_3d_strong(split="val", size=(64, 64, 64))
)

print("train_counts=", train_ds.class_counts.tolist())
print("val_counts=", val_ds.class_counts.tolist())

# Build model 
model = build_pancreatic_expert(pretrained=True, use_gradient_checkpointing=True).to(device)

# Load v5 checkpoint
ckpt_v5 = torch.load(CHECKPOINT_DIR / "expert5_pancreatic_r3d18_v5_best.pth", map_location=device, weights_only=False)
model.load_state_dict(ckpt_v5["model_state_dict"])
print("Loaded v5 checkpoint: val_f1=%.4f" % ckpt_v5["best_val_f1"])

# FocalLoss with alpha based on class distribution
alpha_pan = train_ds.get_focal_alpha()
print("alpha_pan=", alpha_pan)
criterion = FocalLoss(gamma=2.0, alpha=alpha_pan)

print("PANC-v6 sanity:", sanity_check_single_batch(model, train_loader, criterion, device))

# Very low LR for fine-tuning
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-6, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-7)

result = fit_3d_expert(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    epochs=25,
    accum_steps=4,
    mixed_precision=True,
    patience=12,
    checkpoint_path=CKPT_PATH,
    use_mixup=True,
    mixup_alpha=0.25,
)

print("FINAL: panc_v6_result= best_val_f1=%.4f best_epoch=%d" % (result["best_val_f1"], result["best_epoch"]))
print("DONE")
