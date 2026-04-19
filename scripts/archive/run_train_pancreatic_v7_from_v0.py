"""
PANC-v7: Full-volume R3D-18 + FocalLoss + balanced sampler + strong aug + longer training.
Restart from v0 best checkpoint (0.5134) with improved training recipe.
Hypothesis: v0 got 0.5134 in epoch 1 but overfit. Better LR + balanced sampler + stronger aug
can push past 0.65.
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
CKPT_PATH = CHECKPOINT_DIR / "expert5_pancreatic_r3d18_v7_best.pth"

print("=" * 80)
print("TRAINING EXPERT 5 - PANC V7 (R3D-18 from v0, improved recipe)")
print("=" * 80)

train_loader, train_ds = get_dataloader(
    "pancreatic", ROOT, split="train", batch_size=2, num_workers=0,
    transform=get_transform_3d_strong(split="train", size=(64, 64, 64))
)
val_loader, val_ds = get_dataloader(
    "pancreatic", ROOT, split="val", batch_size=2, num_workers=0,
    transform=get_transform_3d_strong(split="val", size=(64, 64, 64))
)

print("train_counts=", train_ds.class_counts.tolist())
print("val_counts=", val_ds.class_counts.tolist())

model = build_pancreatic_expert(pretrained=True, use_gradient_checkpointing=True).to(device)

ckpt_v0 = torch.load(CHECKPOINT_DIR / "expert5_pancreatic_r3d18_best.pth", map_location=device, weights_only=False)
model.load_state_dict(ckpt_v0["model_state_dict"])
print("Loaded v0 checkpoint: val_f1=%.4f" % ckpt_v0["best_val_f1"])

alpha_pan = train_ds.get_focal_alpha()
print("alpha_pan=", alpha_pan)
criterion = FocalLoss(gamma=2.0, alpha=alpha_pan)

print("PANC-v7 sanity:", sanity_check_single_batch(model, train_loader, criterion, device))

# Moderate LR, high weight decay
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=12, T_mult=2, eta_min=1e-7)

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
    checkpoint_path=CKPT_PATH,
    use_mixup=True,
    mixup_alpha=0.3,
)

print("FINAL: panc_v7_result= best_val_f1=%.4f best_epoch=%d" % (result["best_val_f1"], result["best_epoch"]))
print("DONE")
