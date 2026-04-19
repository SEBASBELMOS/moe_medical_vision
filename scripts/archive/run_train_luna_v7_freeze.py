import sys
from pathlib import Path
sys.path.insert(0, "/workspace/moe_medical_vision/src")

import torch
import torch.nn as nn
from data.datasets import get_dataloader, get_transform_3d
from models.experts_3d import build_luna_expert
from train.train_3d import seed_everything, train_3d_epoch, validate_3d_epoch
from torch.amp import GradScaler

seed_everything(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}")

CHECKPOINT_DIR = Path("/workspace/moe_medical_vision/checkpoints")
ROOT = "/workspace/moe_medical_vision/data/raw/luna16"
CKPT_PATH = CHECKPOINT_DIR / "expert4_luna16_r3d18_v7_best.pth"

print("=" * 80)
print("TRAINING EXPERT 4 - LUNA V7 (freeze+unfreeze from v1)")
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

model = build_luna_expert(pretrained=True, use_gradient_checkpointing=True).to(device)

ckpt = torch.load(CHECKPOINT_DIR / "expert4_luna16_r3d18_best.pth", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
bvf = ckpt["best_val_f1"]
print("Loaded v1 checkpoint: val_f1=%.4f" % bvf)

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
    label_smoothing=0.05,
)

scaler = GradScaler("cuda", enabled=True)
best_f1 = -1.0
best_epoch = 0
history = []

print("--- Phase 1: Freeze backbone, train head (10 epochs) ---")
for name, param in model.named_parameters():
    if "fc" not in name:
        param.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_p = sum(p.numel() for p in model.parameters())
print("Trainable: %d / %d (%.1f%%)" % (trainable, total_p, 100*trainable/total_p))

optimizer_head = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3, weight_decay=1e-3
)
scheduler_head = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_head, T_max=10, eta_min=1e-5)

for epoch in range(1, 11):
    tm = train_3d_epoch(model, train_loader, criterion, optimizer_head, device, scaler, accum_steps=4, mixed_precision=True)
    vm = validate_3d_epoch(model, val_loader, criterion, device)
    scheduler_head.step()
    lr = optimizer_head.param_groups[0]["lr"]
    row = {"epoch": epoch, "lr": lr, "train_loss": tm["loss"], "train_f1": tm["f1_macro"],
           "train_acc": tm["accuracy"], "val_loss": vm["loss"], "val_f1": vm["f1_macro"], "val_acc": vm["accuracy"]}
    history.append(row)
    print("[Phase1 E%02d] train_f1=%.4f val_f1=%.4f lr=%.2e" % (epoch, tm["f1_macro"], vm["f1_macro"], lr))
    if vm["f1_macro"] > best_f1:
        best_f1 = vm["f1_macro"]
        best_epoch = epoch
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                     "best_val_f1": best_f1, "history": history, "phase": "head_only"}, CKPT_PATH)
        print("  -> new best: val_f1=%.4f" % best_f1)

print("--- Phase 2: Unfreeze all, differential LR (20 epochs) ---")
print("Best from Phase 1: val_f1=%.4f at epoch %d" % (best_f1, best_epoch))

best_ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
model.load_state_dict(best_ckpt["model_state_dict"])

for param in model.parameters():
    param.requires_grad = True

backbone_params = [p for n, p in model.named_parameters() if "fc" not in n]
head_params = [p for n, p in model.named_parameters() if "fc" in n]

optimizer_full = torch.optim.AdamW([
    {"params": backbone_params, "lr": 2e-6},
    {"params": head_params, "lr": 5e-5},
], weight_decay=5e-3)

scheduler_full = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_full, T_max=20, eta_min=1e-7)

wait = 0
patience = 10
for epoch in range(11, 31):
    tm = train_3d_epoch(model, train_loader, criterion, optimizer_full, device, scaler, accum_steps=4, mixed_precision=True)
    vm = validate_3d_epoch(model, val_loader, criterion, device)
    scheduler_full.step()
    lr = optimizer_full.param_groups[0]["lr"]
    row = {"epoch": epoch, "lr": lr, "train_loss": tm["loss"], "train_f1": tm["f1_macro"],
           "train_acc": tm["accuracy"], "val_loss": vm["loss"], "val_f1": vm["f1_macro"], "val_acc": vm["accuracy"]}
    history.append(row)
    print("[Phase2 E%02d] train_f1=%.4f val_f1=%.4f lr=%.2e" % (epoch, tm["f1_macro"], vm["f1_macro"], lr))
    if vm["f1_macro"] > best_f1:
        best_f1 = vm["f1_macro"]
        best_epoch = epoch
        wait = 0
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                     "best_val_f1": best_f1, "history": history, "phase": "full_finetune"}, CKPT_PATH)
        print("  -> new best: val_f1=%.4f" % best_f1)
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping at epoch %d. Best: epoch=%d f1=%.4f" % (epoch, best_epoch, best_f1))
            break

print("FINAL: luna_v7_result= best_val_f1=%.4f best_epoch=%d" % (best_f1, best_epoch))
print("DONE")
