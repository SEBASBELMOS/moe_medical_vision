"""LUNA FAST training on precomputed .npz volumes."""
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

sys.path.insert(0, '/workspace/moe_medical_vision/src')
from models.experts_3d import build_luna_expert
from train.train_3d import seed_everything, fit_3d_expert, sanity_check_single_batch

seed_everything(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device={device}')

FAST_ROOT = Path('/workspace/moe_medical_vision/data/processed/luna16_fast')
CHECKPOINT_DIR = Path('/workspace/moe_medical_vision/checkpoints')
CKPT_OUT = CHECKPOINT_DIR / 'expert4_luna16_FAST_best.pth'

class LunaFastDataset(Dataset):
    def __init__(self, root, split='train'):
        self.files = sorted(root.glob(f'{split}_*.npz'))
        labels = []
        for f in self.files:
            data = np.load(f)
            labels.append(int(data['label']))
        self.labels = np.array(labels)
        counts = np.bincount(self.labels, minlength=2)
        total = counts.sum()
        self.class_counts = counts
        self.class_weights = torch.tensor(total / (2 * counts + 1e-6), dtype=torch.float32)
        print(f'[LUNA FAST {split}] {len(self.files)} samples | neg={counts[0]} pos={counts[1]}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        x = torch.from_numpy(data['volume'].copy()).float()
        y = torch.tensor(int(data['label']), dtype=torch.long)
        return {'image': x, 'label': y}

    def get_weighted_sampler(self):
        weights = self.class_weights[self.labels]
        return WeightedRandomSampler(weights.tolist(), len(self.files), replacement=True)

    def get_class_weights(self):
        return self.class_weights

print('='*80)
print('LUNA FAST — train on precomputed .npz')
print('='*80)

train_ds = LunaFastDataset(FAST_ROOT, 'train')
val_ds = LunaFastDataset(FAST_ROOT, 'val')
train_loader = DataLoader(train_ds, batch_size=4, sampler=train_ds.get_weighted_sampler(), num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

model = build_luna_expert(pretrained=True, use_gradient_checkpointing=True).to(device)
v1 = torch.load(CHECKPOINT_DIR / 'expert4_luna16_r3d18_best.pth', map_location=device, weights_only=False)
model.load_state_dict(v1['model_state_dict'])
print(f"Loaded v1: val_f1={v1['best_val_f1']:.4f}")

criterion = torch.nn.CrossEntropyLoss(weight=train_ds.get_class_weights().to(device), label_smoothing=0.03)
print('sanity:', sanity_check_single_batch(model, train_loader, criterion, device))
print('START_TRAIN_LOOP', flush=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)

result = fit_3d_expert(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    epochs=25,
    accum_steps=2,
    mixed_precision=True,
    patience=10,
    checkpoint_path=CKPT_OUT,
)
print(f"LUNA FAST: best_val_f1={result['best_val_f1']:.4f} best_epoch={result['best_epoch']}")
print('DONE')
