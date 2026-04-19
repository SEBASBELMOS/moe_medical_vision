import sys
from pathlib import Path
sys.path.insert(0, '/workspace/moe_medical_vision/src')

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from data.datasets import get_transform_3d
from models.experts_3d import build_luna_patch_expert_mc3
from train.train_3d import seed_everything, fit_3d_expert, sanity_check_single_batch

seed_everything(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device={device}')
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'gpu={props.name} vram={props.total_memory/1024**3:.2f}GB')

# step 1 precompute
import subprocess
print('\n[PIPELINE] precomputing LUNA candidate patches if needed...')
subprocess.run(['python3', '/workspace/moe_medical_vision/scripts/precompute_luna_candidate_patches.py'], check=True)

ROOT = Path('/workspace/moe_medical_vision/data/processed/luna_candidates_v3')
CHECKPOINT_DIR = Path('/workspace/moe_medical_vision/checkpoints')
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

class LUNACandidatePatchDataset(Dataset):
    def __init__(self, root, split='train', transform=None, val_frac=0.2, seed=42):
        self.root = Path(root)
        self.transform = transform or get_transform_3d(split=split, augment=(split=='train'))
        df = pd.read_csv(self.root / 'labels.csv')
        rng = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(df), generator=rng).tolist()
        n_val = int(len(df) * val_frac)
        idx = perm[:n_val] if split == 'val' else perm[n_val:]
        self.df = df.iloc[idx].reset_index(drop=True)
        import numpy as np
        counts = np.bincount(self.df['label'].values, minlength=2)
        total = counts.sum()
        self.class_counts = counts
        self.class_weights = torch.tensor(total / (2 * counts + 1e-6), dtype=torch.float32)
        print(f'[LUNA candidates {split}] {len(self.df)} patches | neg:{counts[0]} pos:{counts[1]}')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        import numpy as np
        data = np.load(row['filename'])['volume'].astype('float32')
        x = torch.from_numpy(data).unsqueeze(0)
        x = self.transform(x)
        return {'image': x, 'label': torch.tensor(int(row['label']), dtype=torch.long), 'filename': row['filename']}

    def get_weighted_sampler(self):
        weights = self.class_weights[self.df['label'].values]
        return WeightedRandomSampler(weights.tolist(), len(self.df), replacement=True)

    def get_class_weights(self):
        return self.class_weights

train_ds = LUNACandidatePatchDataset(ROOT, split='train')
val_ds = LUNACandidatePatchDataset(ROOT, split='val', transform=get_transform_3d(split='val', augment=False))
train_loader = DataLoader(train_ds, batch_size=2, sampler=train_ds.get_weighted_sampler(), num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=2, pin_memory=True)

print('class_counts=', train_ds.class_counts.tolist())
print('class_weights=', train_ds.get_class_weights().tolist())
model = build_luna_patch_expert_mc3(pretrained=True, use_gradient_checkpointing=True).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=train_ds.get_class_weights().to(device), label_smoothing=0.02)
print('luna_v3 sanity:', sanity_check_single_batch(model, train_loader, criterion, device))
optimizer = torch.optim.AdamW(model.parameters(), lr=7e-5, weight_decay=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=18, eta_min=1e-6)
result = fit_3d_expert(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    epochs=18,
    accum_steps=4,
    mixed_precision=True,
    patience=7,
    checkpoint_path=CHECKPOINT_DIR / 'expert4_luna16_mc3_candidates_v3_best.pth',
)
print('luna_v3_result=', result['best_val_f1'], result['best_epoch'])
print('DONE')
