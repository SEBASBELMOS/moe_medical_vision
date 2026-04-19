import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import VivitForVideoClassification
from sklearn.metrics import f1_score

sys.path.insert(0, '/workspace/moe_medical_vision/src')
from train.train_3d import seed_everything, fit_3d_expert, sanity_check_single_batch

seed_everything(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device={device}')

FAST_ROOT = Path('/workspace/moe_medical_vision/data/processed/luna16_fast')
CHECKPOINT_DIR = Path('/workspace/moe_medical_vision/checkpoints')
CKPT_OUT = CHECKPOINT_DIR / 'expert4_luna16_ViViT_best.pth'

class LunaViViTDataset(Dataset):
    def __init__(self, root, split='train'):
        self.files = sorted(root.glob(f'{split}_*.npz'))
        self.split = split
        labels = []
        for f in self.files:
            d = np.load(f)
            labels.append(int(d['label']))
        self.labels = np.array(labels)
        counts = np.bincount(self.labels, minlength=2)
        total = counts.sum()
        self.class_weights = torch.tensor(total / (2 * counts + 1e-6), dtype=torch.float32)
        print(f'[LUNA ViViT {split}] {len(self.files)} | neg={counts[0]} pos={counts[1]}')

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        vol = torch.from_numpy(d['volume'][0].copy()).float()
        
        vol = vol.unsqueeze(0).unsqueeze(0)
        vol = F.interpolate(vol, size=(32, 224, 224), mode='trilinear', align_corners=False).squeeze(0)
        vol = vol.repeat(3, 1, 1, 1)
        
        if self.split == 'train':
            if torch.rand(1).item() < 0.5: vol = torch.flip(vol, dims=(2,))
            if torch.rand(1).item() < 0.5: vol = torch.flip(vol, dims=(3,))
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
        vol = (vol - mean) / std
        
        y = torch.tensor(int(d['label']), dtype=torch.long)
        return {'image': vol, 'label': y}

    def sampler(self):
        w = self.class_weights[self.labels]
        return WeightedRandomSampler(w.tolist(), len(self.files), replacement=True)

class WrappedViViT(nn.Module):
    def __init__(self):
        super().__init__()
        # Use safetensors directly to avoid PyTorch security error with from_pretrained
        self.vivit = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400", num_labels=2, ignore_mismatched_sizes=True, use_safetensors=True)
        self.vivit.gradient_checkpointing_enable()
    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        out = self.vivit(pixel_values=x)
        return out.logits

print('='*80)
print('LUNA ViViT FAST (SafeTensors)')
print('='*80)

train_ds = LunaViViTDataset(FAST_ROOT, 'train')
val_ds = LunaViViTDataset(FAST_ROOT, 'val')
train_loader = DataLoader(train_ds, batch_size=2, sampler=train_ds.sampler(), num_workers=0)
val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0)

model = WrappedViViT().to(device)
criterion = nn.CrossEntropyLoss(weight=train_ds.class_weights.to(device), label_smoothing=0.05)

print('sanity check running...', flush=True)
try:
    sanity = sanity_check_single_batch(model, train_loader, criterion, device)
    print('sanity:', sanity, flush=True)
except Exception as e:
    print('sanity check FAILED:', e, flush=True)
    sys.exit(1)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-7)

result = fit_3d_expert(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    epochs=15,
    accum_steps=8,
    mixed_precision=True,
    patience=8,
    checkpoint_path=CKPT_OUT,
)
print(f"LUNA ViViT: best_val_f1={result['best_val_f1']:.4f} best_epoch={result['best_epoch']}")
print('DONE')
