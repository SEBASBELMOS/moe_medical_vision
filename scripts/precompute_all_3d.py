"""Precompute all 3D volumes to fast .npz files at 64^3."""
import sys, time, os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/moe_medical_vision/src')

SIZE = (64, 64, 64)

def resize_volume(vol_np, size=SIZE):
    t = torch.from_numpy(vol_np).float()
    if t.ndim == 3:
        t = t.unsqueeze(0)  # C,D,H,W
    t = t.unsqueeze(0)  # B,C,D,H,W
    t = F.interpolate(t, size=size, mode='trilinear', align_corners=False)
    return t.squeeze(0).numpy()  # C,D,H,W

# === LUNA16 ===
print('='*80)
print('PRECOMPUTING LUNA16')
print('='*80)

from data.datasets import LUNA16Dataset
luna_out = Path('/workspace/moe_medical_vision/data/processed/luna16_fast')
luna_out.mkdir(parents=True, exist_ok=True)

ds = LUNA16Dataset('/workspace/moe_medical_vision/data/raw/luna16', split='train', transform=None)
# also need val
ds_val = LUNA16Dataset('/workspace/moe_medical_vision/data/raw/luna16', split='val', transform=None)

for split_name, split_ds in [('train', ds), ('val', ds_val)]:
    print(f'  {split_name}: {len(split_ds)} samples')
    done = 0
    skipped = 0
    t0 = time.time()
    for i in range(len(split_ds)):
        path, label = split_ds.samples[i]
        out_file = luna_out / f'{split_name}_{i:05d}.npz'
        if out_file.exists():
            skipped += 1
            continue
        try:
            sample = split_ds._load_mhd(path)  # returns tensor
            vol = resize_volume(sample.numpy(), SIZE)
            np.savez_compressed(out_file, volume=vol, label=label)
            done += 1
            if done % 10 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                print(f'    [{split_name}] {done}/{len(split_ds)} done, {skipped} skipped, {rate:.2f} samples/s', flush=True)
        except Exception as e:
            print(f'    ERROR sample {i}: {e}', flush=True)
    print(f'  {split_name} done: {done} new, {skipped} skipped')

# === PANCREATIC ===
print('='*80)
print('PRECOMPUTING PANCREATIC')
print('='*80)

from data.datasets import PancreaticCancerDataset
panc_out = Path('/workspace/moe_medical_vision/data/processed/pancreatic_fast')
panc_out.mkdir(parents=True, exist_ok=True)

ds_p = PancreaticCancerDataset('/workspace/moe_medical_vision/data/raw/pancreatic', split='train', transform=None)
ds_p_val = PancreaticCancerDataset('/workspace/moe_medical_vision/data/raw/pancreatic', split='val', transform=None)

for split_name, split_ds in [('train', ds_p), ('val', ds_p_val)]:
    print(f'  {split_name}: {len(split_ds)} samples')
    done = 0
    skipped = 0
    t0 = time.time()
    for i in range(len(split_ds)):
        row = split_ds.df.iloc[i]
        out_file = panc_out / f'{split_name}_{i:05d}.npz'
        if out_file.exists():
            skipped += 1
            continue
        try:
            vol_tensor = split_ds._load_volume(row['filename'])  # returns tensor
            vol = resize_volume(vol_tensor.numpy(), SIZE)
            np.savez_compressed(out_file, volume=vol, label=int(row['label']))
            done += 1
            if done % 10 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                print(f'    [{split_name}] {done}/{len(split_ds)} done, {skipped} skipped, {rate:.2f} samples/s', flush=True)
        except Exception as e:
            print(f'    ERROR sample {i}: {e}', flush=True)
    print(f'  {split_name} done: {done} new, {skipped} skipped')

print('ALL DONE')
