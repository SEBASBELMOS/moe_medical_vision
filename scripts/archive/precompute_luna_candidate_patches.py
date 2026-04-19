from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

RAW_ROOT = Path('/workspace/moe_medical_vision/data/raw/luna16')
OUT_ROOT = Path('/workspace/moe_medical_vision/data/processed/luna_candidates_v3')
OUT_ROOT.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_ROOT / 'labels.csv'

PATCH_SIZE = 64
NEG_POS_RATIO = 1  # balanced dataset
SEED = 42

cand_df = pd.read_csv(RAW_ROOT / 'candidates.csv')
# keep all positives + balanced random negatives
pos_df = cand_df[cand_df['class'] == 1].copy()
neg_df = cand_df[cand_df['class'] == 0].sample(n=len(pos_df) * NEG_POS_RATIO, random_state=SEED)
sel_df = pd.concat([pos_df, neg_df], ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
print(f'selected candidates: {len(sel_df)} | pos={len(pos_df)} neg={len(neg_df)}')

mhd_dir = RAW_ROOT / 'seg-lungs-LUNA16' / 'seg-lungs-LUNA16'
series_map = {p.stem: p for p in mhd_dir.glob('*.mhd')}
print('series available', len(series_map))


def extract_patch(vol: np.ndarray, center_idx, patch_size=64):
    half = patch_size // 2
    z, y, x = [int(v) for v in center_idx]  # sitk array is z,y,x
    z0, z1 = z - half, z + half
    y0, y1 = y - half, y + half
    x0, x1 = x - half, x + half

    patch = np.zeros((patch_size, patch_size, patch_size), dtype=np.float32)

    src_z0, src_z1 = max(0, z0), min(vol.shape[0], z1)
    src_y0, src_y1 = max(0, y0), min(vol.shape[1], y1)
    src_x0, src_x1 = max(0, x0), min(vol.shape[2], x1)

    dst_z0, dst_y0, dst_x0 = src_z0 - z0, src_y0 - y0, src_x0 - x0
    dst_z1 = dst_z0 + (src_z1 - src_z0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    patch[dst_z0:dst_z1, dst_y0:dst_y1, dst_x0:dst_x1] = vol[src_z0:src_z1, src_y0:src_y1, src_x0:src_x1]
    return patch


existing = {p.stem for p in OUT_ROOT.glob('*.npz')}
rows = []
new_count = 0
cache = {}
for i, rec in tqdm(list(sel_df.iterrows()), total=len(sel_df), desc='precompute_luna_candidates'):
    seriesuid = rec['seriesuid']
    key = f"{seriesuid}_{i}"
    out_file = OUT_ROOT / f'{key}.npz'
    if out_file.exists():
        rows.append({'filename': str(out_file), 'label': int(rec['class']), 'seriesuid': seriesuid})
        continue
    mhd_path = series_map.get(seriesuid)
    if mhd_path is None:
        continue
    if seriesuid not in cache:
        img = sitk.ReadImage(str(mhd_path))
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        arr = np.clip(arr, -1000.0, 400.0)
        arr = (arr + 1000.0) / 1400.0
        cache[seriesuid] = (img, arr)
    img, vol = cache[seriesuid]
    world = (float(rec['coordX']), float(rec['coordY']), float(rec['coordZ']))
    idx_xyz = img.TransformPhysicalPointToIndex(world)
    idx_zyx = (idx_xyz[2], idx_xyz[1], idx_xyz[0])
    patch = extract_patch(vol, idx_zyx, patch_size=PATCH_SIZE)
    np.savez_compressed(out_file, volume=patch)
    rows.append({'filename': str(out_file), 'label': int(rec['class']), 'seriesuid': seriesuid})
    new_count += 1

pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
print(f'labels saved: {OUT_CSV} | rows={len(rows)} | new={new_count}')
