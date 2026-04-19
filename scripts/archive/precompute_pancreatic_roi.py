from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm

RAW_ROOT = Path('/workspace/moe_medical_vision/data/raw/pancreatic')
MASK_ROOT = Path('/workspace/moe_medical_vision/data/raw/panorama_labels')
OUT_ROOT = Path('/workspace/moe_medical_vision/data/processed/pancreatic_roi_v2')
OUT_ROOT.mkdir(parents=True, exist_ok=True)

LABELS_CSV = RAW_ROOT / 'labels.csv'
OUT_CSV = OUT_ROOT / 'labels.csv'
df = pd.read_csv(LABELS_CSV)
label_map = {row['study_id']: row for _, row in df.iterrows()}


def find_mask(study_id: str) -> Path:
    for sub in ['manual_labels', 'automatic_labels']:
        p = MASK_ROOT / sub / f'{study_id}.nii.gz'
        if p.exists():
            return p
    raise FileNotFoundError(f'Mask not found for {study_id}')


def compute_bbox(mask: np.ndarray, margin_ratio=0.15, min_margin=8):
    roi = (mask == 4) | (mask == 1)
    if not roi.any():
        roi = (mask == 4)
    if not roi.any():
        raise ValueError('No pancreas/lesion ROI found in mask')
    coords = np.argwhere(roi)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0) + 1
    size = hi - lo
    margin = np.maximum((size * margin_ratio).astype(int), min_margin)
    lo = np.maximum(lo - margin, 0)
    hi = np.minimum(hi + margin, np.array(mask.shape))
    return tuple(slice(int(lo[i]), int(hi[i])) for i in range(3))


existing = {p.stem for p in OUT_ROOT.glob('*.npz')}
print(f'existing crops: {len(existing)}')
missing = []
new_count = 0
for rec in tqdm(df.to_dict('records'), total=len(df), desc='precompute_pancreatic_roi_resume'):
    study_id = rec['study_id']
    out_file = OUT_ROOT / f'{study_id}.npz'
    if out_file.exists():
        continue
    vol_path = Path(rec['filename'])
    try:
        mask_path = find_mask(study_id)
        vol_img = nib.load(str(vol_path))
        mask_img = nib.load(str(mask_path))
        if vol_img.shape != mask_img.shape:
            raise ValueError(f'shape mismatch vol={vol_img.shape} mask={mask_img.shape}')
        vol = vol_img.get_fdata(dtype=np.float32)
        mask = np.asanyarray(mask_img.dataobj)
        bbox = compute_bbox(mask)
        crop = vol[bbox].astype(np.float32)
        np.savez_compressed(out_file, volume=crop)
        new_count += 1
    except Exception as e:
        missing.append((study_id, str(e)))

rows = []
for npz_file in sorted(OUT_ROOT.glob('*.npz')):
    study_id = npz_file.stem
    rec = label_map.get(study_id)
    if rec is None:
        continue
    rows.append({
        'filename': str(npz_file),
        'study_id': study_id,
        'raw_label': rec['raw_label'],
        'label': rec['label'],
        'source_file': rec['filename'],
    })
pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
print(f'labels saved: {OUT_CSV} | rows={len(rows)} | new={new_count}')
print(f'missing={len(missing)}')
if missing:
    print('sample_missing=', missing[:10])
