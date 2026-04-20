
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import sys
sys.path.insert(0, '/workspace/moe_medical_vision/src')
from data.datasets import _build_pancreatic_labels_from_panorama

ROOT = Path('/workspace/moe_medical_vision')
RAW = ROOT / 'data' / 'raw' / 'pancreatic'
EMB = ROOT / 'embeddings'
ROUT = ROOT / 'data' / 'processed' / 'router_embeddings'
EMB.mkdir(parents=True, exist_ok=True)
ROUT.mkdir(parents=True, exist_ok=True)


def build_manifest(train_per_class=80, val_per_class=20, seed=42):
    label_csv = _build_pancreatic_labels_from_panorama(RAW)
    df = pd.read_csv(label_csv)
    subsets=[]
    for cls in [0,1]:
        cls_df = df[df.label==cls]
        if len(cls_df) == 0:
            continue
        sample_n = min(len(cls_df), train_per_class + val_per_class)
        sub = cls_df.sample(n=sample_n, random_state=seed)
        test_n = min(val_per_class, max(1, len(sub)//5))
        if test_n >= len(sub):
            test_n = max(1, len(sub)-1)
        train,val=train_test_split(sub, test_size=test_n, random_state=seed)
        subsets.append((train,val))
    train_df=pd.concat([x[0] for x in subsets], ignore_index=True)
    val_df=pd.concat([x[1] for x in subsets], ignore_index=True)
    return train_df, val_df


def load_as_router_image(path: str) -> torch.Tensor:
    vol = nib.load(path).get_fdata(dtype=np.float32)
    vol = np.clip(vol, -1000.0, 400.0)
    vol = (vol + 1000.0) / 1400.0
    t = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(64,64,64), mode='trilinear', align_corners=False)
    x = t.squeeze(1)
    mip = torch.stack([x.max(dim=1)[0], x.max(dim=2)[0], x.max(dim=3)[0]], dim=1)
    x = F.interpolate(mip, size=(224,224), mode='bilinear', align_corners=False)
    x = (x - torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)) / torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
    return x


def extract(df: pd.DataFrame, split: str, backbone, device: str):
    z_all=[]; y=[]
    with torch.no_grad():
        for i,row in enumerate(df.itertuples(index=False), start=1):
            x = load_as_router_image(row.filename).to(device)
            z = backbone(x).cpu().numpy()
            z_all.append(z)
            y.append(int(row.label))
            if i % 20 == 0:
                print(f'[{split}] {i}/{len(df)}')
    z_np=np.concatenate(z_all, axis=0).astype(np.float32)
    y_task=np.asarray(y, dtype=np.int64)
    y_expert=np.full(len(z_np), 4, dtype=np.int32)
    np.save(EMB / f'Z_{split}_pancreatic.npy', z_np)
    np.save(EMB / f'y_{split}_pancreatic.npy', y_task)
    np.savez_compressed(ROUT / f'Z_{split}_pancreatic.npz', z=z_np, y_task=y_task, y_expert=y_expert)
    return {'count': int(len(z_np)), 'dim': int(z_np.shape[1])}


def main():
    device='cuda' if torch.cuda.is_available() else 'cpu'
    backbone=timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0).to(device)
    backbone.eval()
    for p in backbone.parameters(): p.requires_grad=False
    train_df, val_df = build_manifest()
    meta={'train_rows':len(train_df),'val_rows':len(val_df)}
    print(json.dumps(meta, indent=2))
    print(json.dumps({'train': extract(train_df, 'train', backbone, device), 'val': extract(val_df, 'val', backbone, device)}, indent=2))

if __name__ == '__main__':
    main()
