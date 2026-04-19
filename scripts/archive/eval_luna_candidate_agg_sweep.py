import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, '/workspace/moe_medical_vision/src')
from data.datasets import LUNA16Dataset
from models.experts_3d import build_luna_patch_expert_mc3

seed=42
device='cuda' if torch.cuda.is_available() else 'cpu'
RAW_ROOT = Path('/workspace/moe_medical_vision/data/raw/luna16')
FAST_ROOT = Path('/workspace/moe_medical_vision/data/processed/luna16_fast')
PATCH_SIZE=32; TARGET_SIZE=64
VAL_NEG_POS_RATIO=8

def build_split_map():
    ds = LUNA16Dataset(RAW_ROOT, split='val', transform=None)
    mapping={}
    for i,(path,study_label) in enumerate(ds.samples):
        mapping[path.stem]={'npz': FAST_ROOT/f'val_{i:05d}.npz', 'mhd': path, 'study_label': int(study_label)}
    return mapping

SPLIT_MAP = build_split_map()
CAND = pd.read_csv(RAW_ROOT/'candidates.csv')

class CandVal(Dataset):
    def __init__(self):
        df=CAND[CAND['seriesuid'].isin(SPLIT_MAP.keys())].copy()
        pos=df[df['class']==1].copy(); neg=df[df['class']==0].copy().sample(n=min(len(df[df['class']==0]), len(pos)*VAL_NEG_POS_RATIO), random_state=seed)
        self.df=pd.concat([pos,neg], ignore_index=True).reset_index(drop=True)
        self.meta={}
        for sid,info in SPLIT_MAP.items():
            img=sitk.ReadImage(str(info['mhd']))
            self.meta[sid]={'size_xyz':img.GetSize()}
    def __len__(self): return len(self.df)
    def _crop(self, vol, c, size=PATCH_SIZE):
        z,y,x=c; h=size//2
        z0,z1=z-h,z+h; y0,y1=y-h,y+h; x0,x1=x-h,x+h
        out=np.zeros((size,size,size),dtype=np.float32)
        sz0,sz1=max(0,z0),min(vol.shape[0],z1); sy0,sy1=max(0,y0),min(vol.shape[1],y1); sx0,sx1=max(0,x0),min(vol.shape[2],x1)
        dz0,dy0,dx0=sz0-z0,sy0-y0,sx0-x0; dz1,dy1,dx1=dz0+(sz1-sz0),dy0+(sy1-sy0),dx0+(sx1-sx0)
        out[dz0:dz1,dy0:dy1,dx0:dx1]=vol[sz0:sz1,sy0:sy1,sx0:sx1]
        return out
    def __getitem__(self, idx):
        row=self.df.iloc[idx]; sid=row['seriesuid']; info=SPLIT_MAP[sid]; meta=self.meta[sid]
        d=np.load(info['npz']); vol=d['volume'][0]
        img=sitk.ReadImage(str(info['mhd']))
        idx_xyz=img.TransformPhysicalPointToIndex((float(row['coordX']),float(row['coordY']),float(row['coordZ'])))
        sx=(TARGET_SIZE-1)/max(meta['size_xyz'][0]-1,1); sy=(TARGET_SIZE-1)/max(meta['size_xyz'][1]-1,1); sz=(TARGET_SIZE-1)/max(meta['size_xyz'][2]-1,1)
        x=int(round(idx_xyz[0]*sx)); y=int(round(idx_xyz[1]*sy)); z=int(round(idx_xyz[2]*sz))
        x=max(0,min(TARGET_SIZE-1,x)); y=max(0,min(TARGET_SIZE-1,y)); z=max(0,min(TARGET_SIZE-1,z))
        patch=self._crop(vol,(z,y,x),PATCH_SIZE)
        x_t=torch.from_numpy(patch.copy()).float().unsqueeze(0)
        x_t=F.interpolate(x_t.unsqueeze(0), size=(TARGET_SIZE,TARGET_SIZE,TARGET_SIZE), mode='trilinear', align_corners=False).squeeze(0)
        return {'image':x_t, 'seriesuid':sid, 'study_label': int(info['study_label'])}

ck='/workspace/moe_medical_vision/checkpoints/expert4_luna16_candidate_REAL_v2_best.pth'
if not Path(ck).exists():
    print('NO_CANDIDATE_MODEL')
    raise SystemExit(0)
model=build_luna_patch_expert_mc3(pretrained=False,use_gradient_checkpointing=False).to(device)
ckpt=torch.load(ck, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict']); model.eval()
loader=DataLoader(CandVal(), batch_size=8, shuffle=False)
by_series=defaultdict(list); by_label={}
with torch.no_grad():
    for batch in loader:
        p=torch.softmax(model(batch['image'].to(device)), dim=1)[:,1].cpu().numpy()
        for i,sid in enumerate(batch['seriesuid']):
            by_series[sid].append(float(p[i])); by_label[sid]=int(batch['study_label'][i])

best=(-1,None,None)
for mode in ['max','top3_mean','top5_mean']:
    for thr in np.linspace(0.05,0.95,37):
        yt=[]; yp=[]
        for sid,plist in by_series.items():
            arr=np.array(sorted(plist, reverse=True))
            if mode=='max': score=arr[0]
            elif mode=='top3_mean': score=arr[:min(3,len(arr))].mean()
            else: score=arr[:min(5,len(arr))].mean()
            yt.append(by_label[sid]); yp.append(1 if score>=thr else 0)
        f1=f1_score(yt,yp,average='macro')
        if f1>best[0]: best=(f1,mode,thr)
    print(f'{mode}: scanned')
print(f'CAND_AGG_BEST val_f1={best[0]:.4f} mode={best[1]} thr={best[2]:.3f}')
