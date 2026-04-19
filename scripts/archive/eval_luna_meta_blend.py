import sys
from pathlib import Path
from collections import defaultdict
import itertools
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision.models import resnet18

sys.path.insert(0, '/workspace/moe_medical_vision/src')
from models.experts_3d import build_luna_expert, build_luna_patch_expert_mc3
from data.datasets import LUNA16Dataset

FAST_ROOT = Path('/workspace/moe_medical_vision/data/processed/luna16_fast')
RAW_ROOT = Path('/workspace/moe_medical_vision/data/raw/luna16')
device='cuda' if torch.cuda.is_available() else 'cpu'

# ---------- datasets ----------
class LunaFast3DVal(Dataset):
    def __init__(self, root): self.files = sorted(root.glob('val_*.npz'))
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        d=np.load(self.files[idx])
        return torch.from_numpy(d['volume'].copy()).float(), int(d['label'])

class Luna2p5DVal(Dataset):
    def __init__(self, root): self.files = sorted(root.glob('val_*.npz'))
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        d=np.load(self.files[idx])
        vol=d['volume'][0]
        zmid=vol.shape[0]//2
        z_ids=[max(0,min(vol.shape[0]-1,zmid+off)) for off in (-8,0,8)]
        stack=torch.from_numpy(vol[z_ids].copy()).float()
        stack=F.interpolate(stack.unsqueeze(0), size=(224,224), mode='bilinear', align_corners=False).squeeze(0)
        stack=(stack-0.5)/0.25
        return stack, int(d['label'])

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
        pos=df[df['class']==1].copy(); neg=df[df['class']==0].copy().sample(n=min(len(df[df['class']==0]), len(pos)*8), random_state=42)
        self.df=pd.concat([pos,neg], ignore_index=True).reset_index(drop=True)
        self.meta={}
        for sid,info in SPLIT_MAP.items():
            img=sitk.ReadImage(str(info['mhd']))
            self.meta[sid]={'size_xyz':img.GetSize()}
    def __len__(self): return len(self.df)
    def _crop(self, vol, c, size=32):
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
        sx=(63)/max(meta['size_xyz'][0]-1,1); sy=(63)/max(meta['size_xyz'][1]-1,1); sz=(63)/max(meta['size_xyz'][2]-1,1)
        x=int(round(idx_xyz[0]*sx)); y=int(round(idx_xyz[1]*sy)); z=int(round(idx_xyz[2]*sz))
        x=max(0,min(63,x)); y=max(0,min(63,y)); z=max(0,min(63,z))
        patch=self._crop(vol,(z,y,x),32)
        x_t=torch.from_numpy(patch.copy()).float().unsqueeze(0)
        x_t=F.interpolate(x_t.unsqueeze(0), size=(64,64,64), mode='trilinear', align_corners=False).squeeze(0)
        return {'image':x_t, 'seriesuid':sid, 'study_label': int(info['study_label'])}

# ---------- load models ----------
m3d=build_luna_expert(pretrained=False,use_gradient_checkpointing=False).to(device)
c3d=torch.load('/workspace/moe_medical_vision/checkpoints/expert4_luna16_FAST_best.pth', map_location=device, weights_only=False)
m3d.load_state_dict(c3d['model_state_dict']); m3d.eval()

m2d=resnet18(weights=None)
m2d.fc=torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(m2d.fc.in_features,2))
m2d=m2d.to(device)
c2d=torch.load('/workspace/moe_medical_vision/checkpoints/expert4_luna16_2p5d_FAST_best.pth', map_location=device, weights_only=False)
m2d.load_state_dict(c2d['model_state_dict']); m2d.eval()

cand_path='/workspace/moe_medical_vision/checkpoints/expert4_luna16_candidate_REAL_v2_best.pth'
mCand=None
if Path(cand_path).exists():
    mCand=build_luna_patch_expert_mc3(pretrained=False,use_gradient_checkpointing=False).to(device)
    cCand=torch.load(cand_path, map_location=device, weights_only=False)
    mCand.load_state_dict(cCand['model_state_dict']); mCand.eval()

# ---------- collect study-level signals ----------
loader3d=DataLoader(LunaFast3DVal(FAST_ROOT), batch_size=8, shuffle=False)
loader2d=DataLoader(Luna2p5DVal(FAST_ROOT), batch_size=8, shuffle=False)

study_ids = sorted(SPLIT_MAP.keys())
study_labels = {sid: info['study_label'] for sid, info in SPLIT_MAP.items()}

# 3D / 2.5D per study directly aligned by val file order
p3d=[]; p2d=[]; ys=[]
with torch.no_grad():
    for (x3d,y1),(x2d,y2) in zip(loader3d, loader2d):
        p3d.extend(torch.softmax(m3d(x3d.to(device)), dim=1)[:,1].cpu().tolist())
        p2d.extend(torch.softmax(m2d(x2d.to(device)), dim=1)[:,1].cpu().tolist())
        ys.extend(y1.tolist())
p3d=np.array(p3d); p2d=np.array(p2d); ys=np.array(ys)

cand_scores = {sid: 0.0 for sid in study_ids}
if mCand is not None:
    loaderCand=DataLoader(CandVal(), batch_size=8, shuffle=False)
    by_series=defaultdict(list)
    with torch.no_grad():
        for batch in loaderCand:
            probs=torch.softmax(mCand(batch['image'].to(device)), dim=1)[:,1].cpu().numpy()
            for i,sid in enumerate(batch['seriesuid']):
                by_series[sid].append(float(probs[i]))
    for sid,plist in by_series.items():
        cand_scores[sid]=max(plist)

cand=np.array([cand_scores[sid] for sid in study_ids])

best=(-1,None,None)
for w3d in np.linspace(0,1,11):
    for w2d in np.linspace(0,1-w3d,11):
        wc = 1.0 - w3d - w2d
        if wc < -1e-9: continue
        combo = w3d*p3d + w2d*p2d + wc*cand
        for thr in np.linspace(0.05,0.95,37):
            pred=(combo>=thr).astype(int)
            f1=f1_score(ys,pred,average='macro')
            if f1>best[0]:
                best=(f1,(w3d,w2d,wc),thr)
print(f'META_BLEND_BEST val_f1={best[0]:.4f} weights={best[1]} thr={best[2]:.3f}')
