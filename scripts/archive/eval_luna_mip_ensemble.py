import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision.models import efficientnet_b0
import torch.nn as nn

sys.path.insert(0, '/workspace/moe_medical_vision/src')
from models.experts_3d import build_luna_expert

FAST_ROOT = Path('/workspace/moe_medical_vision/data/processed/luna16_fast')
device='cuda' if torch.cuda.is_available() else 'cpu'

def norm(t):
    t_min, t_max = t.min(), t.max()
    if t_max > t_min: return (t - t_min) / (t_max - t_min)
    return t

class LunaFast3D(Dataset):
    def __init__(self, root): self.files = sorted(root.glob('val_*.npz'))
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        d=np.load(self.files[idx])
        return torch.from_numpy(d['volume'].copy()).float(), int(d['label'])

class LunaMIPVal(Dataset):
    def __init__(self, root): self.files = sorted(root.glob('val_*.npz'))
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        d=np.load(self.files[idx])
        vol=torch.from_numpy(d['volume'][0].copy()).float()
        img=torch.stack([norm(vol.max(dim=0)[0]), norm(vol.mean(dim=0)), norm(vol.std(dim=0))], dim=0)
        img=F.interpolate(img.unsqueeze(0), size=(224,224), mode='bilinear', align_corners=False).squeeze(0)
        mean=torch.tensor([0.485,0.456,0.406]).view(3,1,1)
        std=torch.tensor([0.229,0.224,0.225]).view(3,1,1)
        img=(img-mean)/std
        return img, int(d['label'])

# 3D best model (0.5911)
m3d=build_luna_expert(pretrained=False,use_gradient_checkpointing=False).to(device)
m3d.load_state_dict(torch.load('/workspace/moe_medical_vision/checkpoints/expert4_luna16_FAST_best.pth', map_location=device, weights_only=False)['model_state_dict']); m3d.eval()

# MIP best model (0.6153)
mMIP=efficientnet_b0(weights=None)
mMIP.classifier[1]=nn.Sequential(nn.Dropout(0.4), nn.Linear(mMIP.classifier[1].in_features, 2))
mMIP=mMIP.to(device)
mMIP.load_state_dict(torch.load('/workspace/moe_medical_vision/checkpoints/expert4_luna16_MIP_best.pth', map_location=device, weights_only=False)['model_state_dict']); mMIP.eval()

loader3d=DataLoader(LunaFast3D(FAST_ROOT), batch_size=8, shuffle=False)
loaderMIP=DataLoader(LunaMIPVal(FAST_ROOT), batch_size=8, shuffle=False)

p3d_all=[]; pMIP_all=[]; labels=[]
with torch.no_grad():
    for (x3d,y1),(xMIP,y2) in zip(loader3d, loaderMIP):
        p3d_all.extend(torch.softmax(m3d(x3d.to(device)), dim=1)[:,1].cpu().tolist())
        pMIP_all.extend(torch.softmax(mMIP(xMIP.to(device)), dim=1)[:,1].cpu().tolist())
        labels.extend(y1.tolist())

p3d=np.array(p3d_all); pMIP=np.array(pMIP_all); ys=np.array(labels)

best=(0, None, None)
for wMIP in np.linspace(0, 1.0, 21):
    w3d = 1.0 - wMIP
    probs = wMIP*pMIP + w3d*p3d
    for thr in np.linspace(0.05, 0.95, 37):
        pred=(probs>=thr).astype(int)
        f1=f1_score(ys, pred, average='macro')
        if f1 > best[0]: best=(f1, wMIP, thr)

print(f'MIP ENSEMBLE BEST val_f1={best[0]:.4f} (wMIP={best[1]:.2f}, w3D={1-best[1]:.2f}) thr={best[2]:.3f}')
