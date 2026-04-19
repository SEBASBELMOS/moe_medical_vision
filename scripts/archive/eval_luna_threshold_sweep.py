import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision.models import resnet18

sys.path.insert(0, '/workspace/moe_medical_vision/src')
from models.experts_3d import build_luna_expert

FAST_ROOT = Path('/workspace/moe_medical_vision/data/processed/luna16_fast')

device='cuda' if torch.cuda.is_available() else 'cpu'

class LunaFast3D(Dataset):
    def __init__(self, root):
        self.files = sorted(root.glob('val_*.npz'))
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        d=np.load(self.files[idx])
        return torch.from_numpy(d['volume'].copy()).float(), int(d['label'])

class Luna2p5DVal(Dataset):
    def __init__(self, root):
        self.files = sorted(root.glob('val_*.npz'))
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

m3d=build_luna_expert(pretrained=False,use_gradient_checkpointing=False).to(device)
c3d=torch.load('/workspace/moe_medical_vision/checkpoints/expert4_luna16_FAST_best.pth', map_location=device, weights_only=False)
m3d.load_state_dict(c3d['model_state_dict'])
m3d.eval()

m2d=resnet18(weights=None)
m2d.fc=torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(m2d.fc.in_features,2))
m2d=m2d.to(device)
c2d=torch.load('/workspace/moe_medical_vision/checkpoints/expert4_luna16_2p5d_FAST_best.pth', map_location=device, weights_only=False)
m2d.load_state_dict(c2d['model_state_dict'])
m2d.eval()

loader3d=DataLoader(LunaFast3D(FAST_ROOT), batch_size=8, shuffle=False)
loader2d=DataLoader(Luna2p5DVal(FAST_ROOT), batch_size=8, shuffle=False)

p3d_all=[]; p2d_all=[]; labels=[]
with torch.no_grad():
    for (x3d,y1),(x2d,y2) in zip(loader3d, loader2d):
        x3d=x3d.to(device); x2d=x2d.to(device)
        p3d=torch.softmax(m3d(x3d), dim=1)[:,1].cpu().numpy()
        p2d=torch.softmax(m2d(x2d), dim=1)[:,1].cpu().numpy()
        p3d_all.extend(p3d.tolist())
        p2d_all.extend(p2d.tolist())
        labels.extend(y1.tolist())

p3d_all=np.array(p3d_all)
p2d_all=np.array(p2d_all)
labels=np.array(labels)
ensemble=0.35*p3d_all + 0.65*p2d_all

best=('none',-1,None)
for name, probs in [('3d',p3d_all), ('2p5d',p2d_all), ('ens_weighted',ensemble)]:
    local_best=(-1,None)
    for thr in np.linspace(0.05,0.95,37):
        pred=(probs>=thr).astype(int)
        f1=f1_score(labels,pred,average='macro')
        if f1>local_best[0]:
            local_best=(f1,thr)
        if f1>best[1]:
            best=(name,f1,thr)
    print(f'{name}: best_f1={local_best[0]:.4f} @ thr={local_best[1]:.3f}')

print(f'GLOBAL_BEST model={best[0]} val_f1={best[1]:.4f} thr={best[2]:.3f}')
