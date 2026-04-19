import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision.models import resnet18

sys.path.insert(0, '/workspace/moe_medical_vision/src')
from models.experts_3d import build_luna_expert

FAST_ROOT = Path('/workspace/moe_medical_vision/data/processed/luna16_fast')

def build_val_indices(n=177, seed=42, val_frac=0.2):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = int(n * val_frac)
    return idx[:n_val]

class LunaFast3DAll(Dataset):
    def __init__(self, root, split='all'):
        self.files = sorted(root.glob('val_*.npz')) if split=='val' else sorted(root.glob('train_*.npz'))
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        d=np.load(self.files[idx])
        return torch.from_numpy(d['volume'].copy()).float(), int(d['label'])

class Luna2p5DAll(Dataset):
    def __init__(self, root, split='all'):
        self.files = sorted(root.glob('val_*.npz')) if split=='val' else sorted(root.glob('train_*.npz'))
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

device='cuda' if torch.cuda.is_available() else 'cpu'

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

# train split for stacking fit
loader3d_tr=DataLoader(LunaFast3DAll(FAST_ROOT,'train'), batch_size=8, shuffle=False)
loader2d_tr=DataLoader(Luna2p5DAll(FAST_ROOT,'train'), batch_size=8, shuffle=False)
loader3d_va=DataLoader(LunaFast3DAll(FAST_ROOT,'val'), batch_size=8, shuffle=False)
loader2d_va=DataLoader(Luna2p5DAll(FAST_ROOT,'val'), batch_size=8, shuffle=False)

def collect(loader3d, loader2d):
    feats=[]; labels=[]
    with torch.no_grad():
        for (x3d,y1),(x2d,y2) in zip(loader3d, loader2d):
            x3d=x3d.to(device); x2d=x2d.to(device)
            p3d=torch.softmax(m3d(x3d), dim=1).cpu().numpy()
            p2d=torch.softmax(m2d(x2d), dim=1).cpu().numpy()
            feat=np.concatenate([p3d,p2d], axis=1)
            feats.append(feat)
            labels.extend(y1.tolist())
    return np.concatenate(feats, axis=0), np.array(labels)

Xtr,ytr=collect(loader3d_tr, loader2d_tr)
Xva,yva=collect(loader3d_va, loader2d_va)
clf=LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
clf.fit(Xtr,ytr)
pred=clf.predict(Xva)
f1=f1_score(yva,pred,average='macro')
print(f'LUNA STACKING val_f1={f1:.4f}')
print('coef=', clf.coef_.tolist())
