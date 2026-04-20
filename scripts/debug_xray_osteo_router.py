
import sys, random, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0,'/workspace/moe_medical_vision/src')
from models.moe_system import MoE_System
from data.datasets import get_dataloader

device='cuda' if torch.cuda.is_available() else 'cpu'
model=MoE_System(device=device)
model.load_all_weights('/workspace/moe_medical_vision/checkpoints')
model.eval()

# NIH and OA only
cfg={0:('nih_chestxray','/workspace/moe_medical_vision/data/raw/nih'),2:('osteoarthritis','/workspace/moe_medical_vision/data/raw/osteoporosis/KLGrade/KLGrade')}
res=[]
for exp_id,(name,root) in cfg.items():
    _, ds = get_dataloader(name, root, split='val', batch_size=1, num_workers=0)
    indices=random.sample(range(len(ds)), min(5,len(ds)))
    for idx in indices:
        item=ds[idx]
        x_exp=item['image'].unsqueeze(0)
        if x_exp.shape[1]==1: x_exp=x_exp.repeat(1,3,1,1)
        if x_exp.shape[-1]!=224: x_exp=F.interpolate(x_exp, size=(224,224), mode='bilinear', align_corners=False)
        x_router=x_exp.to(device)
        x_for_router=model._prepare_for_router(x_router)
        color_gap=(torch.mean(torch.abs(x_for_router[:,0]-x_for_router[:,1])) + torch.mean(torch.abs(x_for_router[:,1]-x_for_router[:,2]))).item()
        z=model.backbone.forward_features(x_for_router)[:,0,:]
        probs=torch.softmax(model.router_xray_osteo(z), dim=-1)
        local=int(torch.argmax(probs, dim=-1).item())
        final=[0,2][local]
        res.append({'true_expert': exp_id, 'idx': idx, 'color_gap': color_gap, 'router_probs': probs[0].detach().cpu().tolist(), 'local_idx': local, 'final_expert': final})
print(json.dumps(res, indent=2))
