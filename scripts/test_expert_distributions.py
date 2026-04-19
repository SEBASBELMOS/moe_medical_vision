import sys
import torch
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from collections import Counter

sys.path.insert(0, '/workspace/moe_medical_vision/src')
from models.moe_system import MoE_System
from data.datasets import get_dataloader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Mapeos de clases
MAPAS = {
    1: {0: 'Melanoma', 1: 'Nevo Melanocítico', 2: 'Carcinoma Basocelular', 3: 'Queratosis Actínica', 4: 'Queratosis Benigna', 5: 'Dermatofibroma', 6: 'Lesión Vascular', 7: 'Carcinoma Espinocelular', 8: 'Otro / Sano'},
    3: {0: 'Sano', 1: 'Nódulo Pulmonar'},
    4: {0: 'Sano', 1: 'Tumor Pancreático'}
}

def test_expert_distribution(dataset_name, expert_idx, root_path, num_samples=100):
    print(f"\n============================================================")
    print(f"TESTING EXPERT {expert_idx} ({dataset_name}) - 100 IMÁGENES VAL")
    print(f"============================================================")
    
    if 'fast' in root_path:
        from torch.utils.data import Dataset, DataLoader
        class NPZFast(Dataset):
            def __init__(self, r, s): self.files = sorted(Path(r).glob(f'{s}_*.npz'))
            def __len__(self): return len(self.files)
            def __getitem__(self, idx):
                d = np.load(self.files[idx])
                vol = torch.from_numpy(d['volume'][0].copy()).float()
                
                # Transformar 3D a MIP 2D como hicimos en app.py
                mip = torch.stack([vol.max(dim=0)[0], vol.mean(dim=0), vol.std(dim=0)], dim=0)
                img_224 = F.interpolate(mip.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_224 = (img_224 - mean) / std
                
                return {'image': img_224, 'image_raw': vol.unsqueeze(0), 'label': int(d['label'])}
                
        ds = NPZFast(root_path, 'val')
        loader = DataLoader(ds, batch_size=1, shuffle=False)
    else:
        loader, _ = get_dataloader(dataset_name, root_path, split='val', batch_size=1, num_workers=0)
        
    y_true = []
    y_pred = []
    
    model = MoE_System(device=device)
    model.load_all_weights('/workspace/moe_medical_vision/checkpoints')
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_samples: break
            
            x_224 = batch['image'].to(device)
            # Control de canal 1 (Ejemplo ISIC grisáceo)
            if x_224.shape[1] == 1: x_224 = x_224.repeat(1, 3, 1, 1)
                
            y_true.append(batch['label'].item())
            
            expert_input = x_224
            if expert_idx == 4: # Páncreas
                expert_input = batch['image_raw'].to(device)
            
            # Predict
            out = model.experts[expert_idx](expert_input)
            pred = torch.argmax(torch.softmax(out, dim=-1), dim=-1).item()
            y_pred.append(pred)

    c_true = Counter(y_true)
    c_pred = Counter(y_pred)
    
    print("\nDistribución REAL de las imágenes:")
    for k, v in c_true.items(): print(f"  - {MAPAS[expert_idx].get(k, k)}: {v}")
        
    print("\nDistribución PREDICHA por el Experto:")
    for k, v in c_pred.items(): print(f"  - {MAPAS[expert_idx].get(k, k)}: {v}")
    
    from sklearn.metrics import f1_score
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"\nF1 Macro calculado: {f1:.4f}")

# Script principal
print("Arrancando Test Diagnóstico...")
test_expert_distribution('isic2019', 1, '/workspace/moe_medical_vision/data/raw/isic', num_samples=100)
test_expert_distribution('luna16', 3, '/workspace/moe_medical_vision/data/processed/luna16_fast', num_samples=100)
test_expert_distribution('pancreatic', 4, '/workspace/moe_medical_vision/data/processed/pancreatic_fast', num_samples=100)
