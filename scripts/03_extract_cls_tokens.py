"""
Task 2.4 - Extracción masiva de CLS tokens (Memoria Protegida y Guardado Incremental)
"""
import sys
import gc
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import timm
from tqdm import tqdm

sys.path.insert(0, '/workspace/moe_medical_vision/src')
from data.datasets import get_dataloader

seed = 42
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

OUT_DIR = Path('/workspace/moe_medical_vision/data/processed/router_embeddings')
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Instanciando ViT-Tiny congelado...", flush=True)
backbone = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0).to(device)
backbone.eval()
for p in backbone.parameters(): p.requires_grad = False

DATASETS = {
    'nih_chestxray': '/workspace/moe_medical_vision/data/raw/nih',
    'isic2019': '/workspace/moe_medical_vision/data/raw/isic',
    'osteoarthritis': '/workspace/moe_medical_vision/data/raw/osteoporosis/train',
    'luna16': '/workspace/moe_medical_vision/data/processed/luna16_fast',
    'pancreatic': '/workspace/moe_medical_vision/data/processed/pancreatic_fast'
}
EXPERT_MAP = {'nih_chestxray': 0, 'isic2019': 1, 'osteoarthritis': 2, 'luna16': 3, 'pancreatic': 4}

def extract_features(loader, split, ds_name, expert_id):
    z_all, y_expert, y_task = [], [], []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader, desc=f'[{split}] {ds_name}', leave=False)):
            x = batch['image'].to(device)
            
            if x.ndim == 5:
                x = x.squeeze(1)
                mip = torch.stack([x.max(dim=1)[0], x.max(dim=2)[0], x.max(dim=3)[0]], dim=1)
                x = torch.nn.functional.interpolate(mip, size=(224, 224), mode='bilinear', align_corners=False)
            elif x.ndim == 4 and x.shape[1] == 1: x = x.repeat(1, 3, 1, 1)
            
            if x.shape[-1] != 224:
                x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            
            z = backbone(x)
            z_all.append(z.cpu().numpy()); y_expert.extend([expert_id]*x.shape[0]); y_task.extend(batch['label'].cpu().tolist())
            if idx % 50 == 0: print(f"[{split} {ds_name}] {idx}/{len(loader)} batches procesados...", flush=True)
    return np.concatenate(z_all, axis=0), np.array(y_expert, dtype=np.int32), np.array(y_task, dtype=np.int32)

for split in ['train', 'val']:
    split_Z, split_Y_exp, split_Y_task = [], [], []
    print(f"\nProcesando split: {split}", flush=True)
    
    for ds_name, root in DATASETS.items():
        # Check if already processed
        dataset_out_file = OUT_DIR / f'Z_{split}_{ds_name}.npz'
        if dataset_out_file.exists():
            print(f"  -> {ds_name} ya procesado. Cargando de disco...", flush=True)
            data = np.load(dataset_out_file)
            split_Z.append(data['z'])
            split_Y_exp.append(data['y_expert'])
            split_Y_task.append(data['y_task'])
            continue

        print(f"  -> Iniciando {ds_name}...", flush=True)
        if 'fast' in root:
            class NPZFast(Dataset):
                def __init__(self, r, s): self.files = sorted(Path(r).glob(f'{s}_*.npz'))
                def __len__(self): return len(self.files)
                def __getitem__(self, idx):
                    d = np.load(self.files[idx])
                    return {'image': torch.from_numpy(d['volume'][0].copy()).float().unsqueeze(0), 'label': int(d['label'])}
            loader = DataLoader(NPZFast(root, split), batch_size=256, num_workers=4)
        else:
            loader, ds_obj = get_dataloader(ds_name, root, split=split, batch_size=256, num_workers=4, transform=None)
            if hasattr(loader, 'sampler') and not isinstance(loader.sampler, torch.utils.data.SequentialSampler):
                loader = DataLoader(ds_obj, batch_size=256, num_workers=4)

        z, ye, yt = extract_features(loader, split, ds_name, EXPERT_MAP[ds_name])
        split_Z.append(z); split_Y_exp.append(ye); split_Y_task.append(yt)
        
        # Save dataset cache to disk to resume later if something fails
        np.savez_compressed(dataset_out_file, z=z, y_expert=ye, y_task=yt)
        
        print(f"  -> Finalizado {ds_name}. Liberando RAM...", flush=True)
        del loader; gc.collect(); torch.cuda.empty_cache()
        
    final_Z = np.concatenate(split_Z, axis=0)
    final_Y_exp = np.concatenate(split_Y_exp, axis=0)
    final_Y_task = np.concatenate(split_Y_task, axis=0)
    np.savez_compressed(OUT_DIR / f'Z_{split}.npz', z=final_Z, y_expert=final_Y_exp, y_task=final_Y_task)
    print(f'Guardado {split} MAESTRO: Z={final_Z.shape}\n', flush=True)

print("Extracción Masiva de Tokens COMPLETADA.", flush=True)
