import sys
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import nibabel as nib
import joblib
import faiss

sys.path.insert(0, '/workspace/moe_medical_vision/src')
from models.moe_system import MoE_System

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = MoE_System(device=device)
model.load_all_weights('/workspace/moe_medical_vision/checkpoints')
model.eval()

PATHS = {
    0: ('NIH (Tórax)', '/workspace/moe_medical_vision/data/raw/nih/images_001/images', '*.png'),
    1: ('ISIC (Piel)', '/workspace/moe_medical_vision/data/raw/isic/ISIC_2019_Training_Input/ISIC_2019_Training_Input', '*.jpg'),
    2: ('Osteo (Rodilla)', '/workspace/moe_medical_vision/data/raw/osteoporosis/train', '**/*.jpg'),
    3: ('LUNA16 (Pulmón 3D)', '/workspace/moe_medical_vision/data/raw/luna16/seg-lungs-LUNA16/seg-lungs-LUNA16', '*.mhd'),
    4: ('Pancreatic (Tumor 3D)', '/workspace/moe_medical_vision/data/raw/pancreatic', '*_0000.nii.gz')
}

def get_random_samples(n=1):
    samples = []
    for exp_id, (name, path, ext) in PATHS.items():
        base_dir = Path(path)
        files = list(base_dir.glob(ext))
        if files:
            selected = random.sample(files, min(n, len(files)))
            for s in selected: samples.append((exp_id, name, str(s)))
    return samples

def process_2d(file_path):
    img = Image.open(file_path).convert('RGB').resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = (img_array - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    x_224 = torch.from_numpy(img_array).float().permute(2, 0, 1).unsqueeze(0)
    return x_224, None

def process_3d(file_path):
    if file_path.endswith('.mhd'):
        import SimpleITK as sitk
        vol = sitk.GetArrayFromImage(sitk.ReadImage(file_path)).astype(np.float32)
    else:
        vol = nib.load(file_path).get_fdata(dtype=np.float32)
        
    vol = np.clip(vol, -1000.0, 400.0)
    vol = (vol + 1000.0) / 1400.0
    x_3d_raw = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)
    x_3d = F.interpolate(x_3d_raw, size=(64, 64, 64), mode='trilinear', align_corners=False)
    x = x_3d.squeeze(1)
    mip = torch.stack([x.max(dim=1)[0], x.max(dim=2)[0], x.max(dim=3)[0]], dim=1)
    x_router = F.interpolate(mip, size=(224, 224), mode='bilinear', align_corners=False)
    return x_router, x_3d

random_samples = get_random_samples(2)
random.shuffle(random_samples)

print("\n" + "="*80)
print("  TEST CIEGO ALEATORIO: 10 IMÁGENES (2 DE CADA DATASET)")
print("="*80)

correct_routes = 0

for exp_id_real, name, ruta in random_samples:
    is_3d = ruta.endswith('.nii.gz') or ruta.endswith('.mhd')
    x_router, x_3d = process_3d(ruta) if is_3d else process_2d(ruta)
    
    with torch.no_grad():
        out, best_idx, probs = model(x_router.to(device), x_3d=x_3d.to(device) if x_3d is not None else None)
        
    router_success = (best_idx == exp_id_real)
    if router_success: correct_routes += 1
        
    print(f"[{name}] -> {Path(ruta).name}")
    print(f"  ➜ Enrutado Experto: {best_idx} | Real: {exp_id_real} | {'✅ ACIERTO' if router_success else '❌ FALLO'}")
    print("-" * 80)

print(f"\nRESUMEN ROUTING ACCURACY: {correct_routes}/{len(random_samples)} ({correct_routes/len(random_samples)*100:.1f}%)\n")
