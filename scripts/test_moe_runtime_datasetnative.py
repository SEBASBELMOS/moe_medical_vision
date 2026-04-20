
import sys, random
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, '/workspace/moe_medical_vision/src')
from models.moe_system import MoE_System
from data.datasets import get_dataloader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('Cargando MoE_System y los 5 Expertos Definitivos...')
model = MoE_System(device=device)
model.load_all_weights('/workspace/moe_medical_vision/checkpoints')
model.eval()

results = {i:{'router_correct':0,'total':0,'y_true':[],'y_pred':[],'classes_pred':set()} for i in range(5)}
print('\nIniciando Test Pequeño del pipeline end-to-end (10 por experto)...\n')

configs = {
 0: ('nih_chestxray','/workspace/moe_medical_vision/data/raw/nih'),
 1: ('isic2019','/workspace/moe_medical_vision/data/raw/isic'),
 2: ('osteoarthritis','/workspace/moe_medical_vision/data/raw/osteoporosis/KLGrade/KLGrade'),
 3: ('luna16','/workspace/moe_medical_vision/data/raw/luna16'),
 4: ('pancreatic','/workspace/moe_medical_vision/data/raw/pancreatic'),
}

for exp_id, (name, root) in configs.items():
    try:
        _, ds = get_dataloader(name, root, split='val', batch_size=1, num_workers=0)
    except Exception as e:
        print(f'[WARN] Could not load {name}: {e}')
        continue
    indices = random.sample(range(len(ds)), min(10, len(ds)))
    for idx in indices:
        item = ds[idx]
        y_true = item['label'].numpy() if exp_id == 0 else item['label']
        if exp_id in [0,1,2]:
            x_2d = item['image'].unsqueeze(0)
            if x_2d.shape[1] == 1:
                x_2d = x_2d.repeat(1,3,1,1)
            if x_2d.shape[-1] != 224:
                x_2d = F.interpolate(x_2d, size=(224,224), mode='bilinear', align_corners=False)
            with torch.no_grad():
                out, best_idx, _ = model(x_2d.to(device))
        else:
            x_3d = item['image'].unsqueeze(0)
            x_dummy = torch.zeros((1,3,224,224), dtype=torch.float32)
            with torch.no_grad():
                out, best_idx, _ = model(x_dummy.to(device), x_3d=x_3d.to(device))
        results[exp_id]['total'] += 1
        if best_idx == exp_id:
            results[exp_id]['router_correct'] += 1
        pred = torch.argmax(out, dim=-1).item()
        results[exp_id]['y_true'].append(y_true)
        results[exp_id]['y_pred'].append(pred)
        results[exp_id]['classes_pred'].add(pred)

NAMES = {0:'NIH ChestX-ray14',1:'ISIC 2019 (Piel)',2:'Osteoarthritis',3:'LUNA16 (Pulmón)',4:'Pancreatic Cancer'}
print('='*80)
print('  RESULTADOS FINALES DEL PIPELINE END-TO-END (50 IMAGENES)')
print('='*80)
total_router=sum(r['router_correct'] for r in results.values())
total_imgs=sum(r['total'] for r in results.values())
for exp_id in range(5):
    r=results[exp_id]; n=r['total']
    if n==0: continue
    print(f"\n--- Experto {exp_id}: {NAMES[exp_id]} ---")
    print(f"➜ Router Accuracy: {r['router_correct']}/{n} ({(r['router_correct']/n)*100:.1f}%)")
    if exp_id == 0:
        correct=sum(1 for yt, yp in zip(r['y_true'], r['y_pred']) if yt[yp]==1)
        print(f"➜ Expert Accuracy (Hit Multi-label): {correct}/{n} ({(correct/n)*100:.1f}%)")
    else:
        acc=accuracy_score(r['y_true'], r['y_pred'])
        f1=f1_score(r['y_true'], r['y_pred'], average='macro')
        print(f"➜ Expert Accuracy: {acc*100:.1f}% | Expert F1 Macro: {f1:.4f}")
    if len(r['classes_pred']) == 1:
        print(f"  ⚠️ ¡ALERTA! El modelo predijo la misma clase ({list(r['classes_pred'])[0]}) para las {n} imágenes.")
    else:
        print(f"  ✅ Modelo sano. Predijo {len(r['classes_pred'])} clases diferentes.")
print('\n' + '='*80)
print(f"ROUTING GLOBAL DEL SISTEMA: {total_router}/{total_imgs} ({(total_router/total_imgs)*100:.1f}%)")
print('='*80)
