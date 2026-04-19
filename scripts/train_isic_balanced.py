import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from sklearn.metrics import f1_score

sys.path.insert(0, '/workspace/moe_medical_vision/src')
from data.datasets import get_dataloader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Cargando ISIC 2019 (Piel) con Balanceo Fuerte...")
ROOT = '/workspace/moe_medical_vision/data/raw/isic'
train_loader, train_ds = get_dataloader('isic2019', ROOT, split='train', batch_size=64, num_workers=4)
val_loader, val_ds = get_dataloader('isic2019', ROOT, split='val', batch_size=64, num_workers=4)

labels_train = [int(train_ds[i]['label']) for i in range(len(train_ds))]
counts = np.bincount(labels_train, minlength=9)
class_weights = 1.0 / (counts + 1e-6)
sample_weights = class_weights[labels_train]

sampler = WeightedRandomSampler(sample_weights.tolist(), len(train_ds), replacement=True)
train_loader_b = DataLoader(train_ds, batch_size=64, sampler=sampler, num_workers=4, pin_memory=True)

model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
model.classifier[1] = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.classifier[1].in_features, 9))
model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=torch.tensor((sum(counts)) / (2 * counts + 1e-6), dtype=torch.float32).to(device), label_smoothing=0.05)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

best_f1 = -1.0
CKPT = Path('/workspace/moe_medical_vision/checkpoints/expert2_isic_best_fixed.pth')

for epoch in range(1, 11):
    model.train()
    tr_y, tr_p = [], []
    for batch in train_loader_b:
        x, y = batch['image'].to(device), batch['label'].to(device)
        out = model(x)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tr_p.extend(out.argmax(1).cpu().tolist())
        tr_y.extend(y.cpu().tolist())
    scheduler.step()
    
    model.eval()
    va_y, va_p = [], []
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch['image'].to(device), batch['label'].to(device)
            out = model(x)
            va_p.extend(out.argmax(1).cpu().tolist())
            va_y.extend(y.cpu().tolist())
            
    val_f1 = f1_score(va_y, va_p, average='macro')
    print(f"[Epoch {epoch:02d}] Train F1: {f1_score(tr_y, tr_p, average='macro'):.4f} | Val F1: {val_f1:.4f}")
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save({'model_state_dict': model.state_dict(), 'best_val_f1': best_f1}, CKPT)

print(f"ISIC Re-entrenado: Best F1 = {best_f1:.4f}")
