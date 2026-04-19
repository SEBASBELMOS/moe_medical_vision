import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, '/workspace/moe_medical_vision/src')
from data.luna_mil_dataset import LUNAMILBagDataset
from models.luna_mil import LUNAMILModel
from train.train_3d import seed_everything

seed_everything(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device={device}')

RAW_ROOT = '/workspace/moe_medical_vision/data/raw/luna16'
FAST_ROOT = '/workspace/moe_medical_vision/data/processed/luna16_fast'
CKPT_OUT = Path('/workspace/moe_medical_vision/checkpoints/expert4_luna16_MIL_v1_best.pth')

train_ds = LUNAMILBagDataset(RAW_ROOT, FAST_ROOT, split='train', k_instances=16, patch_size=32, target_size=64, neg_pos_ratio=1, seed=42)
val_ds = LUNAMILBagDataset(RAW_ROOT, FAST_ROOT, split='val', k_instances=32, patch_size=32, target_size=64, neg_pos_ratio=8, seed=42)
train_loader = DataLoader(train_ds, batch_size=2, sampler=train_ds.get_study_weighted_sampler(), num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)

model = LUNAMILModel(pretrained=True, use_gradient_checkpointing=True, attn_hidden_dim=128, dropout=0.3).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=train_ds.get_class_weights().to(device), label_smoothing=0.02)

batch = next(iter(train_loader))
print('sanity image:', batch['image'].shape, 'mask:', batch['mask'].shape)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)

best=-1.0; best_epoch=0; history=[]; wait=0
for epoch in range(1,21):
    model.train(); tr_loss=[]; tr_p=[]; tr_y=[]
    for batch in train_loader:
        x=batch['image'].to(device, non_blocking=True)
        mask=batch['mask'].to(device, non_blocking=True)
        y=batch['label'].to(device, non_blocking=True)
        logits, attn = model(x, mask=mask)
        loss=criterion(logits,y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward(); optimizer.step()
        tr_loss.append(loss.item())
        tr_p.extend(logits.argmax(1).detach().cpu().tolist())
        tr_y.extend(y.cpu().tolist())
    scheduler.step()

    model.eval(); va_p=[]; va_y=[]; va_loss=[]
    with torch.no_grad():
        for batch in val_loader:
            x=batch['image'].to(device, non_blocking=True)
            mask=batch['mask'].to(device, non_blocking=True)
            y=batch['label'].to(device, non_blocking=True)
            logits, attn = model(x, mask=mask)
            loss=criterion(logits,y)
            va_loss.append(loss.item())
            va_p.extend(logits.argmax(1).cpu().tolist())
            va_y.extend(y.cpu().tolist())
    row={
        'epoch':epoch,
        'lr':optimizer.param_groups[0]['lr'],
        'train_loss':float(np.mean(tr_loss)),
        'train_f1':f1_score(tr_y,tr_p,average='macro'),
        'val_loss':float(np.mean(va_loss)),
        'val_f1':f1_score(va_y,va_p,average='macro'),
        'val_acc':accuracy_score(va_y,va_p),
    }
    history.append(row)
    print(f"[Epoch {epoch:02d}] train_loss={row['train_loss']:.4f} train_f1={row['train_f1']:.4f} val_loss={row['val_loss']:.4f} val_f1={row['val_f1']:.4f} lr={row['lr']:.2e}")
    if row['val_f1']>best:
        best=row['val_f1']; best_epoch=epoch; wait=0
        torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),'best_val_f1':best,'history':history}, CKPT_OUT)
        print(f'  -> nuevo mejor checkpoint: {CKPT_OUT.name} (val_f1={best:.4f})')
    else:
        wait+=1
        if wait>=8:
            print(f'Early stopping activado en epoch {epoch}. Mejor epoch: {best_epoch}')
            break

print(f'LUNA MIL V1: best_val_f1={best:.4f} best_epoch={best_epoch}')
print('DONE')
