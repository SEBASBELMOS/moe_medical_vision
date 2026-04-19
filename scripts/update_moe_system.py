from pathlib import Path

code = """import torch
import torch.nn as nn
import timm
from torchvision.models import densenet121, efficientnet_b3, resnet34, efficientnet_b0, resnet18
import joblib
import faiss
import numpy as np
import torch.nn.functional as F
from models.experts_3d import build_pancreatic_expert

class LunaEnsembleExpert(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.mip_model = efficientnet_b0(weights=None)
        self.mip_model.classifier[1] = nn.Sequential(nn.Dropout(0.4), nn.Linear(self.mip_model.classifier[1].in_features, 2))
        
        self.stack_model = resnet18(weights=None)
        self.stack_model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(self.stack_model.fc.in_features, 2))
        
    def load_ensemble_weights(self, ckpt_dir):
        from pathlib import Path
        ckpt_dir = Path(ckpt_dir)
        try:
            self.mip_model.load_state_dict(torch.load(ckpt_dir/'expert4_luna16_MIP_best.pth', map_location=self.device, weights_only=False)['model_state_dict'], strict=False)
            self.stack_model.load_state_dict(torch.load(ckpt_dir/'expert4_luna16_2p5d_FAST_best.pth', map_location=self.device, weights_only=False)['model_state_dict'], strict=False)
            self.mip_model.to(self.device)
            self.stack_model.to(self.device)
        except Exception as e:
            print("[AVISO LUNA] No se cargaron pesos de Ensemble:", e)
            
    def forward(self, mip_input, x_3d_raw=None):
        self.mip_model.eval()
        self.stack_model.eval()
        
        p_mip = torch.softmax(self.mip_model(mip_input), dim=1)
        
        p_stack = p_mip.clone()
        if x_3d_raw is not None:
            try:
                # x_3d_raw: [B, 1, 64, 64, 64]
                vol = x_3d_raw[0, 0]
                zmid = vol.shape[0]//2
                z_ids = [max(0, min(vol.shape[0]-1, zmid+off)) for off in (-8,0,8)]
                stack = vol[z_ids].clone().float()
                stack = F.interpolate(stack.unsqueeze(0).unsqueeze(0), size=(224,224), mode='bilinear', align_corners=False).squeeze(0)
                stack = (stack - 0.5) / 0.25
                p_stack = torch.softmax(self.stack_model(stack.to(self.device)), dim=1)
            except Exception as e:
                pass

        probs = 0.75 * p_mip + 0.25 * p_stack
        
        adjusted_logits = torch.zeros_like(probs).to(self.device)
        if probs[0, 1] >= 0.375:
            adjusted_logits[0, 1] = 10.0
            adjusted_logits[0, 0] = 0.0
        else:
            adjusted_logits[0, 1] = 0.0
            adjusted_logits[0, 0] = 10.0
            
        return adjusted_logits

class MoE_System(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
        self.backbone = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0).to(device)
        self.backbone.eval()
        for p in self.backbone.parameters(): p.requires_grad = False
            
        self.experts = nn.ModuleList([
            self._load_exp1_nih(),
            self._load_exp2_isic(),
            self._load_exp3_osteo(),
            LunaEnsembleExpert(device).to(device),
            build_pancreatic_expert(pretrained=False, use_gradient_checkpointing=False)
        ]).to(device)
        
        self.pca = None
        self.knn_index = None
        self.knn_labels = None

    def _load_exp1_nih(self):
        model = densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, 14)
        return model
    def _load_exp2_isic(self):
        model = efficientnet_b3(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 9)
        return model
    def _load_exp3_osteo(self):
        model = resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 3)
        return model

    def load_all_weights(self, ckpt_dir):
        from pathlib import Path
        ckpt_dir = Path(ckpt_dir)
        
        ckpts = ['expert1_nih_best.pth', 'expert2_isic_best.pth', 'expert3_oa_best.pth']
        for i, ckpt_name in enumerate(ckpts):
            try:
                state = torch.load(ckpt_dir / ckpt_name, map_location=self.device, weights_only=False)
                state = state.get('model_state_dict', state)
                self.experts[i].load_state_dict(state, strict=False)
            except Exception: pass
                
        try:
            self.experts[3].load_ensemble_weights(ckpt_dir)
            self.experts[4].load_state_dict(torch.load(ckpt_dir / 'expert5_pancreatic_r3d18_v5_best.pth', map_location=self.device, weights_only=False)['model_state_dict'], strict=False)
        except Exception: pass
            
        try:
            self.pca = joblib.load(ckpt_dir / 'router_pca.pkl')
            self.knn_index = faiss.read_index(str(ckpt_dir / 'router_knn.index'))
            self.knn_labels = joblib.load(ckpt_dir / 'router_knn_labels.pkl')
            print("  [OK] Pesos y Router FAISS k-NN cargados (Strict=False).")
        except Exception as e:
            print(f"  [Error Router] {e}")

    def forward(self, x_router, x_3d=None):
        with torch.no_grad():
            z = self.backbone(x_router).cpu().numpy()
            
        z_norm = z / np.linalg.norm(z, axis=1, keepdims=True)
        z_pca = self.pca.transform(z_norm).astype(np.float32)
        faiss.normalize_L2(z_pca)
        
        D, I = self.knn_index.search(z_pca, 5)
        vecinos_etiquetas = self.knn_labels[I[0]]
        
        best_expert_idx = int(np.bincount(vecinos_etiquetas, minlength=5).argmax())
        
        expert_probs_np = np.zeros(5)
        for val in vecinos_etiquetas: expert_probs_np[val] += 0.20
        expert_probs = torch.tensor(expert_probs_np).unsqueeze(0).to(self.device)
        
        expert_input = x_router
        if best_expert_idx == 4:
            if x_3d is not None: expert_input = x_3d
            else:
                import torch.nn.functional as F
                fallback = x_router[:, :1, :, :].unsqueeze(2) 
                expert_input = F.interpolate(fallback, size=(64, 64, 64))
        
        if best_expert_idx == 3:
            expert_output = self.experts[3](x_router, x_3d_raw=x_3d)
        else:
            expert_output = self.experts[best_expert_idx](expert_input)
            
        if best_expert_idx == 1:
            expert_output = expert_output / 3.0 # Smoothing de temperatura para ISIC
            
        return expert_output, best_expert_idx, expert_probs
"""
Path('/workspace/moe_medical_vision/src/models/moe_system.py').write_text(code)
print('MoE System inyectado')
