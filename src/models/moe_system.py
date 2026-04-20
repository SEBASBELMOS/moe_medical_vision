
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import joblib
import faiss
import numpy as np
from torchvision.models import densenet121, efficientnet_b3, resnet34
from torchvision.models.video import mc3_18
from data.adaptive_preprocessor import AdaptivePreprocessor
from models.experts_3d import build_pancreatic_expert

class LinearRouter(nn.Module):
    def __init__(self, d_model=192, n_experts=5):
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts)
    def forward(self, z):
        return self.gate(z)

class SmallLinearRouter(nn.Module):
    def __init__(self, d_model=192, n_classes=3):
        super().__init__()
        self.gate = nn.Linear(d_model, n_classes)
    def forward(self, z):
        return self.gate(z)

class MoE_System(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.preprocessor = AdaptivePreprocessor().to(device)
        self.backbone = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0).to(device)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.router_linear = LinearRouter(d_model=192, n_experts=5).to(device)
        self.router_linear.eval()
        self.router_linear_ready = False
        self.router_2d = SmallLinearRouter(d_model=192, n_classes=3).to(device)
        self.router_xray_osteo = SmallLinearRouter(d_model=192, n_classes=2).to(device)
        self.router_3d = SmallLinearRouter(d_model=192, n_classes=2).to(device)
        self.router_hier_ready = False
        self.experts = nn.ModuleList([
            self._load_exp1_nih(),
            self._load_exp2_isic(),
            self._load_exp3_osteo(),
            self._load_exp4_luna(),
            build_pancreatic_expert(pretrained=False, use_gradient_checkpointing=False),
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
        model.classifier[1] = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.classifier[1].in_features, 9))
        return model

    def _load_exp3_osteo(self):
        model = resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 3)
        return model

    def _load_exp4_luna(self):
        model = mc3_18(weights=None)
        model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.fc.in_features, 2))
        return model

    def load_all_weights(self, ckpt_dir):
        from pathlib import Path
        ckpt_dir = Path(ckpt_dir)
        ckpts = ['expert1_nih_best.pth','expert2_isic_best.pth','expert3_oa_best.pth','expert4_luna_fixed.pth']
        for i, ckpt_name in enumerate(ckpts):
            try:
                state = torch.load(ckpt_dir / ckpt_name, map_location=self.device, weights_only=False)
                state = state.get('model_state_dict', state)
                self.experts[i].load_state_dict(state, strict=False)
            except Exception as e:
                print(f'  [Aviso] No se cargaron pesos para experto {i}: {e}')
        try:
            state = torch.load(ckpt_dir / 'expert5_pancreatic_FAST_best.pth', map_location=self.device, weights_only=False)
            self.experts[4].load_state_dict(state['model_state_dict'], strict=False)
        except Exception as e:
            print(f'  [Aviso] No se cargaron pesos para experto 4: {e}')
        # hierarchical routers first
        try:
            st2 = torch.load('/workspace/router_moe_rebuild/checkpoints/router_2d_linear_v2.pth', map_location=self.device, weights_only=False)
            sd2 = st2.get('model_state_dict', st2)
            clean2 = {('gate.' + k if k in ['weight','bias'] else k): v for k,v in sd2.items()}
            self.router_2d.load_state_dict(clean2, strict=False)
            stxo = torch.load('/workspace/router_moe_rebuild/checkpoints/router_xray_osteo_linear_v1.pth', map_location=self.device, weights_only=False)
            sdxo = stxo.get('model_state_dict', stxo)
            cleanxo = {('gate.' + k if k in ['weight','bias'] else k): v for k,v in sdxo.items()}
            self.router_xray_osteo.load_state_dict(cleanxo, strict=False)
            st3 = torch.load('/workspace/router_moe_rebuild/checkpoints/router_3d_linear_v4.pth', map_location=self.device, weights_only=False)
            sd3 = st3.get('model_state_dict', st3)
            clean3 = {('gate.' + k if k in ['weight','bias'] else k): v for k,v in sd3.items()}
            self.router_3d.load_state_dict(clean3, strict=False)
            self.router_hier_ready = True
            print('  [OK] Router jerárquico cargado.')
        except Exception as e:
            print(f'  [Warn] Router jerárquico no disponible: {e}')
        # legacy flat linear router
        try:
            st = torch.load(ckpt_dir / 'router_linear_balval_alpha02.pth', map_location=self.device, weights_only=False)
            st = st.get('model_state_dict', st)
            clean = {}
            for k,v in st.items():
                if k.startswith('router.'): clean[k.replace('router.','')] = v
                elif k.startswith('gate.'): clean[k] = v
                elif k in ['weight','bias']: clean['gate.'+k] = v
            if not clean: clean = st
            self.router_linear.load_state_dict(clean, strict=False)
            self.router_linear_ready = True
            print('  [OK] Router linear cargado.')
        except Exception as e:
            print(f'  [Warn] Router linear no disponible: {e}')
        try:
            self.pca = joblib.load(ckpt_dir / 'router_pca.pkl')
            self.knn_index = faiss.read_index(str(ckpt_dir / 'router_knn.index'))
            self.knn_labels = joblib.load(ckpt_dir / 'router_knn_labels.pkl')
            print('  [OK] Pesos y Router FAISS k-NN cargados (Strict=False).')
        except Exception as e:
            print(f'  [Error Router] {e}')

    def _prepare_for_router(self, x_router, x_3d=None):
        if x_3d is not None:
            x = x_3d
            if x.ndim == 5 and x.shape[1] == 1:
                x = x.squeeze(1)
            # Match 03_extract_cls_tokens.py: 3 orthogonal MIPs
            mip = torch.stack([x.max(dim=1)[0], x.max(dim=2)[0], x.max(dim=3)[0]], dim=1)
            x = F.interpolate(mip, size=(224, 224), mode='bilinear', align_corners=False)
            return x
        x = x_router
        if len(x.shape) == 4 and x.shape[1] in [1,3] and tuple(x.shape[-2:]) == (224,224):
            return x
        if x.ndim == 4 and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if x.shape[-1] != 224:
            x = F.interpolate(x, size=(224,224), mode='bilinear', align_corners=False)
        return x

    def _route(self, x_router, x_3d=None):
        x_for_router = self._prepare_for_router(x_router, x_3d)
        with torch.no_grad():
            z = self.backbone.forward_features(x_for_router)[:,0,:]
        # hierarchical first
        if self.router_hier_ready:
            if x_3d is not None:
                probs_small = F.softmax(self.router_3d(z), dim=-1)
                local_idx = int(torch.argmax(probs_small, dim=-1).item())
                best_expert_idx = [3,4][local_idx]
                expert_probs = torch.zeros((1,5), device=self.device)
                expert_probs[0, best_expert_idx] = 1.0
                return best_expert_idx, expert_probs
            else:
                # Visual pre-router for 2D: color images -> ISIC directly; grayscale-like -> NIH vs Osteo by dedicated binary router
                # Measure colorfulness on de-normalized image; grayscale replicated images should stay near zero.
                mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1,3,1,1)
                std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1,3,1,1)
                x_rgb = (x_for_router * std + mean).clamp(0, 1)
                color_gap = torch.mean(torch.abs(x_rgb[:,0]-x_rgb[:,1])) + torch.mean(torch.abs(x_rgb[:,1]-x_rgb[:,2]))
                if color_gap.item() > 0.05:
                    best_expert_idx = 1
                    expert_probs = torch.zeros((1,5), device=self.device)
                    expert_probs[0, best_expert_idx] = 1.0
                    return best_expert_idx, expert_probs
                probs_small = F.softmax(self.router_xray_osteo(z), dim=-1)
                local_idx = int(torch.argmax(probs_small, dim=-1).item())
                best_expert_idx = [0,2][local_idx]
                expert_probs = torch.zeros((1,5), device=self.device)
                expert_probs[0, best_expert_idx] = 1.0
                return best_expert_idx, expert_probs
        if self.router_linear_ready:
            logits = self.router_linear(z)
            probs = F.softmax(logits, dim=-1)
            best_expert_idx = int(torch.argmax(probs, dim=-1).item())
            return best_expert_idx, probs
        z_np = z.cpu().numpy()
        z_norm = z_np / np.linalg.norm(z_np, axis=1, keepdims=True)
        z_pca = self.pca.transform(z_norm).astype(np.float32)
        faiss.normalize_L2(z_pca)
        _, I = self.knn_index.search(z_pca, 5)
        neigh = self.knn_labels[I[0]]
        best_expert_idx = int(np.bincount(neigh, minlength=5).argmax())
        expert_probs_np = np.zeros(5)
        for val in neigh: expert_probs_np[val] += 0.20
        return best_expert_idx, torch.tensor(expert_probs_np, device=self.device).unsqueeze(0)

    def forward(self, x_router, x_3d=None):
        best_expert_idx, expert_probs = self._route(x_router, x_3d)
        expert_input = x_router
        if best_expert_idx in [3,4]:
            if x_3d is not None:
                expert_input = x_3d
            else:
                fallback = x_router[:, :1, :, :].unsqueeze(2)
                expert_input = F.interpolate(fallback, size=(64, 64, 64))
            if best_expert_idx == 3:
                expert_input = expert_input.repeat(1, 3, 1, 1, 1)
                mean = torch.tensor([0.43216, 0.394666, 0.37645], device=self.device).view(1,3,1,1,1)
                std = torch.tensor([0.22803, 0.22145, 0.216989], device=self.device).view(1,3,1,1,1)
                expert_input = (expert_input - mean) / std
        expert_output = self.experts[best_expert_idx](expert_input.to(self.device))
        return expert_output, best_expert_idx, expert_probs
