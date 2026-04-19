import torch
import torch.nn as nn
import timm
from torchvision.models import densenet121, efficientnet_b3, resnet34
from torchvision.models.video import mc3_18
import joblib
import faiss
import numpy as np

from models.experts_3d import build_pancreatic_expert


class MoE_System(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device

        self.backbone = timm.create_model(
            "vit_tiny_patch16_224", pretrained=True, num_classes=0
        ).to(device)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.experts = nn.ModuleList(
            [
                self._load_exp1_nih(),
                self._load_exp2_isic(),
                self._load_exp3_osteo(),
                self._load_exp4_luna(),
                build_pancreatic_expert(
                    pretrained=False, use_gradient_checkpointing=False
                ),
            ]
        ).to(device)

        self.pca = None
        self.knn_index = None
        self.knn_labels = None

    def _load_exp1_nih(self):
        model = densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, 2)
        return model

    def _load_exp2_isic(self):
        model = efficientnet_b3(weights=None)
        model.classifier[1] = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(model.classifier[1].in_features, 9)
        )
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

        ckpts = [
            "expert1_nih_best.pth",
            "expert2_isic_best_fixed.pth",
            "expert3_oa_best.pth",
            "expert4_luna_fixed.pth",
        ]
        for i, ckpt_name in enumerate(ckpts):
            try:
                state = torch.load(
                    ckpt_dir / ckpt_name, map_location=self.device, weights_only=False
                )
                state = state.get("model_state_dict", state)
                self.experts[i].load_state_dict(state, strict=False)
            except Exception as e:
                print(f"  [Aviso] No se cargaron pesos para experto {i}: {e}")

        try:
            state = torch.load(
                ckpt_dir / "expert5_pancreatic_r3d18_v5_best.pth",
                map_location=self.device,
                weights_only=False,
            )
            self.experts[4].load_state_dict(state["model_state_dict"], strict=False)
        except Exception as e:
            print(f"  [Aviso] No se cargaron pesos para experto 4: {e}")

        try:
            self.pca = joblib.load(ckpt_dir / "router_pca.pkl")
            self.knn_index = faiss.read_index(str(ckpt_dir / "router_knn.index"))
            self.knn_labels = joblib.load(ckpt_dir / "router_knn_labels.pkl")
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
        for val in vecinos_etiquetas:
            expert_probs_np[val] += 0.20
        expert_probs = torch.tensor(expert_probs_np).unsqueeze(0).to(self.device)

        expert_input = x_router
        if best_expert_idx in [3, 4]:
            if x_3d is not None:
                expert_input = x_3d
            else:
                import torch.nn.functional as F

                fallback = x_router[:, :1, :, :].unsqueeze(2)
                expert_input = F.interpolate(fallback, size=(64, 64, 64))

            if best_expert_idx == 3:
                # LUNA16 expects 3 channels and Kinetics normalization
                expert_input = expert_input.repeat(1, 3, 1, 1, 1)
                mean = (
                    torch.tensor([0.43216, 0.394666, 0.37645])
                    .view(1, 3, 1, 1, 1)
                    .to(self.device)
                )
                std = (
                    torch.tensor([0.22803, 0.22145, 0.216989])
                    .view(1, 3, 1, 1, 1)
                    .to(self.device)
                )
                expert_input = (expert_input - mean) / std

        expert_output = self.experts[best_expert_idx](expert_input)

        return expert_output, best_expert_idx, expert_probs
