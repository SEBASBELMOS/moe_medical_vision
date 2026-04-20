import sys
import time
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import nibabel as nib
from PIL import Image

sys.path.insert(0, "/workspace/moe_medical_vision/src")
from models.moe_system import MoE_System


class LinearGatingHead(nn.Module):
    """Router A: CLS token -> Linear -> Softmax."""

    def __init__(self, d_model=192, n_experts=5):
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts)

    def forward_logits(self, z):
        return self.gate(z)

    def forward(self, z):
        return F.softmax(self.forward_logits(z), dim=-1)


def compute_router_attention_rollout(backbone, router_head, tensor_224, target_idx=None):
    """Gradient-weighted attention rollout for Router A (ViT + Linear).
    Returns a 2D numpy heatmap in [0, 1] of shape (side, side)."""
    attn_maps = []
    original_forwards = []

    def make_forward(attn_mod, storage):
        def new_forward(x, attn_mask=None, **kwargs):
            B, N, C = x.shape
            head_dim = C // attn_mod.num_heads
            qkv = (
                attn_mod.qkv(x)
                .reshape(B, N, 3, attn_mod.num_heads, head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv.unbind(0)

            if hasattr(attn_mod, "q_norm"):
                q = attn_mod.q_norm(q)
            if hasattr(attn_mod, "k_norm"):
                k = attn_mod.k_norm(k)

            q = q * attn_mod.scale
            a = q @ k.transpose(-2, -1)
            a = a.softmax(dim=-1)
            if a.requires_grad:
                a.retain_grad()
            storage.append(a)
            a = attn_mod.attn_drop(a)
            out = (a @ v).transpose(1, 2).reshape(B, N, C)
            out = attn_mod.proj(out)
            out = attn_mod.proj_drop(out)
            return out
        return new_forward

    blocks = getattr(backbone, "blocks", None)
    if blocks is None:
        return None
    for blk in blocks:
        original_forwards.append(blk.attn.forward)
        blk.attn.forward = make_forward(blk.attn, attn_maps)

    try:
        x = tensor_224.detach().clone().requires_grad_(True)
        backbone.zero_grad()
        router_head.zero_grad()
        z = backbone(x)
        logits = router_head.forward_logits(z)
        if target_idx is None:
            target_idx = int(logits.argmax(dim=-1).item())
        score = logits[0, target_idx]
        score.backward()
    finally:
        for blk, fwd in zip(blocks, original_forwards):
            blk.attn.forward = fwd

    if not attn_maps:
        return None

    rollout = None
    for a in attn_maps:
        if a.grad is None:
            continue
        grad = a.grad.detach().clamp(min=0)
        a = (a.detach() * grad).mean(dim=1)
        eye = torch.eye(a.size(-1), device=a.device).unsqueeze(0)
        a = a + eye
        a = a / a.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        rollout = a if rollout is None else rollout @ a

    if rollout is None:
        return None

    cls_to_patches = rollout[0, 0, 1:].cpu().numpy()
    side = int(round(np.sqrt(cls_to_patches.size)))
    if side * side != cls_to_patches.size:
        return None
    heat = np.maximum(cls_to_patches.reshape(side, side), 0.0)

    # Reduce the common ViT border artifact by softly downweighting outer patches.
    window_1d = np.hanning(side)
    border_window = np.outer(window_1d, window_1d).astype(np.float32)
    border_window = 0.35 + 0.65 * border_window
    heat = heat * border_window

    h_min, h_max = float(heat.min()), float(heat.max())
    heat = (heat - h_min) / max(h_max - h_min, 1e-9)

    # Suppress diffuse low-importance activations so anatomy stands out more clearly.
    cutoff = float(np.percentile(heat, 65))
    heat = np.clip(heat - cutoff, 0.0, None)
    h_min, h_max = float(heat.min()), float(heat.max())
    heat = (heat - h_min) / max(h_max - h_min, 1e-9)

    return heat


def compute_plain_attention_rollout(backbone, tensor_224):
    """Non-gradient attention rollout (Abnar & Zuidema 2020) — works with any router."""
    attn_maps = []
    original_forwards = []

    def make_forward(attn_mod, storage):
        def new_forward(x, attn_mask=None, **kwargs):
            B, N, C = x.shape
            head_dim = C // attn_mod.num_heads
            qkv = (
                attn_mod.qkv(x)
                .reshape(B, N, 3, attn_mod.num_heads, head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv.unbind(0)
            if hasattr(attn_mod, "q_norm"):
                q = attn_mod.q_norm(q)
            if hasattr(attn_mod, "k_norm"):
                k = attn_mod.k_norm(k)
            q = q * attn_mod.scale
            a = q @ k.transpose(-2, -1)
            a = a.softmax(dim=-1)
            storage.append(a.detach())
            a = attn_mod.attn_drop(a)
            out = (a @ v).transpose(1, 2).reshape(B, N, C)
            out = attn_mod.proj(out)
            out = attn_mod.proj_drop(out)
            return out
        return new_forward

    blocks = getattr(backbone, "blocks", None)
    if blocks is None:
        return None
    for blk in blocks:
        original_forwards.append(blk.attn.forward)
        blk.attn.forward = make_forward(blk.attn, attn_maps)

    try:
        with torch.no_grad():
            _ = backbone(tensor_224)
    finally:
        for blk, fwd in zip(blocks, original_forwards):
            blk.attn.forward = fwd

    if not attn_maps:
        return None

    rollout = None
    for a in attn_maps:
        a = a.mean(dim=1)
        eye = torch.eye(a.size(-1), device=a.device).unsqueeze(0)
        a = 0.5 * a + 0.5 * eye
        a = a / a.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        rollout = a if rollout is None else rollout @ a

    cls_to_patches = rollout[0, 0, 1:].cpu().numpy()
    side = int(round(np.sqrt(cls_to_patches.size)))
    if side * side != cls_to_patches.size:
        return None
    heat = np.maximum(cls_to_patches.reshape(side, side), 0.0)

    window_1d = np.hanning(side)
    border_window = np.outer(window_1d, window_1d).astype(np.float32)
    border_window = 0.35 + 0.65 * border_window
    heat = heat * border_window

    h_min, h_max = float(heat.min()), float(heat.max())
    heat = (heat - h_min) / max(h_max - h_min, 1e-9)

    cutoff = float(np.percentile(heat, 65))
    heat = np.clip(heat - cutoff, 0.0, None)
    h_min, h_max = float(heat.min()), float(heat.max())
    heat = (heat - h_min) / max(h_max - h_min, 1e-9)
    return heat


def compute_router_input_saliency(backbone, router_head, tensor_224, target_idx=None):
    """Input-gradient saliency map for Router A logit."""
    x = tensor_224.detach().clone().requires_grad_(True)
    backbone.zero_grad()
    router_head.zero_grad()

    z = backbone(x)
    logits = router_head.forward_logits(z)
    if target_idx is None:
        target_idx = int(logits.argmax(dim=-1).item())
    score = logits[0, target_idx]
    score.backward()

    grad_full = x.grad.detach().abs()[0]
    channel_scores = grad_full.mean(dim=(1, 2)).cpu().numpy()
    grad = grad_full.mean(dim=0, keepdim=False)
    grad = F.avg_pool2d(
        grad.unsqueeze(0).unsqueeze(0), kernel_size=9, stride=1, padding=4
    )
    sal = grad.squeeze().cpu().numpy()
    sal = sal - sal.min()
    sal = sal / max(sal.max(), 1e-9)

    cutoff = float(np.percentile(sal, 55))
    sal = np.clip(sal - cutoff, 0.0, None)
    sal = sal / max(sal.max(), 1e-9)
    return sal, channel_scores


def resize_heatmap_to_shape(heatmap, target_shape):
    """Resize a 2D heatmap to target_shape=(H, W) using bilinear interpolation."""
    target_h, target_w = target_shape
    if heatmap.shape == (target_h, target_w):
        return heatmap.astype(np.float32)

    heat_img = Image.fromarray((np.clip(heatmap, 0.0, 1.0) * 255).astype(np.uint8))
    heat_img = heat_img.resize((target_w, target_h), Image.BILINEAR)
    return np.array(heat_img).astype(np.float32) / 255.0


def build_soft_border_mask(target_shape, floor=0.10):
    """Softly suppress edge patches, where ViTs often show spurious attention."""
    target_h, target_w = target_shape
    win_y = np.hanning(max(target_h, 3)).astype(np.float32)
    win_x = np.hanning(max(target_w, 3)).astype(np.float32)
    border = np.outer(win_y, win_x)
    border = border / max(float(border.max()), 1e-9)
    return floor + (1.0 - floor) * border


def build_center_prior(target_shape, strength=0.25):
    """Prefer central anatomy a bit more for radiographs without hard-masking edges."""
    target_h, target_w = target_shape
    yy = np.linspace(-1.0, 1.0, target_h, dtype=np.float32)
    xx = np.linspace(-1.0, 1.0, target_w, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(yy, xx, indexing="ij")
    radius = np.sqrt(grid_x**2 + grid_y**2)
    prior = np.exp(-(radius**2) / 0.85).astype(np.float32)
    prior = prior / max(float(prior.max()), 1e-9)
    return (1.0 - strength) + strength * prior


def fuse_router_heatmaps(attn_heat, saliency_heat, pil_img, expert_idx=None, is_3d=False):
    """Fuse attention rollout with input saliency and suppress background-driven artifacts."""
    target_shape = attn_heat.shape
    saliency_heat = resize_heatmap_to_shape(saliency_heat, target_shape)
    border_mask = build_soft_border_mask(target_shape, floor=0.02 if is_3d else 0.12)
    fused = np.sqrt(np.clip(attn_heat, 0.0, 1.0) * np.clip(saliency_heat, 0.0, 1.0))
    fused = fused * border_mask

    if is_3d:
        # ViT-Tiny accumulates attention sinks on corner patches when fed MIP stats.
        # Kill the outermost 2 patch rings and apply a strong center prior.
        fused[:2, :] = 0.0
        fused[-2:, :] = 0.0
        fused[:, :2] = 0.0
        fused[:, -2:] = 0.0
        fused = fused * build_center_prior(target_shape, strength=0.55)
    else:
        fg_mask = build_foreground_mask(pil_img, target_size=target_shape)
        fused = fused * fg_mask
        if expert_idx in (0, 2):
            center_strength = 0.22 if expert_idx == 0 else 0.18
            fused = fused * build_center_prior(target_shape, strength=center_strength)

    # Keep only the strongest regions to reduce diffuse clouds and corner peaks.
    cutoff = float(np.percentile(fused, 90 if is_3d else 79))
    fused = np.clip(fused - cutoff, 0.0, None)
    fused = fused / max(fused.max(), 1e-9)
    fused = np.array(
        F.avg_pool2d(
            torch.from_numpy(fused).float().unsqueeze(0).unsqueeze(0),
            kernel_size=3,
            stride=1,
            padding=1,
        ).squeeze()
    )
    fused = fused ** (1.10 if is_3d else 1.05)
    fused = fused / max(fused.max(), 1e-9)
    return fused


def build_foreground_mask(pil_img, target_size=(224, 224)):
    """Create a soft foreground mask to suppress black/padded background regions."""
    target_h, target_w = target_size
    gray = pil_img.convert("L").resize((target_w, target_h))
    gray_np = np.array(gray).astype(np.float32) / 255.0

    # Robust threshold that keeps anatomy / lesion content and removes empty borders.
    threshold = max(0.08, float(np.percentile(gray_np, 35)))
    mask = np.clip((gray_np - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0)

    mask_t = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
    mask_t = F.avg_pool2d(mask_t, kernel_size=15, stride=1, padding=7)
    mask = mask_t.squeeze().cpu().numpy()

    mask = np.clip(mask, 0.0, 1.0)
    return 0.15 + 0.85 * mask


def normalize_preview_plane(plane):
    plane = plane.astype(np.float32)
    plane = plane - plane.min()
    plane = plane / max(float(plane.max()), 1e-8)
    return plane


def make_router_channel_image(tensor_224, channel_idx=0):
    """Build a grayscale preview from one router input channel."""
    arr = tensor_224.detach().cpu().squeeze(0).numpy()

    if arr.ndim == 2:
        arr = arr[None, ...]

    channel_idx = int(np.clip(channel_idx, 0, arr.shape[0] - 1))
    base = normalize_preview_plane(arr[channel_idx])
    rgb = np.stack([base, base, base], axis=-1)
    return Image.fromarray((rgb.clip(0, 1) * 255).astype(np.uint8)).convert("RGB")


def make_router_preview_image(tensor_224):
    """Build a preview aligned with the exact tensor seen by the router."""
    arr = tensor_224.detach().cpu().squeeze(0).numpy()

    if arr.ndim == 2:
        arr = arr[None, ...]

    planes = [normalize_preview_plane(arr[i]) for i in range(min(3, arr.shape[0]))]

    if len(planes) == 1:
        rgb = np.stack([planes[0], planes[0], planes[0]], axis=-1)
    else:
        while len(planes) < 3:
            planes.append(planes[-1])
        rgb = np.stack(planes[:3], axis=-1)

    return Image.fromarray((rgb.clip(0, 1) * 255).astype(np.uint8)).convert("RGB")


def make_router_preview_montage(tensor_224):
    """Show the proxy views seen by the 3D router as a small montage."""
    arr = tensor_224.detach().cpu().squeeze(0).numpy()

    if arr.ndim == 2:
        arr = arr[None, ...]

    tiles = []
    for idx in range(min(3, arr.shape[0])):
        plane = normalize_preview_plane(arr[idx])
        tile = np.stack([plane, plane, plane], axis=-1)
        tiles.append(tile)

    if not tiles:
        return make_router_channel_image(tensor_224, 0)

    if len(tiles) == 1:
        montage = tiles[0]
    else:
        spacer = np.full((tiles[0].shape[0], 8, 3), 0.05, dtype=np.float32)
        montage = tiles[0]
        for tile in tiles[1:]:
            montage = np.concatenate([montage, spacer, tile], axis=1)

    return Image.fromarray((montage.clip(0, 1) * 255).astype(np.uint8)).convert("RGB")


def crop_image_to_foreground(pil_img, margin=6):
    """Crop strong empty borders before resizing to reduce border-driven routing."""
    gray = np.array(pil_img.convert("L"))
    threshold = max(8, int(np.percentile(gray, 12)))
    coords = np.argwhere(gray > threshold)
    if coords.size == 0:
        return pil_img

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    # Keep original image if the crop would be almost identical.
    h, w = gray.shape
    if (x1 - x0) >= 0.97 * w and (y1 - y0) >= 0.97 * h:
        return pil_img

    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(w, x1 + margin)
    y1 = min(h, y1 + margin)
    return pil_img.crop((x0, y0, x1, y1))


def overlay_heatmap_on_image(pil_img, heatmap, is_3d=False):
    """Return a PIL RGB image with a jet colormap overlay."""
    import matplotlib.cm as cm
    base = pil_img.convert("RGB").resize((224, 224))
    base_np = np.array(base).astype(np.float32) / 255.0
    heat_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
    heat_np = np.array(heat_img).astype(np.float32) / 255.0
    smooth_k = 21 if is_3d else 11
    heat_np = np.array(
        F.avg_pool2d(
            torch.from_numpy(heat_np).float().unsqueeze(0).unsqueeze(0),
            kernel_size=smooth_k,
            stride=1,
            padding=smooth_k // 2,
        ).squeeze()
    )
    if not is_3d:
        # Foreground mask assumes bright anatomy — valid for RGB 2D,
        # but wrong for MIP proxies where pulmonary parenchyma is dark.
        fg_mask = build_foreground_mask(pil_img)
        heat_np = heat_np * fg_mask
    heat_np = np.clip((heat_np - heat_np.min()) / max(heat_np.max() - heat_np.min(), 1e-9), 0.0, 1.0)
    colored = cm.turbo(heat_np)[:, :, :3]
    alpha = 0.10 + 0.45 * heat_np[..., None]
    overlay = (1.0 - alpha) * base_np + alpha * colored
    overlay = (overlay.clip(0, 1) * 255).astype(np.uint8)
    return Image.fromarray(overlay)

st.set_page_config(page_title="MoE Medical Vision", page_icon="🩺", layout="wide")

if "load_balance" not in st.session_state:
    st.session_state["load_balance"] = [0, 0, 0, 0, 0]

EXPERTOS_INFO = [
    {
        "nombre": "Experto 1 (Tórax 2D)",
        "arq": "ConvNeXt-Tiny",
        "dataset": "NIH ChestX-ray14",
        "clases_map": {
            0: "Atelectasia",
            1: "Cardiomegalia",
            2: "Derrame Pleural",
            3: "Infiltración",
            4: "Masa",
            5: "Nódulo",
            6: "Neumonía",
            7: "Neumotórax",
            8: "Consolidación",
            9: "Edema",
            10: "Enfisema",
            11: "Fibrosis",
            12: "Engrosamiento Pleural",
            13: "Hernia",
        },
    },
    {
        "nombre": "Experto 2 (Piel 2D)",
        "arq": "EfficientNet-B3",
        "dataset": "ISIC 2019",
        "clases_map": {
            0: "Melanoma",
            1: "Nevo Melanocítico",
            2: "Carcinoma Basocelular",
            3: "Queratosis Actínica",
            4: "Queratosis Benigna",
            5: "Dermatofibroma",
            6: "Lesión Vascular",
            7: "Carcinoma Espinocelular",
            8: "Otro / Sano",
        },
    },
    {
        "nombre": "Experto 3 (Rodilla 2D)",
        "arq": "ResNet-34",
        "dataset": "Osteoarthritis Knee",
        "clases_map": {
            0: "Normal / Sano",
            1: "Dudoso",
            2: "Osteoartritis Leve a Severa",
        },
    },
    {
        "nombre": "Experto 4 (Nódulos 3D)",
        "arq": "MC3-18",
        "dataset": "LUNA16",
        "clases_map": {0: "Sano", 1: "Nódulo Pulmonar Positivo"},
    },
    {
        "nombre": "Experto 5 (Páncreas 3D)",
        "arq": "R3D-18",
        "dataset": "Pancreatic Cancer",
        "clases_map": {0: "Sano", 1: "Tumor Pancreático"},
    },
]


@st.cache_data
def load_random_samples(n=30):
    import random
    import itertools

    PATHS = {
        "Tórax (NIH)": (
            "/workspace/moe_medical_vision/data/raw/nih/images_001/images",
            "*.png",
        ),
        "Piel (ISIC)": (
            "/workspace/moe_medical_vision/data/raw/isic/ISIC_2019_Training_Input/ISIC_2019_Training_Input",
            "*.jpg",
        ),
        "Rodilla (Osteo)": (
            "/workspace/moe_medical_vision/data/raw/osteoporosis/KLGrade/KLGrade",
            "**/*.*",
        ),
        "Nódulo (LUNA16)": (
            "/workspace/moe_medical_vision/data/processed/luna16_highres",
            "val_*.npz",
        ),
        "Tumor Páncreas (Pancreatic)": (
            "/workspace/moe_medical_vision/data/raw/pancreatic",
            "*.nii.gz",
        ),
    }

    samples = {}
    for prefix, (path, ext) in PATHS.items():
        base_dir = Path(path)
        files_iterator = base_dir.rglob(ext) if "**" in ext else base_dir.glob(ext)
        first_files = [
            f for f in itertools.islice(files_iterator, 400)
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".npz", ".mhd", ".nii"}
            or str(f).endswith('.nii.gz')
        ]
        if first_files:
            selected = random.sample(first_files, min(n, len(first_files)))
            for s in selected:
                samples[f"{prefix} - {s.name}"] = str(s)
    samples["⚠️ Imagen Ruidosa (OOD Detection)"] = (
        "/workspace/moe_medical_vision/data/raw/isic/ISIC_2019_Training_Input/ISIC_2019_Training_Input/ISIC_0000001.jpg"
    )
    return samples


SAMPLES = load_random_samples()


@st.cache_resource
def load_moe_system():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MoE_System(device=device)
    model.load_all_weights("/workspace/moe_medical_vision/checkpoints")
    model.eval()
    router_ready = all([
        model.pca is not None,
        model.knn_index is not None,
        model.knn_labels is not None,
    ])
    return model, device, router_ready


@st.cache_resource
def load_router_a(device):
    ckpt_dir = Path("/workspace/moe_medical_vision/checkpoints")
    router = LinearGatingHead(d_model=192, n_experts=5).to(device)
    router.eval()

    # Preference order: balanced-alpha checkpoint from router_moe_rebuild → legacy fine-tuned.
    candidates = [
        "router_linear_balval_alpha02.pth",
        "router_a_finetuned.pth",
        "router_a_best.pth",
    ]
    for ckpt_name in candidates:
        ckpt_path = ckpt_dir / ckpt_name
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            router.load_state_dict(state, strict=False)
            meta = {
                "best_acc": ckpt.get("best_acc") if isinstance(ckpt, dict) else None,
                "best_ratio": ckpt.get("best_ratio") if isinstance(ckpt, dict) else None,
                "epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
                "alpha": ckpt.get("alpha") if isinstance(ckpt, dict) else None,
            }
            return router, ckpt_name, meta

    return None, None, {}


def run_expert_by_index(model, best_idx, tensor_224, tensor_3d, device):
    expert_input = tensor_224
    if best_idx in [3, 4]:
        if tensor_3d is not None:
            expert_input = tensor_3d
        else:
            fallback = tensor_224[:, :1, :, :].unsqueeze(2)
            expert_input = F.interpolate(fallback, size=(64, 64, 64))

        if best_idx == 3:
            expert_input = expert_input.repeat(1, 3, 1, 1, 1)
            mean = torch.tensor([0.43216, 0.394666, 0.37645], device=device).view(
                1, 3, 1, 1, 1
            )
            std = torch.tensor([0.22803, 0.22145, 0.216989], device=device).view(
                1, 3, 1, 1, 1
            )
            expert_input = (expert_input - mean) / std

    return model.experts[best_idx](expert_input)


model, device, router_ready = load_moe_system()
router_a, router_a_ckpt, router_a_meta = load_router_a(device)

available_routers = []
if router_a is not None:
    available_routers.append("Router A — ViT + Linear")
if router_ready:
    available_routers.append("Router D — ViT + PCA + k-NN (FAISS)")
if not available_routers:
    available_routers.append("Fallback heurístico")

st.sidebar.markdown("---")
st.sidebar.subheader("Router")
router_choice = st.sidebar.radio(
    "Selecciona el router a usar:",
    available_routers,
    index=0,
)

if router_choice.startswith("Router A"):
    router_mode = "router_a"
elif router_choice.startswith("Router D"):
    router_mode = "knn"
else:
    router_mode = "fallback"

if router_mode == "router_a":
    acc_bal = router_a_meta.get("best_acc")
    ratio_bal = router_a_meta.get("best_ratio")
    meta_bits = []
    if acc_bal is not None:
        meta_bits.append(f"acc_bal={acc_bal:.3f}")
    if ratio_bal is not None:
        meta_bits.append(f"ratio_bal={ratio_bal:.2f}")
    meta_str = f" ({', '.join(meta_bits)})" if meta_bits else ""
    st.success(f"Router A cargado desde `{router_a_ckpt}`{meta_str}.")
elif router_mode == "knn":
    st.success("Router D (k-NN + PCA + FAISS) activo — usando artefactos en `checkpoints/`.")
else:
    st.warning("Router k-NN/FAISS no disponible en este servidor. Se usará routing de respaldo para la demo.")


def process_2d_image(image_path_or_bytes):
    img = Image.open(image_path_or_bytes).convert("RGB")
    img = crop_image_to_foreground(img)
    orig_size = img.size
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    img_array = (img_array - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    tensor_224 = torch.from_numpy(img_array).float().permute(2, 0, 1).unsqueeze(0)
    return tensor_224, None, f"2D ({orig_size[0]}x{orig_size[1]}x3)", img


def process_fast_npz(file_path):
    d = np.load(file_path)
    vol = d["volume"][0]
    tensor_3d = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)

    mip = torch.stack(
        [
            torch.tensor(vol.max(axis=0)),
            torch.tensor(vol.mean(axis=0)),
            torch.tensor(vol.std(axis=0)),
        ],
        dim=0,
    )
    tensor_224 = F.interpolate(
        mip.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
    )

    mid_slice = vol[32, :, :]
    mid_slice = (mid_slice - mid_slice.min()) / (
        mid_slice.max() - mid_slice.min() + 1e-8
    )
    preview_img = Image.fromarray((mid_slice * 255).astype(np.uint8)).convert("L")

    return tensor_224, tensor_3d, "3D Pre-procesado (64x64x64)", preview_img


def process_3d_volume(file_path):
    if str(file_path).endswith(".mhd"):
        import SimpleITK as sitk

        img_sitk = sitk.ReadImage(str(file_path))
        vol = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
    else:
        vol = nib.load(str(file_path)).get_fdata(dtype=np.float32)

    orig_size = vol.shape
    vol = np.clip(vol, -1000.0, 400.0)
    vol = (vol + 1000.0) / 1400.0

    tensor_3d = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)
    tensor_3d_resized = F.interpolate(
        tensor_3d, size=(64, 64, 64), mode="trilinear", align_corners=False
    )

    x = tensor_3d_resized.squeeze(1)
    mip = torch.stack([x.max(dim=1)[0], x.max(dim=2)[0], x.max(dim=3)[0]], dim=1)
    tensor_224 = F.interpolate(
        mip, size=(224, 224), mode="bilinear", align_corners=False
    )

    mid_slice = (
        vol[orig_size[0] // 2, :, :]
        if str(file_path).endswith(".mhd")
        else vol[:, :, orig_size[2] // 2]
    )
    mid_slice = (mid_slice - mid_slice.min()) / (
        mid_slice.max() - mid_slice.min() + 1e-8
    )
    preview_img = Image.fromarray((mid_slice * 255).astype(np.uint8)).convert("L")

    return (
        tensor_224,
        tensor_3d_resized,
        f"3D ({orig_size[0]}x{orig_size[1]}x{orig_size[2]})",
        preview_img,
    )



def infer_expert_fallback(sample_choice, selected_sample_path, file_name, is_3d):
    src = f"{sample_choice} {selected_sample_path or ''} {file_name}".lower()
    if 'nih' in src or 'tórax' in src or 'torax' in src:
        return 0
    if 'isic' in src or 'piel' in src:
        return 1
    if 'osteo' in src or 'rodilla' in src or 'osteoporosis' in src or 'klgrade' in src:
        return 2
    if 'luna' in src or 'nódulo' in src or 'nodulo' in src:
        return 3
    if 'pancre' in src or 'tumor páncreas' in src or 'tumor pancreas' in src:
        return 4
    if is_3d:
        return 3
    return 0


st.title("🩺 Clasificación Médica Multimodal con MoE")

st.sidebar.title("Opciones de Entrada")
input_method = st.sidebar.radio(
    "Selecciona:", ("Usar Muestra (Sample)", "Subir Imagen Propia")
)

uploaded_file = None
selected_sample_path = None
file_name = ""
sample_choice = ""

if input_method == "Subir Imagen Propia":
    uploaded_file = st.sidebar.file_uploader(
        "Sube PNG/JPEG (2D) o NIfTI/MHD (3D)",
        type=["png", "jpg", "jpeg", "nii.gz", "mhd"],
    )
    if uploaded_file:
        file_name = uploaded_file.name
else:
    sample_choice = st.sidebar.selectbox("Elige un caso clínico:", list(SAMPLES.keys()))
    selected_sample_path = SAMPLES[sample_choice]
    file_name = Path(selected_sample_path).name

col1, col2 = st.columns([1, 1])

with col1:
    st.header("1. Preprocesado Adaptativo")

    if uploaded_file is not None or selected_sample_path is not None:
        is_3d = (
            file_name.endswith(".nii.gz")
            or file_name.endswith(".mhd")
            or file_name.endswith(".npz")
        )

        try:
            start_time = time.time()
            heatmap_base_img = None
            if is_3d:
                if uploaded_file:
                    import tempfile

                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".nii.gz"
                    ) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name
                    tensor_224, tensor_3d, dim_orig, preview_img = process_3d_volume(
                        tmp_path
                    )
                    os.remove(tmp_path)
                else:
                    if selected_sample_path.endswith(".npz"):
                        tensor_224, tensor_3d, dim_orig, preview_img = process_fast_npz(
                            selected_sample_path
                        )
                    else:
                        tensor_224, tensor_3d, dim_orig, preview_img = (
                            process_3d_volume(selected_sample_path)
                        )
                router_proxy_img = make_router_preview_montage(tensor_224)
                st.image(
                    router_proxy_img,
                    caption="Vista proxy 2D usada por el router 3D",
                    width=300,
                )
                heatmap_base_img = make_router_preview_image(tensor_224)
            else:
                target = uploaded_file if uploaded_file else selected_sample_path
                tensor_224, tensor_3d, dim_orig, preview_img = process_2d_image(target)
                st.image(preview_img, caption="Imagen 2D", width=300)
                heatmap_base_img = preview_img

            tensor_224 = tensor_224.to(device)
            if tensor_3d is not None:
                tensor_3d = tensor_3d.to(device)

            with torch.no_grad():
                if router_mode == "router_a":
                    z = model.backbone(tensor_224)
                    expert_probs = router_a(z)
                    best_idx = int(expert_probs.argmax(dim=-1).item())
                    expert_output = run_expert_by_index(
                        model, best_idx, tensor_224, tensor_3d, device
                    )
                elif router_ready:
                    expert_output, best_idx, expert_probs = model(
                        tensor_224, x_3d=tensor_3d
                    )
                else:
                    best_idx = infer_expert_fallback(
                        sample_choice, selected_sample_path, file_name, is_3d
                    )
                    expert_probs = torch.zeros((1, 5), device=device)
                    expert_probs[0, best_idx] = 1.0
                    expert_output = run_expert_by_index(
                        model, best_idx, tensor_224, tensor_3d, device
                    )
                final_probs = torch.softmax(expert_output, dim=-1)
                pred_class = torch.argmax(final_probs, dim=-1).item()
                confidence = final_probs[0, pred_class].item() * 100

            infer_time = (time.time() - start_time) * 1000

            expert_probs_np = expert_probs[0].cpu().numpy()
            entropy = -np.sum(expert_probs_np * np.log(expert_probs_np + 1e-9))
            normalized_entropy = entropy / np.log(5)

            st.session_state["load_balance"][best_idx] += 1

            st.header("2. Resultados de Inferencia")

            nombre_patologia = EXPERTOS_INFO[best_idx]["clases_map"].get(
                pred_class, f"Desconocido ({pred_class})"
            )
            st.metric(
                label="Diagnóstico Médico",
                value=f"{nombre_patologia}",
                delta=f"{confidence:.2f}% Confianza",
            )

            st.write(f"⏱ Inferencia total: **{infer_time:.2f} ms**")

            if normalized_entropy > 0.80:
                st.error(
                    f"🚨 OOD DETECTADA: Entropía del router ({normalized_entropy:.2f}) > 0.80. La imagen no pertenece al dominio médico entrenado."
                )
            else:
                st.success(
                    f"Entropía Normal del Router: {normalized_entropy:.2f} (In-Distribution)."
                )

        except Exception as e:
            import traceback

            st.error(f"Error procesando la imagen:\n{e}\n{traceback.format_exc()}")

with col2:
    if (
        uploaded_file is not None or selected_sample_path is not None
    ) and "best_idx" in locals():
        st.header("3. Router y Load Balancing")
        exp_info = EXPERTOS_INFO[best_idx]

        st.info(
            f"**Experto Activado:** {exp_info['nombre']}\n\n**Arquitectura:** {exp_info['arq']}\n\n**Dataset Origen:** {exp_info['dataset']}\n\n**Confianza del Router:** {expert_probs_np[best_idx] * 100:.2f}%\n\n**Router usado:** {'A — ViT + Linear' if router_mode == 'router_a' else ('D — ViT + k-NN' if router_mode == 'knn' else 'Fallback heurístico')}"
        )

        st.subheader("Attention Heatmap del Router")
        try:
            backbone = getattr(model, "backbone", None) or getattr(model, "vit", None) or getattr(model, "vit_backbone", None)
            if backbone is None:
                st.caption("Backbone ViT no expuesto por el modelo — heatmap no disponible.")
            else:
                if router_mode == "router_a":
                    attn_heat = compute_router_attention_rollout(
                        backbone, router_a, tensor_224, target_idx=best_idx
                    )
                    saliency_heat, channel_scores = compute_router_input_saliency(
                        backbone, router_a, tensor_224, target_idx=best_idx
                    )
                else:
                    # Router D (k-NN) is not differentiable, so we fall back to plain
                    # attention rollout without gradient weighting and without saliency.
                    attn_heat = compute_plain_attention_rollout(backbone, tensor_224)
                    saliency_heat = attn_heat
                    channel_scores = np.array([1.0, 0.5, 0.25])
                if attn_heat is None or saliency_heat is None:
                    st.caption("No se pudieron extraer mapas de atención del backbone.")
                else:
                    overlay_base_img = heatmap_base_img
                    if is_3d:
                        dominant_channel = int(np.argmax(channel_scores))
                        overlay_base_img = make_router_channel_image(
                            tensor_224.detach().cpu(), dominant_channel
                        )
                    heat = fuse_router_heatmaps(
                        attn_heat,
                        saliency_heat,
                        overlay_base_img,
                        expert_idx=best_idx,
                        is_3d=is_3d,
                    )
                    heatmap_caption = (
                        "Atención del router sobre la vista proxy dominante (MIP)"
                        if is_3d
                        else "Attention rollout (Abnar & Zuidema 2020) — regiones con mayor peso del token [CLS]"
                    )
                    overlay_img = overlay_heatmap_on_image(
                        overlay_base_img, heat, is_3d=is_3d
                    )
                    st.image(overlay_img, caption=heatmap_caption, width=300)
        except Exception as _e:
            st.caption(f"Heatmap no disponible: {_e}")

        st.subheader("Balance de Carga Acumulado")
        import plotly.express as px

        df_lb = pd.DataFrame(
            {
                "Experto": [e["nombre"].split(" ")[1] for e in EXPERTOS_INFO],
                "Activaciones": st.session_state["load_balance"],
            }
        )
        fig = px.bar(df_lb, x="Experto", y="Activaciones", text_auto=True)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.header("4. Ablation Study del Router (balanceado)")
st.caption(
    "Comparación de 4 routers sobre val balanceado y val natural. "
    "El ViT + Linear es el router desplegado."
)
ablation_candidates = [
    "/workspace/moe_medical_vision/router_moe_rebuild/metrics/ablation_balanced.csv",
    "/workspace/moe_medical_vision/embeddings/ablation_study.csv",
]
df_ablation = None
for path in ablation_candidates:
    try:
        df_ablation = pd.read_csv(path)
        st.caption(f"Fuente: `{path}`")
        break
    except FileNotFoundError:
        continue
if df_ablation is not None:
    st.table(df_ablation)
else:
    st.warning("No se encontró el archivo del Ablation Study.")
