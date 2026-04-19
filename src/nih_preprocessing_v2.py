"""
Preprocesamiento mejorado para NIH ChestX-ray14
================================================
Implementa las recomendaciones del profesor en asesoría:
1. CLAHE — mejora de contraste local
2. Unsharp Mask — realza bordes sin destruir gradientes (reemplaza Otsu)
3. MHA (Multi-Head Attention) — captura características globales
4. Normalización específica para radiografías (percentiles p2/p98)
5. Augmentation balanceado hacia clases con peor rendimiento

Otsu fue removido: demasiado agresivo para radiografías de tórax,
destruye gradientes diagnósticos y genera artefactos.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms
import random


# ─── 1. Funciones de preprocesamiento ────────────────────────────────────────

def normalize_xray(img_np: np.ndarray) -> np.ndarray:
    """
    Normalización específica para radiografías.
    Redistribuye niveles de gris usando percentiles p2/p98 para evitar
    que el modelo confunda hueso con nódulos por densidades similares.
    """
    p2  = np.percentile(img_np, 2)
    p98 = np.percentile(img_np, 98)
    img_clipped = np.clip(img_np, p2, p98)
    img_norm = ((img_clipped - p2) / (p98 - p2 + 1e-6) * 255).astype(np.uint8)
    return img_norm


def apply_clahe(img_np: np.ndarray, clip_limit: float = 2.0,
                tile_size: tuple = (8, 8)) -> np.ndarray:
    """
    CLAHE — Contrast Limited Adaptive Histogram Equalization.
    Mejora el contraste local zona a zona (no global).
    Aclara la zona pulmonar para que los nódulos sean más visibles.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(img_np)


def apply_unsharp_mask(img_np: np.ndarray,
                       sigma: float = 1.0,
                       strength: float = 0.5) -> np.ndarray:
    """
    Unsharp masking — realza bordes suavemente sin binarizar.
    Alternativa clínica al Otsu para radiografías.
    Hace más visibles bordes de nódulos sin destruir gradientes de gris.
    """
    blurred   = cv2.GaussianBlur(img_np.astype(np.float32), (0, 0), sigma)
    sharpened = img_np.astype(np.float32) + \
                strength * (img_np.astype(np.float32) - blurred)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def full_xray_preprocess(pil_img: Image.Image,
                          clip_limit: float = 2.0) -> Image.Image:
    """
    Pipeline completo: Normalización → CLAHE → Unsharp Mask
    Otsu removido — demasiado agresivo para radiografías de tórax.
    Recibe PIL Image, devuelve PIL Image RGB.
    """
    img_gray = np.array(pil_img.convert('L'))

    # Paso 1: Normalización de niveles de gris
    img_norm = normalize_xray(img_gray)

    # Paso 2: CLAHE para mejorar contraste local
    img_clahe = apply_clahe(img_norm, clip_limit=clip_limit)

    # Paso 3: Unsharp mask para realzar bordes de nódulos
    img_sharp = apply_unsharp_mask(img_clahe, sigma=1.0, strength=0.5)

    return Image.fromarray(img_sharp).convert('RGB')


# ─── 2. Transform completo para NIH ──────────────────────────────────────────

class NIHXrayTransform:
    """
    Transform específico para NIH ChestX-ray14.
    Implementa las recomendaciones del profesor:
    - Normalización p2/p98 + CLAHE + Unsharp Mask
    - Gamma correction para simular variaciones de exposición
    - Augmentation agresivo en train
    - Normalización estadística calculada sobre NIH (no ImageNet)
    """

    # Media y std calculadas sobre NIH ChestX-ray14
    NIH_MEAN = [0.5056, 0.5056, 0.5056]
    NIH_STD  = [0.2522, 0.2522, 0.2522]

    def __init__(self, split: str = 'train', clahe_clip: float = 2.0,
                 gamma_range: tuple = (0.7, 1.5)):
        self.split       = split
        self.clahe_clip  = clahe_clip
        self.gamma_range = gamma_range
        self.to_tensor   = transforms.ToTensor()
        self.normalize   = transforms.Normalize(
            mean=self.NIH_MEAN, std=self.NIH_STD
        )

    def apply_gamma(self, img: Image.Image, gamma: float = None) -> Image.Image:
        """Gamma correction — simula variaciones de exposición radiológica."""
        if gamma is None:
            gamma = random.uniform(*self.gamma_range)
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.power(arr, gamma)
        return Image.fromarray((arr * 255).astype(np.uint8))

    def __call__(self, pil_img: Image.Image) -> torch.Tensor:
        # Preprocesamiento base — siempre aplicar
        img = full_xray_preprocess(pil_img, clip_limit=self.clahe_clip)
        img = img.resize((224, 224), Image.BILINEAR)

        # Augmentation solo en train
        if self.split == 'train':
            # Gamma correction aleatoria
            img = self.apply_gamma(img)

            # Flip horizontal (válido en radiografías AP/PA)
            if random.random() > 0.5:
                img = TF.hflip(img)

            # Rotación moderada
            angle = random.uniform(-15, 15)
            img = TF.rotate(img, angle)

            # Zoom aleatorio
            scale = random.uniform(0.88, 1.12)
            new_h = int(224 * scale)
            img = img.resize((new_h, new_h), Image.BILINEAR)
            img = TF.center_crop(img, 224)

            # Brillo y contraste — vital para imágenes médicas
            img = TF.adjust_brightness(img, random.uniform(0.75, 1.25))
            img = TF.adjust_contrast(img, random.uniform(0.75, 1.25))

            # Ruido gaussiano leve para regularización
            tensor = self.to_tensor(img)
            noise  = torch.randn_like(tensor) * 0.02
            tensor = (tensor + noise).clamp(0, 1)
            return self.normalize(tensor)

        return self.normalize(self.to_tensor(img))


# ─── 3. Multi-Head Attention espacial ────────────────────────────────────────

class SpatialMHA(nn.Module):
    """
    Módulo de Multi-Head Attention espacial.
    El profesor dijo: "pásenlo a MHA" — captura relaciones globales
    entre zonas pulmonares que el CNN no modela bien.

    Se inserta después del backbone, antes de la cabeza de clasificación.

    Args:
        d_model:   canales del feature map del backbone (1024 para DenseNet-121)
        num_heads: 4 por defecto, 8 si la GPU aguanta (RTX 4090 sí aguanta)
        dropout:   regularización
    """
    def __init__(self, d_model: int = 1024, num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) debe ser divisible por num_heads ({num_heads})"

        self.num_heads = num_heads
        self.d_head    = d_model // num_heads
        self.scale     = self.d_head ** -0.5

        self.qkv  = nn.Linear(d_model, d_model * 3, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) — feature map del CNN backbone
        Returns:
            (B, C, H, W) — feature map con atención espacial aplicada
        """
        B, C, H, W = x.shape
        N = H * W

        # Aplanar dims espaciales: (B, N, C)
        x_flat = x.flatten(2).transpose(1, 2)

        # QKV projection
        qkv = self.qkv(x_flat).reshape(B, N, 3, self.num_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # cada uno: (B, heads, N, d_head)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)

        # Combinar cabezas
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # Proyección + conexión residual + LayerNorm
        out = self.norm(x_flat + self.proj(out))

        # Devolver a (B, C, H, W)
        return out.transpose(1, 2).reshape(B, C, H, W)


# ─── 4. Experto NIH: DenseNet-121 + MHA ──────────────────────────────────────

class NIHExpertWithMHA(nn.Module):
    """
    DenseNet-121 + SpatialMHA antes de la cabeza de clasificación.

    El MHA captura dependencias de largo alcance entre zonas pulmonares
    que el DenseNet solo con convoluciones no puede modelar:
    p.ej. correlación entre zona perihiliar y ápice para Tuberculosis.

    Args:
        num_classes: 2 patologias NIH (Mass, Nodule)
        num_heads:   8 cabezas de atención (RTX 4090 lo soporta)
        pretrained:  cargar pesos ImageNet del DenseNet-121
    """
    def __init__(self, num_classes: int = 2, num_heads: int = 8,
                 pretrained: bool = True):
        super().__init__()
        import timm

        # DenseNet-121 sin pooling global — necesitamos el feature map (B,1024,7,7)
        self.backbone = timm.create_model(
            'densenet121',
            pretrained=pretrained,
            num_classes=0,
            global_pool=''
        )

        d_model = 1024  # canales de salida de DenseNet-121

        # MHA espacial: 8 heads como sugirió el profesor
        self.mha = SpatialMHA(d_model=d_model, num_heads=num_heads)

        # Cabeza multilabel con regularización
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(d_model, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)    # (B, 1024, 7, 7)
        features = self.mha(features)  # atención espacial
        return self.head(features)     # (B, num_classes)


# ─── 5. Muestra pequeña — validar antes de entrenar todo ──────────────────

def visualize_preprocessing_sample(nih_root: str, n_samples: int = 4,
                                    save_path: str = '/workspace/moe_medical_vision/preprocessing_sample.png'):
    """
    Muestra 4 columnas: Original | CLAHE+Norm | +Unsharp | +Augmentation
    Corre PRIMERO para validar que el preprocesamiento mejora visualmente.
    """
    import matplotlib.pyplot as plt
    from pathlib import Path

    root     = Path(nih_root)
    all_imgs = list(root.rglob('*.png'))
    samples  = random.sample(all_imgs, min(n_samples, len(all_imgs)))

    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    titles = ['Original', 'Norm + CLAHE', 'Norm + CLAHE\n+ Unsharp', 'Con Augmentation']

    transform_aug = NIHXrayTransform(split='train', clahe_clip=2.0)

    for i, img_path in enumerate(samples):
        original  = Image.open(img_path).convert('L')
        arr_orig  = np.array(original)

        # Columna 2: solo normalización + CLAHE
        arr_norm  = normalize_xray(arr_orig)
        arr_clahe = apply_clahe(arr_norm)

        # Columna 3: + unsharp mask
        arr_sharp = apply_unsharp_mask(arr_clahe, sigma=1.0, strength=0.5)

        # Columna 4: transform completo con augmentation (volver a numpy)
        tensor_aug = transform_aug(original)
        arr_aug = (tensor_aug.permute(1, 2, 0).numpy() * 0.2522 + 0.5056)
        arr_aug = np.clip(arr_aug[:, :, 0], 0, 1)

        for j, (arr, title) in enumerate(zip(
            [arr_orig, arr_clahe, arr_sharp, arr_aug], titles
        )):
            cmap = 'gray'
            axes[i, j].imshow(arr, cmap=cmap)
            axes[i, j].set_title(title, fontsize=9)
            axes[i, j].axis('off')

    plt.suptitle('Validación del preprocesamiento NIH — muestra pequeña',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Guardado en {save_path}')


# ─── 6. Sampler balanceado por clase ─────────────────────────────────────────

def get_nih_balanced_sampler(dataset, oversample_factor: dict = None):
    """
    WeightedRandomSampler que sobre-muestrea clases con peor F1.
    El profesor: "aumento de datos hacia el que les esté yendo peor".

    Args:
        dataset:           NIHChestXray14Dataset
        oversample_factor: {class_idx: factor} para clases específicas
                           Ej: {13: 5.0} — Hernia x5 (la más rara)
                               {6: 3.0}  — Pneumonia x3
                               {10: 3.0} — Emphysema x3
    """
    from torch.utils.data import WeightedRandomSampler

    labels_matrix  = np.array(dataset.df['labels_list'].tolist())  # (N, num_classes)
    N              = len(labels_matrix)
    sample_weights = np.ones(N)
    
    num_classes = len(dataset.df['labels_list'].iloc[0])

    for class_idx in range(num_classes):
        class_mask = labels_matrix[:, class_idx] == 1
        n_pos      = class_mask.sum()
        if n_pos > 0:
            class_weight = N / (num_classes * n_pos)
            sample_weights[class_mask] *= class_weight

    if oversample_factor:
        for class_idx, factor in oversample_factor.items():
            mask = labels_matrix[:, class_idx] == 1
            sample_weights[mask] *= factor

    return WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=N,
        replacement=True
    )