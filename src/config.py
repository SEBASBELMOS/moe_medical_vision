"""
config.py — Configuración centralizada del proyecto MoE
--------------------------------------------------------
EDITA SOLO ESTE ARCHIVO para cambiar rutas, hiperparámetros
o configuración del clúster.
"""

from pathlib import Path

# ─── Rutas de datos ──────────────────────────────────────────────────────────

DATA_ROOT = Path("/workspace/moe_medical_vision/data/raw")

DATASET_PATHS = {
    "nih_chestxray":  DATA_ROOT / "nih",
    "isic2019":       DATA_ROOT / "isic",
    "osteoarthritis": DATA_ROOT / "osteoporosis" / "KLGrade" / "KLGrade",
    "luna16":         DATA_ROOT / "luna16",
    "pancreatic":     DATA_ROOT / "pancreatic",
}

CHECKPOINT_DIR = Path("/workspace/moe_medical_vision/checkpoints")
LOG_DIR        = Path("/workspace/logs")

# ─── Hiperparámetros de entrenamiento ────────────────────────────────────────

SEED = 42

# Batch sizes (ajustar según VRAM disponible)
BATCH_SIZE_2D    = 32    # Para NIH, ISIC, Osteoarthritis
BATCH_SIZE_3D    = 4     # Para LUNA16, Pancreatic (VRAM limitada)
BATCH_SIZE_MIXED = 8     # Por dataset en el DataLoader mixto del router

# Gradient Accumulation (batch efectivo = BATCH_SIZE * ACCUMULATION_STEPS)
ACCUMULATION_STEPS = 4   # Batch efectivo = 32 para 2D, = 16 para 3D

# Mixed Precision
USE_FP16 = True

# Learning rates (Método B — Por Partes)
LR_EXPERT_FASE1 = 1e-3   # Entrenamiento individual de expertos
LR_ROUTER_FASE2 = 1e-3   # Router aislado con expertos congelados
LR_FINETUNE_FASE3_EXPERT = 1e-5   # Fine-tuning global — expertos
LR_FINETUNE_FASE3_ROUTER = 1e-4   # Fine-tuning global — router

# Épocas por fase (Método B)
EPOCHS_EXPERTO_2D = 20
EPOCHS_EXPERTO_3D = 15   # Menos épocas, más lento por volumen
EPOCHS_ROUTER     = 30
EPOCHS_FINETUNE   = 10

# ─── Configuración del Sistema MoE ───────────────────────────────────────────

N_EXPERTS = 5

# Auxiliary Loss (Switch Transformer)
AUX_ALPHA_INITIAL = 0.01   # Empezar conservador
AUX_ALPHA_MAX     = 0.1    # Máximo si el balance no converge
BALANCE_RATIO_LIMIT = 1.30  # ⚠ Penalización del 40% si se supera

# Router — ViT backbone
VIT_MODEL_NAME = "vit_tiny_patch16_224"   # De timm
VIT_FROZEN     = True    # Congelar pesos del backbone

# ─── OOD Detection ───────────────────────────────────────────────────────────
OOD_ENTROPY_THRESHOLD = 0.80   # Entropía > 80% → imagen OOD

# ─── Hardware ────────────────────────────────────────────────────────────────
NUM_WORKERS    = 4
PIN_MEMORY     = True
NUM_GPUS       = 2       # 2 GPUs × 12 GB VRAM
VRAM_PER_GPU   = 12.0   # GB

# ─── Tracking ────────────────────────────────────────────────────────────────
WANDB_PROJECT  = "moe-medico-analitica2"
WANDB_ENTITY   = None    # Tu username de W&B (o None para no usar)
USE_WANDB      = False   # Cambiar a True cuando tengas las credenciales

# ─── Seeds ───────────────────────────────────────────────────────────────────

def set_seed(seed: int = SEED):
    """Fija todas las seeds para reproducibilidad."""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Config] Seed fijada: {seed}")
