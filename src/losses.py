"""
losses.py — Funciones de pérdida del proyecto MoE
---------------------------------------------------
- FocalLoss          → Pancreatic Cancer (imbalance severo)
- WeightedBCELoss    → NIH ChestX-ray14 (multilabel + desbalanceo)
- AuxiliaryLoss      → Load Balancing (Switch Transformer)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Focal Loss ───────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss para clasificación binaria/multiclase desbalanceada.
    Referencia: Lin et al. (2017) — RetinaNet.

    Uso para Pancreatic Cancer:
        criterion = FocalLoss(gamma=2.0, alpha=0.25)
        loss = criterion(logits, labels)

    Args:
        gamma: factor de enfoque (típico: 2.0)
        alpha: peso de la clase positiva (None = sin peso)
        reduction: "mean" | "sum" | "none"
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, num_classes) sin softmax
            targets: (B,) con índices de clase
        """
        log_probs = F.log_softmax(logits, dim=1)          # (B, C)
        probs = torch.exp(log_probs)                       # (B, C)

        # Gather prob de la clase correcta
        log_p_t = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)          # (B,)

        focal_weight = (1 - p_t) ** self.gamma
        loss = -focal_weight * log_p_t

        if self.alpha is not None:
            alpha_t = torch.full_like(loss, self.alpha)
            # Para clase 0, usar (1 - alpha)
            alpha_t[targets == 0] = 1.0 - self.alpha
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ─── Weighted BCE Loss (NIH multilabel) ──────────────────────────────────────

class WeightedBCEWithLogitsLoss(nn.Module):
    """
    BCEWithLogitsLoss con pos_weight por clase.
    Para NIH ChestX-ray14 (multilabel, 2 patologías muy desbalanceadas).

    Uso:
        pos_weight = dataset.get_pos_weight()   # (2,) tensor
        criterion  = WeightedBCEWithLogitsLoss(pos_weight=pos_weight)
        loss = criterion(logits, labels)        # logits: (B,2), labels: (B,2) float
    """

    def __init__(self, pos_weight: torch.Tensor):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, targets)


# ─── Auxiliary Loss (Switch Transformer Load Balancing) ──────────────────────

class AuxiliaryLoadBalancingLoss(nn.Module):
    """
    Auxiliary Loss del paper Switch Transformer (Fedus et al., 2021).

    Previene Expert Collapse penalizando cuando un experto recibe
    desproporcionadamente más tráfico que otros.

    L_aux = α · N · Σ(i=1 to N) f_i · P_i

        f_i = fracción de tokens enrutados al experto i (dura, no diferenciable)
        P_i = probabilidad media asignada al experto i (suave, diferenciable)
        N   = número de expertos

    ⚠ Penalización del proyecto: max(f_i)/min(f_i) > 1.30 → -40% nota.

    Args:
        n_experts: número de expertos (5 en este proyecto)
        alpha:     peso de la auxiliary loss (rango: 0.01 – 0.1)
    """

    def __init__(self, n_experts: int = 5, alpha: float = 0.01):
        super().__init__()
        self.n_experts = n_experts
        self.alpha = alpha

    def forward(self, gate_probs: torch.Tensor) -> tuple:
        """
        Args:
            gate_probs: (B, N) tensor de probabilidades del router (después de softmax)

        Returns:
            (aux_loss, load_balance_ratio)
                aux_loss:           escalar para sumar a L_task
                load_balance_ratio: max(f_i)/min(f_i) — monitorear < 1.30
        """
        B, N = gate_probs.shape
        assert N == self.n_experts, f"Esperaba {self.n_experts} expertos, recibí {N}"

        # f_i: fracción de tokens al experto i (argmax = routing duro)
        expert_assignments = gate_probs.argmax(dim=-1)  # (B,)
        f = torch.zeros(N, device=gate_probs.device)
        for i in range(N):
            f[i] = (expert_assignments == i).float().sum() / B

        # P_i: probabilidad media asignada al experto i (diferenciable)
        P = gate_probs.mean(dim=0)  # (N,)

        # L_aux = α · N · Σ f_i · P_i
        aux_loss = self.alpha * N * (f * P).sum()

        # Ratio de balanceo (para monitorear)
        f_max = f.max().item()
        f_min = f.min().item() + 1e-6
        balance_ratio = f_max / f_min

        return aux_loss, balance_ratio

    def set_alpha(self, alpha: float):
        """Permite calibrar α durante el entrenamiento."""
        self.alpha = alpha
