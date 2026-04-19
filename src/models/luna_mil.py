from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torchvision.models.video import mc3_18, MC3_18_Weights


class MC3BackboneEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, pretrained: bool = True, use_gradient_checkpointing: bool = True):
        super().__init__()
        weights = MC3_18_Weights.DEFAULT if pretrained else None
        backbone = mc3_18(weights=weights)

        if in_channels != 3:
            old_conv = backbone.stem[0]
            new_conv = nn.Conv3d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=(old_conv.bias is not None),
            )
            with torch.no_grad():
                if in_channels == 1:
                    new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
                else:
                    repeats = (in_channels + 2) // 3
                    weight = old_conv.weight.repeat(1, repeats, 1, 1, 1)[:, :in_channels]
                    weight = weight / repeats
                    new_conv.weight.copy_(weight)
                if old_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)
            backbone.stem[0] = new_conv

        self.backbone = backbone
        self.out_dim = backbone.fc.in_features
        self.use_gradient_checkpointing = use_gradient_checkpointing

    def _run_block(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.use_gradient_checkpointing and self.training:
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.stem(x)
        x = self._run_block(self.backbone.layer1, x)
        x = self._run_block(self.backbone.layer2, x)
        x = self._run_block(self.backbone.layer3, x)
        x = self._run_block(self.backbone.layer4, x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class AttentionMILHead(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor, mask: torch.Tensor | None = None):
        # z: [B, K, D]
        scores = self.attn(z).squeeze(-1)  # [B, K]
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        weights = torch.softmax(scores, dim=1)
        pooled = torch.einsum('bk,bkd->bd', weights, z)
        return pooled, weights


class LUNAMILModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 1,
        pretrained: bool = True,
        attn_hidden_dim: int = 128,
        dropout: float = 0.3,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.encoder = MC3BackboneEncoder(
            in_channels=in_channels,
            pretrained=pretrained,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
        self.mil = AttentionMILHead(self.encoder.out_dim, hidden_dim=attn_hidden_dim, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.encoder.out_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        # x: [B, K, C, D, H, W]
        b, k, c, d, h, w = x.shape
        x = x.view(b * k, c, d, h, w)
        z = self.encoder(x).view(b, k, -1)
        pooled, weights = self.mil(z, mask=mask)
        logits = self.classifier(pooled)
        return logits, weights
