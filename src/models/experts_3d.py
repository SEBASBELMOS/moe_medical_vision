from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torchvision.models.video import r3d_18, mc3_18, R3D_18_Weights, MC3_18_Weights


class R3D18Expert(nn.Module):
    """R3D-18 adaptado para volúmenes médicos 1 canal."""

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 1,
        pretrained: bool = True,
        dropout: float = 0.2,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        weights = R3D_18_Weights.DEFAULT if pretrained else None
        backbone = r3d_18(weights=weights)

        # Adaptar primer conv a 1 canal si aplica.
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

        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

        self.backbone = backbone
        self.use_gradient_checkpointing = use_gradient_checkpointing

    def enable_gradient_checkpointing(self, enabled: bool = True):
        self.use_gradient_checkpointing = enabled

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
        x = self.backbone.fc(x)
        return x


def build_luna_expert(pretrained: bool = True, use_gradient_checkpointing: bool = True) -> nn.Module:
    return R3D18Expert(
        num_classes=2,
        in_channels=1,
        pretrained=pretrained,
        dropout=0.2,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )



def build_luna_expert_v2(pretrained: bool = True, use_gradient_checkpointing: bool = True) -> nn.Module:
    return R3D18Expert(
        num_classes=2,
        in_channels=1,
        pretrained=pretrained,
        dropout=0.35,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )

def build_pancreatic_expert(pretrained: bool = True, use_gradient_checkpointing: bool = True) -> nn.Module:
    return R3D18Expert(
        num_classes=2,
        in_channels=1,
        pretrained=pretrained,
        dropout=0.3,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )


class MC318Expert(nn.Module):
    """MC3-18 adaptado para volúmenes médicos 1 canal."""

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 1,
        pretrained: bool = True,
        dropout: float = 0.25,
        use_gradient_checkpointing: bool = False,
    ):
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

        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
        self.backbone = backbone
        self.use_gradient_checkpointing = use_gradient_checkpointing

    def enable_gradient_checkpointing(self, enabled: bool = True):
        self.use_gradient_checkpointing = enabled

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
        x = self.backbone.fc(x)
        return x


def build_luna_patch_expert_mc3(pretrained: bool = True, use_gradient_checkpointing: bool = True) -> nn.Module:
    return MC318Expert(
        num_classes=2,
        in_channels=1,
        pretrained=pretrained,
        dropout=0.25,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )


def build_pancreatic_expert_mc3(pretrained: bool = True, use_gradient_checkpointing: bool = True) -> nn.Module:
    return MC318Expert(
        num_classes=2,
        in_channels=1,
        pretrained=pretrained,
        dropout=0.2,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )


from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights


class R2Plus1D18Expert(nn.Module):
    """R(2+1)D-18 adapted for single-channel 3D medical volumes."""

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 1,
        pretrained: bool = True,
        dropout: float = 0.25,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        weights = R2Plus1D_18_Weights.DEFAULT if pretrained else None
        backbone = r2plus1d_18(weights=weights)

        # R(2+1)D stem has factored convs: stem[0] is Conv3d(3,45,(1,7,7))
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

        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
        self.backbone = backbone
        self.use_gradient_checkpointing = use_gradient_checkpointing

    def enable_gradient_checkpointing(self, enabled: bool = True):
        self.use_gradient_checkpointing = enabled

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
        x = self.backbone.fc(x)
        return x


def build_luna_expert_r2plus1d(pretrained: bool = True, use_gradient_checkpointing: bool = True) -> nn.Module:
    return R2Plus1D18Expert(
        num_classes=2,
        in_channels=1,
        pretrained=pretrained,
        dropout=0.25,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )


def build_pancreatic_expert_r2plus1d(pretrained: bool = True, use_gradient_checkpointing: bool = True) -> nn.Module:
    return R2Plus1D18Expert(
        num_classes=2,
        in_channels=1,
        pretrained=pretrained,
        dropout=0.3,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )
