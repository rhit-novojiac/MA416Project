"""
Model definitions for transfer learning on the DeFungi dataset using ResNet-18.
"""

import sys
from pathlib import Path
from typing import Tuple, cast

# Ensure current directory is in sys.path
_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_resnet18(
    num_classes: int = 5,
    freeze_backbone: bool = True,
    dropout: float = 0.3,
    decoder_type: str = "linear"
) -> nn.Module:
    """
    Constructs a ResNet-18 model with ImageNet pre-trained weights and a customizable decoder head.

    Args:
        num_classes: Number of target categories (5 for DeFungi).
        freeze_backbone: If True, freezes all backbone convolutional layers.
        dropout: Dropout probability in the decoder.
        decoder_type: Architecture for the decoder head:
                      - 'linear': Single projection layer (Dropout -> Linear)
                      - 'mlp': 2-layer MLP with BatchNorm and ReLU
                      - 'mlp_deep': 3-layer MLP with BatchNorm and GELU

    Returns:
        model: Configured PyTorch nn.Module.
    """
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace final classification head (decoder) based on selected depth/architecture
    in_features = model.fc.in_features  # 512 for ResNet-18

    if decoder_type == "linear":
        model.fc = cast(
            nn.Linear,
            nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes)
            )
        )
    elif decoder_type == "mlp":
        model.fc = cast(
            nn.Linear,
            nn.Sequential(
                nn.Linear(in_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(256, num_classes)
            )
        )
    elif decoder_type == "mlp_deep":
        model.fc = cast(
            nn.Linear,
            nn.Sequential(
                nn.Linear(in_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(256, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(256, num_classes)
            )
        )
    else:
        raise ValueError(f"Unknown decoder_type '{decoder_type}'. Must be 'linear', 'mlp', or 'mlp_deep'.")

    return model


def get_parameter_counts(model: nn.Module) -> Tuple[int, int, int]:
    """
    Computes trainable, frozen, and total parameter counts.

    Returns:
        (trainable_params, frozen_params, total_params)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = trainable + frozen
    return trainable, frozen, total


def unfreeze_backbone(model: nn.Module, unfreeze_from_layer4_only: bool = False) -> None:
    """
    Utility for Reasonable / Stretch goals: unfreezes layers for fine-tuning.

    Args:
        model: ResNet model.
        unfreeze_from_layer4_only: If True, only unfreezes layer4 and fc.
                                   If False, unfreezes the whole backbone.
    """
    if unfreeze_from_layer4_only:
        for name, param in model.named_parameters():
            if "layer4" in name or "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True


if __name__ == "__main__":
    net = build_resnet18(num_classes=5, freeze_backbone=True, dropout=0.3)
    trainable_p, frozen_p, total_p = get_parameter_counts(net)
    print("ResNet-18 Model Initialized:")
    print(f"  Trainable Parameters: {trainable_p:,} ({trainable_p / total_p * 100:.2f}%)")
    print(f"  Frozen Parameters:    {frozen_p:,} ({frozen_p / total_p * 100:.2f}%)")
    print(f"  Total Parameters:     {total_p:,}")

    # Forward pass test
    dummy_input = torch.randn(2, 3, 224, 224)
    output = net(dummy_input)
    print(f"  Forward pass output shape: {output.shape} (Expected: [2, 5])")
