"""
Model definitions for transfer learning on the DeFungi dataset using ResNet-18.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, cast

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
    Constructs a ResNet-18 backbone with a custom classification decoder head.

    Args:
        num_classes: Number of target classes.
        freeze_backbone: Whether to freeze feature extraction layers.
        dropout: Dropout rate applied before linear layers.
        decoder_type: Architecture of classification head ('linear', 'mlp', 'mlp_deep', 'mlp_deeper').
    """
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # ResNet-18 has 512 fully connected input features
    in_features = model.fc.in_features

    if decoder_type == "linear":
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

    elif decoder_type == "mlp":
        model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )

    elif decoder_type == "mlp_deep":
        model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )

    elif decoder_type == "mlp_deeper":
        model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )

    else:
        raise ValueError(
            f"Unknown decoder_type '{decoder_type}'. "
            "Must be 'linear', 'mlp', 'mlp_deep', or 'mlp_deeper'."
        )

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


def unfreeze_backbone(
    model: nn.Module,
    unfreeze_from_layer4_only: bool = False,
    layers: Optional[str] = None
) -> None:
    """
    Selectively unfreezes model parameters for fine-tuning.

    Args:
        model: ResNet-18 model instance.
        unfreeze_from_layer4_only: Legacy boolean flag. If True, unfreezes layer4 and fc only.
        layers: Fine-tuning mode: 'layer4', 'layer3_4', or 'all'.
                If provided, overrides unfreeze_from_layer4_only.
    """
    mode = layers if layers is not None else ("layer4" if unfreeze_from_layer4_only else "all")

    if mode == "layer4":
        for name, param in model.named_parameters():
            if name.startswith("layer4") or name.startswith("fc"):
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif mode == "layer3_4":
        for name, param in model.named_parameters():
            if name.startswith("layer3") or name.startswith("layer4") or name.startswith("fc"):
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif mode == "all":
        for param in model.parameters():
            param.requires_grad = True
    else:
        raise ValueError(
            f"Unknown unfreeze mode '{mode}'. Must be 'layer4', 'layer3_4', or 'all'."
        )


def inspect_trainable_layers(model: nn.Module) -> Dict[str, Dict[str, int]]:
    """
    Summarizes trainable vs frozen parameters across model layer groups.
    """
    groups: Dict[str, Dict[str, int]] = {
        "stem (conv1/bn1)": {"trainable": 0, "frozen": 0},
        "layer1": {"trainable": 0, "frozen": 0},
        "layer2": {"trainable": 0, "frozen": 0},
        "layer3": {"trainable": 0, "frozen": 0},
        "layer4": {"trainable": 0, "frozen": 0},
        "decoder_head (fc)": {"trainable": 0, "frozen": 0},
    }

    for name, param in model.named_parameters():
        count = param.numel()
        bucket = "trainable" if param.requires_grad else "frozen"

        if name.startswith("conv1") or name.startswith("bn1"):
            groups["stem (conv1/bn1)"][bucket] += count
        elif name.startswith("layer1"):
            groups["layer1"][bucket] += count
        elif name.startswith("layer2"):
            groups["layer2"][bucket] += count
        elif name.startswith("layer3"):
            groups["layer3"][bucket] += count
        elif name.startswith("layer4"):
            groups["layer4"][bucket] += count
        elif name.startswith("fc"):
            groups["decoder_head (fc)"][bucket] += count

    return groups


if __name__ == "__main__":
    net = build_resnet18(
        num_classes=5,
        freeze_backbone=True,
        dropout=0.3
    )

    trainable_p, frozen_p, total_p = get_parameter_counts(net)

    print("ResNet-18 Model Initialized (Frozen Backbone):")
    print(f"  Trainable Parameters (Head): {trainable_p:,} ({trainable_p / total_p * 100:.2f}%)")
    print(f"  Frozen Parameters (Backbone): {frozen_p:,} ({frozen_p / total_p * 100:.2f}%)")
    print(f"  Total Parameters:             {total_p:,}")

    dummy_input = torch.randn(2, 3, 224, 224)
    output = net(dummy_input)

    print(f"  Forward pass output shape: {output.shape}")
    print("  Expected: [2, 5]")
