"""
Unit tests for dynamic fine-tuning: parameter freezing/unfreezing,
automatic checkpoint discovery, architecture detection, and optimizer grouping.
"""

import shutil
import sys
import tempfile
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import torch
import torch.nn as nn

from src.transfer.model import (
    build_resnet18,
    get_parameter_counts,
    unfreeze_backbone,
    inspect_trainable_layers
)
from src.transfer.train import (
    detect_decoder_type_from_checkpoint,
    find_best_checkpoint
)


@pytest.fixture
def temp_ckpt_dir():
    temp_path = Path(tempfile.mkdtemp(prefix="test_ft_ckpts_"))
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


def test_backbone_freezing_and_layer_unfreezing():
    """Verify that freeze_backbone accurately freezes the backbone and selective unfreezing works."""
    model = build_resnet18(num_classes=5, freeze_backbone=True, decoder_type="linear")
    trainable_p, frozen_p, total_p = get_parameter_counts(model)

    # Initial state: only linear head (512 * 5 + 5 = 2,565 params) is trainable
    assert trainable_p == 2565
    assert frozen_p == 11176512
    assert total_p == 11179077

    # Layer inspection breakdown
    breakdown = inspect_trainable_layers(model)
    assert breakdown["stem (conv1/bn1)"]["frozen"] > 0
    assert breakdown["stem (conv1/bn1)"]["trainable"] == 0
    assert breakdown["layer1"]["trainable"] == 0
    assert breakdown["layer4"]["trainable"] == 0
    assert breakdown["decoder_head (fc)"]["trainable"] == 2565

    # Unfreeze layer4 only
    unfreeze_backbone(model, layers="layer4")
    bd_l4 = inspect_trainable_layers(model)
    assert bd_l4["stem (conv1/bn1)"]["trainable"] == 0
    assert bd_l4["layer1"]["trainable"] == 0
    assert bd_l4["layer3"]["trainable"] == 0
    assert bd_l4["layer4"]["trainable"] > 0
    assert bd_l4["decoder_head (fc)"]["trainable"] > 0

    # Unfreeze layer3 and layer4
    unfreeze_backbone(model, layers="layer3_4")
    bd_l34 = inspect_trainable_layers(model)
    assert bd_l34["layer2"]["trainable"] == 0
    assert bd_l34["layer3"]["trainable"] > 0
    assert bd_l34["layer4"]["trainable"] > 0

    # Unfreeze all
    unfreeze_backbone(model, layers="all")
    trainable_all, frozen_all, _ = get_parameter_counts(model)
    assert frozen_all == 0
    assert trainable_all == total_p


def test_detect_decoder_type():
    """Test dynamic detection of decoder architecture from metadata, weights, and filename."""
    # From explicit metadata
    assert detect_decoder_type_from_checkpoint({"decoder_type": "mlp_deep"}, Path("model.pt")) == "mlp_deep"

    # From filename
    assert detect_decoder_type_from_checkpoint({}, Path("resnet18_mlp_deeper_best.pt")) == "mlp_deeper"
    assert detect_decoder_type_from_checkpoint({}, Path("resnet18_mlp_best.pt")) == "mlp"
    assert detect_decoder_type_from_checkpoint({}, Path("resnet18_linear_best.pt")) == "linear"

    # From state_dict weight layers
    dummy_mlp_state = {
        "model_state_dict": {
            "fc.0.weight": torch.randn(256, 512),
            "fc.3.weight": torch.randn(5, 256)
        }
    }
    assert detect_decoder_type_from_checkpoint(dummy_mlp_state, Path("unknown.pt")) == "mlp"


def test_dynamic_best_checkpoint_discovery(temp_ckpt_dir):
    """Test finding and selecting the best checkpoint dynamically across multiple candidates."""
    # Create 3 candidate checkpoints with different performance metrics
    m_linear = build_resnet18(num_classes=5, decoder_type="linear")
    m_mlp = build_resnet18(num_classes=5, decoder_type="mlp")
    m_deeper = build_resnet18(num_classes=5, decoder_type="mlp_deeper")

    torch.save({
        "epoch": 10,
        "decoder_type": "linear",
        "best_macro_f1": 0.6401,
        "val_acc": 0.6111,
        "model_state_dict": m_linear.state_dict()
    }, temp_ckpt_dir / "resnet18_linear_best.pt")

    torch.save({
        "epoch": 25,
        "decoder_type": "mlp",
        "best_macro_f1": 0.7540,
        "val_acc": 0.7159,
        "model_state_dict": m_mlp.state_dict()
    }, temp_ckpt_dir / "resnet18_mlp_best.pt")

    torch.save({
        "epoch": 50,
        "decoder_type": "mlp_deeper",
        "best_macro_f1": 0.8798,
        "val_acc": 0.8607,
        "model_state_dict": m_deeper.state_dict()
    }, temp_ckpt_dir / "resnet18_mlp_deeper_best.pt")

    # 1. Automatic Champion Discovery (should pick mlp_deeper with F1 87.98%)
    best_path, dec_type, f1 = find_best_checkpoint(temp_ckpt_dir)
    assert best_path.name == "resnet18_mlp_deeper_best.pt"
    assert dec_type == "mlp_deeper"
    assert f1 == pytest.approx(0.8798)

    # 2. Preferred Decoder Selection (should pick mlp even though mlp_deeper has higher F1)
    pref_path, pref_dec, pref_f1 = find_best_checkpoint(temp_ckpt_dir, preferred_decoder="mlp")
    assert pref_path.name == "resnet18_mlp_best.pt"
    assert pref_dec == "mlp"
    assert pref_f1 == pytest.approx(0.7540)

