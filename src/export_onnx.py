"""
Script to export trained PyTorch ResNet-18 checkpoints to optimized ONNX models
for client-side browser inference via ONNX Runtime Web on GitHub Pages.
"""

import argparse
from pathlib import Path
import sys

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.transfer.model import build_resnet18
from src.transfer.train import detect_decoder_type_from_checkpoint, get_base_module


class SoftmaxResNet(nn.Module):
    """Wraps ResNet-18 model to output softmax probabilities directly."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return torch.softmax(logits, dim=1)


def export_checkpoint_to_onnx(
    ckpt_path: Path,
    output_path: Path,
    num_classes: int = 5,
    opset_version: int = 14
):
    print(f"\nExporting {ckpt_path.name} -> {output_path.name}...")
    ckpt_data = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt_data["model_state_dict"] if (isinstance(ckpt_data, dict) and "model_state_dict" in ckpt_data) else ckpt_data
    
    decoder_type = detect_decoder_type_from_checkpoint(ckpt_data, ckpt_path)
    print(f"  Detected decoder architecture: {decoder_type}")
    
    model = build_resnet18(num_classes=num_classes, freeze_backbone=False, decoder_type=decoder_type)
    raw_m = get_base_module(model)
    raw_m.load_state_dict(state_dict)
    raw_m.eval()
    
    wrapped_model = SoftmaxResNet(raw_m)
    wrapped_model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # First attempt with standard TorchScript exporter (dynamo=False)
        torch.onnx.export(
            wrapped_model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["probabilities"],
            dynamo=False,
            dynamic_axes={
                "input": {0: "batch_size"},
                "probabilities": {0: "batch_size"}
            }
        )
    except Exception as e:
        print(f"  Legacy export notice ({e}), falling back to default exporter...")
        torch.onnx.export(
            wrapped_model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=18,
            input_names=["input"],
            output_names=["probabilities"]
        )
    
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Exported successfully! File size: {size_mb:.2f} MB")
    
    # Verify inference output sum is 1.0
    with torch.no_grad():
        py_out = wrapped_model(dummy_input)
    print(f"  PyTorch test output sum: {py_out.sum().item():.4f}")


def main():
    root = Path(__file__).resolve().parent.parent
    ckpt_dir = root / "checkpoints"
    docs_models_dir = root / "docs" / "models"
    docs_models_dir.mkdir(parents=True, exist_ok=True)
    
    targets = [
        (ckpt_dir / "resnet18_mlp_finetuned_best.pt", docs_models_dir / "resnet18_mlp_finetuned_best.onnx"),
        (ckpt_dir / "resnet18_mlp_deeper_best.pt", docs_models_dir / "resnet18_mlp_deeper_best.onnx"),
        (ckpt_dir / "resnet18_linear_best.pt", docs_models_dir / "resnet18_linear_best.onnx")
    ]
    
    for ckpt_path, onnx_path in targets:
        if ckpt_path.exists():
            export_checkpoint_to_onnx(ckpt_path, onnx_path)
        else:
            print(f"Warning: Checkpoint not found: {ckpt_path}")
            
    print("\nAll target ONNX models generated in:", docs_models_dir)


if __name__ == "__main__":
    main()
