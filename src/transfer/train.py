"""
Training, fine-tuning, evaluation, and benchmark verification script for ResNet-18 on DeFungi.
Supports both Stage 1 (decoder feature extraction) and Stage 2 (dynamic backbone fine-tuning).
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sized, Tuple, cast

# Ensure current directory is in sys.path for direct script and standalone execution
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm  # type: ignore[import-untyped]

# Fallback between relative and absolute imports
try:
    from .dataset import get_dataloaders
    from .model import build_resnet18, get_parameter_counts, unfreeze_backbone, inspect_trainable_layers
except (ImportError, ValueError):
    from dataset import get_dataloaders
    from model import build_resnet18, get_parameter_counts, unfreeze_backbone, inspect_trainable_layers


def get_base_module(model: nn.Module) -> nn.Module:
    """Unwraps DataParallel or compiled wrappers to access the root nn.Module."""
    if hasattr(model, "module"):
        return cast(nn.Module, getattr(model, "module"))
    if hasattr(model, "_orig_mod"):
        return cast(nn.Module, getattr(model, "_orig_mod"))
    return model


def detect_decoder_type_from_checkpoint(checkpoint_state: Dict[str, Any], file_path: Path) -> str:
    """
    Infers the decoder architecture ('linear', 'mlp', 'mlp_deep', 'mlp_deeper')
    from checkpoint metadata, state dict layer structures, or filename.
    """
    if isinstance(checkpoint_state, dict):
        if "decoder_type" in checkpoint_state and checkpoint_state["decoder_type"]:
            return str(checkpoint_state["decoder_type"])

        # Inspect model_state_dict layer shapes
        if "model_state_dict" in checkpoint_state and isinstance(checkpoint_state["model_state_dict"], dict):
            sd = checkpoint_state["model_state_dict"]
            fc_weight_keys = [k for k in sd.keys() if (k.startswith("fc.") or ".fc." in k) and "weight" in k]
            if len(fc_weight_keys) == 4:
                return "mlp_deeper"
            elif len(fc_weight_keys) == 3:
                return "mlp_deep"
            elif len(fc_weight_keys) == 2:
                return "mlp"
            elif len(fc_weight_keys) == 1:
                return "linear"

    # Infer from filename
    name = file_path.stem.lower()
    for candidate in ["mlp_deeper", "mlp_deep", "mlp", "linear"]:
        if candidate in name:
            return candidate

    return "linear"


def find_best_checkpoint(
    checkpoint_dir: Path,
    preferred_decoder: Optional[str] = None
) -> Tuple[Path, str, float]:
    """
    Scans checkpoint_dir for candidate models, displays a summary table,
    and returns (best_checkpoint_path, detected_decoder_type, best_macro_f1).
    """
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory '{checkpoint_dir}' does not exist.")

    # Search for best checkpoints first; fallback to all .pt
    candidates = [p for p in checkpoint_dir.glob("resnet18_*_best.pt") if "_finetuned" not in p.name]
    if not candidates:
        candidates = list(checkpoint_dir.glob("resnet18_*_best.pt"))
    if not candidates:
        candidates = list(checkpoint_dir.glob("*.pt"))

    if not candidates:
        raise FileNotFoundError(f"No checkpoint files (.pt) found in '{checkpoint_dir}'.")

    records = []
    for p in sorted(candidates):
        try:
            data = torch.load(p, map_location="cpu")
            macro_f1 = -1.0
            val_acc = -1.0
            dec = None
            if isinstance(data, dict):
                macro_f1 = float(data.get("best_macro_f1", -1.0))
                val_acc = float(data.get("val_acc", -1.0))
                dec = detect_decoder_type_from_checkpoint(data, p)
                if macro_f1 <= 0 and "history" in data and isinstance(data["history"], dict):
                    f1_hist = data["history"].get("val_macro_f1", [])
                    if f1_hist:
                        macro_f1 = float(max(f1_hist))
            if dec is None:
                dec = detect_decoder_type_from_checkpoint({}, p)
            records.append({
                "path": p,
                "decoder_type": dec,
                "macro_f1": macro_f1,
                "val_acc": val_acc,
            })
        except Exception as e:
            print(f"[Warning] Could not inspect checkpoint {p.name}: {e}")

    if not records:
        raise RuntimeError(f"No valid checkpoints could be loaded from '{checkpoint_dir}'.")

    print("\n" + "=" * 70)
    print(f"{'DISCOVERED CHECKPOINTS IN':<30} {str(checkpoint_dir)}")
    print("=" * 70)
    print(f"{'Checkpoint Filename':<32} | {'Decoder':<12} | {'Macro F1':<10} | {'Val Acc':<9}")
    print("-" * 70)
    for r in records:
        f1_str = f"{r['macro_f1']*100:.2f}%" if r["macro_f1"] >= 0 else "N/A"
        acc_str = f"{r['val_acc']*100:.2f}%" if r["val_acc"] >= 0 else "N/A"
        print(f"{r['path'].name:<32} | {r['decoder_type']:<12} | {f1_str:<10} | {acc_str:<9}")
    print("=" * 70)

    # Match preferred decoder if specified
    if preferred_decoder:
        matching = [r for r in records if r["decoder_type"] == preferred_decoder]
        if matching:
            matching.sort(key=lambda x: (x["macro_f1"], x["val_acc"]), reverse=True)
            chosen = matching[0]
            print(f"[Dynamic Selection] Selected preferred '{preferred_decoder}' checkpoint: {chosen['path'].name}")
            return chosen["path"], chosen["decoder_type"], chosen["macro_f1"]
        else:
            print(f"[Dynamic Selection] No exact match for preferred decoder '{preferred_decoder}'. Picking overall champion.")

    records.sort(key=lambda x: (x["macro_f1"], x["val_acc"]), reverse=True)
    champion = records[0]
    print(f"[Dynamic Selection] Automatically selected champion checkpoint: {champion['path'].name}")
    print(f"                     Architecture: {champion['decoder_type'].upper()} | Val Macro-F1: {champion['macro_f1']*100:.2f}%\n")
    return champion["path"], champion["decoder_type"], champion["macro_f1"]


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    use_amp: bool = False,
    max_batches: Optional[int] = None
) -> Tuple[float, float]:
    """Runs a single training epoch with optional mixed precision."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch_idx, (images, labels) in enumerate(pbar):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct / total:.4f}"})

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False,
    max_batches: Optional[int] = None
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """Evaluates the model on test/validation set and computes metrics."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    total = len(all_labels)
    eval_loss = running_loss / max(total, 1)
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    if total > 0:
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    else:
        acc = 0.0
        macro_f1 = 0.0

    return float(eval_loss), float(acc), float(macro_f1), y_true, y_pred


def save_history_files(history: Dict[str, List[Any]], json_path: Path, csv_path: Path) -> None:
    """
    Saves training history dictionary to both JSON and CSV files for human inspection
    and persistent multi-session recovery.
    """
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Save JSON
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"[Warning] Failed to write history JSON ({json_path}): {e}")

    # Save CSV
    try:
        epochs = history.get("epoch", [])
        if not epochs and "train_loss" in history:
            epochs = list(range(1, len(history["train_loss"]) + 1))

        fieldnames = ["epoch"] + [k for k in history.keys() if k != "epoch"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(len(epochs)):
                row = {k: history[k][i] if i < len(history[k]) else "" for k in fieldnames}
                writer.writerow(row)
    except Exception as e:
        print(f"[Warning] Failed to write history CSV ({csv_path}): {e}")


def load_history(json_path: Path) -> Dict[str, List[Any]]:
    """Loads training history dictionary from a JSON file if it exists."""
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception as e:
            print(f"[Warning] Could not load history from {json_path}: {e}")
    return {}


def align_history_to_epoch(history: Dict[str, List[Any]], target_epoch: int) -> Dict[str, List[Any]]:
    """
    Ensures history is aligned to target_epoch by trimming any trailing entries
    beyond target_epoch (useful when rolling back or restarting from an earlier checkpoint).
    """
    if not history:
        return {}

    cleaned: Dict[str, List[Any]] = {}
    if "epoch" in history and history["epoch"]:
        valid_indices = [i for i, ep in enumerate(history["epoch"]) if int(ep) <= target_epoch]
        for k, vals in history.items():
            cleaned[k] = [vals[i] for i in valid_indices if i < len(vals)]
    else:
        for k, vals in history.items():
            cleaned[k] = vals[:target_epoch] if len(vals) > target_epoch else list(vals)
    return cleaned


def save_training_plots(
    history: Dict[str, List[Any]],
    save_path: Path,
    verbose: bool = False
) -> None:
    """
    Saves loss, accuracy, and macro-F1 curves across epochs.
    Preserves and extends history continuously without resetting x-axis.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = history.get("epoch", [])
    if not epochs or len(epochs) != len(history.get("train_loss", [])):
        epochs = list(range(1, len(history.get("train_loss", [])) + 1))

    if not epochs:
        return

    marker_loss = "o" if len(epochs) <= 30 else None
    marker_val = "s" if len(epochs) <= 30 else None
    markersize = 4 if len(epochs) <= 30 else 2

    has_f1 = "val_macro_f1" in history and len(history["val_macro_f1"]) == len(epochs)
    num_plots = 3 if has_f1 else 2
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
    ax1, ax2 = axes[0], axes[1]

    # Loss subplot
    ax1.plot(epochs, history["train_loss"], marker=marker_loss, markersize=markersize,
             label="Train Loss", color="#1f77b4", linewidth=1.8)
    ax1.plot(epochs, history["val_loss"], marker=marker_val, markersize=markersize,
             label="Val Loss", color="#ff7f0e", linewidth=1.8)
    ax1.set_title("Cross-Entropy Loss vs. Epoch", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Accuracy subplot
    ax2.plot(epochs, [float(a) * 100 for a in history["train_acc"]], marker=marker_loss, markersize=markersize,
             label="Train Acc", color="#1f77b4", linewidth=1.8)
    ax2.plot(epochs, [float(a) * 100 for a in history["val_acc"]], marker=marker_val, markersize=markersize,
             label="Val Acc", color="#ff7f0e", linewidth=1.8)
    ax2.axhline(y=48.33, color="red", linestyle="--", linewidth=1.5, label="Majority Baseline (48.33%)")
    ax2.set_title("Accuracy (%) vs. Epoch", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Optional 3rd subplot: Macro-F1
    if has_f1:
        ax3 = axes[2]
        ax3.plot(epochs, [float(f) * 100 for f in history["val_macro_f1"]], marker="^" if len(epochs) <= 30 else None,
                 markersize=markersize, label="Val Macro F1", color="#2ca02c", linewidth=1.8)
        ax3.axhline(y=13.03, color="red", linestyle="--", linewidth=1.5, label="Baseline Macro F1 (13.03%)")
        ax3.set_title("Validation Macro F1 (%) vs. Epoch", fontweight="bold")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Macro F1 (%)")
        ax3.legend()
        ax3.grid(True, linestyle="--", alpha=0.6)
        ax3.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    if verbose:
        print(f"Saved training curves to {save_path}")


def save_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str],
    save_path: Path,
    verbose: bool = False
) -> None:
    """Saves normalized and raw confusion matrix visualization."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if len(y_true) == 0 or len(y_pred) == 0:
        return

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(cmap="Blues", ax=cast(Any, ax), colorbar=True)
    plt.title("ResNet-18 Transfer Learning - Confusion Matrix")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    if verbose:
        print(f"Saved confusion matrix plot to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train or fine-tune ResNet-18 on DeFungi Dataset.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of additional training epochs (default: 10)")
    parser.add_argument("--total-epochs", type=int, default=None,
                        help="Target total epochs count (overrides --epochs to reach this specific epoch)")
    parser.add_argument("--batch-size", type=int, default=32, help="Input batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for initial feature extraction training")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 regularization")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate before linear head")
    parser.add_argument("--decoder-type", type=str, default="linear", choices=["linear", "mlp", "mlp_deep", "mlp_deeper"],
                        help="Decoder head architecture: 'linear', 'mlp', 'mlp_deep', or 'mlp_deeper'")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine", "plateau"],
                        help="Learning rate scheduler")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes")
    parser.add_argument("--no-class-weights", action="store_true", help="Disable inverse class weighting")
    parser.add_argument("--amp", action="store_true", default=torch.cuda.is_available(),
                        help="Enable Automatic Mixed Precision (AMP)")
    parser.add_argument("--no-amp", action="store_false", dest="amp",
                        help="Disable Automatic Mixed Precision")
    parser.add_argument("--no-multi-gpu", action="store_true",
                        help="Disable DataParallel across multiple GPU shards")
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile() for PyTorch 2.x acceleration")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save/load models")
    parser.add_argument("--plots-dir", type=str, default="plots", help="Directory to save figures and history")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Explicit path to defungi directory")
    parser.add_argument("--resume", type=str, nargs="?", const="auto", default=None,
                        help="Path to checkpoint to resume from, or pass without argument to auto-resume latest")
    parser.add_argument("--auto-resume", action="store_true",
                        help="Automatically resume from latest checkpoint if one exists")

    # Fine-tuning parameters
    parser.add_argument(
        "--fine-tune",
        nargs="?",
        const="auto",
        default=None,
        help="Enable fine-tuning mode. Pass without argument or 'auto' to dynamically find and load the best feature-extracted checkpoint, or pass a specific checkpoint path (e.g. checkpoints/resnet18_mlp_best.pt)."
    )
    parser.add_argument(
        "--fine-tune-layers",
        type=str,
        default="layer4",
        choices=["layer4", "layer3_4", "all"],
        help="Backbone layers to unfreeze during fine-tuning: 'layer4' (layer4 + decoder head), 'layer3_4' (layer3 + layer4 + decoder), or 'all' (entire backbone)."
    )
    parser.add_argument(
        "--fine-tune-lr",
        type=float,
        default=1e-4,
        help="Learning rate for classification decoder head during fine-tuning (default: 1e-4)"
    )
    parser.add_argument(
        "--fine-tune-backbone-lr",
        type=float,
        default=1e-5,
        help="Learning rate for unfrozen backbone convolutional layers during fine-tuning (default: 1e-5)"
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Maximum number of batches per epoch (useful for dry runs and fast testing)"
    )

    args = parser.parse_args()

    # Hardware setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    print("=" * 60)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"Primary GPU:                 {torch.cuda.get_device_name(0)}")
        print(f"Available GPUs / Shards:     {num_gpus}")
        torch.backends.cudnn.benchmark = True
        print("cuDNN Benchmark:             Enabled (optimal conv algorithm tuning)")
    print("=" * 60)

    # Output directories and asset paths
    ckpt_dir = Path(args.checkpoint_dir)
    plots_dir = Path(args.plots_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    is_finetune = args.fine_tune is not None
    base_checkpoint_to_load: Optional[Path] = None

    if is_finetune:
        # Check if user specified a path or "auto"
        ft_val = args.fine_tune.strip()
        user_specified_decoder = any(arg.startswith("--decoder-type") for arg in sys.argv)
        preferred_decoder = args.decoder_type if user_specified_decoder else None

        if ft_val.lower() == "auto":
            base_checkpoint_to_load, detected_decoder, _ = find_best_checkpoint(
                ckpt_dir,
                preferred_decoder=preferred_decoder
            )
            args.decoder_type = detected_decoder
        else:
            cand_path = Path(ft_val)
            if not cand_path.is_file():
                # Try relative to ckpt_dir
                cand_path = ckpt_dir / ft_val
            if not cand_path.is_file():
                raise FileNotFoundError(f"Specified fine-tuning checkpoint not found: '{ft_val}'")
            base_checkpoint_to_load = cand_path
            # Detect architecture from checkpoint
            temp_data = torch.load(base_checkpoint_to_load, map_location="cpu")
            args.decoder_type = detect_decoder_type_from_checkpoint(temp_data, base_checkpoint_to_load)
            print(f"[Fine-Tuning] Loaded explicit checkpoint: {base_checkpoint_to_load.name} (Detected architecture: {args.decoder_type.upper()})")

    file_tag = f"resnet18_{args.decoder_type}_finetuned" if is_finetune else f"resnet18_{args.decoder_type}"

    best_model_path = ckpt_dir / f"{file_tag}_best.pt"
    latest_model_path = ckpt_dir / f"{file_tag}_latest.pt"
    curves_path = plots_dir / f"{file_tag}_training_curves.png"
    cm_path = plots_dir / f"{file_tag}_confusion_matrix.png"
    history_json_path = plots_dir / f"{file_tag}_history.json"
    history_csv_path = plots_dir / f"{file_tag}_history.csv"

    # Data preparation
    print("\n[1/4] Preparing DataLoaders...")
    train_loader, test_loader, classes, class_weights = get_dataloaders(
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Train samples: {len(cast(Sized, train_loader.dataset)):,}")
    print(f"Test samples:  {len(cast(Sized, test_loader.dataset)):,}")

    # Model initialization
    mode_desc = f"Fine-Tuning ({args.fine_tune_layers})" if is_finetune else "Feature Extraction (Frozen Backbone)"
    print(f"\n[2/4] Initializing ResNet-18 with '{args.decoder_type}' Decoder [{mode_desc}]...")

    model: nn.Module = build_resnet18(
        num_classes=len(classes),
        freeze_backbone=True,
        dropout=args.dropout,
        decoder_type=args.decoder_type
    ).to(device)

    # If fine-tuning, load the weights from base checkpoint and unfreeze target layers
    if is_finetune and base_checkpoint_to_load is not None:
        print(f"Loading base pretrained weights from: {base_checkpoint_to_load.name}")
        base_data = torch.load(base_checkpoint_to_load, map_location=device)
        raw_m = get_base_module(model)
        state_d = base_data["model_state_dict"] if (isinstance(base_data, dict) and "model_state_dict" in base_data) else base_data
        raw_m.load_state_dict(state_d)

        unfreeze_backbone(raw_m, layers=args.fine_tune_layers)
        print(f"Unfrozen target layers: '{args.fine_tune_layers}'")

    trainable_p, frozen_p, total_p = get_parameter_counts(model)
    print(f"Decoder Type:                        {args.decoder_type}")
    print(f"Trainable Parameters:                {trainable_p:,} ({trainable_p / total_p * 100:.2f}%)")
    print(f"Frozen Parameters:                   {frozen_p:,} ({frozen_p / total_p * 100:.2f}%)")

    # Layer trainability breakdown
    raw_m = get_base_module(model)
    layer_summary = inspect_trainable_layers(raw_m)
    print("\nLayer Trainability Summary:")
    for group_name, counts in layer_summary.items():
        print(f"  {group_name:<20} | Trainable: {counts['trainable']:>10,} | Frozen: {counts['frozen']:>10,}")

    # Loss function & Optimizer
    if args.no_class_weights:
        criterion = nn.CrossEntropyLoss()
        print("\nUsing standard CrossEntropyLoss (unweighted)")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print("\nUsing weighted CrossEntropyLoss (inverse-frequency weights):")
        for cls_name, w in zip(classes, class_weights.tolist()):
            print(f"  {cls_name}: {w:.3f}")

    if is_finetune:
        # Differential learning rates: backbone gets fine-tune-backbone-lr, head gets fine-tune-lr
        backbone_params = []
        head_params = []
        for name, param in raw_m.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("fc"):
                head_params.append(param)
            else:
                backbone_params.append(param)

        param_groups = []
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": args.fine_tune_backbone_lr})
            print(f"Backbone Optimizer LR:   {args.fine_tune_backbone_lr} ({len(backbone_params)} param tensors)")
        if head_params:
            param_groups.append({"params": head_params, "lr": args.fine_tune_lr})
            print(f"Decoder Head Optimizer LR: {args.fine_tune_lr} ({len(head_params)} param tensors)")

        optimizer = torch.optim.Adam(param_groups, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay
        )

    # Multi-GPU / Multi-shard parallelization
    if num_gpus > 1 and not args.no_multi_gpu:
        print(f"\n[Parallelism] Distributing model across all {num_gpus} GPU devices/shards with DataParallel.")
        model = cast(nn.Module, nn.DataParallel(model))

    if args.compile and hasattr(torch, "compile"):
        print("[Optimization] Applying torch.compile() kernel fusion.")
        try:
            model = cast(nn.Module, torch.compile(model))
        except Exception as e:
            print(f"[Warning] torch.compile() skipped: {e}")

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = GradScaler("cuda", enabled=use_amp)
    if use_amp:
        print("[Optimization] Automatic Mixed Precision (AMP) enabled for Tensor Cores.")

    # Determine checkpoint resume path (if user is resuming an ongoing session)
    resume_path: Optional[Path] = None
    if args.resume:
        if args.resume == "auto":
            if latest_model_path.exists():
                resume_path = latest_model_path
            else:
                print(f"[Auto-Resume] No existing latest checkpoint found at '{latest_model_path}'. Starting from scratch.")
        else:
            resume_path = Path(args.resume)
    elif args.auto_resume and latest_model_path.exists():
        resume_path = latest_model_path
        print(f"[Auto-Resume] Found existing latest checkpoint: {resume_path}")

    start_epoch = 1
    best_macro_f1 = -1.0
    history: Dict[str, List[Any]] = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "lr": []
    }

    ckpt: Optional[Dict[str, Any]] = None
    resumed_epoch = 0

    if resume_path:
        if resume_path.exists():
            print(f"\nResuming training from checkpoint: {resume_path}")
            loaded_obj = torch.load(resume_path, map_location=device)
            if isinstance(loaded_obj, dict):
                ckpt = cast(Dict[str, Any], loaded_obj)
                raw_model = get_base_module(model)
                raw_model.load_state_dict(ckpt["model_state_dict"])
                if "optimizer_state_dict" in ckpt:
                    try:
                        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    except Exception as e:
                        print(f"[Warning] Could not restore optimizer state: {e}")

                resumed_epoch = int(ckpt.get("epoch", 0))
                start_epoch = resumed_epoch + 1
                best_macro_f1 = float(ckpt.get("best_macro_f1", -1.0))

                # Load history: prioritize embedded checkpoint history, fallback to JSON file
                if "history" in ckpt and isinstance(ckpt["history"], dict) and ckpt["history"].get("train_loss"):
                    history = {k: list(v) for k, v in ckpt["history"].items()}
                    print(f"Restored training history ({len(history['train_loss'])} epochs) from checkpoint.")
                elif history_json_path.exists():
                    loaded_hist = load_history(history_json_path)
                    if loaded_hist and loaded_hist.get("train_loss"):
                        history = loaded_hist
                        print(f"Restored training history ({len(history['train_loss'])} epochs) from {history_json_path}.")

                # Align history to resumed epoch to prevent future duplicates on rollback
                history = align_history_to_epoch(history, resumed_epoch)
                if "epoch" not in history or len(history["epoch"]) != len(history.get("train_loss", [])):
                    history["epoch"] = list(range(1, len(history.get("train_loss", [])) + 1))
                if "lr" not in history:
                    history["lr"] = [args.lr] * len(history["epoch"])

                print(f"Resumed from epoch {resumed_epoch} (Best Macro F1 so far: {best_macro_f1:.4f})")
            else:
                print(f"Warning: Checkpoint at '{resume_path}' is not a valid dictionary. Starting from scratch.")
        else:
            print(f"Warning: Checkpoint '{resume_path}' not found. Starting from scratch.")

    # Determine end_epoch based on --total-epochs or --epochs
    if args.total_epochs is not None:
        end_epoch = args.total_epochs
    else:
        end_epoch = start_epoch + args.epochs - 1

    # Learning rate scheduler
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(end_epoch, 1), eta_min=1e-6)
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    else:
        scheduler = None

    # Restore scheduler state if resuming
    if ckpt is not None and scheduler is not None:
        if "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"]:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                    scheduler.T_max = max(end_epoch, 1)
                print("Restored learning rate scheduler state from checkpoint.")
            except Exception as e:
                print(f"[Warning] Could not restore scheduler state: {e}")

    # Restore scaler state if resuming
    if ckpt is not None and use_amp and scaler is not None:
        if "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"]:
            try:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
                print("Restored AMP GradScaler state from checkpoint.")
            except Exception as e:
                print(f"[Warning] Could not restore AMP scaler state: {e}")

    num_to_train = max(0, end_epoch - start_epoch + 1)
    if num_to_train == 0:
        print(f"\nTarget end epoch ({end_epoch}) has already been reached. Skipping training loop.")
    else:
        print(f"\n[3/4] Training for {num_to_train} Epochs (Epoch {start_epoch} to {end_epoch})...")

    last_completed_epoch = start_epoch - 1
    interrupted = False

    if num_to_train > 0:
        try:
            for epoch in range(start_epoch, end_epoch + 1):
                train_loss, train_acc = train_one_epoch(
                    model=model,
                    dataloader=train_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device,
                    scaler=scaler,
                    use_amp=use_amp,
                    max_batches=args.max_batches
                )
                val_loss, val_acc, val_macro_f1, y_true_val, y_pred_val = evaluate(
                    model=model,
                    dataloader=test_loader,
                    criterion=criterion,
                    device=device,
                    use_amp=use_amp,
                    max_batches=args.max_batches
                )

                curr_lr = float(optimizer.param_groups[0]["lr"])
                history["epoch"].append(epoch)
                history["train_loss"].append(float(train_loss))
                history["train_acc"].append(float(train_acc))
                history["val_loss"].append(float(val_loss))
                history["val_acc"].append(float(val_acc))
                history["val_macro_f1"].append(float(val_macro_f1))
                history["lr"].append(curr_lr)

                is_best = val_macro_f1 > best_macro_f1
                if is_best:
                    best_macro_f1 = val_macro_f1
                    flag = " [BEST SAVED]"
                else:
                    flag = ""

                raw_model = get_base_module(model)
                checkpoint_state = {
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "scaler_state_dict": scaler.state_dict() if (use_amp and scaler) else None,
                    "best_macro_f1": best_macro_f1,
                    "val_acc": val_acc,
                    "classes": classes,
                    "history": history,
                    "decoder_type": args.decoder_type,
                    "is_finetuned": is_finetune,
                    "fine_tune_layers": args.fine_tune_layers if is_finetune else None
                }

                # Save latest checkpoint after every epoch
                torch.save(checkpoint_state, latest_model_path)

                if is_best:
                    torch.save(checkpoint_state, best_model_path)
                    save_confusion_matrix_plot(y_true_val, y_pred_val, classes, cm_path, verbose=False)

                # Persist history JSON and CSV immediately
                save_history_files(history, history_json_path, history_csv_path)

                # Update training curves plot immediately
                save_training_plots(history, curves_path, verbose=False)

                if scheduler is not None:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_macro_f1)
                    else:
                        scheduler.step()

                print(
                    f"Epoch {epoch:2d}/{end_epoch:2d} | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
                    f"Val F1: {val_macro_f1:.4f}{flag}"
                )
                last_completed_epoch = epoch

        except KeyboardInterrupt:
            interrupted = True
            print(f"\n[Interrupt] Training interrupted by user at epoch {last_completed_epoch + 1}.")
            print(f"Checkpoints, charts, and history are safely preserved up to epoch {last_completed_epoch}.")
            print(f"To resume training seamlessly, run:")
            print(f"  python src/transfer/train.py --resume {latest_model_path} --decoder-type {args.decoder_type}")

    # Final evaluation using best model (or latest if best not present)
    eval_model_path = best_model_path if best_model_path.exists() else latest_model_path
    if eval_model_path.exists() and not interrupted:
        print(f"\n[4/4] Final Evaluation on Test Images using '{eval_model_path.name}'...")
        checkpoint = torch.load(eval_model_path, map_location=device)
        raw_model = get_base_module(model)
        raw_model.load_state_dict(checkpoint["model_state_dict"])
        _, test_acc, test_macro_f1, y_true, y_pred = evaluate(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            max_batches=args.max_batches
        )

        if len(y_true) > 0 and len(y_pred) > 0:
            bal_acc = balanced_accuracy_score(y_true, y_pred)
            macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
            macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
            weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

            mode_label = f"ResNet-18 [{args.decoder_type.upper()}{' FT' if is_finetune else ''}]"
            print("\n" + "=" * 60)
            print(f"{mode_label} FINAL BENCHMARK RESULTS")
            print("=" * 60)
            print(f"Overall Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")
            print(f"Balanced Accuracy:     {bal_acc:.4f} ({bal_acc * 100:.2f}%)")
            print(f"Macro Precision:       {macro_prec:.4f} ({macro_prec * 100:.2f}%)")
            print(f"Macro Recall:          {macro_rec:.4f} ({macro_rec * 100:.2f}%)")
            print(f"Macro F1-Score:        {test_macro_f1:.4f} ({test_macro_f1 * 100:.2f}%)")
            print(f"Weighted F1-Score:     {weighted_f1:.4f} ({weighted_f1 * 100:.2f}%)")

            print("\nClassification Report:")
            print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))

            # Benchmark vs Baseline
            print("=" * 60)
            print("COMPARISON: MAJORITY CLASS BASELINE vs. RESNET-18 TRANSFER LEARNING")
            print("=" * 60)
            print(f"{'Metric':<22} | {'Baseline (Zero-R)':<18} | {f'{mode_label}':<24} | {'Change':<10}")
            print("-" * 81)
            print(f"{'Accuracy':<22} | {'48.33%':<18} | {f'{test_acc*100:.2f}%':<24} | {f'+{(test_acc*100 - 48.33):.2f}%':<10}")
            print(f"{'Balanced Accuracy':<22} | {'20.00%':<18} | {f'{bal_acc*100:.2f}%':<24} | {f'+{(bal_acc*100 - 20.00):.2f}%':<10}")
            print(f"{'Macro Precision':<22} | {'9.67%':<18} | {f'{macro_prec*100:.2f}%':<24} | {f'+{(macro_prec*100 - 9.67):.2f}%':<10}")
            print(f"{'Macro Recall':<22} | {'20.00%':<18} | {f'{macro_rec*100:.2f}%':<24} | {f'+{(macro_rec*100 - 20.00):.2f}%':<10}")
            print(f"{'Macro F1':<22} | {'13.03%':<18} | {f'{test_macro_f1*100:.2f}%':<24} | {f'+{(test_macro_f1*100 - 13.03):.2f}%':<10}")
            print("=" * 60)

            # Re-save visual assets at end
            save_training_plots(history, curves_path, verbose=True)
            save_confusion_matrix_plot(y_true, y_pred, classes, cm_path, verbose=True)

        if not is_finetune and args.decoder_type == "linear":
            torch.save(checkpoint, ckpt_dir / "resnet18_defungi_best.pt")
        print(f"\nArtifacts generated successfully in {plots_dir}/ and {ckpt_dir}/")


if __name__ == "__main__":
    main()
