"""
Training, evaluation, and benchmark verification script for ResNet-18 on DeFungi.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sized, Tuple, cast

# Ensure current directory is in sys.path for direct script and standalone execution
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
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
    from .model import build_resnet18, get_parameter_counts
except (ImportError, ValueError):
    from dataset import get_dataloaders
    from model import build_resnet18, get_parameter_counts


def get_base_module(model: nn.Module) -> nn.Module:
    """Unwraps DataParallel or compiled wrappers to access the root nn.Module."""
    if hasattr(model, "module"):
        return cast(nn.Module, getattr(model, "module"))
    if hasattr(model, "_orig_mod"):
        return cast(nn.Module, getattr(model, "_orig_mod"))
    return model


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    use_amp: bool = False
) -> Tuple[float, float]:
    """Runs a single training epoch with optional mixed precision."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in pbar:
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

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """Evaluates the model on test/validation set and computes metrics."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
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
    eval_loss = running_loss / total
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return float(eval_loss), float(acc), float(macro_f1), y_true, y_pred


def save_training_plots(history: Dict[str, List[float]], save_path: Path) -> None:
    """Saves loss and accuracy curves across epochs."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss subplot
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#1f77b4")
    ax1.plot(epochs, history["val_loss"], "s-", label="Val Loss", color="#ff7f0e")
    ax1.set_title("Cross-Entropy Loss vs. Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Accuracy subplot
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "o-", label="Train Acc", color="#1f77b4")
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "s-", label="Val Acc", color="#ff7f0e")
    ax2.axhline(y=48.33, color="red", linestyle="--", label="Majority Baseline (48.33%)")
    ax2.set_title("Accuracy (%) vs. Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved training curves to {save_path}")


def save_confusion_matrix_plot(y_true: np.ndarray, y_pred: np.ndarray, classes: List[str], save_path: Path) -> None:
    """Saves normalized and raw confusion matrix visualization."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(cmap="Blues", ax=cast(Any, ax), colorbar=True)
    plt.title("ResNet-18 Transfer Learning - Confusion Matrix")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrix plot to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train ResNet-18 on DeFungi Dataset.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Input batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for decoder head")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 regularization")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate before linear head")
    parser.add_argument("--decoder-type", type=str, default="linear", choices=["linear", "mlp", "mlp_deep"],
                        help="Decoder head architecture: 'linear' (1-layer), 'mlp' (2-layer), 'mlp_deep' (3-layer)")
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
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save models")
    parser.add_argument("--plots-dir", type=str, default="plots", help="Directory to save figures")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Explicit path to defungi directory")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
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

    # Output directories
    ckpt_dir = Path(args.checkpoint_dir)
    plots_dir = Path(args.plots_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = ckpt_dir / f"resnet18_{args.decoder_type}_best.pt"

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
    print(f"\n[2/4] Initializing Pre-trained ResNet-18 with '{args.decoder_type}' Decoder...")
    model: nn.Module = build_resnet18(
        num_classes=len(classes),
        freeze_backbone=True,
        dropout=args.dropout,
        decoder_type=args.decoder_type
    ).to(device)

    trainable_p, frozen_p, total_p = get_parameter_counts(model)
    print(f"Decoder Type:                        {args.decoder_type}")
    print(f"Trainable Parameters (Decoder Head): {trainable_p:,}")
    print(f"Frozen Parameters (Backbone):        {frozen_p:,}")

    # Loss function & Optimizer
    if args.no_class_weights:
        criterion = nn.CrossEntropyLoss()
        print("Using standard CrossEntropyLoss (unweighted)")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print("Using weighted CrossEntropyLoss (inverse-frequency weights):")
        for cls_name, w in zip(classes, class_weights.tolist()):
            print(f"  {cls_name}: {w:.3f}")

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

    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    else:
        scheduler = None

    # Resume from checkpoint if requested
    start_epoch = 1
    best_macro_f1 = -1.0
    latest_model_path = ckpt_dir / f"resnet18_{args.decoder_type}_latest.pt"

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"\nResuming training from checkpoint: {resume_path}")
            ckpt = torch.load(resume_path, map_location=device)
            raw_model = get_base_module(model)
            raw_model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_macro_f1 = ckpt.get("best_macro_f1", -1.0)
            print(f"Resumed from epoch {ckpt['epoch']} (Best Macro F1 so far: {best_macro_f1:.4f})")
        else:
            print(f"Warning: Checkpoint '{args.resume}' not found. Starting from scratch.")

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": []
    }

    print(f"\n[3/4] Training for {args.epochs} Epochs (Starting from Epoch {start_epoch})...")
    end_epoch = start_epoch + args.epochs - 1

    for epoch in range(start_epoch, end_epoch + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp
        )
        val_loss, val_acc, val_macro_f1, _, _ = evaluate(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_macro_f1)

        raw_model = get_base_module(model)
        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_macro_f1": best_macro_f1,
            "val_acc": val_acc,
            "classes": classes
        }

        # Save latest checkpoint after every epoch
        torch.save(checkpoint_state, latest_model_path)

        is_best = val_macro_f1 > best_macro_f1
        if is_best:
            best_macro_f1 = val_macro_f1
            checkpoint_state["best_macro_f1"] = best_macro_f1
            torch.save(checkpoint_state, best_model_path)
            flag = " [BEST SAVED]"
        else:
            flag = ""

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

    # Final evaluation using best model
    print("\n[4/4] Final Evaluation on 1,823 Test Images...")
    checkpoint = torch.load(best_model_path, map_location=device)
    raw_model = get_base_module(model)
    raw_model.load_state_dict(checkpoint["model_state_dict"])
    _, test_acc, test_macro_f1, y_true, y_pred = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        use_amp=use_amp
    )

    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("\n" + "=" * 60)
    print(f"RESNET-18 [{args.decoder_type.upper()}] FINAL BENCHMARK RESULTS")
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
    print(f"{'Metric':<22} | {'Baseline (Zero-R)':<18} | {f'ResNet-18 ({args.decoder_type})':<18} | {'Change':<10}")
    print("-" * 75)
    print(f"{'Accuracy':<22} | {'48.33%':<18} | {f'{test_acc*100:.2f}%':<18} | {f'+{(test_acc*100 - 48.33):.2f}%':<10}")
    print(f"{'Balanced Accuracy':<22} | {'20.00%':<18} | {f'{bal_acc*100:.2f}%':<18} | {f'+{(bal_acc*100 - 20.00):.2f}%':<10}")
    print(f"{'Macro Precision':<22} | {'9.67%':<18} | {f'{macro_prec*100:.2f}%':<18} | {f'+{(macro_prec*100 - 9.67):.2f}%':<10}")
    print(f"{'Macro Recall':<22} | {'20.00%':<18} | {f'{macro_rec*100:.2f}%':<18} | {f'+{(macro_rec*100 - 20.00):.2f}%':<10}")
    print(f"{'Macro F1':<22} | {'13.03%':<18} | {f'{test_macro_f1*100:.2f}%':<18} | {f'+{(test_macro_f1*100 - 13.03):.2f}%':<10}")
    print("=" * 60)

    # Save visual assets
    curves_path = plots_dir / f"resnet18_{args.decoder_type}_training_curves.png"
    cm_path = plots_dir / f"resnet18_{args.decoder_type}_confusion_matrix.png"
    save_training_plots(history, curves_path)
    save_confusion_matrix_plot(y_true, y_pred, classes, cm_path)

    if args.decoder_type == "linear":
        torch.save(checkpoint, ckpt_dir / "resnet18_defungi_best.pt")
    print(f"\nArtifacts generated successfully in {plots_dir}/ and {ckpt_dir}/")


if __name__ == "__main__":
    main()
