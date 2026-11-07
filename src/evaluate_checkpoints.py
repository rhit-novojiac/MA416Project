"""
Evaluation script to benchmark and compare ResNet-18 checkpoints on the DeFungi test set.
Computes Test Accuracy, Balanced Accuracy, Macro F1, Precision, Recall, and generates Confusion Matrices.
"""

import json
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
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

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.transfer.dataset import get_dataloaders
from src.transfer.model import build_resnet18
from src.transfer.train import detect_decoder_type_from_checkpoint, get_base_module


def evaluate_checkpoint(
    ckpt_path: Path,
    test_loader,
    classes: list,
    device: torch.device,
    plots_dir: Path
):
    print(f"\n=======================================================")
    print(f"Evaluating: {ckpt_path.name}")
    print(f"=======================================================")
    
    ckpt_data = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt_data["model_state_dict"] if (isinstance(ckpt_data, dict) and "model_state_dict" in ckpt_data) else ckpt_data
    
    decoder_type = detect_decoder_type_from_checkpoint(ckpt_data, ckpt_path)
    print(f"Detected Decoder: {decoder_type.upper()}")
    
    model = build_resnet18(num_classes=len(classes), freeze_backbone=False, decoder_type=decoder_type)
    raw_m = get_base_module(model)
    raw_m.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(targets.numpy())
            
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    print(f"Overall Test Accuracy: {acc * 100:.2f}%")
    print(f"Balanced Accuracy:     {bal_acc * 100:.2f}%")
    print(f"Macro Precision:       {macro_prec * 100:.2f}%")
    print(f"Macro Recall:          {macro_rec * 100:.2f}%")
    print(f"Macro F1-Score:        {macro_f1 * 100:.2f}%")
    print(f"Weighted F1-Score:     {weighted_f1 * 100:.2f}%")
    
    print("\nClassification Report:")
    report_dict = classification_report(y_true, y_pred, target_names=classes, zero_division=0, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))
    
    # Save Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap="Blues", ax=ax, colorbar=True, values_format="d")
    ax.set_title(f"Confusion Matrix: {ckpt_path.stem}\nAcc: {acc*100:.2f}% | Macro-F1: {macro_f1*100:.2f}%", fontweight="bold")
    plt.tight_layout()
    cm_path = plots_dir / f"{ckpt_path.stem}_eval_confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close(fig)
    print(f"Saved confusion matrix: {cm_path.name}")
    
    return {
        "checkpoint": ckpt_path.name,
        "decoder_type": decoder_type,
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "report": report_dict
    }


def main():
    root = Path(__file__).resolve().parent.parent
    ckpt_dir = root / "checkpoints"
    plots_dir = root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    _, test_loader, classes, _ = get_dataloaders(
        dataset_dir=str(root / "defungi"),
        batch_size=64,
        num_workers=2
    )
    
    models_to_eval = [
        ckpt_dir / "resnet18_linear_best.pt",
        ckpt_dir / "resnet18_mlp_best.pt",
        ckpt_dir / "resnet18_mlp_deeper_best.pt",
        ckpt_dir / "resnet18_mlp_deep_finetuned_best.pt"
    ]
    
    results = []
    for model_path in models_to_eval:
        if model_path.exists():
            res = evaluate_checkpoint(model_path, test_loader, classes, device, plots_dir)
            results.append(res)
        else:
            print(f"Skipping missing model: {model_path.name}")
            
    summary_path = plots_dir / "checkpoint_benchmark_comparison.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Checkpoint Name':<35} | {'Decoder':<12} | {'Test Acc':<10} | {'Macro F1':<10} | {'Bal Acc':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['checkpoint']:<35} | {r['decoder_type']:<12} | {r['accuracy']*100:>7.2f}%  | {r['macro_f1']*100:>7.2f}%  | {r['balanced_accuracy']*100:>7.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()

