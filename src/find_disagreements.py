"""
Script to scan the DeFungi test set and find specific images where:
- Some models identify the species correctly, while others fail.
- Specifically highlighting cases where Fine-Tuning resolves mistakes made by Baselines.
"""

from pathlib import Path
import sys
import numpy as np
import torch

# Ensure repo root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.transfer.dataset import get_dataloaders
from src.transfer.model import build_resnet18
from src.transfer.train import detect_decoder_type_from_checkpoint, get_base_module


def main():
    root = Path(__file__).resolve().parent.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Scanning test set on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})...")

    _, test_loader, classes, _ = get_dataloaders(
        dataset_dir=str(root / "defungi"),
        batch_size=64,
        num_workers=2
    )
    test_dataset = test_loader.dataset

    model_configs = [
        ("1-Layer Decoder (Baseline)", root / "saved_models/resnet18_linear_best.pt"),
        ("2-Layer Decoder (Stage 1)", root / "saved_models/resnet18_mlp_best.pt"),
        ("4-Layer Decoder", root / "saved_models/resnet18_mlp_deeper_best.pt"),
        ("2-Layer Decoder (Fine-Tuned)", root / "saved_models/resnet18_mlp_finetuned_best.pt"),
        ("3-Layer Decoder (Fine-Tuned Champion)", root / "saved_models/resnet18_mlp_deep_finetuned_best.pt")
    ]

    models = []
    for name, p in model_configs:
        data = torch.load(p, map_location=device)
        sd = data["model_state_dict"] if "model_state_dict" in data else data
        dec = detect_decoder_type_from_checkpoint(data, p)
        m = build_resnet18(num_classes=len(classes), freeze_backbone=False, decoder_type=dec)
        get_base_module(m).load_state_dict(sd)
        m.to(device)
        m.eval()
        models.append((name, m))

    print(f"Loaded {len(models)} models. Running predictions on {len(test_dataset)} test images...")

    all_preds = {name: [] for name, _ in models}
    all_probs = {name: [] for name, _ in models}

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            for name, m in models:
                logits = m(images)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs[name].extend(probs)
                all_preds[name].extend(probs.argmax(axis=1))

    y_true = np.array(test_dataset.labels)
    filepaths = test_dataset.filepaths

    # Find examples where models disagree
    disagreements = []
    for i in range(len(y_true)):
        gt = y_true[i]
        gt_class = classes[gt]
        preds = {name: all_preds[name][i] for name, _ in models}
        correct_count = sum(1 for name, p in preds.items() if p == gt)

        if 1 <= correct_count <= 4:
            disagreements.append({
                "index": i,
                "filepath": filepaths[i],
                "filename": Path(filepaths[i]).name,
                "gt": gt_class,
                "correct_count": correct_count,
                "details": {
                    name: {
                        "pred": classes[preds[name]],
                        "correct": preds[name] == gt,
                        "prob": float(all_probs[name][i][preds[name]])
                    }
                    for name, _ in models
                }
            })

    print(f"\nTotal test images with model disagreements: {len(disagreements)} / {len(y_true)}")

    # Category 1: Clear progression (Baseline and Stage 1 FAIL, but Fine-Tuned SUCCEEDS)
    progression_examples = [
        d for d in disagreements
        if d["details"]["3-Layer Decoder (Fine-Tuned Champion)"]["correct"]
        and not d["details"]["1-Layer Decoder (Baseline)"]["correct"]
        and not d["details"]["2-Layer Decoder (Stage 1)"]["correct"]
    ]
    print(f"Found {len(progression_examples)} examples where Baselines FAIL but Fine-Tuned SUCCEEDS!")

    print("\n" + "=" * 80)
    print("TOP RECOMMENDED DISAGREEMENT EXAMPLES FOR PRESENTATION / DEMO")
    print("=" * 80)

    for idx, ex in enumerate(progression_examples[:5], 1):
        print(f"\nExample #{idx}: {ex['filename']}")
        print(f"  True Species Label: {ex['gt']}")
        print(f"  Full File Path:     {ex['filepath']}")
        for mname, info in ex["details"].items():
            mark = "CORRECT" if info["correct"] else "WRONG"
            print(f"    {mname:<38}: Pred={info['pred']} ({info['prob']*100:5.1f}%) -> [{mark}]")

    # Category 2: A challenging edge case where only the 3-Layer Champion succeeds
    champ_only = [
        d for d in disagreements
        if d["details"]["3-Layer Decoder (Fine-Tuned Champion)"]["correct"]
        and d["correct_count"] == 1
    ]
    if champ_only:
        print("\n" + "=" * 80)
        print("RARE EDGE CASE: ONLY THE 3-LAYER CHAMPION GETS IT RIGHT!")
        print("=" * 80)
        ex = champ_only[0]
        print(f"Image: {ex['filename']} (True: {ex['gt']})")
        print(f"Path:  {ex['filepath']}")
        for mname, info in ex["details"].items():
            mark = "CORRECT" if info["correct"] else "WRONG"
            print(f"    {mname:<38}: Pred={info['pred']} ({info['prob']*100:5.1f}%) -> [{mark}]")


if __name__ == "__main__":
    main()

