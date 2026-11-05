"""
End-to-end CLI workflow tests running train.py directly as a subprocess.
Tests standard feature extraction, checkpoint resumption, and dynamic fine-tuning.
"""

import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import pytest
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.transfer.model import build_resnet18


@pytest.fixture
def cli_temp_dir():
    temp_path = Path(tempfile.mkdtemp(prefix="test_cli_"))
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


def test_cli_train_and_resume_continuity(cli_temp_dir):
    """
    Runs train.py for 1 epoch with capped batches for ultra-fast verification,
    verifies artifacts, then resumes for 1 more epoch.
    Verifies that artifacts show 2 continuous epochs instead of resetting.
    """
    ckpt_dir = cli_temp_dir / "checkpoints"
    plots_dir = cli_temp_dir / "plots"

    # Session 1: Train 1 epoch (capped to 2 batches for fast CI testing)
    cmd_s1 = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "transfer" / "train.py"),
        "--epochs", "1",
        "--batch-size", "32",
        "--num-workers", "0",
        "--max-batches", "2",
        "--checkpoint-dir", str(ckpt_dir),
        "--plots-dir", str(plots_dir),
    ]

    res_s1 = subprocess.run(cmd_s1, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    assert res_s1.returncode == 0, f"Session 1 failed:\nSTDOUT:\n{res_s1.stdout}\nSTDERR:\n{res_s1.stderr}"
    assert "Epoch  1/ 1" in res_s1.stdout

    latest_ckpt = ckpt_dir / "resnet18_linear_latest.pt"
    hist_json = plots_dir / "resnet18_linear_history.json"
    hist_csv = plots_dir / "resnet18_linear_history.csv"
    curves_png = plots_dir / "resnet18_linear_training_curves.png"

    assert latest_ckpt.exists(), "Latest checkpoint was not created"
    assert hist_json.exists(), "History JSON was not created"
    assert hist_csv.exists(), "History CSV was not created"
    assert curves_png.exists(), "Training curves PNG was not created"

    with open(hist_json, "r", encoding="utf-8") as f:
        data_s1 = json.load(f)
        assert data_s1["epoch"] == [1]

    # Session 2: Resume for 1 additional epoch using --resume auto
    cmd_s2 = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "transfer" / "train.py"),
        "--epochs", "1",
        "--batch-size", "32",
        "--num-workers", "0",
        "--max-batches", "2",
        "--checkpoint-dir", str(ckpt_dir),
        "--plots-dir", str(plots_dir),
        "--resume", "auto",
    ]

    res_s2 = subprocess.run(cmd_s2, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    assert res_s2.returncode == 0, f"Session 2 failed:\nSTDOUT:\n{res_s2.stdout}\nSTDERR:\n{res_s2.stderr}"
    assert "Resumed from epoch 1" in res_s2.stdout
    assert "Epoch  2/ 2" in res_s2.stdout

    # Verify that history was appended, NOT deleted or overwritten with only epoch 2
    with open(hist_json, "r", encoding="utf-8") as f:
        data_s2 = json.load(f)
        assert data_s2["epoch"] == [1, 2], f"Expected epochs [1, 2], got {data_s2.get('epoch')}"
        assert len(data_s2["train_loss"]) == 2

    with open(hist_csv, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 2
        assert [r["epoch"] for r in reader] == ["1", "2"]

    assert curves_png.exists()
    assert curves_png.stat().st_size > 0


def test_cli_dynamic_fine_tune_workflow(cli_temp_dir):
    """
    Tests --fine-tune CLI workflow:
    1. Prepares a candidate best model checkpoint in ckpt_dir.
    2. Runs train.py with --fine-tune auto.
    3. Verifies automatic checkpoint discovery, backbone unfreezing, differential LR,
       and that outputs are cleanly saved with _finetuned suffixes without overwriting the base checkpoint.
    """
    ckpt_dir = cli_temp_dir / "checkpoints"
    plots_dir = cli_temp_dir / "plots"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy base checkpoint representing the best Stage 1 MLP model
    base_mlp = build_resnet18(num_classes=5, decoder_type="mlp")
    base_ckpt_path = ckpt_dir / "resnet18_mlp_best.pt"
    torch.save({
        "epoch": 20,
        "decoder_type": "mlp",
        "best_macro_f1": 0.7540,
        "val_acc": 0.7159,
        "classes": ["H1", "H2", "H3", "H5", "H6"],
        "model_state_dict": base_mlp.state_dict()
    }, base_ckpt_path)
    base_mtime = base_ckpt_path.stat().st_mtime

    # Run fine-tuning using --fine-tune auto
    cmd_ft = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "transfer" / "train.py"),
        "--fine-tune", "auto",
        "--fine-tune-layers", "layer4",
        "--epochs", "1",
        "--batch-size", "32",
        "--num-workers", "0",
        "--max-batches", "2",
        "--checkpoint-dir", str(ckpt_dir),
        "--plots-dir", str(plots_dir),
    ]

    res_ft = subprocess.run(cmd_ft, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    assert res_ft.returncode == 0, f"Fine-tuning CLI run failed:\nSTDOUT:\n{res_ft.stdout}\nSTDERR:\n{res_ft.stderr}"

    # Verify discovery logs
    assert "DISCOVERED CHECKPOINTS IN" in res_ft.stdout
    assert "Dynamic Selection" in res_ft.stdout
    assert "resnet18_mlp_best.pt" in res_ft.stdout
    assert "Fine-Tuning (layer4)" in res_ft.stdout
    assert "Backbone Optimizer LR" in res_ft.stdout
    assert "Decoder Head Optimizer LR" in res_ft.stdout

    # Verify fine-tuned artifacts exist
    ft_best_ckpt = ckpt_dir / "resnet18_mlp_finetuned_best.pt"
    ft_latest_ckpt = ckpt_dir / "resnet18_mlp_finetuned_latest.pt"
    ft_hist_json = plots_dir / "resnet18_mlp_finetuned_history.json"
    ft_hist_csv = plots_dir / "resnet18_mlp_finetuned_history.csv"
    ft_curves = plots_dir / "resnet18_mlp_finetuned_training_curves.png"

    assert ft_best_ckpt.exists(), "Fine-tuned best checkpoint was not created"
    assert ft_latest_ckpt.exists(), "Fine-tuned latest checkpoint was not created"
    assert ft_hist_json.exists(), "Fine-tuned history JSON was not created"
    assert ft_hist_csv.exists(), "Fine-tuned history CSV was not created"
    assert ft_curves.exists(), "Fine-tuned training curves PNG was not created"

    # Verify that the base checkpoint was PRESERVED (unaltered)
    assert base_ckpt_path.exists()
    assert base_ckpt_path.stat().st_mtime == base_mtime

    # Verify checkpoint internal metadata
    loaded_ft = torch.load(ft_best_ckpt, map_location="cpu")
    assert loaded_ft.get("is_finetuned") is True
    assert loaded_ft.get("fine_tune_layers") == "layer4"
    assert loaded_ft.get("decoder_type") == "mlp"
