"""
Unit and integration tests for continuous training history, multi-session checkpointing,
and crash-resilient chart generation.
"""

import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless testing
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Import functions from train module
from src.transfer.train import (
    save_history_files,
    load_history,
    align_history_to_epoch,
    save_training_plots,
    save_confusion_matrix_plot,
)


@pytest.fixture
def temp_dir():
    """Provides a clean temporary directory for each test."""
    temp_path = Path(tempfile.mkdtemp(prefix="test_transfer_"))
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


def test_save_and_load_history_json_and_csv(temp_dir):
    """Test saving history to JSON and CSV and reloading accurately."""
    json_path = temp_dir / "plots" / "resnet18_linear_history.json"
    csv_path = temp_dir / "plots" / "resnet18_linear_history.csv"

    sample_history = {
        "epoch": [1, 2, 3],
        "train_loss": [1.45, 1.20, 0.95],
        "train_acc": [0.45, 0.55, 0.65],
        "val_loss": [1.30, 1.10, 0.88],
        "val_acc": [0.50, 0.58, 0.67],
        "val_macro_f1": [0.35, 0.48, 0.60],
        "lr": [0.001, 0.00095, 0.00085],
    }

    save_history_files(sample_history, json_path, csv_path)

    assert json_path.exists(), "history.json was not created"
    assert csv_path.exists(), "history.csv was not created"

    # Verify JSON load
    loaded = load_history(json_path)
    assert loaded["epoch"] == [1, 2, 3]
    assert loaded["train_loss"] == pytest.approx([1.45, 1.20, 0.95])
    assert loaded["val_macro_f1"] == pytest.approx([0.35, 0.48, 0.60])

    # Verify CSV content
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 3
        assert reader[0]["epoch"] == "1"
        assert float(reader[0]["train_loss"]) == pytest.approx(1.45)
        assert float(reader[2]["val_macro_f1"]) == pytest.approx(0.60)


def test_align_history_to_epoch():
    """Test that rolling back or resuming properly truncates future epochs."""
    history = {
        "epoch": [1, 2, 3, 4, 5],
        "train_loss": [1.5, 1.3, 1.1, 0.9, 0.7],
        "train_acc": [0.4, 0.5, 0.6, 0.7, 0.8],
        "val_loss": [1.4, 1.2, 1.0, 0.85, 0.75],
        "val_acc": [0.45, 0.52, 0.61, 0.68, 0.77],
        "val_macro_f1": [0.3, 0.4, 0.5, 0.6, 0.7],
        "lr": [0.001, 0.001, 0.001, 0.001, 0.001],
    }

    # Truncate to epoch 3
    aligned = align_history_to_epoch(history, target_epoch=3)
    assert aligned["epoch"] == [1, 2, 3]
    assert len(aligned["train_loss"]) == 3
    assert len(aligned["val_macro_f1"]) == 3

    # If target_epoch >= current max, no truncation
    aligned_noop = align_history_to_epoch(history, target_epoch=10)
    assert aligned_noop["epoch"] == [1, 2, 3, 4, 5]


def test_save_training_plots_multirun(temp_dir):
    """Test plotting when history is extended across multiple sessions."""
    curves_path = temp_dir / "plots" / "resnet18_linear_training_curves.png"

    # Session 1: epochs 1-3
    history = {
        "epoch": [1, 2, 3],
        "train_loss": [1.5, 1.2, 1.0],
        "train_acc": [0.45, 0.55, 0.62],
        "val_loss": [1.4, 1.15, 0.95],
        "val_acc": [0.48, 0.57, 0.65],
        "val_macro_f1": [0.38, 0.49, 0.59],
    }
    save_training_plots(history, curves_path)
    assert curves_path.exists()
    size_session1 = curves_path.stat().st_size
    assert size_session1 > 0

    # Session 2: continue to epochs 4-5
    history["epoch"].extend([4, 5])
    history["train_loss"].extend([0.85, 0.70])
    history["train_acc"].extend([0.70, 0.76])
    history["val_loss"].extend([0.88, 0.80])
    history["val_acc"].extend([0.71, 0.78])
    history["val_macro_f1"].extend([0.66, 0.74])

    save_training_plots(history, curves_path)
    assert curves_path.exists()
    size_session2 = curves_path.stat().st_size
    assert size_session2 > 0


def test_save_confusion_matrix_plot(temp_dir):
    """Test confusion matrix visualization generation."""
    cm_path = temp_dir / "plots" / "resnet18_linear_confusion_matrix.png"
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 1, 2])
    classes = ["H1", "H2", "H3"]

    save_confusion_matrix_plot(y_true, y_pred, classes, cm_path)
    assert cm_path.exists()
    assert cm_path.stat().st_size > 0


def test_checkpoint_state_persistence(temp_dir):
    """Test that checkpoint dictionary correctly bundles history, scheduler, and scaler."""
    ckpt_path = temp_dir / "checkpoints" / "test_ckpt.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_model = nn.Linear(10, 5)
    dummy_optimizer = torch.optim.Adam(dummy_model.parameters(), lr=0.001)
    dummy_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(dummy_optimizer, T_max=10)
    dummy_scheduler.step()

    history = {
        "epoch": [1, 2],
        "train_loss": [1.2, 0.9],
        "train_acc": [0.5, 0.65],
        "val_loss": [1.1, 0.85],
        "val_acc": [0.52, 0.68],
        "val_macro_f1": [0.4, 0.58],
        "lr": [0.001, 0.00095],
    }

    state = {
        "epoch": 2,
        "model_state_dict": dummy_model.state_dict(),
        "optimizer_state_dict": dummy_optimizer.state_dict(),
        "scheduler_state_dict": dummy_scheduler.state_dict(),
        "scaler_state_dict": None,
        "best_macro_f1": 0.58,
        "val_acc": 0.68,
        "classes": ["H1", "H2", "H3", "H5", "H6"],
        "history": history,
    }

    torch.save(state, ckpt_path)
    assert ckpt_path.exists()

    loaded = torch.load(ckpt_path, map_location="cpu")
    assert loaded["epoch"] == 2
    assert loaded["history"]["epoch"] == [1, 2]
    assert loaded["history"]["val_macro_f1"] == pytest.approx([0.4, 0.58])
    assert "scheduler_state_dict" in loaded
