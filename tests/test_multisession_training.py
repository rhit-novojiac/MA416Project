"""
Integration tests for multi-session continuous training and resume functionality.
"""

import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.transfer.train import (
    train_one_epoch,
    evaluate,
    save_history_files,
    load_history,
    align_history_to_epoch,
    save_training_plots,
    save_confusion_matrix_plot,
    get_base_module,
)


@pytest.fixture
def temp_workspace():
    temp_path = Path(tempfile.mkdtemp(prefix="test_multisession_"))
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


def create_synthetic_dataloaders(num_samples=40, num_classes=5, input_dim=512, batch_size=8):
    """Creates synthetic embedding DataLoader for fast training simulation."""
    torch.manual_seed(42)
    X_train = torch.randn(num_samples, input_dim)
    y_train = torch.randint(0, num_classes, (num_samples,))
    X_val = torch.randn(num_samples // 2, input_dim)
    y_val = torch.randint(0, num_classes, (num_samples // 2,))

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    classes = [f"H{i}" for i in range(1, num_classes + 1)]
    return train_loader, val_loader, classes


def test_multisession_continuous_training(temp_workspace):
    """
    Simulates:
    - Session 1: Train 2 epochs (1 to 2) and save checkpoint, plots, and history.
    - Session 2: Resume from Session 1 checkpoint and train 2 more epochs (3 to 4).
    Verifies that metrics and charts are continuous and NOT wiped out.
    """
    ckpt_dir = temp_workspace / "checkpoints"
    plots_dir = temp_workspace / "plots"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    latest_ckpt_path = ckpt_dir / "model_latest.pt"
    best_ckpt_path = ckpt_dir / "model_best.pt"
    curves_path = plots_dir / "training_curves.png"
    cm_path = plots_dir / "confusion_matrix.png"
    history_json_path = plots_dir / "history.json"
    history_csv_path = plots_dir / "history.csv"

    device = torch.device("cpu")
    train_loader, val_loader, classes = create_synthetic_dataloaders()
    criterion = nn.CrossEntropyLoss()

    # -------------------------------------------------------------------------
    # SESSION 1: Train 2 Epochs (Epoch 1 and 2)
    # -------------------------------------------------------------------------
    model = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, len(classes))).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "lr": []
    }
    best_macro_f1 = -1.0

    for epoch in range(1, 3):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_macro_f1, y_true, y_pred = evaluate(model, val_loader, criterion, device)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_macro_f1)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        is_best = val_macro_f1 > best_macro_f1
        if is_best:
            best_macro_f1 = val_macro_f1

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": None,
            "best_macro_f1": best_macro_f1,
            "val_acc": val_acc,
            "classes": classes,
            "history": history
        }
        torch.save(state, latest_ckpt_path)
        if is_best:
            torch.save(state, best_ckpt_path)
            save_confusion_matrix_plot(y_true, y_pred, classes, cm_path)

        save_history_files(history, history_json_path, history_csv_path)
        save_training_plots(history, curves_path)
        scheduler.step()

    # Verify Session 1 artifacts
    assert latest_ckpt_path.exists()
    assert history_json_path.exists()
    assert history_csv_path.exists()
    assert curves_path.exists()

    s1_ckpt = torch.load(latest_ckpt_path, map_location="cpu")
    assert s1_ckpt["epoch"] == 2
    assert s1_ckpt["history"]["epoch"] == [1, 2]

    with open(history_csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert [r["epoch"] for r in rows] == ["1", "2"]

    # -------------------------------------------------------------------------
    # SESSION 2: Resume from Session 1 checkpoint for 2 more Epochs (Epoch 3 and 4)
    # -------------------------------------------------------------------------
    # Re-instantiate fresh model, optimizer, scheduler to test true resume
    model_resumed = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, len(classes))).to(device)
    optimizer_resumed = torch.optim.Adam(model_resumed.parameters(), lr=1e-3)
    scheduler_resumed = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_resumed, T_max=4)

    # Load checkpoint
    ckpt_resumed = torch.load(latest_ckpt_path, map_location=device)
    model_resumed.load_state_dict(ckpt_resumed["model_state_dict"])
    optimizer_resumed.load_state_dict(ckpt_resumed["optimizer_state_dict"])
    scheduler_resumed.load_state_dict(ckpt_resumed["scheduler_state_dict"])

    resumed_epoch = ckpt_resumed["epoch"]
    assert resumed_epoch == 2
    start_epoch = resumed_epoch + 1  # 3
    best_macro_f1_resumed = ckpt_resumed["best_macro_f1"]

    # Restore history from checkpoint
    history_resumed = ckpt_resumed["history"]
    history_resumed = align_history_to_epoch(history_resumed, resumed_epoch)
    assert history_resumed["epoch"] == [1, 2]

    # Train epochs 3 and 4
    for epoch in range(start_epoch, start_epoch + 2):
        train_loss, train_acc = train_one_epoch(model_resumed, train_loader, criterion, optimizer_resumed, device)
        val_loss, val_acc, val_macro_f1, y_true, y_pred = evaluate(model_resumed, val_loader, criterion, device)

        history_resumed["epoch"].append(epoch)
        history_resumed["train_loss"].append(train_loss)
        history_resumed["train_acc"].append(train_acc)
        history_resumed["val_loss"].append(val_loss)
        history_resumed["val_acc"].append(val_acc)
        history_resumed["val_macro_f1"].append(val_macro_f1)
        history_resumed["lr"].append(optimizer_resumed.param_groups[0]["lr"])

        is_best = val_macro_f1 > best_macro_f1_resumed
        if is_best:
            best_macro_f1_resumed = val_macro_f1

        state = {
            "epoch": epoch,
            "model_state_dict": model_resumed.state_dict(),
            "optimizer_state_dict": optimizer_resumed.state_dict(),
            "scheduler_state_dict": scheduler_resumed.state_dict(),
            "scaler_state_dict": None,
            "best_macro_f1": best_macro_f1_resumed,
            "val_acc": val_acc,
            "classes": classes,
            "history": history_resumed
        }
        torch.save(state, latest_ckpt_path)
        if is_best:
            torch.save(state, best_ckpt_path)
            save_confusion_matrix_plot(y_true, y_pred, classes, cm_path)

        save_history_files(history_resumed, history_json_path, history_csv_path)
        save_training_plots(history_resumed, curves_path)
        scheduler_resumed.step()

    # -------------------------------------------------------------------------
    # VERIFICATION: Verify continuous 4-epoch history
    # -------------------------------------------------------------------------
    final_ckpt = torch.load(latest_ckpt_path, map_location="cpu")
    assert final_ckpt["epoch"] == 4
    assert final_ckpt["history"]["epoch"] == [1, 2, 3, 4], "Epoch sequence must be continuous 1..4"
    assert len(final_ckpt["history"]["train_loss"]) == 4

    # Check JSON
    final_json = load_history(history_json_path)
    assert final_json["epoch"] == [1, 2, 3, 4]
    assert len(final_json["train_loss"]) == 4

    # Check CSV
    with open(history_csv_path, "r", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))
        assert len(csv_rows) == 4
        assert [r["epoch"] for r in csv_rows] == ["1", "2", "3", "4"]

    # Verify curves file was maintained and valid
    assert curves_path.exists()
    assert curves_path.stat().st_size > 0

