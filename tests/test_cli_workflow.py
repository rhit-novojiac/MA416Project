"""
End-to-end CLI workflow tests running train.py directly as a subprocess.
"""

import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def cli_temp_dir():
    temp_path = Path(tempfile.mkdtemp(prefix="test_cli_"))
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


def test_cli_train_and_resume_continuity(cli_temp_dir):
    """
    Runs train.py for 1 epoch, verifies artifacts, then resumes for 1 more epoch.
    Verifies that artifacts show 2 continuous epochs instead of resetting.
    """
    ckpt_dir = cli_temp_dir / "checkpoints"
    plots_dir = cli_temp_dir / "plots"

    # Session 1: Train 1 epoch
    cmd_s1 = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "transfer" / "train.py"),
        "--epochs", "1",
        "--batch-size", "128",
        "--num-workers", "0",
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
        "--batch-size", "128",
        "--num-workers", "0",
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

