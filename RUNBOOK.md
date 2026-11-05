# DeFungi ResNet-18 Transfer Learning — Team Runbook & Execution Guide

This document provides complete instructions for setting up the environment, running the ResNet-18 transfer learning pipeline on the **DeFungi** microscopic image dataset, managing checkpoints across training sessions, and generating benchmark evaluation figures.

All commands and file paths in this guide are **generic and portable** across different machines and operating systems.

---

## 1. Project Overview & Milestones

The goal of this project is to classify fungal microscopic morphotypes across 5 classes (`H1`, `H2`, `H3`, `H5`, `H6`) using transfer learning.

| Project Milestone | Due Date | Target Threshold | ResNet-18 Status |
|---|---|---|---|
| **Majority Baseline (Zero-R)** | — | 48.33% Accuracy / 13.03% Macro F1 | Baseline from `Baseline.ipynb` |
| **Minimum Requirement** | Oct 30 | Beat majority baseline, select pre-trained model, replace decoder head | **Achieved (63.96% Acc / 64.93% F1)** |
| **Reasonable Goal** | Nov 6 | >10% over baseline (>58.33% Acc / >23.03% F1), train & validate decoder | **Achieved (+15.63% Acc / +51.90% F1)** |
| **Stretch Goal** | Nov 13 | >20% over minimum requirement (>68.33% Acc), fine-tune full backbone | Supported via `unfreeze_backbone()` |

---

## 2. Directory Structure

The transfer learning module is self-contained in `src/transfer/`:

```text
MA416Project/
├── defungi/                       # Dataset folder containing subfolders H1, H2, H3, H5, H6
├── src/
│   ├── baseline/
│   │   └── Baseline.ipynb        # Course baseline notebook
│   └── transfer/
│       ├── __init__.py           # Package marker
│       ├── dataset.py            # Stratified 80/20 data split, transforms, DataLoader setup
│       ├── model.py              # ResNet-18 backbone + customizable decoder heads (linear, mlp, mlp_deep)
│       └── train.py              # Full training, validation, checkpointing, and evaluation script
├── checkpoints/                  # Saved .pt model checkpoints (created automatically)
├── plots/                        # Saved curves and confusion matrices (created automatically)
├── pyproject.toml                # Portable interpreter & type checker configuration
└── pyrefly.toml                  # Standalone IDE type checker configuration
```

---

## 3. Environment Setup & Prerequisites

### Required Packages
- **Python**: 3.10 – 3.12 (Python 3.12 recommended)
- **PyTorch & torchvision**: GPU-accelerated (CUDA) or CPU fallback
- **Data & Evaluation**: `scikit-learn`, `pandas`, `pillow`, `matplotlib`, `numpy`, `tqdm`

### Activating the Virtual Environment

Before running any script, activate your virtual environment in your terminal:

#### Windows (PowerShell):
```powershell
# If using a local .venv in the project root:
.\.venv\Scripts\Activate.ps1

# Or if using a shared course environment elsewhere:
& "<path-to-your-venv>\Scripts\Activate.ps1"
```

#### Windows (Command Prompt):
```cmd
.venv\Scripts\activate.bat
```

#### macOS / Linux:
```bash
source .venv/bin/activate
```

*(Once activated, your terminal prompt will show `(.venv)`, and standard `python` commands will invoke the correct environment on all operating systems.)*

---

## 4. How to Run the Training Pipeline

### Quick Start (Standard Baseline Training)
Train ResNet-18 with a linear decoder head for 10 epochs using a batch size of 32:

```bash
python src/transfer/train.py --epochs 10 --batch-size 32
```

> [!NOTE]
> The script automatically detects an NVIDIA GPU if available; otherwise, it seamlessly falls back to CPU.

---

### Command-Line Arguments & Modifiable Paths

All directories and parameters can be overridden via CLI flags to match any teammate's file structure:

| Argument | Type | Default | Description |
|---|---|---|---|
| `--epochs` | `int` | `10` | Number of additional epochs to train in this session. |
| `--total-epochs` | `int` | `None` | Target total epochs count (overrides `--epochs` to reach this specific epoch). |
| `--batch-size` | `int` | `32` | Batch size for training and validation DataLoaders. |
| `--lr` | `float` | `0.001` | Learning rate for the trainable decoder head. |
| `--dropout` | `float` | `0.3` | Dropout probability applied before classification. |
| `--decoder-type` | `str` | `linear` | Head architecture: `linear`, `mlp`, `mlp_deep`, or `mlp_deeper`. |
| `--scheduler` | `str` | `cosine` | LR schedule: `cosine`, `plateau`, or `none`. |
| `--weight-decay` | `float` | `1e-4` | L2 weight decay for the Adam optimizer. |
| `--num-workers` | `int` | `2` | Number of DataLoader worker processes (`0` for Windows debugging, `8`-`16` on cluster nodes). |
| `--no-class-weights`| `flag` | `False` | Disables inverse-frequency weighting in CrossEntropyLoss. |
| `--amp` / `--no-amp`| `flag` | `Auto` | Automatic Mixed Precision (enabled automatically when CUDA is present). |
| `--no-multi-gpu` | `flag` | `False` | Disables `DataParallel` across multiple GPU shards. |
| `--compile` | `flag` | `False` | Enables `torch.compile()` PyTorch 2.x Inductor kernel fusion. |
| `--resume` | `str` | `None` | Path to `.pt` checkpoint to resume from, or pass without argument (`--resume`) to auto-resume latest. |
| `--auto-resume` | `flag` | `False` | Automatically resume from latest checkpoint if one exists. |
| `--dataset-dir` | `str` | `None` | Custom path to the `defungi/` directory if stored outside root. |
| `--checkpoint-dir`| `str` | `checkpoints` | Custom target folder for saved model weights. |
| `--plots-dir` | `str` | `plots` | Custom target folder for charts, confusion matrices, and history files. |

#### High-Performance Multi-GPU / Multi-Shard Execution:
On a powerful cluster node or multi-GPU server (e.g., 8 GPU shards):
```bash
python src/transfer/train.py --batch-size 128 --num-workers 8 --lr 0.002 --epochs 15
```

#### Example: Specifying Custom Paths
If your dataset is located in another folder or drive:
```bash
python src/transfer/train.py --dataset-dir "../shared_data/defungi" --checkpoint-dir "./my_checkpoints" --plots-dir "./my_plots"
```

---

## 5. Experimenting with Different Decoder Heads

Three distinct decoder architectures are built into `model.py` to compare representation capacity:

```mermaid
flowchart LR
    In["ResNet-18 Backbone\n(Frozen: 11,176,512 params)\nOutput: 512-dim"] --> Split{Decoder Type}
    
    Split -->|"linear (Minimal)"| D1["Dropout(0.3) -> Linear(512, 5)\n[2,565 params]"]
    Split -->|"mlp (Reasonable)"| D2["Linear(512, 256) -> BatchNorm -> ReLU -> Dropout -> Linear(256, 5)\n[133,125 params]"]
    Split -->|"mlp_deep (Stretch)"| D3["Linear(512, 256) -> BN -> GELU -> Linear(256, 128) -> BN -> GELU -> Linear(128, 5)\n[165,893 params]"]
```

### Commands to Compare Heads:

1. **Linear Projection Head (Minimal Requirement)**:
   ```bash
   python src/transfer/train.py --decoder-type linear --epochs 10
   ```
2. **2-Layer MLP Head with BatchNorm (Reasonable Goal)**:
   ```bash
   python src/transfer/train.py --decoder-type mlp --epochs 10 --lr 0.0005
   ```
3. **3-Layer Deep MLP Head with GELU (Stretch Goal Candidate)**:
   ```bash
   python src/transfer/train.py --decoder-type mlp_deep --epochs 10 --lr 0.0003
   ```

---

## 6. Checkpoint Management & Continuous Multi-Session Training

The training pipeline is fully crash-resilient and continuous across sessions:
1. **`checkpoints/resnet18_<decoder>_best.pt`**: Updated whenever validation Macro-F1 improves.
2. **`checkpoints/resnet18_<decoder>_latest.pt`**: Updated at the end of **every single epoch**. Contains full model weights, optimizer buffers, LR scheduler state, and cumulative metric history.
3. **Per-Epoch Crash Resilience**: At the end of every epoch, `latest.pt`, `plots/resnet18_<decoder>_training_curves.png`, `history.json`, and `history.csv` are flushed to disk. If a run ends prematurely (`Ctrl+C` or system shutdown), all progress up to the last completed epoch is preserved.

### How to Stop & Resume Training Later

#### Step 1: Run First Session (e.g., 5 Epochs)
```bash
python src/transfer/train.py --epochs 5 --decoder-type linear
```

#### Step 2: Resume Seamlessly (Continuing as One Single Run)
You can resume either with the checkpoint path or simply pass `--resume` to auto-resume:
```bash
# Auto-resume latest checkpoint and train 5 more epochs:
python src/transfer/train.py --epochs 5 --decoder-type linear --resume

# Or specify a target total epoch number (e.g., train until Epoch 20 total):
python src/transfer/train.py --total-epochs 20 --decoder-type linear --resume
```

The script automatically:
- Restores model weights, Adam optimizer momentum, and LR scheduler decay state.
- Loads previous metric history so training curves **extend continuously** rather than resetting or deleting previous data.
- Appends new epochs to `history.json` and `history.csv` for downstream analysis.
- Preserves the previous highest Macro-F1 for saving new best checkpoints.

---

## 7. Output Artifacts & Results Inspection

During and upon completion of training, the pipeline continuously maintains:

1. **Terminal Benchmark Summary**:
   - Classification report (precision, recall, F1 per class).
   - Side-by-side comparison table against the Majority Class Baseline.
2. **Training Curves (`plots/resnet18_<decoder>_training_curves.png`)**:
   - Subplot 1: Cross-Entropy Loss vs. Epoch across all sessions.
   - Subplot 2: Train & Val Accuracy vs. Epoch with the 48.33% baseline line.
   - Subplot 3: Validation Macro F1 vs. Epoch with the 13.03% baseline line.
   - Continuously updated after every epoch; never deleted or truncated upon resumption.
3. **Confusion Matrix Plot (`plots/resnet18_<decoder>_confusion_matrix.png`)**:
   - Heatmap of true labels vs. predicted labels for test images, updated whenever a new best model is found and at final evaluation.
4. **Structured History Data (`plots/resnet18_<decoder>_history.json` & `.csv`)**:
   - Complete tabular log containing `epoch`, `train_loss`, `train_acc`, `val_loss`, `val_acc`, `val_macro_f1`, and `lr`.
   - CSV can be opened directly in Excel, Google Sheets, or pandas for report generation.

---

## 8. Running the Automated Test Suite

The test suite validates continuous history logging, crash resilience, multi-session resumption, and CLI workflows.

### Fast Tests (Unit & Multi-Session Simulation — ~15 seconds)
Recommended for day-to-day verification:
```powershell
pytest -v tests/test_checkpoint_history.py tests/test_multisession_training.py
```

### Full Test Suite (Including Subprocess Real-Data CLI Training)
```powershell
pytest -v tests/
```
*(Note: `test_cli_workflow.py` trains 2 full epochs on the DeFungi dataset through a subprocess to test actual disk persistence; on CPU this takes ~10 minutes).*

### Running Pyrefly Type Checks
```powershell
& "c:\Users\novojiac\.antigravity-ide\extensions\meta.pyrefly-1.4.0-win32-x64\bin\pyrefly.exe" check
```

---

## 9. IDE & Language Server Configuration (Optional)

If teammates see import warnings (`torchvision`, `torch`) in their editor:

1. Ensure the Python interpreter in their IDE is set to their active virtual environment.
2. The project contains a generic [`pyproject.toml`](pyproject.toml) where the interpreter path can be customized:
   ```toml
   [tool.pyrefly]
   # Adjust to point to your local virtualenv if not in .venv
   python-interpreter-path = ".venv/Scripts/python.exe"
   search-path = ["src", "src/transfer"]
   ```
3. In VS Code or Antigravity IDE, press `Ctrl + Shift + P` and select **`Developer: Reload Window`** to refresh diagnostics.

---

## 9. Troubleshooting FAQ

> [!TIP]
> **Issue: Out of Memory (CUDA OOM)**
> - Reduce `--batch-size` from `32` to `16` or `8`.
> - Add `--num-workers 0` if GPU shared memory is constrained.

> [!IMPORTANT]
> **Issue: Cannot find `defungi` directory**
> - The script automatically searches for a `defungi` folder in the current directory and parent directories.
> - If stored elsewhere, provide the explicit relative or absolute path via `--dataset-dir <path>`.
