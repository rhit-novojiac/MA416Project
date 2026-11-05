# DeFungi ResNet-18 Transfer Learning — Team Runbook & Execution Guide

This document provides complete instructions for setting up the environment, running the ResNet-18 transfer learning pipeline on the **DeFungi** microscopic image dataset, executing Stage 1 feature extraction and Stage 2 dynamic fine-tuning, managing checkpoints across training sessions, and generating benchmark evaluation figures.

All commands and file paths in this guide are **generic and portable** across different machines and operating systems.

---

## 1. Project Overview & Milestones

The goal of this project is to classify fungal microscopic morphotypes across 5 classes (`H1`, `H2`, `H3`, `H5`, `H6`) using transfer learning and deep convolutional feature fine-tuning.

| Project Milestone | Target Threshold | Status | Implementation Details |
|---|---|---|---|
| **Majority Baseline (Zero-R)** | 48.33% Accuracy / 13.03% Macro F1 | Baseline | Calculated from empirical class frequencies in `Baseline.ipynb` |
| **Minimum Requirement** | Beat majority baseline, select pre-trained model, replace decoder head | **Achieved (67.92% F1)** | ResNet-18 + Linear Decoder Head (`--decoder-type linear`) |
| **Reasonable Goal** | >10% over baseline (>58.33% Acc / >23.03% F1), train & validate decoder | **Achieved (78.84% F1)** | ResNet-18 + MLP Decoder Head (`--decoder-type mlp`) |
| **Stretch Goal (Fine-Tuning)** | >20% over minimum requirement, fine-tune backbone representations | **Achieved (87.98% F1)** | ResNet-18 + Deep MLP + Dynamic Fine-Tuning (`--fine-tune`) |

---

## 2. Directory Structure

```text
MA416Project/
├── defungi/                       # Dataset folder containing subfolders H1, H2, H3, H5, H6
├── src/
│   ├── baseline/
│   │   └── Baseline.ipynb        # Course baseline notebook
│   └── transfer/
│       ├── __init__.py           # Package marker
│       ├── dataset.py            # Stratified 80/20 data split, transforms, DataLoader setup
│       ├── model.py              # ResNet-18 backbone + custom heads (linear, mlp, mlp_deep, mlp_deeper)
│       └── train.py              # Training, dynamic fine-tuning, checkpointing, and evaluation CLI
├── checkpoints/                  # Saved .pt model checkpoints (created automatically)
│   ├── resnet18_*_best.pt        # Stage 1 feature extraction best checkpoints
│   └── resnet18_*_finetuned_best.pt # Stage 2 fine-tuned best checkpoints
├── plots/                        # Saved curves, confusion matrices, and history (created automatically)
├── hf_space/                     # Hugging Face Spaces Gradio web application
│   ├── app.py                    # Interactive demo interface
│   └── model_utils.py            # Model loader & inference wrapper
├── tests/                        # Automated pytest suite
│   ├── test_checkpoint_history.py
│   ├── test_fine_tune.py
│   ├── test_cli_workflow.py
│   └── test_multisession_training.py
├── pyproject.toml                # Portable interpreter & type checker configuration
└── RUNBOOK.md                    # This document
```

---

## 3. Environment Setup & Prerequisites

### Required Packages
- **Python**: 3.10 – 3.13 (Python 3.12 recommended)
- **PyTorch & torchvision**: GPU-accelerated (CUDA) or CPU fallback
- **Data & Evaluation**: `scikit-learn`, `pandas`, `pillow`, `matplotlib`, `numpy`, `tqdm`, `gradio`, `pytest`

### Activating the Virtual Environment

Before running any script, activate your virtual environment in your terminal:

#### Windows (PowerShell):
```powershell
# If using the project's local .venv:
.\.venv\Scripts\Activate.ps1
```

#### Windows (Command Prompt):
```cmd
.venv\Scripts\activate.bat
```

#### macOS / Linux:
```bash
source .venv/bin/activate
```

> [!TIP]
> Always use `.\.venv\Scripts\python.exe` (or ensure your shell prompt shows `(.venv)`) to utilize NVIDIA CUDA acceleration rather than unaccelerated CPU execution.

---

## 4. Stage 1: Feature Extraction (Decoder Training)

In Stage 1, the ResNet-18 convolutional backbone is **frozen** (all 11,176,512 parameters frozen), and only the classification decoder head is trained.

### Architecture Options:
- `--decoder-type linear`: 1-layer projection (`Dropout(0.3) -> Linear(512, 5)`) — *2,565 parameters*
- `--decoder-type mlp`: 2-layer MLP (`Linear(512, 256) -> ReLU -> Dropout -> Linear(256, 5)`) — *132,613 parameters*
- `--decoder-type mlp_deep`: 3-layer MLP — *198,405 parameters*
- `--decoder-type mlp_deeper`: 4-layer MLP — *264,197 parameters*

### Example Stage 1 Commands:
```powershell
# Train MLP decoder for 15 epochs
python src/transfer/train.py --decoder-type mlp --epochs 15

# Train Linear decoder for 10 epochs
python src/transfer/train.py --decoder-type linear --epochs 10

# Train with specific batch size and learning rate
python src/transfer/train.py --decoder-type mlp --batch-size 64 --lr 0.001 --epochs 20
```

---

## 5. Stage 2: Dynamic Fine-Tuning Pipeline

In Stage 2, the convolutional backbone layers are **selectively unfrozen** and trained with differential learning rates starting from the best Stage 1 model weights.

### How Dynamic Fine-Tuning Works:
1. **Automatic Discovery & Ranking**: Pass `--fine-tune` (or `--fine-tune auto`). The script inspects all candidate checkpoints in `checkpoints/`, ranks them by validation Macro-F1 score, and selects the champion model.
2. **Auto-Architecture Detection**: Detects whether the checkpoint is `linear`, `mlp`, `mlp_deep`, or `mlp_deeper` directly from weights/metadata.
3. **Selective Layer Unfreezing**: Unfreezes the target convolutional blocks (`layer4` by default).
4. **Differential Optimizer**: Assigns a smaller learning rate to the backbone (e.g. `1e-5`) and a standard rate to the head (e.g. `1e-4`) to prevent catastrophic forgetting.
5. **Separate Output Artifacts**: Saves to `resnet18_{decoder}_finetuned_*`, leaving your Stage 1 baseline checkpoints completely intact.

### Common Fine-Tuning Commands:

#### 1. Automatic Champion Fine-Tuning (Easiest)
Automatically scans checkpoints, picks the top model, and fine-tunes for 10 epochs:
```powershell
python src/transfer/train.py --fine-tune --epochs 10
```

#### 2. Fine-Tune a Specific Decoder Architecture
Picks the best checkpoint matching that decoder:
```powershell
python src/transfer/train.py --fine-tune --decoder-type mlp --epochs 10
```

#### 3. Fine-Tune an Explicit Checkpoint Path
```powershell
python src/transfer/train.py --fine-tune checkpoints/resnet18_mlp_best.pt --epochs 10
```

#### 4. Custom Unfreezing Layers & Learning Rates
```powershell
# Unfreeze layer4 only (Recommended):
python src/transfer/train.py --fine-tune --fine-tune-layers layer4 --fine-tune-backbone-lr 1e-5 --fine-tune-lr 1e-4 --epochs 15

# Unfreeze both layer3 and layer4:
python src/transfer/train.py --fine-tune --fine-tune-layers layer3_4 --epochs 15

# Unfreeze entire network:
python src/transfer/train.py --fine-tune --fine-tune-layers all --fine-tune-backbone-lr 5e-6 --epochs 15
```

---

## 6. Complete CLI Options Reference

| Argument | Type | Default | Description |
|---|---|---|---|
| `--fine-tune` | `str` (optional) | `None` | Enable fine-tuning mode. Pass `'auto'` or omit value to pick best model, or pass a `.pt` path. |
| `--fine-tune-layers` | `layer4` \| `layer3_4` \| `all` | `layer4` | Backbone layers to unfreeze during fine-tuning. |
| `--fine-tune-backbone-lr` | `float` | `1e-5` | Learning rate for unfrozen convolutional backbone layers. |
| `--fine-tune-lr` | `float` | `1e-4` | Learning rate for decoder classification head during fine-tuning. |
| `--decoder-type` | `linear` \| `mlp` \| `mlp_deep` \| `mlp_deeper` | `linear` | Classification head architecture. |
| `--epochs` | `int` | `10` | Number of epochs to train in this session. |
| `--total-epochs` | `int` | `None` | Target cumulative epoch count (overrides `--epochs` when resuming). |
| `--batch-size` | `int` | `32` | Batch size for training and validation DataLoaders. |
| `--lr` | `float` | `1e-3` | Initial learning rate for Stage 1 feature extraction. |
| `--weight-decay` | `float` | `1e-4` | L2 weight regularization for Adam optimizer. |
| `--dropout` | `float` | `0.3` | Dropout probability applied before classification layers. |
| `--scheduler` | `cosine` \| `plateau` \| `none` | `cosine` | Learning rate scheduler. |
| `--num-workers` | `int` | `2` | DataLoader worker processes (`0` for Windows debugging, `4`-`8` on Linux). |
| `--no-class-weights` | `flag` | `False` | Disables inverse-frequency class weighting in CrossEntropyLoss. |
| `--amp` / `--no-amp` | `flag` | `Auto` | Enable/disable Automatic Mixed Precision (AMP). |
| `--no-multi-gpu` | `flag` | `False` | Disables DataParallel across multiple GPU devices. |
| `--compile` | `flag` | `False` | Enables `torch.compile()` kernel fusion (PyTorch 2.x). |
| `--max-batches` | `int` | `None` | Caps batches per epoch (useful for dry runs and fast testing). |
| `--resume` | `str` (optional) | `None` | Resume from specific checkpoint or pass `'auto'` for latest. |
| `--auto-resume` | `flag` | `False` | Automatically resume from latest checkpoint if one exists. |
| `--dataset-dir` | `str` | `None` | Custom path to the `defungi/` directory if stored outside root. |
| `--checkpoint-dir` | `str` | `checkpoints` | Directory to save and load model weights. |
| `--plots-dir` | `str` | `plots` | Directory to save training curves, confusion matrices, and history files. |

---

## 7. Checkpoint Management & Multi-Session Resumption

The pipeline is fully crash-resilient and continuous across sessions:
1. **`checkpoints/resnet18_<decoder>_best.pt`**: Updated whenever validation Macro-F1 improves.
2. **`checkpoints/resnet18_<decoder>_latest.pt`**: Updated at the end of **every single epoch**. Contains full model weights, optimizer buffers, LR scheduler state, and cumulative history.
3. **`checkpoints/resnet18_<decoder>_finetuned_*.pt`**: Separate fine-tuning checkpoints that preserve Stage 1 models.

### Resuming an Interrupted Run:
```powershell
# Automatically resume latest checkpoint for 5 more epochs:
python src/transfer/train.py --resume auto --epochs 5

# Or resume fine-tuning:
python src/transfer/train.py --fine-tune --resume auto --epochs 5
```

The script automatically:
- Restores model weights, Adam optimizer momentum, and LR scheduler decay state.
- Loads previous metric history so training curves **extend continuously** without resetting or overwriting.
- Preserves the previous highest Macro-F1 score.

---

## 8. Output Artifacts & Results Inspection

After every epoch and upon completion, the pipeline generates:

1. **Training Curves (`plots/resnet18_<decoder>_training_curves.png`)**:
   - Subplot 1: Cross-Entropy Loss vs. Epoch across all sessions.
   - Subplot 2: Train & Val Accuracy vs. Epoch (with 48.33% baseline reference line).
   - Subplot 3: Validation Macro F1 vs. Epoch (with 13.03% baseline reference line).
2. **Confusion Matrix Plot (`plots/resnet18_<decoder>_confusion_matrix.png`)**:
   - Normalized heatmap of true labels vs. predicted labels on the 1,823 test images.
3. **Structured History Data (`plots/resnet18_<decoder>_history.json` & `.csv`)**:
   - Tabular log containing `epoch`, `train_loss`, `train_acc`, `val_loss`, `val_acc`, `val_macro_f1`, and `lr`.
   - CSV can be opened directly in Excel or Google Sheets for report plotting.
4. **Terminal Benchmark Summary**:
   - Full classification report (precision, recall, F1 per class).
   - Side-by-side comparison table against the Majority Class Baseline.

---

## 9. Interactive Web Demo (Gradio)

To launch the web interface locally:

```powershell
python hf_space/app.py
```
Open **`http://127.0.0.1:7860`** in your browser.

- Switch between models (MLP Decoder, Linear Decoder, Fine-Tuned).
- Drag-and-drop any microscopic fungal image or test preloaded sample images.
- See real-time prediction confidence bars and scientific species details.

For deployment to Hugging Face Spaces, see [`hf_space/DEPLOYMENT.md`](file:///c:/Users/novojiac/gitOrganized/MA416Project/hf_space/DEPLOYMENT.md).

---

## 10. Running the Automated Test Suite

The test suite validates layer freezing/unfreezing, dynamic model discovery, history recovery, crash resilience, and CLI subprocess execution:

```powershell
# Run all tests (all 11 tests complete in ~80 seconds)
pytest -v

# Run only fine-tuning unit tests:
pytest -v tests/test_fine_tune.py

# Run only checkpoint history persistence tests:
pytest -v tests/test_checkpoint_history.py
```

---

## 11. Troubleshooting FAQ

> [!TIP]
> **Issue: Out of Memory (CUDA OOM)**
> - Reduce `--batch-size` from `32` to `16` or `8`.
> - Add `--num-workers 0` if GPU shared memory is constrained.

> [!IMPORTANT]
> **Issue: Cannot find `defungi` directory**
> - The script automatically searches for a `defungi` folder in the current directory and parent directories.
> - If stored elsewhere, specify: `--dataset-dir "path/to/defungi"`.

> [!NOTE]
> **Issue: Training is running slowly on CPU**
> - Ensure your virtual environment is activated (`.\.venv\Scripts\Activate.ps1`).
> - Verify GPU availability by running: `python -c "import torch; print(torch.cuda.is_available())"`. It should print `True`.
