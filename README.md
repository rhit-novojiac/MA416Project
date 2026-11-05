# MA416 Project - Microscopic Fungi Classification

A Deep Learning / Computer Vision project utilizing the **DeFungi** microscopic image dataset for fungal classification, transfer learning, and fine-tuning using ResNet-18.

---

## Table of Contents
1. [Dataset Overview](#dataset-overview)
2. [Environment Setup](#environment-setup)
3. [Quick Start & Common Commands](#quick-start--common-commands)
4. [Training Pipeline Guide](#training-pipeline-guide)
   - [Stage 1: Feature Extraction (Decoder Heads)](#stage-1-feature-extraction-decoder-heads)
   - [Stage 2: Dynamic Fine-Tuning](#stage-2-dynamic-fine-tuning)
   - [Resuming Interrupted Runs](#resuming-interrupted-runs)
5. [CLI Options Reference](#cli-options-reference)
6. [Interactive Web Demo (Gradio)](#interactive-web-demo-gradio)
7. [Running Tests & Verification](#running-tests--verification)
8. [Repository Directory Structure](#repository-directory-structure)

---

## Dataset Overview

The project uses the **DeFungi** dataset containing microscopic images across 5 fungal dermatophyte species classes:
- **H1**: *Trichophyton rubrum* (most common cause of athlete's foot and nail infections)
- **H2**: *Trichophyton mentagrophytes* (causes inflammatory skin lesions)
- **H3**: *Epidermophyton floccosum* (dermatophyte affecting skin and nails)
- **H5**: *Microsporum canis* (zoophilic ringworm from cats and dogs)
- **H6**: *Microsporum gypseum* (geophilic dermatophyte found in soil)

### Local Dataset Folder
The dataset is stored locally in the `defungi/` directory (excluded from Git):
```
defungi/
├── H1/   (3,524 images)
├── H2/   (1,867 images)
├── H3/   (1,324 images)
├── H5/   (1,232 images)
└── H6/   (1,167 images)
```
*Total: 9,114 images (80% train / 20% test split automatically managed by data loader).*

---

## Environment Setup

### 1. Activate the Virtual Environment
The repository contains a pre-configured `.venv` with CUDA GPU acceleration enabled:

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

*(If you ever need to create a fresh environment from scratch):*
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> [!TIP]
> Always run scripts with `.\.venv\Scripts\python.exe` (or with the virtual environment activated) so that PyTorch detects your NVIDIA CUDA GPU (`cuda`) rather than falling back to CPU (`cpu`).

---

## Quick Start & Common Commands

| Task | PowerShell Command |
|---|---|
| **Fine-Tune Best Model (Auto)** | `python src/transfer/train.py --fine-tune --epochs 10` |
| **Fine-Tune Specific Architecture** | `python src/transfer/train.py --fine-tune --decoder-type mlp --epochs 10` |
| **Fine-Tune Specific Checkpoint** | `python src/transfer/train.py --fine-tune checkpoints/resnet18_mlp_best.pt --epochs 10` |
| **Train Stage 1 MLP Decoder** | `python src/transfer/train.py --decoder-type mlp --epochs 10` |
| **Train Stage 1 Linear Head** | `python src/transfer/train.py --decoder-type linear --epochs 10` |
| **Resume Interrupted Run** | `python src/transfer/train.py --resume auto` |
| **Launch Gradio Web App** | `python hf_space/app.py` |
| **Run All Unit & CLI Tests** | `pytest -v` |

---

## Training Pipeline Guide

The training workflow consists of two main stages:

```
[Pretrained ResNet-18 Backbone]
              │
              ▼  (Stage 1: Feature Extraction)
[Train Custom Decoder Head with Frozen Backbone: linear, mlp, mlp_deep, mlp_deeper]
              │
              ▼  (Checkpoints saved to checkpoints/resnet18_*_best.pt)
              │
              ▼  (Stage 2: Dynamic Fine-Tuning)
[python src/transfer/train.py --fine-tune]
   ├── 1. Automatically inspects checkpoints/ & ranks candidates by Val Macro-F1
   ├── 2. Auto-detects model architecture (e.g. MLP)
   ├── 3. Unfreezes layer4 (or chosen layers)
   ├── 4. Applies differential learning rates (1e-5 backbone / 1e-4 head)
   └── 5. Saves outputs to *_finetuned (original checkpoints preserved)
```

---

### Stage 1: Feature Extraction (Decoder Heads)

In Stage 1, the ResNet-18 convolutional backbone is **completely frozen** (11,176,512 parameters frozen), and only the custom classification head is trained.

Choose from four supported decoder head architectures:
1. `linear`: 1-layer linear projection (Dropout + Linear)
2. `mlp`: 2-layer MLP (Linear -> ReLU -> Dropout -> Linear)
3. `mlp_deep`: 3-layer MLP
4. `mlp_deeper`: 4-layer MLP

**Example commands:**
```powershell
# Train 2-layer MLP decoder head for 15 epochs
python src/transfer/train.py --decoder-type mlp --epochs 15

# Train Linear head for 10 epochs
python src/transfer/train.py --decoder-type linear --epochs 10
```

Artifacts produced:
- Checkpoints: `checkpoints/resnet18_{decoder}_best.pt`, `resnet18_{decoder}_latest.pt`
- Plots: `plots/resnet18_{decoder}_training_curves.png`, `resnet18_{decoder}_confusion_matrix.png`
- History: `plots/resnet18_{decoder}_history.json`, `resnet18_{decoder}_history.csv`

---

### Stage 2: Dynamic Fine-Tuning

In Stage 2, the backbone convolutional layers are **selectively unfrozen** and fine-tuned with a lower learning rate.

#### Mode 1: Automatic Champion Selection (Recommended)
Pass `--fine-tune` (or `--fine-tune auto`). The script will scan `checkpoints/`, display a comparison table, and pick the best model automatically:
```powershell
python src/transfer/train.py --fine-tune --epochs 10
```

Example terminal output:
```text
======================================================================
DISCOVERED CHECKPOINTS IN      checkpoints
======================================================================
Checkpoint Filename              | Decoder      | Macro F1   | Val Acc  
----------------------------------------------------------------------
resnet18_linear_best.pt          | linear       | 67.92%     | 66.21%   
resnet18_mlp_best.pt             | mlp          | 78.84%     | 75.10%   
======================================================================
[Dynamic Selection] Automatically selected champion checkpoint: resnet18_mlp_best.pt
                     Architecture: MLP | Val Macro-F1: 78.84%
```

#### Mode 2: Fine-Tune a Specific Architecture
To fine-tune the best checkpoint of a particular decoder type:
```powershell
python src/transfer/train.py --fine-tune --decoder-type mlp --epochs 10
```

#### Mode 3: Fine-Tune an Explicit Checkpoint Path
To fine-tune any specific `.pt` file:
```powershell
python src/transfer/train.py --fine-tune checkpoints/resnet18_mlp_best.pt --epochs 10
```

#### Layer Unfreezing Options (`--fine-tune-layers`)
- `--fine-tune-layers layer4` *(Default & Recommended)*: Unfreezes `layer4` and the decoder head. Freezes `stem` and `layers 1-3` to avoid overfitting on smaller microscopic datasets.
- `--fine-tune-layers layer3_4`: Unfreezes `layer3`, `layer4`, and head.
- `--fine-tune-layers all`: Unfreezes all parameters across the entire network.

#### Differential Learning Rates
To prevent catastrophic forgetting of pretrained ImageNet visual features:
- `--fine-tune-backbone-lr`: Learning rate for backbone layers (default: `1e-5`).
- `--fine-tune-lr`: Learning rate for classification head (default: `1e-4`).

**Example with custom learning rates and layers:**
```powershell
python src/transfer/train.py --fine-tune --fine-tune-layers layer4 --fine-tune-backbone-lr 5e-6 --fine-tune-lr 5e-5 --epochs 15
```

> [!NOTE]
> Fine-tuning outputs are saved with `_finetuned` in their names (e.g. `checkpoints/resnet18_mlp_finetuned_best.pt`). Your Stage 1 checkpoints are **never overwritten**.

---

### Resuming Interrupted Runs

Training is crash-resilient and multi-session continuous:
- Checkpoints, training curve charts, and history files are saved after **every single epoch**.
- To resume any interrupted training or fine-tuning run:
  ```powershell
  python src/transfer/train.py --resume auto
  ```
- Or specify an exact checkpoint:
  ```powershell
  python src/transfer/train.py --resume checkpoints/resnet18_mlp_latest.pt
  ```

---

## CLI Options Reference

| Argument | Type | Default | Description |
|---|---|---|---|
| `--fine-tune` | string (optional) | `None` | Enable fine-tuning. Pass without args or `'auto'` for best model, or pass a `.pt` path. |
| `--fine-tune-layers` | `layer4` \| `layer3_4` \| `all` | `layer4` | Backbone layers to unfreeze during fine-tuning. |
| `--fine-tune-backbone-lr` | float | `1e-5` | Learning rate for unfrozen convolutional backbone layers. |
| `--fine-tune-lr` | float | `1e-4` | Learning rate for decoder classification head during fine-tuning. |
| `--decoder-type` | `linear` \| `mlp` \| `mlp_deep` \| `mlp_deeper` | `linear` | Classification head architecture. |
| `--epochs` | int | `10` | Number of epochs to train. |
| `--total-epochs` | int | `None` | Target total cumulative epoch count (useful when resuming). |
| `--batch-size` | int | `32` | Batch size. |
| `--lr` | float | `1e-3` | Initial learning rate for Stage 1 feature extraction. |
| `--weight-decay` | float | `1e-4` | L2 weight regularization. |
| `--dropout` | float | `0.3` | Dropout probability before classification layers. |
| `--scheduler` | `cosine` \| `plateau` \| `none` | `cosine` | Learning rate scheduler. |
| `--num-workers` | int | `2` | DataLoader worker processes. |
| `--no-class-weights` | flag | `False` | Disables inverse-frequency class weighting in CrossEntropyLoss. |
| `--amp` / `--no-amp` | flag | Auto | Enable/disable Automatic Mixed Precision (AMP). |
| `--max-batches` | int | `None` | Capping batches per epoch (great for dry runs and fast testing). |
| `--resume` | string (optional) | `None` | Resume from specific checkpoint or pass `'auto'` for latest. |
| `--auto-resume` | flag | `False` | Automatically resume from latest checkpoint if one exists. |

---

## Interactive Web Demo (Gradio)

The repository includes a web application for classifying microscopic fungi images interactively:

```powershell
# Launch the Gradio app
python hf_space/app.py
```
Open your browser to: **`http://127.0.0.1:7860`**

Features:
- Select between model checkpoints (Linear, MLP, Fine-Tuned).
- Drag-and-drop any microscopic fungal image or click sample images.
- Real-time prediction probability distribution and scientific species information.

For Hugging Face Spaces deployment instructions, see [`hf_space/DEPLOYMENT.md`](file:///c:/Users/novojiac/gitOrganized/MA416Project/hf_space/DEPLOYMENT.md).

---

## Running Tests & Verification

The project includes unit and integration tests covering parameter freezing/unfreezing, checkpoint persistence, history recovery, and CLI workflows:

```powershell
# Run the complete test suite
pytest -v
```

All 11 tests should pass:
- `tests/test_checkpoint_history.py` (5 tests): History JSON/CSV persistence, plot regeneration, and rollback alignment.
- `tests/test_fine_tune.py` (3 tests): Parameter counts, selective layer unfreezing, architecture detection, and champion model selection.
- `tests/test_cli_workflow.py` (2 tests): Subprocess CLI execution, multi-session continuous training, and `--fine-tune` pipeline.
- `tests/test_multisession_training.py` (1 test): Multi-session resumption without resetting metrics.

---

## Repository Directory Structure

```
MA416Project/
├── .gitignore                      # Git ignore rules (excludes defungi/, checkpoints, venv)
├── README.md                       # Main documentation (this file)
├── requirements.txt                # Project dependencies
├── checkpoints/                    # Saved model checkpoints (.pt)
│   ├── resnet18_linear_best.pt
│   ├── resnet18_mlp_best.pt
│   └── resnet18_*_finetuned_best.pt
├── plots/                          # Visualizations and metrics
│   ├── *_training_curves.png
│   ├── *_confusion_matrix.png
│   ├── *_history.json
│   └── *_history.csv
├── hf_space/                       # Hugging Face Spaces Gradio web demo
│   ├── app.py                      # Gradio UI application
│   ├── model_utils.py              # Inference logic and model loading
│   ├── DEPLOYMENT.md               # Step-by-step HF Spaces deployment runbook
│   ├── examples/                   # Pre-packaged sample images (H1, H2, H3, H5, H6)
│   └── checkpoints/                # Lightweight checkpoints for deployment
├── src/
│   ├── baseline/
│   │   └── Baseline.ipynb          # Zero-R baseline analysis
│   └── transfer/
│       ├── dataset.py              # PyTorch Dataset, transforms & DataLoader
│       ├── model.py                # ResNet-18 builder, freezing & unfreezing
│       └── train.py                # Training, fine-tuning & evaluation CLI
└── tests/                          # Automated pytest suite
    ├── test_checkpoint_history.py
    ├── test_cli_workflow.py
    ├── test_fine_tune.py
    └── test_multisession_training.py
```
