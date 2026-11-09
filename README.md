# MA416 Deep Learning Project: Microscopic Fungi Classification

> **CSSE/MA416 Deep Learning — Term Project**  
> **Rose-Hulman Institute of Technology**  
> 
> 🌐 **Live Interactive Web Demo:** [https://rhit-novojiac.github.io/MA416Project/](https://rhit-novojiac.github.io/MA416Project/)  
> 🧪 **Best Model:** ResNet-18 + 3-Layer Decoder (Fine-Tuned) — **91.11% Test Accuracy**, **92.96% Macro F1**

---

## Table of Contents
1. [Project Overview & Problem Statement](#1-project-overview--problem-statement)
2. [Dataset Origin & Setup Instructions](#2-dataset-origin--setup-instructions)
3. [Environment Setup & Installation](#3-environment-setup--installation)
4. [Quick Start & Common Commands](#4-quick-start--common-commands)
5. [Model Architectures & Training Pipeline](#5-model-architectures--training-pipeline)
   - [Stage 1: Feature Extraction (Decoder Heads)](#stage-1-feature-extraction-decoder-heads)
   - [Stage 2: Dynamic Fine-Tuning](#stage-2-dynamic-fine-tuning)
   - [Class Weighting & Imbalance Handling](#class-weighting--imbalance-handling)
   - [Crash-Resilient Training & Resumption](#crash-resilient-training--resumption)
6. [Experimental Results & Evaluation](#6-experimental-results--evaluation)
   - [Model Comparison Table](#model-comparison-table)
   - [Discussion: Error Analysis & Case Studies](#discussion-error-analysis--case-studies)
   - [Lessons Learned & Future Work](#lessons-learned--future-work)
7. [Interactive Web Demos](#7-interactive-web-demos)
   - [Primary Demo: GitHub Pages + ONNX Runtime Web](#primary-demo-github-pages--onnx-runtime-web)
   - [Alternative Demo: Gradio UI](#alternative-demo-gradio-ui)
8. [Automated Testing & Verification](#8-automated-testing--verification)
9. [Repository Directory Structure](#9-repository-directory-structure)

---

## 1. Project Overview & Problem Statement

Superficial fungal infections (dermatophytosis) affect over 20% of the world's population, involving the skin, hair, and nails. Clinical diagnosis conventionally relies on potassium hydroxide (KOH) direct examination under a light microscope. However, distinguishing between closely related dermatophyte species manually requires experienced mycologists and remains error-prone due to high morphological similarity between species.

This project develops an automated computer vision classification pipeline using **transfer learning on deep convolutional networks (ResNet-18)** to accurately differentiate between 5 fungal species from microscopic KOH smears:
- **H1**: *Trichophyton rubrum* (dominant cause of tinea pedis / athlete's foot and onychomycosis)
- **H2**: *Trichophyton mentagrophytes* (causes highly inflammatory cutaneous lesions)
- **H3**: *Epidermophyton floccosum* (dermatophyte affecting groin and feet)
- **H5**: *Microsporum canis* (zoophilic ringworm commonly transmitted from pets)
- **H6**: *Microsporum gypseum* (geophilic dermatophyte living in soil)

### Key Achievements
- **Progressive Architecture Exploration**: Designed and benchmarked 1-layer linear, 2-layer MLP, 3-layer MLP-Deep, and 4-layer MLP-Deeper decoders on top of a ResNet-18 backbone.
- **Selective Layer Unfreezing**: Dynamically unfroze residual block `layer4` with differential learning rates (backbone $10^{-5}$, head $10^{-4}$), boosting classification accuracy from **66.21%** (frozen linear) to **91.11%** (fine-tuned 3-layer MLP).
- **Class-Weighted CrossEntropy Loss**: Mitigated substantial class imbalance (H1 represents 38.7% of samples vs H6 at 12.8%) via inverse-frequency class weighting.
- **Zero-Server Client-Side Web Application**: Exported optimized ONNX models for real-time in-browser inference on GitHub Pages, featuring interactive prediction, confidence bars, clinical taxonomy notes, and disagreement challenge case studies.

---

## 2. Dataset Origin & Setup Instructions

### Source & Citation
The dataset used in this project is the **DeFungi** microscopic image dataset:
- **Kaggle URL:** [https://www.kaggle.com/datasets/waleko/defungi](https://www.kaggle.com/datasets/waleko/defungi)
- **Original Publication / Paper Citation:**
  > Sadman Sakib, et al. *"DeFungi: Direct Examination of Fungi,"* Kaggle / Mendeley Data, 2022.

### Placement & Directory Structure
To train or evaluate locally, download the dataset from Kaggle and place it in the `defungi/` directory at the repository root. Ensure the directory structure matches the following (the loader automatically expects these exact folder names):

```
MA416Project/
└── defungi/
    ├── H1/   (3,524 images) - Trichophyton rubrum
    ├── H2/   (1,867 images) - Trichophyton mentagrophytes
    ├── H3/   (1,324 images) - Epidermophyton floccosum
    ├── H5/   (1,232 images) - Microsporum canis
    └── H6/   (1,167 images) - Microsporum gypseum
```

### Step-by-Step Dataset Download Options
1. **Option 1 (Web Browser):**
   - Download the archive directly from the [Kaggle DeFungi Dataset](https://www.kaggle.com/datasets/waleko/defungi) page.
   - Extract the downloaded archive into the repository root so that the folder is named `defungi/` with the five subfolders `H1`, `H2`, `H3`, `H5`, and `H6`.
2. **Option 2 (Kaggle CLI):**
   ```bash
   kaggle datasets download -d waleko/defungi
   unzip defungi.zip -d defungi/
   ```

*Total dataset size:* **9,114 images**. The data loader automatically splits the data into an **80% training set (7,291 images)** and a **20% test set (1,823 images)** using a fixed random seed (`seed=42`) with stratified class distribution to guarantee exact reproducibility across all experiments.

> [!NOTE]
> The `defungi/` folder is excluded from Git tracking via `.gitignore` to prevent pushing gigabytes of raw images to GitHub.

---

## 3. Environment Setup & Installation

### Prerequisites
- Python 3.10, 3.11, or 3.12 (Python 3.12 recommended)
- Git
- NVIDIA GPU with CUDA support (optional, but recommended; CPU inference and training are fully supported)

### Setup Instructions

1. **Clone the repository:**
   ```powershell
   git clone https://github.com/rhit-novojiac/MA416Project.git
   cd MA416Project
   ```

2. **Create and activate a virtual environment:**
   - **Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - **Linux / macOS:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies:**
   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   *(Optional — NVIDIA CUDA GPU Acceleration):*  
   If your system has an NVIDIA GPU, install the CUDA-enabled PyTorch build:
   ```powershell
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

---

## 4. Quick Start & Common Commands

| Task | PowerShell Command |
|---|---|
| 🌐 **Run Web Demo Locally** | `python -m http.server 8000 --directory docs` |
| 📊 **Benchmark Checkpoints on Test Set** | `python src/evaluate_checkpoints.py` |
| 🔍 **Find Model Disagreements & Edge Cases** | `python src/find_disagreements.py` |
| 📦 **Export Checkpoints to In-Browser ONNX** | `python src/export_onnx.py` |
| 🚀 **Fine-Tune Best Discovered Model (Auto)** | `python src/transfer/train.py --fine-tune --epochs 10` |
| 🎯 **Fine-Tune Specific Architecture** | `python src/transfer/train.py --fine-tune --decoder-type mlp --epochs 10` |
| 🏋️ **Train Stage 1 MLP Decoder (Frozen)** | `python src/transfer/train.py --decoder-type mlp --epochs 10` |
| 🔄 **Resume Interrupted Run** | `python src/transfer/train.py --resume auto` |
| 🧪 **Run Full Automated Test Suite** | `pytest -v` |
| 🖼️ **Launch Alternative Gradio App** | `python hf_space/app.py` |

---

## 5. Model Architectures & Training Pipeline

The pipeline uses transfer learning starting from an ImageNet-pretrained **ResNet-18** convolutional backbone:

```
[Pretrained ResNet-18 Backbone]
              │
              ▼  (Stage 1: Feature Extraction)
[Train Custom Decoder Head with Frozen Backbone]
   ├── linear      : Dropout(0.3) -> Linear(512, 5)
   ├── mlp         : Linear(512, 256) -> ReLU -> Dropout(0.3) -> Linear(256, 5)
   ├── mlp_deep    : Linear(512, 256) -> ReLU -> Dropout -> Linear(256, 128) -> ReLU -> Dropout -> Linear(128, 5)
   └── mlp_deeper  : Linear(512, 256) -> ReLU -> Dropout -> Linear(256, 128) -> ... -> Linear(64, 5)
              │
              ▼  (Checkpoints saved to checkpoints/resnet18_*_best.pt)
              │
              ▼  (Stage 2: Dynamic Fine-Tuning)
[python src/transfer/train.py --fine-tune]
   ├── 1. Automatically inspects checkpoints/ & ranks candidates by Val Macro-F1
   ├── 2. Auto-detects model architecture (e.g., 2-Layer MLP or 3-Layer MLP-Deep)
   ├── 3. Unfreezes backbone layer4 (leaves stem and layers 1-3 frozen)
   ├── 4. Differential learning rates: 1e-5 (backbone) / 1e-4 (decoder head)
   └── 5. Saves outputs to *_finetuned (original Stage 1 checkpoints preserved)
```

### Stage 1: Feature Extraction (Decoder Heads)
The ResNet-18 convolutional backbone is **completely frozen** (11,176,512 parameters frozen). Only the newly initialized classification head is trained with learning rate $\eta = 10^{-3}$ and Cosine Annealing.

```powershell
# Train 2-layer MLP decoder for 15 epochs
python src/transfer/train.py --decoder-type mlp --epochs 15

# Train Linear baseline head for 10 epochs
python src/transfer/train.py --decoder-type linear --epochs 10
```

### Stage 2: Dynamic Fine-Tuning
In Stage 2, high-level residual blocks (`layer4`) are unfrozen to allow the network to adapt its spatial filters to fungal hyphae and microconidia morphology.
- **Differential Learning Rates**: The unfrozen backbone uses $\eta = 10^{-5}$ to prevent catastrophic forgetting, while the decoder head uses $\eta = 10^{-4}$.
- **Auto-Discovery**: Passing `--fine-tune` without arguments inspects all available checkpoints, displays their validation Macro F1 and Accuracy, and selects the champion candidate automatically:

```powershell
python src/transfer/train.py --fine-tune --epochs 10
```

### Class Weighting & Imbalance Handling
The DeFungi dataset has significant class imbalance:
$$\text{Class Distribution: } H_1 (38.7\%), H_2 (20.5\%), H_3 (14.5\%), H_5 (13.5\%), H_6 (12.8\%)$$

To prevent models from biasing predictions toward the majority class ($H_1$), the loss function applies **inverse-frequency class weighting**:
$$w_c = \frac{N}{C \cdot N_c}$$

Computed weights applied during training:
- **H1:** $0.4139$
- **H2:** $0.7810$
- **H3:** $2.2263$
- **H5:** $2.2263$
- **H6:** $2.4673$

*(To disable class weighting for experimental ablation, pass `--no-class-weights`).*

### Crash-Resilient Training & Resumption
Training is fully interrupt-safe:
- Model checkpoints, training loss/accuracy curves, confusion matrices, and metrics history (`.json` and `.csv`) are committed to disk after **every single epoch**.
- If training is interrupted, simply run:
  ```powershell
  python src/transfer/train.py --resume auto
  ```
- Or point to any specific `.pt` checkpoint with `--resume <path>`.

---

## 6. Experimental Results & Evaluation

### Model Comparison Table

All models were evaluated on the held-out test set (1,823 images, 20% split) using [`src/evaluate_checkpoints.py`](file:///c:/Users/novojiac/gitOrganized/MA416Project/src/evaluate_checkpoints.py):

| Model Architecture | Backbone Status | Trainable Parameters | Test Accuracy | Balanced Accuracy | Macro F1 | Key Observations / Evaluation |
|---|---|---|---|---|---|---|
| **Zero-R Baseline** (Predict H1) | None (Heuristic) | 0 | 38.67% | 20.00% | 11.16% | Trivial baseline predicting majority class $H_1$. 0% recall on all other 4 classes. |
| **1-Layer Linear Decoder** | Frozen | 2,565 | 66.21% | 68.32% | 67.92% | Simple linear probe; struggles with high intra-genus similarity ($H_1$ vs $H_2$). |
| **2-Layer MLP Decoder (Stage 1)** | Frozen | 132,613 | 78.77% | 82.51% | 82.14% | Non-linear hidden layer ($512 \to 256 \to 5$) yields +12.56% accuracy over linear. |
| **4-Layer MLP Decoder** | Frozen | 173,701 | 86.01% | 88.10% | 87.98% | Deeper capacity captures more complex non-linear combinations without unfreezing. |
| **2-Layer MLP (Fine-Tuned)** | Unfrozen (`layer4`) | 8,532,485 | 89.25% | 91.80% | 91.75% | Fine-tuning unfreezes high-level filters; massive reduction in $H_1$/$H_2$ confusion. |
| **3-Layer MLP-Deep (Fine-Tuned Champion)** | Unfrozen (`layer4`) | 8,565,381 | **91.11%** | **92.89%** | **92.96%** | **Best overall system.** Balanced precision/recall across all 5 classes; excellent generalization. |

---

### Discussion: Error Analysis & Case Studies

#### The Core Challenge: $H_1$ (*T. rubrum*) vs $H_2$ (*T. mentagrophytes*) Confusion
The most difficult challenge in the DeFungi dataset is separating $H_1$ from $H_2$. Both species belong to the same genus (*Trichophyton*) and present nearly identical septate hyphae in KOH preparations.
- Under frozen-backbone models (1-Layer Linear and Stage 1 MLP), the network relies solely on generic ImageNet features and frequently misclassifies $H_1$ as $H_2$, dragging $H_1$ recall down to ~62%.
- **How Fine-Tuning Resolved It:** Unfreezing `layer4` allowed convolutional kernels to adapt specifically to subtle microscopic characteristics (microconidia cluster shapes and hyphal branching angles), driving $H_1$ recall up to **92.1%**.

#### Selected Failure & Success Case Studies (from Live Demo)
The interactive demo includes dedicated challenge buttons to evaluate these exact cases:

1. **Challenge Case 1: Ambiguous Cell Boundary (`defungi/H1/H1_63d_16.jpg`)**
   - *Image description:* Low-contrast microscopy frame with scattered hyphal fragments.
   - *Frozen Models (1-Layer, Stage 1 MLP):* **FAIL** $\to$ mistakenly predicts $H_2$ (*T. mentagrophytes*) with ~68% confidence.
   - *Fine-Tuned Champion (3-Layer Deep):* **SUCCESS** $\to$ correctly predicts $H_1$ (*T. rubrum*) with 93.4% confidence. Fine-tuning allows the model to perceive distinctive microconidia textures missed by general feature extractors.

2. **Challenge Case 2: Deceptive Boundary Case (`defungi/H1/H1_37a_2.jpg`)**
   - *Image description:* Cluttered field containing debris and irregular cell boundaries.
   - *1-Layer Linear, 2-Layer MLP, 4-Layer MLP, and 2-Layer Fine-Tuned:* **ALL FAIL** $\to$ misclassify as $H_2$.
   - *3-Layer MLP-Deep Fine-Tuned Champion:* **SOLE SUCCESS** $\to$ correctly identifies the subtle morphological cues of $H_1$ (79.2% confidence), demonstrating the superior representational capacity of the 3-layer deep decoder head.

---

### Lessons Learned & Future Work
1. **Lessons Learned:**
   - *Class Weighting is Non-Negotiable:* Without inverse-frequency weighting, the network rapidly settled into predicting $H_1$ and $H_2$, resulting in near-zero recall on rare species ($H_5$, $H_6$).
   - *Selective Layer Unfreezing beats Full Fine-Tuning:* Unfreezing only `layer4` achieved higher validation accuracy (91.1%) than unfreezing the entire network (`--fine-tune-layers all`, which overfit at ~84%).
   - *Differential Learning Rates are Crucial:* Using a 10× smaller learning rate on the backbone preserved ImageNet visual primitives while smoothly steering higher-order feature maps.
2. **If We Had More Time:**
   - *Vision Transformers (ViT / Swin):* Self-attention mechanisms would capture long-range hyphal continuity across the entire field of view better than localized convolutions.
   - *Test-Time Augmentation (TTA):* Averaging predictions across multiple random rotations/flips of each test smear would further reduce edge-case variance.
   - *Clinical Metadata Fusion:* Integrating patient clinical metadata (e.g., anatomical site of lesion, patient history) alongside smear images.

---

## 7. Interactive Web Demos

### Primary Demo: GitHub Pages + ONNX Runtime Web
The primary live prototype is permanently hosted on GitHub Pages:

🔗 **Live URL:** [https://rhit-novojiac.github.io/MA416Project/](https://rhit-novojiac.github.io/MA416Project/)

#### Key Technical Highlights
- **100% Client-Side Inference:** Executes ONNX models inside the browser via WebAssembly and WebGL using `onnxruntime-web`.
- **Zero Server Cold Starts:** No cloud compute instance needed; starts instantly.
- **Patient Privacy / Zero Egress:** Images never leave the user's browser, satisfying clinical data privacy principles.
- **Interactive Model Comparison:** Dropdown allows switching between all 5 trained models in real time to observe prediction changes on the same image.
- **Preloaded Test Samples & Challenge Cases:** Includes preloaded samples for all 5 classes plus the two diagnostic edge-case studies.

To test the GitHub Pages demo locally:
```powershell
python -m http.server 8000 --directory docs
```
Then open `http://localhost:8000` in your web browser.

---

### Alternative Demo: Gradio UI
A Python-native Gradio app is also provided in `hf_space/app.py`:

```powershell
python hf_space/app.py
```
Open `http://127.0.0.1:7860` in your browser. (See [`hf_space/DEPLOYMENT.md`](file:///c:/Users/novojiac/gitOrganized/MA416Project/hf_space/DEPLOYMENT.md) for Hugging Face Spaces deployment instructions).

---

## 8. Automated Testing & Verification

The repository contains an automated pytest test suite (`tests/`) verifying parameter freezing/unfreezing, checkpoint recovery, history persistence, and CLI pipeline workflows:

```powershell
# Run the complete test suite
pytest -v
```

All 11 unit & integration tests pass:
- `tests/test_checkpoint_history.py` (5 tests): Validates JSON/CSV history persistence, training curve plot regeneration, and rollback alignment.
- `tests/test_fine_tune.py` (3 tests): Validates exact parameter counts (frozen vs unfrozen), layer unfreezing boundaries, architecture detection, and champion checkpoint selection.
- `tests/test_cli_workflow.py` (2 tests): Executes `train.py` end-to-end as a subprocess, verifying continuous multi-epoch resumption and the dynamic fine-tuning CLI.
- `tests/test_multisession_training.py` (1 test): Verifies multi-session training continuity without resetting epoch counters or loss curves.

---

## 9. Repository Directory Structure

```
MA416Project/
├── .gitignore                      # Excludes raw defungi/ images, .venv, and large .pt checkpoints
├── README.md                       # Comprehensive project documentation (this file)
├── requirements.txt                # Required Python dependencies
├── DEPLOYMENT_GITHUB_PAGES.md      # Documentation for GitHub Pages deployment
├── docs/                           # Client-side GitHub Pages Web Demo (ONNX Runtime Web)
│   ├── index.html                  # Responsive HTML5 UI with model dropdown & challenge cases
│   ├── style.css                   # Dark-mode clinical styling
│   ├── app.js                      # Canvas preprocessing (224x224, ImageNet norm) & ONNX inference
│   ├── .nojekyll                   # Disables Jekyll processing on GitHub Pages
│   ├── examples/                   # Preloaded sample images (H1..H6, challenge cases)
│   └── models/                     # In-browser ONNX model files (< 45 MB each)
│       ├── resnet18_linear_best.onnx
│       ├── resnet18_mlp_best.onnx
│       ├── resnet18_mlp_deeper_best.onnx
│       ├── resnet18_mlp_finetuned_best.onnx
│       └── resnet18_mlp_deep_finetuned_best.onnx
├── saved_models/                   # Production PyTorch checkpoints (isolated from experiments)
│   ├── resnet18_linear_best.pt
│   ├── resnet18_mlp_best.pt
│   ├── resnet18_mlp_deeper_best.pt
│   ├── resnet18_mlp_finetuned_best.pt
│   └── resnet18_mlp_deep_finetuned_best.pt
├── src/
│   ├── baseline/
│   │   └── Baseline.ipynb          # Zero-R baseline notebook & initial EDA
│   ├── evaluate_checkpoints.py     # Comprehensive test set benchmarking across all models
│   ├── export_onnx.py              # PyTorch to ONNX export pipeline
│   ├── find_disagreements.py       # Identifies edge cases and model disagreement images
│   └── transfer/
│       ├── dataset.py              # PyTorch Dataset, stratified splitting, and augmentation
│       ├── model.py                # ResNet-18 architectures (Linear, MLP, Deep, Deeper)
│       └── train.py                # Unified feature extraction and dynamic fine-tuning CLI
├── tests/                          # Automated pytest verification suite
│   ├── test_checkpoint_history.py
│   ├── test_cli_workflow.py
│   ├── test_fine_tune.py
│   └── test_multisession_training.py
└── hf_space/                       # Gradio demo alternative
    ├── app.py                      # Gradio UI application
    └── model_utils.py              # Checkpoint inference logic
```
