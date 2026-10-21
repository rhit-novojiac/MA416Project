# MA416 Project - Fungi Classification

A Deep Learning / Computer Vision project utilizing the **DeFungi** microscopic image dataset for fungal classification and analysis.

---

## Dataset Overview

The project uses the **DeFungi** dataset containing microscopic images across multiple fungal species classes:
- **H1**: *Trichophyton rubrum*
- **H2**: *Trichophyton mentagrophytes*
- **H3**: *Epidermophyton floccosum*
- **H5**: *Microsporum canis*
- **H6**: *Microsporum gypseum*

### Dataset Directory Structure

The dataset is stored locally in the `defungi/` directory (which is excluded from version control via `.gitignore` to keep repository size optimal):

```
defungi/
├── H1/
├── H2/
├── H3/
├── H5/
└── H6/
```

---

## Getting Started

### 1. Environment Setup

Create and activate a virtual environment:

```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
```

### 2. Dependencies

Install required machine learning and imaging packages:

```bash
pip install torch torchvision numpy matplotlib pandas scikit-learn
```

---

## Repository Structure

```
MA416Project/
├── .gitignore       # Standard Python, checkpoint, and dataset ignore rules
├── README.md        # Project documentation
└── defungi/         # Local dataset folder (ignored from Git)
```
