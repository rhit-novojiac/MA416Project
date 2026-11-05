---
title: DeFungi Microscopic Classifier
emoji: 🍄
colorFrom: green
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
short_description: Microscopic fungi classifier using ResNet-18 transfer learning & fine-tuning
---

# DeFungi Microscopic Classifier Demo

Interactive deep learning web demo for microscopic fungi classification based on the clinical **DeFungi** dataset.

## Supported Fungal Species
- **H1**: *Trichophyton rubrum* (common cause of athlete's foot and nail infections)
- **H2**: *Trichophyton mentagrophytes* (causes inflammatory skin lesions)
- **H3**: *Epidermophyton floccosum* (dermatophyte affecting skin and nails)
- **H5**: *Microsporum canis* (zoophilic ringworm from cats and dogs)
- **H6**: *Microsporum gypseum* (geophilic dermatophyte found in soil)

## Models Supported
- **ResNet-18 (MLP Decoder - Recommended)**: Val Acc: 75.10%, Macro F1: 78.84%
- **ResNet-18 (Linear Decoder)**: Val Acc: 66.21%, Macro F1: 67.92%
- **ResNet-18 (Fine-Tuned)**: Fine-tuned backbone on layer4 / all layers for enhanced microscopic feature representation.

## Quick Local Run
```bash
python app.py
```
Open `http://127.0.0.1:7860` in your web browser.

See `DEPLOYMENT.md` for full Hugging Face Spaces deployment instructions.
