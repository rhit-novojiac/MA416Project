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
short_description: Microscopic fungi classifier using ResNet-18
---

# DeFungi Microscopic Classifier Demo

Interactive deep learning classifier for microscopic fungi based on the **DeFungi** clinical dataset.

## Supported Fungal Species
- **H1**: *Trichophyton rubrum*
- **H2**: *Trichophyton mentagrophytes*
- **H3**: *Epidermophyton floccosum*
- **H5**: *Microsporum canis*
- **H6**: *Microsporum gypseum*

## Models Included
- **ResNet-18 (MLP Decoder - Recommended)**: Val Acc: 75.10%, Macro F1: 78.84%
- **ResNet-18 (Linear Decoder)**: Val Acc: 66.21%, Macro F1: 67.92%

