# GitHub Pages Demo Deployment Guide

This guide explains how to run the client-side ONNX web demo locally and how to host it permanently on **GitHub Pages** (`https://<username>.github.io/<repo>/`).

---

## 1. Local Testing

To test the web demo locally in your browser:

### Option A: From Repository Root (Recommended)
```powershell
python -m http.server 8000 --directory docs
```

### Option B: From the `docs/` Directory
```powershell
cd docs
python -m http.server 8000
```

Open your browser to: **`http://localhost:8000`**

### Features Available in the Demo:
- **Client-Side WebAssembly (WASM)**: Inference runs locally in the browser with zero cloud server dependencies.
- **Model Switching**: Compare in real-time between:
  1. **ResNet-18 (Fine-Tuned MLP Champion)** &bull; **91.11%** Test Accuracy, **92.96%** Macro F1
  2. **ResNet-18 (Deeper 4-Layer MLP)** &bull; **86.01%** Test Accuracy, **87.98%** Macro F1
  3. **ResNet-18 (Linear Head Baseline)** &bull; **66.21%** Test Accuracy, **67.92%** Macro F1
- **Click-to-Test Sample Gallery**: Instant evaluation on preloaded microscopic examples (H1, H2, H3, H5, H6).
- **Drag-and-Drop / Clipboard**: Upload any external microscopic fungi image or paste directly from clipboard.
- **Interactive Probabilities & Clinical Notes**: Real-time confidence bars and scientific descriptions for dermatophyte species.

---

## 2. Deploying to GitHub Pages (Permanent URL)

GitHub Pages serves static web apps directly from the `/docs` directory on your repository's `main` branch.

### Step 2.1: Stage and Push Changes to GitHub
Run the following commands in your terminal:

```powershell
# 1. Stage all changes (including the /docs folder and ONNX models)
git add .

# 2. Commit the changes
git commit -m "Deploy client-side ONNX web demo to GitHub Pages"

# 3. Push to your GitHub repository
git push origin main
```

---

### Step 2.2: Enable GitHub Pages in Repository Settings

1. Open your repository on [GitHub](https://github.com).
2. Click on the **Settings** tab (gear icon at the top right of the repo).
3. In the left sidebar, click **Pages** (under the "Code and automation" section).
4. Under **Build and deployment**:
   - **Source**: Select `Deploy from a branch`.
   - **Branch**: Select `main`.
   - **Folder**: Select `/docs` from the dropdown.
   - Click **Save**.
5. Wait 60–90 seconds for GitHub Actions to build and deploy.
6. Your live demo URL will appear at the top of the Pages settings page:
   ```
   https://<your-github-username>.github.io/<your-repository-name>/
   ```

---

## 3. Why GitHub Pages + ONNX vs. Hugging Face Spaces?

| Feature | Hugging Face Spaces | GitHub Pages + ONNX Runtime Web |
|---|---|---|
| **Server Cost** | Free tier sleeps after inactivity | 100% Free permanent uptime (zero sleep) |
| **Cold Starts** | 30–60 second wake-up delays | **Instant** (loads immediately in browser) |
| **Privacy** | Images sent to external cloud server | **100% Private** (images never leave client device) |
| **Portfolio Longevity** | Can be deleted if unused | **Permanent** academic portfolio link |
| **Offline Support** | Requires internet connection | Can run offline as a Progressive Web App |

---

## 4. Re-exporting ONNX Models (Optional)

If you train additional checkpoints and want to update the models featured on GitHub Pages:

```powershell
# Exports PyTorch checkpoints to docs/models/*.onnx
python src/export_onnx.py
```

