# Hugging Face Space Deployment Runbook

This guide explains how to test the Gradio demo locally and deploy it to Hugging Face Spaces.

---

## 1. Local Testing

To launch the web interface locally on your machine:

### Option A: From Repository Root (Recommended & Easiest)
```powershell
# Direct invocation using the project's virtual environment (bypasses Windows PATH conflicts):
.\.venv\Scripts\python.exe hf_space/app.py
```

### Option B: From `hf_space/` Directory
```powershell
cd hf_space

# Use the parent .venv Python directly:
..\.venv\Scripts\python.exe app.py
```

> [!TIP]
> If you see `ModuleNotFoundError: No module named 'torch'`, your terminal is invoking the global/Windows Store `python` instead of `.venv`. Always use `.\.venv\Scripts\python.exe` or run `.\.venv\Scripts\python.exe -m pip install torch torchvision gradio`.

Open your browser to: **`http://127.0.0.1:7860`**

You can:
- Switch between **ResNet-18 (MLP Decoder)** and **ResNet-18 (Linear Decoder)**.
- Upload any microscopic fungi image or click the preloaded samples under **Sample Images (Click to Test)**.
- See real-time classification confidences and scientific species info.

---

## 2. Deploying to Hugging Face Spaces

### Step 2.1: Create the Space on Hugging Face
1. Log into your account at [Hugging Face](https://huggingface.co).
2. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
3. Name your space (e.g. `defungi-classifier`).
4. Select **Gradio** as the Space SDK.
5. Choose **Public** visibility and click **Create Space**.

---

### Step 2.2: Push the Space via Git

Clone your newly created Space repository to a separate folder or clone directly:

```powershell
# In any directory (e.g. your projects folder or temporary folder):
git clone https://huggingface.co/spaces/<YOUR_USERNAME>/<SPACE_NAME>
cd <SPACE_NAME>

# Copy all contents of hf_space/ into this folder
Copy-Item -Path "path\to\MA416Project\hf_space\*" -Destination . -Recurse -Force

# Stage, commit, and push
git add .
git commit -m "Deploy DeFungi classification demo"
git push
```

*(If prompted for credentials, use your Hugging Face username and an Access Token from https://huggingface.co/settings/tokens as your password).*

---

### Step 2.3: Alternative - Automated Deployment via Agent or HF CLI

If you prefer the agent or CLI to push directly:
1. Generate a **Write** access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
2. Run in terminal:
   ```powershell
   hf auth login
   ```
3. Run the upload command:
   ```powershell
   hf space upload <YOUR_USERNAME>/<SPACE_NAME> hf_space/
   ```
   Or provide the token to the assistant to complete the creation and push automatically.

---

## 3. Directory Layout in `hf_space/`

```
hf_space/
├── README.md           # Hugging Face Space YAML config & documentation
├── app.py              # Main Gradio application entrypoint
├── model_utils.py      # ResNet-18 inference logic & species mapping
├── requirements.txt    # PyTorch, Gradio, Pillow dependencies
├── checkpoints/        # Best model weights (~45 MB each)
│   ├── resnet18_mlp_best.pt
│   └── resnet18_linear_best.pt
└── examples/           # Test images for H1, H2, H3, H5, H6
    ├── H1_example.jpg
    ├── H2_example.jpg
    ├── H3_example.jpg
    ├── H5_example.jpg
    └── H6_example.jpg
```

