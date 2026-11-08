"""
Model inference utilities for the DeFungi classification demo.
Loads ResNet-18 checkpoints trained on the DeFungi microscopic dataset.
Supports Linear, MLP, MLP-Deep, MLP-Deeper, and Fine-Tuned checkpoints.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image

# Directory where models are located
_BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = _BASE_DIR / "checkpoints"

# Fungal classes & descriptive scientific names
CLASS_INFO = {
    "H1": {"name": "Trichophyton rubrum", "notes": "Most common cause of athlete's foot and fungal nail infections."},
    "H2": {"name": "Trichophyton mentagrophytes", "notes": "Zoophilic/anthropophilic dermatophyte causing inflammatory lesions."},
    "H3": {"name": "Epidermophyton floccosum", "notes": "Anthropophilic dermatophyte affecting skin and nails, club-shaped macroconidia."},
    "H5": {"name": "Microsporum canis", "notes": "Zoophilic dermatophyte primarily found in cats and dogs, highly contagious to humans."},
    "H6": {"name": "Microsporum gypseum", "notes": "Geophilic dermatophyte living in soil, produces abundant rough-walled macroconidia."}
}

CLASS_NAMES = ["H1", "H2", "H3", "H5", "H6"]

# Normalization constants used during training
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


def build_model(decoder_type: str = "mlp", num_classes: int = 5, dropout: float = 0.3) -> nn.Module:
    """Instantiate a ResNet-18 architecture with specified classification head."""
    model = resnet18(weights=None)
    in_features = model.fc.in_features

    if decoder_type == "linear":
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
    elif decoder_type == "mlp":
        model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )
    elif decoder_type == "mlp_deep":
        model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )
    elif decoder_type == "mlp_deeper":
        model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )
    else:
        raise ValueError(f"Unsupported decoder type: {decoder_type}")

    return model


_CACHED_MODELS: Dict[str, nn.Module] = {}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(model_choice: str) -> nn.Module:
    """
    Loads and caches the requested model checkpoint.
    """
    choice_lower = model_choice.lower()
    is_ft = "fine-tune" in choice_lower or "finetuned" in choice_lower

    if "deeper" in choice_lower:
        decoder_type = "mlp_deeper"
    elif "deep" in choice_lower:
        decoder_type = "mlp_deep"
    elif "mlp" in choice_lower:
        decoder_type = "mlp"
    else:
        decoder_type = "linear"

    cache_key = f"{decoder_type}_ft" if is_ft else decoder_type

    if cache_key in _CACHED_MODELS:
        return _CACHED_MODELS[cache_key]

    filename = f"resnet18_{decoder_type}_finetuned_best.pt" if is_ft else f"resnet18_{decoder_type}_best.pt"
    # Search order: local hf_space/checkpoints -> saved_models -> checkpoints
    search_dirs = [
        CHECKPOINT_DIR,
        _BASE_DIR.parent / "saved_models",
        _BASE_DIR.parent / "checkpoints"
    ]
    ckpt_path = None
    for d in search_dirs:
        cand = d / filename
        if cand.exists():
            ckpt_path = cand
            break

    if ckpt_path is None and is_ft:
        # Fallback to base model if finetuned not found
        fallback_name = f"resnet18_{decoder_type}_best.pt"
        for d in search_dirs:
            cand = d / fallback_name
            if cand.exists():
                ckpt_path = cand
                break

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    net = build_model(decoder_type=decoder_type, num_classes=len(CLASS_NAMES))
    state = torch.load(str(ckpt_path), map_location=DEVICE)

    if isinstance(state, dict) and "model_state_dict" in state:
        state_dict = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state_dict = state["state_dict"]
    else:
        state_dict = state

    net.load_state_dict(state_dict)
    net.to(DEVICE)
    net.eval()
    _CACHED_MODELS[cache_key] = net

    return _CACHED_MODELS[cache_key]


def predict(image: Image.Image, model_choice: str = "ResNet-18 (MLP Decoder - Recommended)") -> Tuple[Dict[str, float], str]:
    """
    Runs prediction on a PIL Image.
    Returns:
      - Dictionary mapping label string -> probability (for Gradio gr.Label)
      - Markdown description of top predicted class and context
    """
    if image is None:
        return {}, "Please upload or select an image to classify."

    model = get_model(model_choice)

    if image.mode != "RGB":
        image = image.convert("RGB")

    tensor = inference_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    # Create probability dictionary with full descriptive labels
    results = {}
    for idx, cls_code in enumerate(CLASS_NAMES):
        species = CLASS_INFO[cls_code]["name"]
        label = f"{cls_code}: {species}"
        results[label] = float(probs[idx])

    top_idx = int(probs.argmax())
    top_code = CLASS_NAMES[top_idx]
    top_species = CLASS_INFO[top_code]["name"]
    top_prob = float(probs[top_idx]) * 100.0
    notes = CLASS_INFO[top_code]["notes"]

    markdown_details = f"""
### Predicted Species: **{top_code} - *{top_species}***
- **Confidence:** `{top_prob:.2f}%`
- **Model Selected:** `{model_choice}`
- **Fungal Description:** {notes}
"""

    return results, markdown_details
