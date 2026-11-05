"""
Interactive Gradio Web Demo for DeFungi Microscopic Classification.
Configured for local execution and Hugging Face Spaces deployment.
"""

from pathlib import Path
import gradio as gr
from model_utils import predict, CLASS_INFO

# Locate example images
BASE_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = BASE_DIR / "examples"

example_images = []
if EXAMPLES_DIR.exists():
    for f in sorted(EXAMPLES_DIR.glob("*.jpg")):
        example_images.append([str(f), "ResNet-18 (MLP Decoder - Recommended)"])

DESCRIPTION = """
# 🍄 Microscopic Fungi Classifier (DeFungi Dataset)
### CSSE/MA416 Deep Learning Term Project

Upload a microscopic fungi image or pick one from the example gallery below to classify the species and view model confidence.

**Supported Species:**
- **H1:** *Trichophyton rubrum* (common nail & skin infections)
- **H2:** *Trichophyton mentagrophytes* (inflammatory lesions)
- **H3:** *Epidermophyton floccosum* (skin & nails)
- **H5:** *Microsporum canis* (zoophilic ringworm)
- **H6:** *Microsporum gypseum* (soil-borne dermatophyte)
"""

ARTICLE = """
---
### 🔬 Model Details
- **Architecture**: Transfer learning with ResNet-18 pretrained backbone.
- **Trained heads**:
  - **MLP Decoder**: 2-layer MLP with ReLU & Dropout (**75.10%** test accuracy, **78.84%** macro F1)
  - **Linear Decoder**: Single-layer linear projection (**66.21%** test accuracy, **67.92%** macro F1)
- **Resolution**: Images resized & normalized to standard 224x224 RGB tensors.
"""


def create_demo():
    with gr.Blocks(title="DeFungi Classifier") as demo:
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Microscopic Fungi Image",
                    sources=["upload", "clipboard"]
                )

                model_selector = gr.Radio(
                    choices=[
                        "ResNet-18 (MLP Decoder - Recommended)",
                        "ResNet-18 (Linear Decoder)"
                    ],
                    value="ResNet-18 (MLP Decoder - Recommended)",
                    label="Model Architecture / Decoder"
                )

                submit_btn = gr.Button("🔍 Classify Fungi", variant="primary")
                clear_btn = gr.ClearButton([image_input], value="Clear Image")

            with gr.Column(scale=1):
                label_output = gr.Label(num_top_classes=5, label="Prediction Probabilities")
                markdown_output = gr.Markdown(label="Species Information")

        submit_btn.click(
            fn=predict,
            inputs=[image_input, model_selector],
            outputs=[label_output, markdown_output]
        )

        image_input.change(
            fn=predict,
            inputs=[image_input, model_selector],
            outputs=[label_output, markdown_output]
        )

        model_selector.change(
            fn=predict,
            inputs=[image_input, model_selector],
            outputs=[label_output, markdown_output]
        )

        if example_images:
            gr.Examples(
                examples=example_images,
                inputs=[image_input, model_selector],
                outputs=[label_output, markdown_output],
                fn=predict,
                cache_examples=False,
                label="Sample Images (Click to Test)"
            )

        gr.Markdown(ARTICLE)

    return demo


if __name__ == "__main__":
    demo = create_demo()
    # launch locally on port 7860
    demo.launch(theme=gr.themes.Soft(), share=False)
