"""
A Gradio web app with drag and drop for image and returns the output of the `prediction()` function.
"""

import requests
from typing import Union
import gradio as gr
from PIL import Image
import io


def prediction(img: Union[Image.Image, None]) -> str:
    """Return a message based on the provided image.
    Replace this body with your actual model call.
    """
    if img is None:
        return "Aucune image reçue. Glissez-déposez une image à gauche."

    # Convert the image to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    api_url = "http://localhost:8000/predict"
    files = {"file": ("image.png", img_bytes, "image/png")}
    try:
        response = requests.post(api_url, files=files)
        response.raise_for_status()
        result = response.json()
        return f"Prédiction : {result['prediction']} (Confiance : {result['probability']:.2f})"
    except requests.RequestException as e:
        return f"Erreur lors de l'appel à l'API : {str(e)}"


beige_theme = gr.themes.Soft().set(
    body_background_fill="#f8f5eb",
    block_background_fill="#fffdf8",
    block_radius="12px",
    shadow_drop="0 2px 4px rgba(0,0,0,0.05)",
    button_primary_background_fill="#d2b48c",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#e8dcc8",
)

CSS = """
.gr-box, .gr-block {
    border: 1px solid #e5decf !important;
}
.gr-button-primary {
    color: #ffffff !important;
}
/* Animated gradient hero banner */
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    100% { background-position: 100% 50%; }
}
.hero {
    background: linear-gradient(135deg, #f0e6d8 0%, #e7d8c4 50%, #d9c6b0 100%);
    background-size: 200% 200%;
    animation: gradientShift 8s ease-in-out infinite alternate;
    padding: 2rem 1.5rem;
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    text-align: center;
    margin-bottom: 2rem;
}
.hero h1 {
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0 0 .6rem;
    color: #4b463f;
}
.hero p {
    font-size: 1.05rem;
    font-weight: 500;
    margin: 0;
    color: #4b463f;
}
"""


def build_interface() -> gr.Blocks:
    """Construct and return the Gradio interface."""

    with gr.Blocks(theme=beige_theme, css=CSS, title="Image Prediction") as demo:
        gr.Markdown(
            """
            <div class="hero">
                <h1>Classifier les images de pissenlits et d'herbes</h1>
            </div>
            <p>1. Déposer votre image à gauche.</p>
            <p>2. Le résultat apparait à votre droite.</p>
            """
        )

        # Layout – input on the left, output on the right
        with gr.Row(equal_height=True):
            image_input = gr.Image(type="pil", label="Image")
            output_text = gr.Textbox(
                label="Résultat de la prédiction",
                placeholder="Pissenlit ou herbe ?",
                lines=2,
                interactive=False,
            )

        # Call for the prediction function
        image_input.change(fn=prediction, inputs=image_input, outputs=output_text)

        # Utility buttons
        with gr.Row():
            clear_btn = gr.Button("Effacer")
            clear_btn.click(lambda: (None, ""), outputs=[image_input, output_text])

    return demo


if __name__ == "__main__":
    demo = build_interface()

    # Enable the task queue for concurrency
    demo.queue()

    # Launch the app locally on URL (by default http://localhost:7860)
    demo.launch(server_name="0.0.0.0", server_port=7860)
