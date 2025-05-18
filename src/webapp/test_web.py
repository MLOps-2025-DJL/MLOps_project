import io
from PIL import Image
import pytest
import gradio as gr
from unittest.mock import patch, Mock
from web import prediction, build_interface


def test_prediction_none():
    result = prediction(None)
    assert result == "Aucune image reçue. Glissez-déposez une image à gauche."


@patch("web.requests.post")
def test_prediction_success(mock_post):
    # Create a dummy image
    img = Image.new("RGB", (10, 10), color="red")
    # Mock response
    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"prediction": "pissenlit", "probability": 0.95}
    mock_post.return_value = mock_response

    result = prediction(img)
    assert "Prédiction : pissenlit" in result
    assert "(Confiance : 0.95)" in result


def test_build_interface_returns_blocks():
    interface = build_interface()
    assert isinstance(interface, gr.Blocks)
