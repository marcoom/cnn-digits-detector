"""
app.py — Gradio sketchpad + MNIST CNN inference (Plotly + Clear Canvas).

- Sketchpad ready to draw (white background, fixed black brush, thin stroke).
- "Clear Canvas" button resets the pad and hides outputs.
- Preprocess to (28, 28, 1) grayscale [0..1], inverted to white-on-black.
- Inference with `mnist_cnn_albu.keras`.
- Outputs: a prominent predicted digit + an interactive Plotly horizontal bar chart (0–9).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from PIL import Image, ImageOps
from tensorflow import keras


# -----------------------------
# Config
# -----------------------------
APP_TITLE = "CNN Digits Detector"
MODEL_FILENAME = "models/mnist_cnn_albu.keras"
CANVAS_SIZE = 520            # Larger drawing area
DEFAULT_BRUSH_SIZE = 6       # Thinner default stroke
BRUSH_COLOR = "#000000"      # Black
BG_COLOR = 255               # White background (uint8)
BAR_COLOR = "#ea580c"        # Plotly bar color


# -----------------------------
# Model loading & inference
# -----------------------------
def _resolve_model_path(filename: str) -> Path:
    """Resolve model path in a way that works in scripts and notebooks."""
    candidates = []
    try:
        candidates.append(Path(__file__).parent / filename)
    except NameError:
        pass
    candidates.append(Path.cwd() / filename)

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Model file '{filename}' not found. "
        f"Tried: {', '.join(str(p) for p in candidates)}"
    )


@lru_cache(maxsize=1)
def load_model() -> keras.Model:
    """Load and cache the trained Keras model from disk."""
    return keras.models.load_model(_resolve_model_path(MODEL_FILENAME))


def predict_probs(x: np.ndarray) -> np.ndarray:
    """Run model inference and return class probabilities as a 1D float array of length 10."""
    model = load_model()
    preds = model.predict(x, verbose=0)
    probs = preds[0].astype(float)
    s = probs.sum()
    if s <= 0 or not np.isfinite(s):
        probs = np.full_like(probs, 1.0 / probs.size)
    else:
        probs = probs / s
    return probs


# -----------------------------
# Image preprocessing
# -----------------------------
def _to_pil(img_like: Any) -> Image.Image:
    """Convert an ImageEditor value to a PIL Image."""
    if isinstance(img_like, Image.Image):
        return img_like
    if isinstance(img_like, np.ndarray):
        if img_like.ndim == 2:
            return Image.fromarray(img_like.astype(np.uint8), "L")
        return Image.fromarray(img_like.astype(np.uint8))
    if isinstance(img_like, str):
        return Image.open(img_like)
    return Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=BG_COLOR)


def preprocess_editor_value(editor_value: Optional[Dict[str, Any]]) -> np.ndarray:
    """
    Convert the ImageEditor's EditorValue into a normalized tensor for the CNN.

    Steps:
    - Take 'composite', convert to 'L', invert (white strokes on black),
      resize to 28x28, normalize to [0,1], shape -> (1, 28, 28, 1).
    """
    if not editor_value:
        pil_img = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=BG_COLOR)
    else:
        composite = editor_value.get("composite", None)
        pil_img = _to_pil(composite)

    pil_img = pil_img.convert("L")
    pil_img = ImageOps.invert(pil_img)
    pil_img = pil_img.resize((28, 28), resample=Image.Resampling.LANCZOS)

    arr = np.asarray(pil_img, dtype=np.float32) / 255.0
    arr = arr[..., np.newaxis]   # (28, 28, 1)
    arr = arr[np.newaxis, ...]   # (1, 28, 28, 1)
    return arr


# -----------------------------
# Visualizations
# -----------------------------
def make_bar_chart_plotly(probs: np.ndarray) -> go.Figure:
    """Create a Plotly horizontal bar chart with transparent background and value labels."""
    digits_desc = list(range(9, -1, -1))  # 9, 8, ..., 0
    labels = [str(d) for d in digits_desc]
    percents = [float(probs[d]) * 100.0 for d in digits_desc]
    text_labels = [f"{p:.1f}%" for p in percents]

    fig = go.Figure(
        data=[
            go.Bar(
                x=percents,
                y=labels,
                orientation="h",
                marker=dict(color=BAR_COLOR),
                text=text_labels,               # show % labels on bars
                textposition="auto",
                hovertemplate="Digit %{y}: %{x:.2f}%<extra></extra>",
            )
        ]
    )
    # Transparent background, no vertical grid/zero lines
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title="Probability (%)",
            range=[0, 100],
            showgrid=False,       # no vertical grid lines
            zeroline=False,       # no zero line
            showline=False,       # no axis line
        ),
        yaxis=dict(
            title="Digit",
            autorange="reversed"  # 9 at top
        ),
        margin=dict(l=60, r=20, t=50, b=50),
        height=420,
    )
    return fig


def render_big_digit_html(digit: int) -> str:
    """Return an HTML snippet displaying the digit prominently."""
    return f"""
    <div style="text-align:center;">
      <div style="font-size:88px; font-weight:900; line-height:1; margin:8px 0;">{digit}</div>
      <div style="font-size:14px; opacity:0.75;">Most Probable Digit</div>
    </div>
    """


# -----------------------------
# Gradio callbacks
# -----------------------------
def run_prediction(editor_value: Optional[Dict[str, Any]]):
    """Preprocess the sketch, predict, and render outputs (and make them visible)."""
    x = preprocess_editor_value(editor_value)
    probs = predict_probs(x)
    top_digit = int(np.argmax(probs))
    fig = make_bar_chart_plotly(probs)
    return (
        gr.update(value=render_big_digit_html(top_digit), visible=True),
        gr.update(value=fig, visible=True),
    )


def on_clear():
    """Reset the canvas and hide outputs again."""
    blank = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=BG_COLOR)
    return (
        blank,
        gr.update(value="", visible=False),
        gr.update(value=None, visible=False),
    )


# -----------------------------
# Gradio UI
# -----------------------------
def build_ui() -> gr.Blocks:
    """Build the Gradio Blocks application."""
    white_bg = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=BG_COLOR)

    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(
            f"# {APP_TITLE}\n"
            "Draw a single digit (0-9) in the sketchpad, then click **Predict Digit**."
        )

        with gr.Row():
            with gr.Column(scale=1):
                editor = gr.ImageEditor(
                    value=white_bg,
                    label="Sketchpad",
                    image_mode="L",
                    width=CANVAS_SIZE,
                    height=CANVAS_SIZE,
                    sources=(),                 # no upload/webcam/clipboard
                    transforms=(),              # keep drawing tool ready
                    layers=False,               # hide layers UI (single-layer)
                    brush=gr.Brush(
                        default_size=DEFAULT_BRUSH_SIZE,
                        colors=[BRUSH_COLOR],    # single, fixed color
                        default_color=BRUSH_COLOR,
                        color_mode="fixed",
                    ),
                    eraser=gr.Eraser(default_size=24),
                )

                with gr.Row():
                    clear_btn = gr.Button("Clear Canvas")
                    predict_btn = gr.Button("Predict Digit", variant="primary")

            with gr.Column(scale=1):
                # Outputs start hidden, become visible on prediction
                predicted_digit_html = gr.HTML(visible=False)
                bar_plot = gr.Plot(label="Prediction Probabilities", visible=False)

        # Wire actions
        clear_btn.click(
            fn=on_clear,
            inputs=None,
            outputs=[editor, predicted_digit_html, bar_plot],
        )
        predict_btn.click(
            fn=run_prediction,
            inputs=editor,
            outputs=[predicted_digit_html, bar_plot],
            api_name="predict_digit",
        )

        with gr.Accordion("How it works", open=False):
            gr.Markdown(
                "A trained **Convolutional Neural Network (CNN)** is used to recognize handwritten digits. "
                "The network was trained on **MNIST**, a classic dataset of handwritten numbers, with **data "
                "augmentations** during training to make it more robust, and it was evaluated extensively. "
                "When you draw a digit, your sketch is sent to this trained model, which returns a probability for "
                "each number from 0 to 9. The model is designed for digits—if you draw something else, it will still "
                "choose the digit it considers the closest match."
            )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch()
