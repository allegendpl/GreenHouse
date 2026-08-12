"""
Bell Pepper Leaf Damage Identifier — Streamlit test UI.

A minimal drag-and-drop interface for exercising a trained model. The headline
verdict (HEALTHY / DAMAGED) is rendered first and largest; the image preview and
probability breakdown follow as supporting detail.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import streamlit as st

from bell_pepper_pipeline import CLASS_NAMES, load_image
from predict_pepper import DEFAULT_MODEL_PATH, load_bundle, predict_from_array

# --------------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------------- #

HEALTHY_COLOR = "#1a7f37"
DAMAGED_COLOR = "#b42318"

# Any format OpenCV can decode. Kept permissive so you can drop in whatever
# photo you have; unreadable files are caught by decode_upload() instead.
ACCEPTED_TYPES = [
    "jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp", "jp2", "ppm", "pgm",
]

# Strips the Streamlit toolbar/footer so this looks like a plain local app
# rather than something about to be deployed to a cloud service.
_HIDE_CHROME_CSS = """
<style>
  /* Drop the Deploy button, the hamburger menu and the gradient top bar, but
     keep the header itself so the sidebar toggle stays reachable. */
  div[data-testid="stToolbar"] {display: none;}
  div[data-testid="stDecoration"] {display: none;}
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
  header[data-testid="stHeader"] {background: transparent;}
  .block-container {padding-top: 3rem;}
</style>
"""


@st.cache_resource(show_spinner="Loading model ...")
def get_bundle(model_path: str, mtime: float) -> dict:
    """
    Load and cache the model bundle.

    ``mtime`` is part of the cache key (not used in the body) so that retraining
    the model invalidates the cache automatically on the next rerun.
    """
    return load_bundle(model_path)


def decode_upload(raw: bytes) -> np.ndarray:
    """Decode uploaded image bytes into an RGB uint8 array."""
    buf = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode that file as an image.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def render_verdict(result: dict) -> None:
    """Render the big headline verdict banner."""
    color = HEALTHY_COLOR if result["is_healthy"] else DAMAGED_COLOR
    icon = "✅" if result["is_healthy"] else "⚠️"
    confidence = (
        f"{result['confidence']:.1%} confidence"
        if result["confidence"] is not None
        else "confidence unavailable"
    )

    st.markdown(
        f"""
        <div style="
            background-color:{color};
            border-radius:14px;
            padding:1.6rem 1rem;
            text-align:center;
            margin-bottom:1.2rem;">
            <div style="font-size:3.2rem;line-height:1.1;">{icon}</div>
            <div style="
                color:#ffffff;
                font-size:2.6rem;
                font-weight:800;
                letter-spacing:0.06em;">{result['verdict']}</div>
            <div style="color:#ffffffcc;font-size:1.05rem;margin-top:0.35rem;">
                {confidence}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_details(result: dict) -> None:
    """Render the prediction detail block below the verdict banner."""
    st.subheader(f"Prediction: {result['label_name']}")
    if result["confidence"] is not None:
        st.write(f"Confidence: {result['confidence']:.4f}")

    if result["probabilities"]:
        st.write("Class probabilities:")
        for name, p in sorted(
            result["probabilities"].items(), key=lambda kv: kv[1], reverse=True
        ):
            st.markdown(f"- {name}: {p:.4f}")


def show_missing_model_help(model_path: str) -> None:
    """Explain how to produce a model when none exists yet."""
    st.error(f"No trained model found at `{model_path}`.")
    st.markdown(
        "Train one first, then reload this page:\n"
        "```bash\n"
        "python bell_pepper_pipeline.py --data-dir data\n"
        "```\n"
        "See `README.md` for where to download the dataset."
    )


def render_sidebar() -> tuple:
    """Draw the sidebar and return ``(model_path, bundle_or_None)``."""
    st.sidebar.header("Settings")
    model_path = st.sidebar.text_input("Model bundle path", value=DEFAULT_MODEL_PATH)

    if not os.path.isfile(model_path):
        return model_path, None

    bundle = get_bundle(model_path, os.path.getmtime(model_path))
    st.sidebar.success(f"Loaded: {bundle.get('model_name', 'unknown')}")
    cfg = bundle.get("feature_config", {})
    if cfg:
        st.sidebar.caption(
            f"Input size {cfg.get('img_size')} · HSV bins {cfg.get('hsv_bins')} "
            f"· LBP P={cfg.get('lbp_points')}, R={cfg.get('lbp_radius')}"
        )
    st.sidebar.caption("Classes: " + ", ".join(bundle.get("class_names", CLASS_NAMES)))
    return model_path, bundle


def main() -> None:
    """Entry point. Kept in a function so the module stays importable."""
    st.set_page_config(
        page_title="Bell Pepper Leaf Damage Identifier",
        page_icon="🌿",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    st.markdown(_HIDE_CHROME_CSS, unsafe_allow_html=True)

    model_path, bundle = render_sidebar()

    st.title("🌿 Bell Pepper Leaf Damage Identifier")
    st.caption(
        "Upload a bell pepper (*Capsicum annuum*) leaf image to check whether it "
        "looks healthy or damaged. Runs entirely on this machine."
    )

    if bundle is None:
        show_missing_model_help(model_path)
        return

    uploaded = st.file_uploader("Choose an image...", type=ACCEPTED_TYPES)

    # Fallback for images already on disk — handy for batch spot-checking
    # straight out of data/ without dragging files around.
    local_path = st.text_input(
        "...or paste a path to an image on this machine",
        placeholder="/Users/you/Pictures/leaf.jpg",
    ).strip()

    rgb = None
    try:
        if uploaded is not None:
            rgb = decode_upload(uploaded.getvalue())
        elif local_path:
            if not os.path.isfile(local_path):
                st.error(f"No such file: `{local_path}`")
                return
            rgb = load_image(local_path)
    except ValueError as exc:
        st.error(str(exc))
        return

    if rgb is None:
        st.info("Choose an image above to run a prediction.")
        return

    with st.spinner("Extracting features and classifying ..."):
        result = predict_from_array(rgb, bundle)

    # Verdict before the preview: a full-width image would otherwise push the
    # headline below the fold, which is the one thing that must be seen first.
    render_verdict(result)
    render_details(result)

    st.image(rgb, caption="Uploaded Image", width=380)

    st.divider()
    st.caption(
        "Baseline model: HSV color histogram + Local Binary Pattern texture "
        "features with a classic scikit-learn classifier. Screening aid only — "
        "not a substitute for agronomic diagnosis."
    )


if __name__ == "__main__":
    main()
