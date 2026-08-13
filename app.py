"""
GreenHouse Leaf Checker — single Streamlit UI for both crops.

One drag-and-drop interface covering bell pepper and tomato. Pick the crop in the
sidebar; the headline verdict (HEALTHY / DAMAGED) is rendered first and largest,
with the specific condition, confidence and probability breakdown as supporting
detail, and the image preview last.

The two models are completely different underneath — bell pepper is a
scikit-learn bundle over hand-engineered features, tomato is a Keras CNN — so
each crop supplies its own loader and predictor, and both return the same result
dict. Everything downstream of that is shared, which is what keeps the UI single.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import streamlit as st

from bell_pepper_pipeline import load_image
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

TOMATO_MODEL_PATH = "app/tomato_model.keras"
TOMATO_INPUT_SIZE = (224, 224)

# Class order must match training exactly — index i of the model's output vector
# is this list's element i. Reordering these silently mislabels every prediction.
TOMATO_CLASS_NAMES = [
    "Tomato__Bacterial_spot",
    "Tomato__Early_blight",
    "Tomato__Late_blight",
    "Tomato__Leaf_Mold",
    "Tomato__Septoria_leaf_spot",
    "Tomato__Spider_mites",
    "Tomato__Target_Spot",
    "Tomato__Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato__healthy",
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


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def decode_upload(raw: bytes) -> np.ndarray:
    """Decode uploaded image bytes into an RGB uint8 array."""
    buf = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode that file as an image.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def prettify(raw_name: str) -> str:
    """
    Turn a training folder label into something readable.

    ``Tomato__Tomato_Yellow_Leaf_Curl_Virus`` -> ``Yellow leaf curl virus``.
    The crop prefix is dropped because the sidebar already says which crop this
    is, and repeating it in every row just adds noise.
    """
    name = raw_name.replace("Tomato__", "").replace("Tomato_", "")
    name = name.replace("_", " ").strip()
    return name[:1].upper() + name[1:] if name else raw_name


# --------------------------------------------------------------------------- #
# Bell pepper: scikit-learn bundle over HSV + LBP features
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner="Loading pepper model ...")
def load_pepper_model(model_path: str, mtime: float):
    """
    Load and cache the pepper bundle.

    ``mtime`` is part of the cache key (not used in the body) so that retraining
    the model invalidates the cache automatically on the next rerun.
    """
    return load_bundle(model_path)


def predict_pepper_image(rgb: np.ndarray, model) -> dict:
    """Classify a pepper leaf. Already returns the shared result shape."""
    return predict_from_array(rgb, model)


def describe_pepper(bundle) -> list:
    """Sidebar caption lines describing a loaded pepper bundle."""
    lines = []
    cfg = bundle.get("feature_config", {})
    if cfg:
        lines.append(
            f"Input size {cfg.get('img_size')} · HSV bins {cfg.get('hsv_bins')} "
            f"· LBP P={cfg.get('lbp_points')}, R={cfg.get('lbp_radius')}"
        )
        if cfg.get("segment_leaf"):
            lines.append(
                "Background removed before scoring · white-balanced · "
                "exposure-normalised"
            )
    stability = bundle.get("consistency", {}).get("stable_fraction")
    if stability is not None:
        lines.append(
            f"Verdict stable on {stability:.0%} of leaves under rotation, "
            f"exposure and colour shifts"
        )
    lines.append("Classes: " + ", ".join(bundle.get("class_names", [])))
    return lines


# --------------------------------------------------------------------------- #
# Tomato: Keras CNN, 10-class disease identification
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner="Loading tomato model ...")
def load_tomato_model(model_path: str, mtime: float):
    """
    Load and cache the tomato Keras model.

    TensorFlow is imported here rather than at module scope on purpose: it is a
    heavy optional dependency that is not in requirements.txt, and importing it
    up top would break the pepper half of this app for anyone who has not
    installed it.
    """
    try:
        from tensorflow import keras
    except ImportError as exc:  # noqa: BLE001 - surfaced to the user as guidance
        raise ImportError(
            "TensorFlow is required for the tomato model but is not installed. "
            "The bell pepper model does not need it."
        ) from exc

    return keras.models.load_model(model_path)


def predict_tomato_image(rgb: np.ndarray, model) -> dict:
    """
    Classify a tomato leaf into one of ten conditions.

    Preprocessing mirrors the training notebook exactly — PIL resize to 224x224
    and a raw float32 array with no rescaling, because the model carries its own
    normalisation layer. Changing either would quietly shift every prediction.
    """
    from PIL import Image

    img = Image.fromarray(rgb).convert("RGB").resize(TOMATO_INPUT_SIZE)
    batch = np.expand_dims(np.array(img, dtype=np.float32), axis=0)

    probabilities = model.predict(batch, verbose=0)[0]
    index = int(np.argmax(probabilities))
    raw_name = TOMATO_CLASS_NAMES[index]
    is_healthy = raw_name.endswith("healthy")

    return {
        "verdict": "HEALTHY" if is_healthy else "DAMAGED",
        "is_healthy": is_healthy,
        # Class index, matching the pepper predictor's key set so both crops
        # return an identical dict shape and the renderers stay crop-agnostic.
        "label": index,
        "label_name": prettify(raw_name),
        "confidence": float(probabilities[index]),
        "probabilities": {
            prettify(name): float(p)
            for name, p in zip(TOMATO_CLASS_NAMES, probabilities)
        },
    }


def describe_tomato(model) -> list:
    """Sidebar caption lines describing the loaded tomato model."""
    return [
        f"Keras CNN · input {TOMATO_INPUT_SIZE[0]}×{TOMATO_INPUT_SIZE[1]}",
        f"{len(TOMATO_CLASS_NAMES)} classes: "
        + ", ".join(prettify(n) for n in TOMATO_CLASS_NAMES),
    ]


# --------------------------------------------------------------------------- #
# Crop registry — the only place the two models differ
# --------------------------------------------------------------------------- #

CROPS = {
    "Bell pepper": {
        "icon": "🌶️",
        "species": "Capsicum annuum",
        "default_path": DEFAULT_MODEL_PATH,
        "loader": load_pepper_model,
        "predictor": predict_pepper_image,
        "describer": describe_pepper,
        "train_hint": "python bell_pepper_pipeline.py --data-dir data",
        "footer": (
            "HSV color histogram + Local Binary Pattern texture features over the "
            "segmented leaf, with a scikit-learn classifier."
        ),
    },
    "Tomato": {
        "icon": "🍅",
        "species": "Solanum lycopersicum",
        "default_path": TOMATO_MODEL_PATH,
        "loader": load_tomato_model,
        "predictor": predict_tomato_image,
        "describer": describe_tomato,
        "train_hint": "See notebooks/plant_damage_classifier.ipynb",
        "footer": "Keras convolutional network trained on 224×224 leaf images.",
    },
}


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

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
        ranked = sorted(
            result["probabilities"].items(), key=lambda kv: kv[1], reverse=True
        )
        # Ten rows of near-zero probabilities is noise; the top few carry the
        # information. Binary models fall through and show both classes.
        shown = ranked if len(ranked) <= 3 else ranked[:3]
        label = "Class probabilities:" if len(ranked) <= 3 else "Top 3 probabilities:"
        st.write(label)
        for name, p in shown:
            st.markdown(f"- {name}: {p:.4f}")


def show_model_problem(crop_name: str, crop: dict, model_path: str, error) -> None:
    """
    Explain why no model is loaded, distinguishing the two very different causes.

    A missing file and a file that exists but would not load need opposite
    actions, so they must not share a message — telling someone to retrain when
    the model is sitting right there just sends them down the wrong path.
    """
    if error is None:
        st.error(f"No trained {crop_name.lower()} model found at `{model_path}`.")
        st.markdown(
            "Train one first, then reload this page:\n"
            f"```bash\n{crop['train_hint']}\n```\n"
            "See `README.md` for where to download the dataset."
        )
        return

    st.error(f"Found `{model_path}`, but it could not be loaded.")
    st.markdown(str(error))
    if isinstance(error, ImportError):
        st.markdown("```bash\npip install tensorflow\n```")


def render_sidebar() -> tuple:
    """
    Draw the sidebar.

    Returns ``(crop_name, crop, model_path, model_or_None, error_or_None)``. The
    error is carried out rather than only printed so the main panel can explain
    the right thing.
    """
    st.sidebar.header("Settings")

    crop_name = st.sidebar.radio("Crop", list(CROPS), index=0)
    crop = CROPS[crop_name]

    model_path = st.sidebar.text_input(
        "Model path",
        value=crop["default_path"],
        # Keyed per crop so switching crops swaps the remembered path instead of
        # carrying the pepper path over to tomato.
        key=f"model_path_{crop_name}",
    ).strip()

    if not os.path.isfile(model_path):
        return crop_name, crop, model_path, None, None

    try:
        model = crop["loader"](model_path, os.path.getmtime(model_path))
    except (ValueError, ImportError) as exc:
        # ValueError: bundle descriptor version predates this code.
        # ImportError: TensorFlow missing for the tomato model.
        # Both are actionable, so show the message rather than a raw traceback.
        st.sidebar.error(str(exc))
        return crop_name, crop, model_path, None, exc

    st.sidebar.success(f"Loaded {crop_name.lower()} model")
    for line in crop["describer"](model):
        st.sidebar.caption(line)

    return crop_name, crop, model_path, model, None


def main() -> None:
    """Entry point. Kept in a function so the module stays importable."""
    st.set_page_config(
        page_title="GreenHouse Leaf Checker",
        page_icon="🌿",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    st.markdown(_HIDE_CHROME_CSS, unsafe_allow_html=True)

    crop_name, crop, model_path, model, load_error = render_sidebar()

    st.title("🌿 GreenHouse Leaf Checker")
    st.caption(
        f"{crop['icon']} &nbsp;Upload a **{crop_name.lower()}** "
        f"(*{crop['species']}*) leaf image to check whether it looks healthy or "
        "damaged. Runs entirely on this machine."
    )

    if model is None:
        show_model_problem(crop_name, crop, model_path, load_error)
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

    with st.spinner("Classifying ..."):
        result = crop["predictor"](rgb, model)

    # Verdict before the preview: a full-width image would otherwise push the
    # headline below the fold, which is the one thing that must be seen first.
    render_verdict(result)
    render_details(result)

    st.image(rgb, caption="Uploaded Image", width=380)

    st.divider()
    st.caption(
        f"{crop['footer']} Each crop has its own model — a leaf is only meaningful "
        "to the crop selected in the sidebar. Screening aid only, not a substitute "
        "for agronomic diagnosis."
    )


if __name__ == "__main__":
    main()
