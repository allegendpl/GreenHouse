"""
Single-image inference for the Bell Pepper Leaf Damage Identifier.

Loads a trained ``.joblib`` bundle produced by ``bell_pepper_pipeline.py`` and
classifies one bell pepper leaf image, leading with the headline verdict:
**HEALTHY** or **DAMAGED**.

Feature extraction is imported directly from the training module so inference
is guaranteed to use the identical descriptor.

Usage
-----
    python predict_pepper.py --image path/to/leaf.jpg
    python predict_pepper.py --image leaf.jpg --model models/best_pepper_model.joblib
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from joblib import load

from bell_pepper_pipeline import (
    CLASS_DAMAGED,
    CLASS_NAMES,
    CLASS_UNDAMAGED,
    FEATURE_VERSION,
    extract_features,
    load_image,
)

DEFAULT_MODEL_PATH = "models/best_pepper_model.joblib"

# Headline verdict strings, kept here so the CLI and the Streamlit UI agree.
VERDICT = {CLASS_UNDAMAGED: "HEALTHY", CLASS_DAMAGED: "DAMAGED"}


def load_bundle(model_path: str = DEFAULT_MODEL_PATH) -> dict:
    """
    Load the persisted model bundle, validating the path and descriptor version.

    A bundle trained on an older feature definition would still *run* — the
    vector lengths can coincide — while quietly scoring nonsense, because the
    bins no longer mean what the model learned. Refusing to load it turns a
    silent accuracy collapse into an actionable error.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path!r}. Train one first:\n"
            f"  python bell_pepper_pipeline.py --data-dir data"
        )

    bundle = load(model_path)

    bundle_version = bundle.get("feature_version", 1)
    if bundle_version != FEATURE_VERSION:
        raise ValueError(
            f"Model {model_path!r} was trained with feature version "
            f"{bundle_version}, but this code produces version {FEATURE_VERSION}. "
            f"Retrain it:\n  python bell_pepper_pipeline.py --data-dir data"
        )
    return bundle


def predict_from_array(rgb: np.ndarray, bundle: dict) -> dict:
    """
    Classify an already-decoded RGB image array using a preloaded bundle.

    Split out from :func:`predict` so a long-running UI can load the model once
    and reuse it across many images instead of paying disk I/O per request.

    Returns
    -------
    dict
        ``{'verdict', 'is_healthy', 'label', 'label_name', 'confidence',
        'probabilities'}``. ``confidence`` / ``probabilities`` are ``None`` for
        estimators without ``predict_proba``.
    """
    model = bundle["model"]
    class_names = bundle.get("class_names", CLASS_NAMES)

    features = extract_features(rgb).reshape(1, -1)
    label = int(model.predict(features)[0])

    confidence, prob_map = None, None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        # Report the probability of the predicted class, not just the max, so
        # the number always corresponds to the verdict shown.
        classes = list(model.classes_)
        confidence = float(proba[classes.index(label)])
        prob_map = {class_names[int(c)]: float(p) for c, p in zip(classes, proba)}

    return {
        "verdict": VERDICT[label],
        "is_healthy": label == CLASS_UNDAMAGED,
        "label": label,
        "label_name": class_names[label],
        "confidence": confidence,
        "probabilities": prob_map,
    }


def predict(image_path: str, model_path: str = DEFAULT_MODEL_PATH) -> dict:
    """Classify a single leaf image given a path on disk."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path!r}")
    bundle = load_bundle(model_path)
    # extract_features accepts a path directly, but decoding here keeps the
    # array-based code path the single source of truth for prediction logic.
    return predict_from_array(load_image(image_path), bundle)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify a single bell pepper leaf image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", required=True, help="Path to the leaf image.")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_PATH, help="Trained model bundle."
    )
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = predict(args.image, args.model)

    # --- Headline verdict first -------------------------------------------- #
    mark = "[OK]" if result["is_healthy"] else "[!!]"
    banner = f"  {mark}  {result['verdict']}  "
    rule = "=" * max(len(banner), 40)
    print(f"\n{rule}\n{banner}\n{rule}")

    if result["confidence"] is not None:
        print(f"Confidence : {result['confidence']:.1%}")

    # --- Supporting detail afterwards -------------------------------------- #
    print(f"\nImage      : {args.image}")
    print(f"Class      : [{result['label']}] {result['label_name']}")
    if result["probabilities"]:
        print("\nProbability breakdown:")
        for name, p in sorted(
            result["probabilities"].items(), key=lambda kv: kv[1], reverse=True
        ):
            bar = "#" * int(round(p * 30))
            print(f"  {name:<22} {p:>6.1%}  {bar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
