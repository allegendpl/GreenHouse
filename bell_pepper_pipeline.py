"""
Bell Pepper Leaf Damage Identifier — Baseline Binary Classifier
================================================================

A classic-ML computer-vision pipeline (scikit-learn + scikit-image) that
distinguishes *Healthy* bell pepper leaves (Capsicum annuum) from *Damaged*
ones (Bacterial Spot, pest damage, nutrient deficiency, etc.).

The pipeline is intentionally free of deep learning. Instead it relies on
hand-engineered visual descriptors:

    * HSV color histograms   -> capture the yellow/brown chlorotic halos and
                                necrotic lesions typical of Bacterial Spot.
    * Local Binary Patterns  -> capture surface roughness / texture change
                                introduced by lesions and tissue breakdown.

Stages
------
    1. Dataset parsing & filtering (bell-pepper folders only)
    2. Feature extraction (HSV histogram + LBP), per image
    3. Model training (RandomForest + SVM) with an 80/20 stratified split
    4. Evaluation (classification_report + confusion matrix)
    5. Persistence of the best model as a .joblib bundle

This module is also imported by ``predict_pepper.py`` so that training and
inference share *exactly* the same feature-extraction code.

Usage
-----
    python bell_pepper_pipeline.py --data-dir data --model-out models/best_pepper_model.joblib
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import matplotlib
import numpy as np
import pandas as pd
from joblib import dump

# Non-interactive backend so plots save fine over SSH / in CI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
from skimage.feature import local_binary_pattern
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Class label convention used throughout the project.
CLASS_UNDAMAGED = 0  # "healthy"
CLASS_DAMAGED = 1    # any symptom folder (bacterial spot, pest, deficiency...)
CLASS_NAMES = ["Undamaged (Healthy)", "Damaged"]

# Punctuation stripped before matching folder names. Real datasets are wildly
# inconsistent: PlantVillage mirrors ship both "Pepper__bell___healthy" and
# "Pepper,_bell___healthy" (note the comma), so we flatten all of it away.
_PUNCT = str.maketrans("", "", "_-, .()[]'\"")

# Tokens (post-normalisation) that identify a bell-pepper directory.
_PEPPER_TOKENS = ("pepperbell", "bellpepper", "capsicumannuum")

# Substrings that mark a folder as "healthy" (class 0).
_HEALTHY_TOKENS = ("healthy", "undamaged", "normal")

# Controlled vocabulary from LABELS.md, normalised. Used only to warn about
# folders whose condition slug looks like a typo — never to reject images.
_KNOWN_CONDITIONS = (
    "healthy", "undamaged", "normal",
    "bacterialspot", "cercosporaleafspot", "powderymildew",
    "phytophthorablight", "anthracnose", "mosaicvirus", "leafcurlvirus",
    "aphiddamage", "thripsdamage", "spidermitedamage", "whiteflydamage",
    "leafminer", "nitrogendeficiency", "potassiumdeficiency",
    "magnesiumdeficiency", "calciumdeficiency", "sunscald",
    "physicaldamage", "herbicideinjury",
)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# Feature-extraction hyper-parameters. Keeping them here (rather than as bare
# literals) makes the descriptor reproducible between training and inference.
IMG_SIZE = (256, 256)     # resize target (W, H) before feature extraction
HSV_BINS = (16, 16, 16)   # histogram bins per H, S, V channel
LBP_RADIUS = 3
LBP_POINTS = 8 * LBP_RADIUS
LBP_METHOD = "uniform"


# --------------------------------------------------------------------------- #
# Stage 1: dataset parsing & filtering
# --------------------------------------------------------------------------- #

def _normalise(name: str) -> str:
    """Lower-case a folder name and strip punctuation for robust matching."""
    return name.lower().translate(_PUNCT)


def _is_pepper_folder(folder_name: str) -> bool:
    """
    True if the directory name refers to bell peppers, any separator style.

    Normalising punctuation away means one token list covers every variant we
    have seen in the wild: ``Pepper__bell``, ``Pepper_bell``, ``Pepper Bell``,
    ``Pepper,_bell``, ``bell-pepper`` and ``Capsicum annuum``.
    """
    normalised = _normalise(folder_name)
    return any(tok in normalised for tok in _PEPPER_TOKENS)


def _label_for_folder(folder_name: str) -> int:
    """Map a bell-pepper subfolder to CLASS_UNDAMAGED (0) or CLASS_DAMAGED (1)."""
    normalised = _normalise(folder_name)
    if any(tok in normalised for tok in _HEALTHY_TOKENS):
        return CLASS_UNDAMAGED
    # Everything that is a pepper folder but not "healthy" is treated as damaged
    # (Bacterial_spot and any other symptom / pest / deficiency folder).
    return CLASS_DAMAGED


@dataclass
class Dataset:
    """Container for the parsed image paths, labels and provenance."""

    paths: List[str] = field(default_factory=list)
    labels: List[int] = field(default_factory=list)
    source_folders: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.paths)

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "path": self.paths,
                "label": self.labels,
                "source_folder": self.source_folders,
            }
        )


def parse_dataset(data_dir: str) -> Dataset:
    """
    Scan ``data_dir`` recursively and collect only bell-pepper leaf images.

    The scanner walks the tree; whenever it encounters a directory whose name
    identifies bell peppers, every image inside it is assigned a class label
    based on the healthy/symptom heuristic.

    Parameters
    ----------
    data_dir : str
        Root directory holding the (nested) dataset folders.

    Returns
    -------
    Dataset
        Parsed paths, integer labels and the originating folder names.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir!r}")

    dataset = Dataset()

    for root, _dirs, files in os.walk(data_dir):
        folder_name = os.path.basename(os.path.normpath(root))
        if not _is_pepper_folder(folder_name):
            continue

        label = _label_for_folder(folder_name)
        images = [f for f in files if f.lower().endswith(IMAGE_EXTENSIONS)]
        for fname in images:
            dataset.paths.append(os.path.join(root, fname))
            dataset.labels.append(label)
            dataset.source_folders.append(folder_name)

    if len(dataset) == 0:
        raise RuntimeError(
            f"No bell-pepper images found under {data_dir!r}. Expected folders "
            f"such as 'Pepper__bell___healthy' and 'Pepper__bell___Bacterial_spot'."
        )

    return dataset


def describe_distribution(dataset: Dataset, title: str = "Dataset") -> None:
    """Pretty-print the per-class and per-folder image distribution."""
    df = dataset.as_frame()
    print(f"\n{'=' * 60}")
    print(f"{title}: {len(df)} images")
    print(f"{'=' * 60}")

    print("\nClass distribution:")
    counts = df["label"].value_counts().sort_index()
    for label, count in counts.items():
        pct = 100.0 * count / len(df)
        print(f"  [{label}] {CLASS_NAMES[label]:<22} : {count:5d}  ({pct:5.1f}%)")

    print("\nSource folders contributing images:")
    folder_counts = (
        df.groupby(["source_folder", "label"]).size().reset_index(name="count")
    )
    for _, row in folder_counts.iterrows():
        print(
            f"  {row['source_folder']:<40} -> class {row['label']} "
            f"({row['count']} imgs)"
        )

    _warn_on_unknown_conditions(df)


def _warn_on_unknown_conditions(df: pd.DataFrame) -> None:
    """
    Flag folders whose condition slug is not in the LABELS.md vocabulary.

    A typo like ``Pepper_bell___helthy`` would otherwise be silently swept into
    class 1 (Damaged), quietly poisoning the training set. Warn loudly instead.
    """
    unknown = sorted(
        {
            folder
            for folder in df["source_folder"].unique()
            if not any(tok in _normalise(folder) for tok in _KNOWN_CONDITIONS)
        }
    )
    if unknown:
        print(
            "\n[warning] These folders matched 'bell pepper' but their condition "
            "slug is not in the LABELS.md vocabulary.\n"
            "          They default to class 1 (Damaged) — check for typos:"
        )
        for folder in unknown:
            print(f"            - {folder}")


# --------------------------------------------------------------------------- #
# Stage 2: feature extraction (scikit-image + OpenCV)
# --------------------------------------------------------------------------- #

def load_image(path: str) -> np.ndarray:
    """Load an image as an RGB uint8 array, raising on failure."""
    # cv2 reads BGR; convert to RGB so color semantics are intuitive downstream.
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {path!r}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _hsv_color_histogram(rgb: np.ndarray) -> np.ndarray:
    """
    Compute a normalised 3D HSV color histogram, flattened to 1D.

    HSV separates chromaticity (H, S) from brightness (V), which makes the
    yellow-brown halos of Bacterial Spot far more separable than in RGB.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist(
        [hsv],
        channels=[0, 1, 2],
        mask=None,
        histSize=list(HSV_BINS),
        # OpenCV HSV ranges: H in [0,180), S and V in [0,256).
        ranges=[0, 180, 0, 256, 0, 256],
    )
    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype(np.float32)


def _lbp_texture_histogram(rgb: np.ndarray) -> np.ndarray:
    """
    Compute a normalised Local Binary Pattern histogram, flattened to 1D.

    LBP encodes local micro-texture; lesions and necrotic tissue change the
    surface roughness relative to smooth healthy leaf tissue.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, P=LBP_POINTS, R=LBP_RADIUS, method=LBP_METHOD)
    # "uniform" method yields LBP_POINTS + 2 distinct codes.
    n_bins = LBP_POINTS + 2
    hist, _ = np.histogram(
        lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True
    )
    return hist.astype(np.float32)


def extract_features(image_or_path) -> np.ndarray:
    """
    Extract the full feature vector for a single image.

    Accepts either a file path (str) or an already-loaded RGB uint8 array, so
    the same function serves both batch training and single-image inference.

    Returns
    -------
    np.ndarray
        1D concatenation of [HSV color histogram | LBP texture histogram].
    """
    rgb = load_image(image_or_path) if isinstance(image_or_path, str) else image_or_path
    rgb = cv2.resize(rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)

    color_feat = _hsv_color_histogram(rgb)
    texture_feat = _lbp_texture_histogram(rgb)
    return np.concatenate([color_feat, texture_feat])


def build_feature_matrix(dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Turn a :class:`Dataset` into an (X, y) pair suitable for scikit-learn.

    Images that fail to load are skipped with a warning rather than aborting
    the whole run, which is common with large agricultural dumps.
    """
    features: List[np.ndarray] = []
    labels: List[int] = []
    n_total = len(dataset)

    print(f"\nExtracting features from {n_total} images ...")
    for i, (path, label) in enumerate(zip(dataset.paths, dataset.labels), start=1):
        try:
            features.append(extract_features(path))
            labels.append(label)
        except Exception as exc:  # noqa: BLE001 - keep the batch resilient
            print(f"  [skip] {path}: {exc}", file=sys.stderr)

        if i % 200 == 0 or i == n_total:
            print(f"  processed {i}/{n_total}")

    if not features:
        raise RuntimeError("Feature extraction produced no vectors.")

    X = np.vstack(features)
    y = np.asarray(labels, dtype=int)
    print(f"Feature matrix ready: X={X.shape}, y={y.shape}")
    return X, y


# --------------------------------------------------------------------------- #
# Stage 3 + 4: training & evaluation
# --------------------------------------------------------------------------- #

def build_models() -> dict:
    """Return the baseline estimators keyed by name."""
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        # SVMs need scaled inputs; wrap in a Pipeline so scaling travels with
        # the model when persisted (important for consistent inference).
        # SVC(probability=True) is deprecated in scikit-learn 1.9 and removed in
        # 1.11, so calibrate explicitly to get predict_proba for the UI.
        "SVM": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "svc",
                    CalibratedClassifierCV(
                        SVC(
                            kernel="rbf",
                            C=10.0,
                            gamma="scale",
                            class_weight="balanced",
                            random_state=42,
                        ),
                        ensemble=False,
                    ),
                ),
            ]
        ),
    }


def plot_confusion_matrix(cm: np.ndarray, model_name: str, out_dir: str) -> str:
    """Render the confusion matrix to a PNG and return the written path."""
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, cmap="Greens")
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=range(len(CLASS_NAMES)),
        yticks=range(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted",
        ylabel="True",
        title=f"Confusion Matrix — {model_name}",
    )
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    # Annotate each cell, flipping text colour on dark squares for contrast.
    threshold = cm.max() / 2.0 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"confusion_matrix_{model_name.lower()}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def evaluate_model(name: str, model, X_test, y_test, report_dir: str = None) -> float:
    """Print a classification report + confusion matrix; return macro F1."""
    y_pred = model.predict(X_test)

    print(f"\n{'-' * 60}")
    print(f"Evaluation: {name}")
    print(f"{'-' * 60}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test, y_pred, target_names=CLASS_NAMES, zero_division=0
        )
    )
    print("Confusion matrix (rows = true, cols = predicted):")
    cm = confusion_matrix(y_test, y_pred, labels=[CLASS_UNDAMAGED, CLASS_DAMAGED])
    header = "            " + "".join(f"{n[:10]:>12}" for n in CLASS_NAMES)
    print(header)
    for i, row in enumerate(cm):
        print(f"{CLASS_NAMES[i][:10]:>12}" + "".join(f"{v:>12}" for v in row))

    if report_dir:
        png = plot_confusion_matrix(cm, name, report_dir)
        print(f"\nSaved confusion matrix plot -> {png}")

    return f1_score(y_test, y_pred, average="macro", zero_division=0)


def train_and_select(
    X, y, test_size: float = 0.20, random_state: int = 42, report_dir: str = None
):
    """
    Train all baselines on an 80/20 stratified split and return the best.

    Returns
    -------
    tuple
        (best_name, best_model, best_macro_f1)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(
        f"\nSplit: {len(X_train)} train / {len(X_test)} test "
        f"(stratified, test_size={test_size})"
    )

    best_name, best_model, best_f1 = None, None, -1.0
    for name, model in build_models().items():
        print(f"\nTraining {name} ...")
        model.fit(X_train, y_train)
        macro_f1 = evaluate_model(name, model, X_test, y_test, report_dir=report_dir)
        print(f"{name} macro-F1 = {macro_f1:.4f}")

        if macro_f1 > best_f1:
            best_name, best_model, best_f1 = name, model, macro_f1

    print(f"\nBest model: {best_name} (macro-F1 = {best_f1:.4f})")
    return best_name, best_model, best_f1


def save_model(model, model_name: str, out_path: str) -> None:
    """Persist the model + metadata needed for consistent inference."""
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)

    bundle = {
        "model": model,
        "model_name": model_name,
        "class_names": CLASS_NAMES,
        "feature_config": {
            "img_size": IMG_SIZE,
            "hsv_bins": HSV_BINS,
            "lbp_radius": LBP_RADIUS,
            "lbp_points": LBP_POINTS,
            "lbp_method": LBP_METHOD,
        },
    }
    dump(bundle, out_path)
    print(f"\nSaved best model bundle -> {out_path}")


# --------------------------------------------------------------------------- #
# Stage 5: CLI
# --------------------------------------------------------------------------- #

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the Bell Pepper Leaf Damage Identifier baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", default="data", help="Root directory containing the dataset."
    )
    parser.add_argument(
        "--model-out",
        default="models/best_pepper_model.joblib",
        help="Where to save the best-performing model bundle.",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.20, help="Held-out test fraction."
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Reproducibility seed."
    )
    parser.add_argument(
        "--report-dir",
        default="reports",
        help="Directory for confusion-matrix PNGs.",
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Parse and print the class distribution, then exit without training.",
    )
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    print("Bell Pepper Leaf Damage Identifier — training pipeline")
    dataset = parse_dataset(args.data_dir)
    describe_distribution(dataset, title="Full dataset (bell pepper only)")

    if args.scan_only:
        print("\n--scan-only set: stopping before feature extraction.")
        return 0

    X, y = build_feature_matrix(dataset)
    best_name, best_model, _ = train_and_select(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        report_dir=args.report_dir,
    )
    save_model(best_model, best_name, args.model_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
