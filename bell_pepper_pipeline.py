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

Both are computed over *leaf pixels only*. Studio datasets shoot leaves against
a flat backdrop, and a whole-image histogram will happily learn that backdrop
instead of the leaf — scoring well on the benchmark while flipping its verdict
the moment lighting or background changes. Segmenting first is what makes the
prediction depend on the plant.

Stages
------
    1. Dataset parsing & filtering (bell-pepper folders only)
    2. Preprocessing: resize -> segment leaf -> white balance -> normalise
       exposure  (order matters; see ``preprocess``)
    3. Feature extraction (masked HSV histogram + masked LBP), per image
    4. Model training, selected on group-aware k-fold cross-validated macro-F1
    5. Evaluation: classification report, confusion matrix, and a consistency
       score under label-preserving transforms
    6. Persistence of the refit model as a version-stamped .joblib bundle

This module is also imported by ``predict_pepper.py`` so that training and
inference share *exactly* the same feature-extraction code.

Usage
-----
    python bell_pepper_pipeline.py --data-dir data
    python bell_pepper_pipeline.py --data-dir data --augment --models RandomForest
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
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
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

# Descriptor version. Bumped whenever the feature vector changes meaning, so a
# stale model bundle can never be silently fed vectors it was not trained on —
# the single worst source of train/inference inconsistency.
#   1 = whole-image HSV histogram + whole-image LBP (background included)
#   2 = white-balanced, leaf-segmented HSV + LBP (background excluded)
FEATURE_VERSION = 2

# --- Consistency controls -------------------------------------------------- #
# The v1 descriptor was computed over the entire frame, so the uniform grey
# PlantVillage background leaked straight into the colour histogram. Any change
# of backdrop, exposure or white balance therefore moved the feature vector and
# could flip the verdict. v2 removes both shortcuts:
#
#   1. Saturation-Otsu leaf mask  -> background pixels never reach either
#                                    histogram, so only leaf tissue is scored.
#   2. Background-referenced WB   -> neutralises colour casts between cameras,
#                                    using the (known-neutral) backdrop as the
#                                    white reference rather than the whole frame.
WHITE_BALANCE = True
SEGMENT_LEAF = True
# White balance equalises the channels *relative to each other* but preserves
# the overall exposure level, so a photo taken 25% darker still lands in
# different V bins. Rescaling the value channel to a fixed mean over the leaf
# closes that gap; relative light/dark structure within the leaf (what actually
# marks necrotic tissue) is untouched.
NORMALISE_LUMA = True
LUMA_TARGET = 128.0

# A mask this small or this large is implausible for a leaf photo and usually
# means segmentation failed (e.g. a near-monochrome image). Fall back to the
# full frame rather than feeding the classifier a handful of stray pixels.
MASK_MIN_COVERAGE = 0.02
MASK_MAX_COVERAGE = 0.995
# Kernel for the morphological close/open that removes lesion pinholes and
# speckle from the raw threshold.
MASK_MORPH_KERNEL = 7
# Connected components at least this fraction of the largest one are kept, so a
# leaf split by a lesion survives intact while speckle is discarded.
MASK_COMPONENT_RATIO = 0.15


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


def white_balance(rgb: np.ndarray, reference_mask: np.ndarray = None) -> np.ndarray:
    """
    Neutralise a colour cast by forcing a reference region to equal channel means.

    ``reference_mask`` selects pixels that *should* be neutral — in these studio
    datasets, the backdrop. Using the background as the white reference is much
    safer than plain grey-world: grey-world assumes the whole frame averages to
    grey, which a leaf-filled frame flatly violates, so it bleeds the leaf's own
    greenness into the correction and partly cancels the signal we classify on.

    With no reference mask (or too few reference pixels to be reliable) this
    falls back to grey-world over the whole frame.
    """
    out = rgb.astype(np.float32)

    if reference_mask is not None and np.count_nonzero(reference_mask) >= 0.05 * reference_mask.size:
        means = out[reference_mask > 0].reshape(-1, 3).mean(axis=0)
    else:
        means = out.reshape(-1, 3).mean(axis=0)

    grey = float(means.mean())
    # A channel mean of zero only happens on degenerate images; leave those be.
    if grey <= 0 or np.any(means <= 0):
        return rgb
    out *= grey / means
    return np.clip(out, 0, 255).astype(np.uint8)


def segment_leaf(rgb: np.ndarray) -> np.ndarray:
    """
    Return a uint8 mask (255 = leaf tissue) isolating the leaf from the backdrop.

    Studio plant-disease datasets shoot a single detached leaf against a flat,
    near-neutral background. Neutral means *low saturation*, while leaf tissue —
    green, chlorotic yellow or necrotic brown alike — is comparatively saturated.
    Otsu on the saturation channel therefore separates the two without needing a
    hand-tuned hue window that would exclude diseased tissue.

    Morphological closing then opening fills lesion pinholes and drops speckle,
    and only the largest connected component is kept so stray corner blobs do not
    contribute pixels. If the result is implausible as a leaf, an all-ones mask
    is returned so the caller degrades to whole-image behaviour rather than
    scoring noise.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]

    _thresh, mask = cv2.threshold(
        saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MASK_MORPH_KERNEL, MASK_MORPH_KERNEL)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Drop speckle, but keep every substantial blob: a leaf bisected by a large
    # necrotic lesion legitimately fragments, and taking only the single largest
    # component would silently throw away half the tissue we need to score.
    n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    if n_labels > 1:
        # Row 0 is the background component; measure against the biggest of the rest.
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = 1 + np.flatnonzero(areas >= MASK_COMPONENT_RATIO * areas.max())
        mask = np.where(np.isin(labels, keep), 255, 0).astype(np.uint8)

    # Fill interior holes so lesion centres — the most diagnostic pixels on the
    # leaf — are not punched out of the very mask meant to select leaf tissue.
    filled = mask.copy()
    flood = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), np.uint8)
    cv2.floodFill(filled, flood, (0, 0), 255)
    mask = mask | cv2.bitwise_not(filled)

    coverage = float(np.count_nonzero(mask)) / mask.size
    if not (MASK_MIN_COVERAGE <= coverage <= MASK_MAX_COVERAGE):
        return np.full(mask.shape, 255, dtype=np.uint8)
    return mask


def normalise_luma(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Rescale brightness so the mean value *inside the mask* hits LUMA_TARGET.

    Measuring the mean over leaf pixels only matters: a bright background would
    otherwise drag the correction around and reintroduce the very backdrop
    dependence the mask exists to remove.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    value = hsv[:, :, 2]

    selected = value[mask > 0]
    if selected.size == 0:
        selected = value.ravel()
    mean_v = float(selected.mean())
    if mean_v <= 1.0:  # essentially black; scaling would just amplify noise
        return rgb

    hsv[:, :, 2] = np.clip(value * (LUMA_TARGET / mean_v), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def preprocess(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resize, white-balance, segment and exposure-normalise an RGB image.

    Returns ``(rgb_resized, mask)``. Centralised so training, inference and the
    consistency harness cannot drift apart — the descriptor is only reproducible
    if every caller runs exactly these steps in exactly this order.
    """
    rgb = cv2.resize(rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)

    # Segment FIRST, on the untouched image. The mask keys on saturation, and
    # any white-balance pass applied beforehand shifts the neutral backdrop off
    # neutral — which hands the background saturation and shreds the mask.
    mask = (
        segment_leaf(rgb)
        if SEGMENT_LEAF
        else np.full(rgb.shape[:2], 255, dtype=np.uint8)
    )

    # With the leaf located, the complement is a known-neutral white reference.
    if WHITE_BALANCE:
        rgb = white_balance(rgb, reference_mask=cv2.bitwise_not(mask))

    if NORMALISE_LUMA:
        rgb = normalise_luma(rgb, mask)
    return rgb, mask


def _hsv_color_histogram(rgb: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """
    Compute a normalised 3D HSV color histogram, flattened to 1D.

    HSV separates chromaticity (H, S) from brightness (V), which makes the
    yellow-brown halos of Bacterial Spot far more separable than in RGB. When
    ``mask`` is supplied only leaf pixels are counted, so the backdrop cannot
    contribute to the descriptor.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist(
        [hsv],
        channels=[0, 1, 2],
        mask=mask,
        histSize=list(HSV_BINS),
        # OpenCV HSV ranges: H in [0,180), S and V in [0,256).
        ranges=[0, 180, 0, 256, 0, 256],
    )
    # L1-normalise to a distribution: unlike cv2.normalize's default max-scaling,
    # this makes the vector invariant to how many leaf pixels the mask kept, so
    # a tightly cropped photo and a wide one land in the same place.
    total = float(hist.sum())
    if total > 0:
        hist /= total
    return hist.flatten().astype(np.float32)


def _lbp_texture_histogram(rgb: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """
    Compute a normalised Local Binary Pattern histogram, flattened to 1D.

    LBP encodes local micro-texture; lesions and necrotic tissue change the
    surface roughness relative to smooth healthy leaf tissue. With ``mask`` set,
    only codes computed at leaf pixels are histogrammed — otherwise the flat
    background would dominate the "uniform" bins and wash out the signal.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, P=LBP_POINTS, R=LBP_RADIUS, method=LBP_METHOD)
    # "uniform" method yields LBP_POINTS + 2 distinct codes.
    n_bins = LBP_POINTS + 2

    values = lbp[mask > 0] if mask is not None else lbp.ravel()
    if values.size == 0:
        values = lbp.ravel()

    hist, _ = np.histogram(values, bins=n_bins, range=(0, n_bins), density=True)
    return np.nan_to_num(hist).astype(np.float32)


def extract_features(image_or_path) -> np.ndarray:
    """
    Extract the full feature vector for a single image.

    Accepts either a file path (str) or an already-loaded RGB uint8 array, so
    the same function serves both batch training and single-image inference.

    Returns
    -------
    np.ndarray
        1D concatenation of [HSV color histogram | LBP texture histogram],
        computed over leaf pixels only (see :data:`FEATURE_VERSION`).
    """
    rgb = load_image(image_or_path) if isinstance(image_or_path, str) else image_or_path
    rgb, mask = preprocess(rgb)

    color_feat = _hsv_color_histogram(rgb, mask)
    texture_feat = _lbp_texture_histogram(rgb, mask)
    return np.concatenate([color_feat, texture_feat])


def build_feature_matrix(
    dataset: Dataset, augment: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Turn a :class:`Dataset` into an (X, y, groups) triple for scikit-learn.

    Images that fail to load are skipped with a warning rather than aborting
    the whole run, which is common with large agricultural dumps.

    With ``augment`` set, each image also contributes photometric variants (see
    :data:`AUGMENT_TRANSFORMS`) so the classifier is taught that a warmer, dimmer
    or more heavily compressed photo of a leaf is still the same leaf.

    ``groups`` holds the index of the source image for every row. Augmented
    copies share their original's group, which lets cross-validation keep them on
    the same side of every fold — otherwise a rotated near-duplicate of a training
    image lands in the test set and the reported score is inflated by leakage.
    """
    features: List[np.ndarray] = []
    labels: List[int] = []
    groups: List[int] = []
    n_total = len(dataset)

    variant_note = f" (+{len(AUGMENT_TRANSFORMS)} augmented each)" if augment else ""
    print(f"\nExtracting features from {n_total} images{variant_note} ...")

    for i, (path, label) in enumerate(zip(dataset.paths, dataset.labels), start=1):
        try:
            rgb = load_image(path)
            features.append(extract_features(rgb))
            labels.append(label)
            groups.append(i - 1)

            if augment:
                for transform in AUGMENT_TRANSFORMS.values():
                    features.append(extract_features(transform(rgb)))
                    labels.append(label)
                    groups.append(i - 1)
        except Exception as exc:  # noqa: BLE001 - keep the batch resilient
            print(f"  [skip] {path}: {exc}", file=sys.stderr)

        if i % 200 == 0 or i == n_total:
            print(f"  processed {i}/{n_total}")

    if not features:
        raise RuntimeError("Feature extraction produced no vectors.")

    X = np.vstack(features)
    y = np.asarray(labels, dtype=int)
    g = np.asarray(groups, dtype=int)
    print(f"Feature matrix ready: X={X.shape}, y={y.shape}, groups={len(set(groups))}")
    return X, y, g


# --------------------------------------------------------------------------- #
# Stage 2b: consistency harness
# --------------------------------------------------------------------------- #
#
# Accuracy on a held-out split says nothing about whether the *same leaf*, shot
# a little differently, keeps its verdict. These transforms are all
# label-preserving — a rotated, dimmer, re-compressed photo of a diseased leaf
# is still a diseased leaf — so any change in prediction is pure instability.

def _adjust_brightness(rgb: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(rgb.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def _adjust_gamma(rgb: np.ndarray, gamma: float) -> np.ndarray:
    table = ((np.arange(256) / 255.0) ** gamma * 255.0).astype(np.uint8)
    return cv2.LUT(rgb, table)


def _jpeg_recompress(rgb: np.ndarray, quality: int) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return rgb
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def _color_cast(rgb: np.ndarray, scales: Tuple[float, float, float]) -> np.ndarray:
    """Simulate a different camera white balance by scaling R, G, B."""
    out = rgb.astype(np.float32) * np.asarray(scales, dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def _rotate_arbitrary(rgb: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate about the centre, replicating the border to avoid black corners."""
    h, w = rgb.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), degrees, 1.0)
    return cv2.warpAffine(rgb, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _centre_zoom(rgb: np.ndarray, keep: float) -> np.ndarray:
    """Centre-crop to ``keep`` of each side, then resize back — a framing change."""
    h, w = rgb.shape[:2]
    ch, cw = int(h * keep), int(w * keep)
    top, left = (h - ch) // 2, (w - cw) // 2
    return cv2.resize(
        rgb[top:top + ch, left:left + cw], (w, h), interpolation=cv2.INTER_AREA
    )


# Photometric transforms used to AUGMENT TRAINING. Kept separate from the
# evaluation set below so it stays clear which nuisances the model was taught.
AUGMENT_TRANSFORMS = {
    "aug_gamma_lo": lambda x: _adjust_gamma(x, 0.75),
    "aug_gamma_hi": lambda x: _adjust_gamma(x, 1.35),
    "aug_warm": lambda x: _color_cast(x, (1.12, 1.0, 0.88)),
    "aug_cool": lambda x: _color_cast(x, (0.88, 1.0, 1.12)),
    "aug_jpeg": lambda x: _jpeg_recompress(x, 40),
    "aug_blur": lambda x: cv2.GaussianBlur(x, (5, 5), 0),
}

# Transforms in the consistency harness that are deliberately NOT trained on, so
# the score stays an honest test of generalised stability even after augmentation
# rather than a check that training memorised its own augmentations.
HELDOUT_VARIANTS = ("rot17", "zoom_0.85", "jpeg_q25")


def consistency_variants(rgb: np.ndarray) -> dict:
    """
    Build the label-preserving variants of one image, keyed by name.

    Covers what realistically differs between two photos of the same leaf:
    orientation, framing, exposure, camera colour response, and compression.
    Names in :data:`HELDOUT_VARIANTS` are never used to augment training.
    """
    return {
        "identity": rgb,
        "rot90": np.rot90(rgb, 1).copy(),
        "rot180": np.rot90(rgb, 2).copy(),
        "rot270": np.rot90(rgb, 3).copy(),
        "flip_h": np.fliplr(rgb).copy(),
        "flip_v": np.flipud(rgb).copy(),
        "bright_x0.75": _adjust_brightness(rgb, 0.75),
        "bright_x1.25": _adjust_brightness(rgb, 1.25),
        "gamma_0.75": _adjust_gamma(rgb, 0.75),
        "gamma_1.35": _adjust_gamma(rgb, 1.35),
        "warm_cast": _color_cast(rgb, (1.12, 1.0, 0.88)),
        "cool_cast": _color_cast(rgb, (0.88, 1.0, 1.12)),
        "jpeg_q40": _jpeg_recompress(rgb, 40),
        "blur": cv2.GaussianBlur(rgb, (5, 5), 0),
        # --- held out from training augmentation ---
        "rot17": _rotate_arbitrary(rgb, 17.0),
        "zoom_0.85": _centre_zoom(rgb, 0.85),
        "jpeg_q25": _jpeg_recompress(rgb, 25),
    }


def evaluate_consistency(model, paths: List[str], sample: int = 200,
                         random_state: int = 42) -> dict:
    """
    Measure how stable the model's verdict is under label-preserving change.

    For each sampled image every variant is classified. The headline number is
    the fraction of images whose verdict is identical across *all* variants —
    the practical question of "will this app give me the same answer if I retake
    the photo slightly differently".

    Returns a dict with the aggregate scores plus a per-transform flip rate, so
    it is obvious *which* nuisance factor the model is still sensitive to.
    """
    rng = np.random.default_rng(random_state)
    if len(paths) > sample:
        chosen = [paths[i] for i in rng.choice(len(paths), sample, replace=False)]
    else:
        chosen = list(paths)

    names = list(consistency_variants(np.zeros((8, 8, 3), np.uint8)).keys())
    flips = {n: 0 for n in names}
    stable_images = 0
    prob_spreads: List[float] = []
    n_scored = 0

    print(f"\nConsistency check on {len(chosen)} images x {len(names)} variants ...")
    for path in chosen:
        try:
            rgb = load_image(path)
        except Exception:  # noqa: BLE001 - a bad file should not abort the run
            continue

        variants = consistency_variants(rgb)
        feats = np.vstack([extract_features(v) for v in variants.values()])
        preds = model.predict(feats)

        baseline = preds[0]  # the "identity" variant leads the dict
        for name, pred in zip(names, preds):
            if pred != baseline:
                flips[name] += 1
        if np.all(preds == baseline):
            stable_images += 1

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(feats)
            classes = list(model.classes_)
            prob_spreads.append(
                float(np.std(proba[:, classes.index(CLASS_DAMAGED)]))
            )
        n_scored += 1

    if n_scored == 0:
        raise RuntimeError("Consistency check scored no images.")

    result = {
        "n_images": n_scored,
        "n_variants": len(names),
        "stable_fraction": stable_images / n_scored,
        "mean_prob_std": float(np.mean(prob_spreads)) if prob_spreads else None,
        "flip_rate_by_transform": {
            n: flips[n] / n_scored for n in names if n != "identity"
        },
    }
    return result


def print_consistency(result: dict) -> None:
    """Pretty-print the output of :func:`evaluate_consistency`."""
    print(f"\n{'-' * 60}")
    print("Consistency under label-preserving transforms")
    print(f"{'-' * 60}")
    print(
        f"Images fully stable across all {result['n_variants']} variants: "
        f"{result['stable_fraction']:.1%}  ({result['n_images']} images)"
    )
    if result["mean_prob_std"] is not None:
        print(f"Mean std of P(Damaged) across variants : {result['mean_prob_std']:.4f}")

    print("\nVerdict flip rate per transform (lower is better):")
    for name, rate in sorted(
        result["flip_rate_by_transform"].items(), key=lambda kv: -kv[1]
    ):
        bar = "#" * int(round(rate * 40))
        print(f"  {name:<14} {rate:>6.1%}  {bar}")


# --------------------------------------------------------------------------- #
# Stage 3 + 4: training & evaluation
# --------------------------------------------------------------------------- #

def build_models(only: List[str] = None) -> dict:
    """
    Return the baseline estimators keyed by name.

    ``only`` filters the selection. The RBF SVM is O(n^2)-ish in the sample count,
    so on an augmented set (7x the rows) it dominates runtime — being able to
    train just the RandomForest keeps that run practical.
    """
    models = {
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

    if only:
        missing = [n for n in only if n not in models]
        if missing:
            raise ValueError(
                f"Unknown model(s): {missing}. Available: {sorted(models)}"
            )
        models = {n: models[n] for n in only}
    return models


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


def cross_validated_f1(model, X, y, groups=None, folds: int = 5,
                       random_state: int = 42):
    """
    Return ``(mean, std)`` macro-F1 over stratified k-fold cross-validation.

    A single 80/20 split is a sample of size one: rerun it with another seed and
    the headline number moves. Averaging over k folds — and reporting the spread
    — makes the reported score itself consistent, which is a precondition for
    trusting any comparison between the two estimators.

    When ``groups`` is supplied, folds are group-aware so augmented copies never
    straddle the train/test boundary. Without that, every augmented variant of a
    training image would appear in the test fold and the score would measure
    memorisation rather than generalisation.
    """
    if groups is not None and len(set(groups)) < len(groups):
        cv = StratifiedGroupKFold(
            n_splits=folds, shuffle=True, random_state=random_state
        )
        scores = cross_val_score(
            model, X, y, groups=groups, cv=cv, scoring="f1_macro", n_jobs=-1
        )
    else:
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
        scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
    return float(scores.mean()), float(scores.std())


def train_and_select(
    X,
    y,
    groups=None,
    test_size: float = 0.20,
    random_state: int = 42,
    report_dir: str = None,
    cv_folds: int = 5,
    only_models: List[str] = None,
):
    """
    Train all baselines, select on cross-validated macro-F1, and refit on all data.

    Selection uses the k-fold mean rather than the single held-out split so the
    winner is not an artefact of one lucky partition. The held-out split is still
    evaluated and printed, because a confusion matrix on unseen images is what
    makes the error profile legible.

    Returns
    -------
    tuple
        (best_name, best_model, best_macro_f1, metrics_by_model)
    """
    if groups is not None and len(set(groups)) < len(groups):
        # Split on groups so an image and its augmented copies stay together.
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        split_note = "grouped"
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        split_note = "stratified"

    print(
        f"\nSplit: {len(X_train)} train / {len(X_test)} test "
        f"({split_note}, test_size={test_size})"
    )

    best_name, best_model, best_cv, metrics = None, None, -1.0, {}
    for name, model in build_models(only=only_models).items():
        print(f"\nTraining {name} ...")

        cv_mean, cv_std = cross_validated_f1(
            model, X, y, groups=groups, folds=cv_folds, random_state=random_state
        )
        print(f"{name} {cv_folds}-fold macro-F1 = {cv_mean:.4f} +/- {cv_std:.4f}")

        model.fit(X_train, y_train)
        holdout_f1 = evaluate_model(name, model, X_test, y_test, report_dir=report_dir)
        print(f"{name} hold-out macro-F1 = {holdout_f1:.4f}")

        metrics[name] = {
            "cv_macro_f1_mean": cv_mean,
            "cv_macro_f1_std": cv_std,
            "holdout_macro_f1": holdout_f1,
        }
        if cv_mean > best_cv:
            best_name, best_model, best_cv = name, model, cv_mean

    print(f"\nBest model: {best_name} ({cv_folds}-fold macro-F1 = {best_cv:.4f})")

    # Refit the winner on the full dataset: the split existed to estimate
    # performance, and holding 20% back from the shipped model costs accuracy
    # for no remaining benefit now that the estimate is in hand.
    print(f"Refitting {best_name} on all {len(X)} samples for the shipped model ...")
    best_model = build_models(only=only_models)[best_name]
    best_model.fit(X, y)

    return best_name, best_model, best_cv, metrics


def save_model(
    model,
    model_name: str,
    out_path: str,
    metrics: dict = None,
    consistency: dict = None,
    augmented: bool = False,
) -> None:
    """Persist the model + metadata needed for consistent inference."""
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)

    bundle = {
        "model": model,
        "model_name": model_name,
        "class_names": CLASS_NAMES,
        # Stamped so inference can refuse a bundle whose descriptor predates the
        # current one instead of silently scoring mismatched vectors.
        "feature_version": FEATURE_VERSION,
        "feature_config": {
            "img_size": IMG_SIZE,
            "hsv_bins": HSV_BINS,
            "lbp_radius": LBP_RADIUS,
            "lbp_points": LBP_POINTS,
            "lbp_method": LBP_METHOD,
            "white_balance": WHITE_BALANCE,
            "segment_leaf": SEGMENT_LEAF,
            "normalise_luma": NORMALISE_LUMA,
            "luma_target": LUMA_TARGET,
        },
        "training_config": {
            "augmented": augmented,
            "augment_transforms": sorted(AUGMENT_TRANSFORMS) if augmented else [],
        },
        "metrics": metrics or {},
        "consistency": consistency or {},
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
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Stratified k-fold count used to select the winning model.",
    )
    parser.add_argument(
        "--consistency-sample",
        type=int,
        default=200,
        help="Images to score in the consistency check (0 disables it).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Subset of estimators to train, e.g. --models RandomForest.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help=(
            "Add photometric variants of each training image so the model learns "
            "to ignore exposure, colour cast and compression."
        ),
    )
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    print("Bell Pepper Leaf Damage Identifier — training pipeline")
    print(f"Descriptor version {FEATURE_VERSION} "
          f"(white_balance={WHITE_BALANCE}, segment_leaf={SEGMENT_LEAF}, "
          f"normalise_luma={NORMALISE_LUMA})")

    dataset = parse_dataset(args.data_dir)
    describe_distribution(dataset, title="Full dataset (bell pepper only)")

    if args.scan_only:
        print("\n--scan-only set: stopping before feature extraction.")
        return 0

    X, y, groups = build_feature_matrix(dataset, augment=args.augment)
    best_name, best_model, _best_f1, metrics = train_and_select(
        X,
        y,
        groups=groups,
        test_size=args.test_size,
        random_state=args.random_state,
        report_dir=args.report_dir,
        cv_folds=args.cv_folds,
        only_models=args.models,
    )

    consistency = {}
    if args.consistency_sample > 0:
        consistency = evaluate_consistency(
            best_model,
            dataset.paths,
            sample=args.consistency_sample,
            random_state=args.random_state,
        )
        print_consistency(consistency)

    save_model(
        best_model,
        best_name,
        args.model_out,
        metrics=metrics,
        consistency=consistency,
        augmented=args.augment,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
