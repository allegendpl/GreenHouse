# GreenHouse

Can help farmers determine specifically what the type of damage is, and if this was an app it could then offer some solutions (i.e. which pesticide brand to buy)

The repo currently holds two independent classifiers, one per crop. They do not
share code or dependencies yet — pick the one you want to run.

| Track | Crop | Approach | Entry point |
|---|---|---|---|
| **Tomato** | *Solanum lycopersicum* | Keras CNN, 10-class disease ID | `streamlit run app/app.py` |
| **Bell pepper** | *Capsicum annuum* | scikit-learn + scikit-image, binary healthy/damaged | `streamlit run app.py` |

---

# 🍅 Tomato Leaf Disease Classifier

A 10-class Keras CNN trained on a tomato-leaf dataset at 224×224, covering:

`Bacterial_spot`, `Early_blight`, `Late_blight`, `Leaf_Mold`,
`Septoria_leaf_spot`, `Spider_mites`, `Target_Spot`,
`Tomato_Yellow_Leaf_Curl_Virus`, `Tomato_mosaic_virus`, `healthy`

| File | Purpose |
|---|---|
| `notebooks/plant_damage_classifier.ipynb` | Colab training notebook |
| `app/tomato_model.keras` | Trained model loaded by the app |
| `app/app.py` | Streamlit UI — upload a leaf, get top-3 predictions |

```bash
streamlit run app/app.py
```

Requires `tensorflow`, `streamlit`, `pillow`, `numpy`. Note this track is *not*
covered by the root `requirements.txt`, which is pepper-only.

---

# 🌿 Bell Pepper Leaf Damage Identifier

A baseline **binary classifier** (Healthy vs. Damaged) for bell pepper
(*Capsicum annuum*) leaves, built with classic computer vision — no deep
learning. Uses **scikit-image** for feature extraction and **scikit-learn** for
classification, with a **Streamlit** UI for testing single images.

| File | Purpose |
|---|---|
| `bell_pepper_pipeline.py` | Dataset parsing, feature extraction, training, evaluation |
| `predict_pepper.py` | CLI inference on one image |
| `app.py` | Streamlit web UI (drag-and-drop testing) |
| `LABELS.md` | Shared labeling standard for the team |
| `requirements.txt` | Dependencies |

---

## 1. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2. Download a dataset

Use **one** of these. The first is recommended — its folder names match this
pipeline's expectations exactly, with no renaming needed.

### Recommended: PlantVillage (emmarex)

**https://www.kaggle.com/datasets/emmarex/plantdisease**

Contains exactly the two folders you need:
- `Pepper__bell___Bacterial_spot`
- `Pepper__bell___healthy`

15 classes total across several crops; the pipeline auto-filters to pepper only.

### Alternative: PlantVillage full (abdallahalidev)

**https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset**

Larger, with `color/`, `grayscale/` and `segmented/` variants. Uses the
**comma** folder naming (`Pepper,_bell___Bacterial_spot`) — the parser handles
this. Use the `color/` subfolder.

### Alternative: New Plant Diseases Dataset (vipoooool)

**https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset**

Augmented PlantVillage with a pre-made `train/` + `valid/` split. Note: it is
already augmented, so the 80/20 split here may leak near-duplicates between
train and test — expect optimistic scores.

### Non-Kaggle mirror

Mendeley Data (original PlantVillage): **https://data.mendeley.com/datasets/tywbtsjrjv/1**

### Downloading via the Kaggle CLI

```bash
pip install kaggle
# Put your kaggle.json API token in ~/.kaggle/ first (Kaggle > Settings > API)
kaggle datasets download -d emmarex/plantdisease -p data/archives
```

---

## 3. Extract into `data/`

```bash
for z in data/archives/*.zip; do unzip -q -o "$z" -d data/raw/; done
```

No manual sorting needed. The parser walks `data/` recursively, keeps only
bell-pepper folders, and derives labels from folder names:

- folder contains `healthy` → **class 0 (Undamaged)**
- any other pepper folder → **class 1 (Damaged)**

Non-pepper folders (tomato, potato, …) are ignored automatically.

---

## 4. Check the parse before training

Confirm the right folders were found and labeled, without waiting for feature
extraction:

```bash
python bell_pepper_pipeline.py --data-dir data --scan-only
```

This prints per-class and per-folder counts, and warns about folders whose
condition slug isn't in the `LABELS.md` vocabulary (catches typos like
`helthy` that would otherwise be silently filed as Damaged).

---

## 5. Train

```bash
python bell_pepper_pipeline.py --data-dir data
```

Trains a `RandomForestClassifier` and an `SVC`, prints a `classification_report`
plus confusion matrix for each, saves confusion-matrix PNGs to `reports/`, and
writes the higher-macro-F1 model to `models/best_pepper_model.joblib`.

Useful flags: `--model-out`, `--report-dir`, `--test-size`, `--random-state`.

---

## 6. Test a single image

**Web UI** (shows a big HEALTHY / DAMAGED verdict first):

```bash
streamlit run app.py
```

Then open <http://localhost:8501>. The app runs **entirely locally** — bound to
`127.0.0.1`, no tunnel (ngrok/Colab), no Deploy button, no telemetry. See
`.streamlit/config.toml`. Dark theme is on by default.

Two ways to supply an image:
- **Upload** — drag and drop any JPG/PNG/BMP/TIF/WEBP from your machine.
- **Paste a path** — point at a file already on disk, e.g. `samples/healthy_leaf.jpg`.

**CLI:**

```bash
python predict_pepper.py --image samples/bacterial_spot_leaf.jpg
```

---

## Measured baseline results

Trained on the 2,475 PlantVillage bell-pepper images (1,478 healthy / 997
bacterial spot), 80/20 stratified split → 495 held-out test images:

| Model | Accuracy | Macro-F1 |
|---|---|---|
| **SVM** (RBF, calibrated) — selected | **0.9960** | **0.9958** |
| RandomForest | 0.9939 | 0.9937 |

RandomForest confusion matrix on the test split:

|  | Pred. Undamaged | Pred. Damaged |
|---|---|---|
| **True Undamaged** | 296 | 0 |
| **True Damaged** | 3 | 196 |

Both models are strongest at *not* false-alarming on healthy leaves; the few
errors are missed bacterial-spot leaves. For a screening tool you likely want
the opposite bias — see "Tuning" below.

### Tuning the healthy/damaged trade-off

Missing a diseased leaf costs more than a false alarm in most greenhouse
settings. To trade precision for recall on the Damaged class, threshold the
probability instead of using `predict`:

```python
proba = model.predict_proba(features)[0][1]   # P(Damaged)
label = 1 if proba > 0.35 else 0              # more sensitive than 0.5
```

---

## How it works

Each image is reduced to one 1-D feature vector:

1. **HSV color histogram** (16×16×16 bins, normalised) — HSV separates hue from
   brightness, which makes the yellow-brown chlorotic halos of bacterial spot
   far more separable than RGB.
2. **Local Binary Pattern histogram** (P=24, R=3, uniform) — captures the
   surface-roughness change caused by lesions and necrotic tissue.

Both are concatenated and fed to the classifiers. The SVM is wrapped in a
`Pipeline` with `StandardScaler` so scaling is persisted with the model.

Class imbalance is handled with `class_weight="balanced"`, and model selection
uses **macro-F1** rather than accuracy so a skewed test set doesn't flatter the
result.

---

## Caveats

- On PlantVillage alone, "Damaged" effectively means *bacterial spot*, since
  that is the only pepper disease class present. Add pest/deficiency folders
  (see `LABELS.md`) to broaden class 1 — no code change required.
- **The 99% above is not field accuracy.** PlantVillage images are single
  detached leaves shot against uniform grey backgrounds under even lighting.
  HSV color histograms are a global descriptor, so the model is partly keying
  on that consistent background. Expect a large drop on real greenhouse photos
  with soil, multiple overlapping leaves, and variable sun.
- **Bell pepper only.** There is no "not a pepper" class. Feed it a tomato,
  potato, or non-leaf image and it will still emit HEALTHY or DAMAGED with high
  apparent confidence. The output is meaningless outside *Capsicum annuum*.
- This is a screening baseline, not an agronomic diagnosis.
