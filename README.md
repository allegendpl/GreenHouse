# GreenHouse

Can help farmers determine specifically what the type of damage is, and if this was an app it could then offer some solutions (i.e. which pesticide brand to buy)

Both crops share **one** interface. Pick the crop in the sidebar:

```bash
streamlit run app.py
```

| Crop | Species | Model | Predicts |
|---|---|---|---|
| **Bell pepper** | *Capsicum annuum* | scikit-learn over HSV + LBP features | Healthy / Damaged |
| **Tomato** | *Solanum lycopersicum* | Keras CNN, 224×224 | 10 specific conditions |

The two models are entirely different underneath, so each crop supplies its own
loader and predictor and both return the same result shape. Everything after
that — the verdict banner, the probability breakdown, the image preview — is
shared, which is what keeps it a single UI rather than two bolted together.

Every prediction leads with the same **HEALTHY / DAMAGED** banner. For tomato the
specific condition (early blight, leaf mold, …) appears directly beneath it, so
one glance answers "is something wrong?" and the next line answers "what?".

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

Select **Tomato** in the sidebar of `app.py`. This half needs a deep-learning
runtime, deliberately **not** in `requirements.txt` — it is large and the pepper
half does not need it. `app.py` imports it lazily and shows an install prompt if
you pick Tomato without it:

```bash
pip install keras jax jaxlib
```

**Use Keras + JAX, not TensorFlow.** `pip install tensorflow` fails on this
project's interpreter with `No matching distribution found` — TensorFlow
publishes no wheels for Python 3.14. That is not a broken install: there is
simply nothing to download.

It doesn't matter, because `tomato_model.keras` was saved by **Keras 3.13** in
the backend-agnostic `.keras` format. Keras 3 runs on JAX, PyTorch or
TensorFlow and loads the identical file either way, so JAX (which *does* ship
3.14 wheels) is a drop-in substitute. `app.py` prefers standalone Keras and sets
`KERAS_BACKEND=jax`, falling back to `tensorflow.keras` if you already have TF.

---

# 🌿 Bell Pepper Leaf Damage Identifier

A **binary classifier** (Healthy vs. Damaged) for bell pepper (*Capsicum
annuum*) leaves, built with classic computer vision — no deep learning. Uses
**scikit-image** for feature extraction and **scikit-learn** for classification,
with a **Streamlit** UI for testing single images.

The pipeline optimises for **consistency**, not just accuracy: the leaf is
segmented from its background before any feature is computed, lighting and
colour are normalised, and the model is scored on whether it holds the same
verdict when the same leaf is rephotographed differently. See
[Measured results](#measured-results) for what that bought — and what it cost.

| File | Purpose |
|---|---|
| `bell_pepper_pipeline.py` | Dataset parsing, preprocessing, feature extraction, training, evaluation, consistency harness |
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

Trains a `RandomForestClassifier` and an `SVC`, reports **5-fold cross-validated**
macro-F1 for each plus a hold-out `classification_report` and confusion matrix,
saves confusion-matrix PNGs to `reports/`, runs the consistency check, refits the
winner on all images, and writes it to `models/best_pepper_model.joblib`.

For the most stable model, train with photometric augmentation:

```bash
python bell_pepper_pipeline.py --data-dir data --augment --models RandomForest
```

`--augment` adds gamma, colour-cast, JPEG and blur variants of every training
image so the classifier learns to ignore them. Augmented copies share a *group*
with their source, and both the k-fold and the hold-out split are group-aware, so
a variant of a training image can never land in the test set — that leakage would
inflate the score without improving anything.

`--models RandomForest` matters here: the RBF SVM scales roughly quadratically in
sample count, and the augmented set is 7× larger, so including it turns a ~40
minute run into a multi-hour one for an estimator that loses on this descriptor
anyway.

Useful flags: `--model-out`, `--report-dir`, `--test-size`, `--random-state`,
`--cv-folds`, `--consistency-sample`, `--scan-only`.

---

## 6. Test a single image

**Web UI** (shows a big HEALTHY / DAMAGED verdict first):

```bash
streamlit run app.py
```

Then open <http://localhost:8501> and leave **Bell pepper** selected in the
sidebar. The app runs **entirely locally** — bound to `127.0.0.1`, no tunnel
(ngrok/Colab), no Deploy button, no telemetry. See `.streamlit/config.toml`.
Dark theme is on by default.

Two ways to supply an image:
- **Upload** — drag and drop any JPG/PNG/BMP/TIF/WEBP from your machine.
- **Paste a path** — point at a file already on disk, e.g. `samples/healthy_leaf.jpg`.

**CLI:**

```bash
python predict_pepper.py --image samples/bacterial_spot_leaf.jpg
```

---

## Measured results

Trained on the 2,475 PlantVillage bell-pepper images (1,478 healthy / 997
bacterial spot).

### Accuracy is not the interesting number

Accuracy on a held-out split says nothing about whether the *same leaf*, shot a
little differently, keeps its verdict. So alongside macro-F1 the pipeline
measures **consistency**: each image is re-classified under 17 label-preserving
transforms (rotation, flip, brightness, gamma, warm/cool colour cast, JPEG
recompression, blur, centre-zoom), and the headline figure is the fraction of
leaves whose verdict is identical across *all* of them.

| Descriptor | Rows | CV macro-F1 | Fully stable | P(Damaged) spread |
|---|---|---|---|---|
| v1 whole-image (previous) | 2,475 | **0.9945** ± 0.0028 | 63.3% | 0.0987 |
| v2 leaf-segmented | 2,475 | 0.9899 ± 0.0028 | 94.2% | 0.0374 |
| **v2 + augmented** — shipped | 17,325 | 0.9883 ± 0.0032 | **96.7%** | **0.0242** |

All three rows are measured on the **same 120-image sample with the same seed**,
so they are directly comparable. The shipped bundle records its own figure from a
larger 200-image sample — **98.5% stable, spread 0.0192** — which is why
`models/best_pepper_model.joblib` reports a slightly better number than the
comparison table above. Different sample, not a different model.

The old model scored the *highest* macro-F1 while flipping its verdict on more
than a third of leaves. That extra accuracy was the background shortcut, and it
is exactly what made it fragile — so the 0.6-point F1 drop is the price of
removing it, not a regression.

### Where the instability was

Verdict flip rate per transform (lower is better):

| Transform | v1 | v2 | v2 + augmented |
|---|---|---|---|
| warm cast | 27.5% | 2.5% | 0.0% |
| cool cast | 9.2% | 1.7% | 0.0% |
| brightness ×0.75 | 2.5% | 0.0% | 0.0% |
| gamma 0.75 | 0.8% | 1.7% | 0.0% |
| JPEG q40 | 0.8% | 1.7% | 0.0% |
| blur | 0.8% | 0.8% | 0.0% |
| rotations / flips | 0.0% | ≤0.8% | ≤0.8% |
| **zoom 0.85** *(held out)* | 0.8% | 3.3% | **2.5%** |
| **JPEG q25** *(held out)* | 0.8% | 0.8% | **0.8%** |
| **rot 17°** *(held out)* | 0.0% | 0.8% | **0.8%** |

Two honest caveats on this table:

- The trained transforms reaching exactly 0.0% is expected — the model was
  trained on them. Read the **held-out** rows instead: `zoom_0.85`, `jpeg_q25`
  and `rot17` are deliberately excluded from `AUGMENT_TRANSFORMS`, so they
  measure generalised stability rather than memorised augmentation.
- Segmentation introduced a *new* sensitivity of its own. Gamma and JPEG perturb
  saturation, which shifts the Otsu mask, which moves the features — visible as
  v2 getting slightly worse than v1 on `gamma_0.75` (0.8% → 1.7%) and
  `zoom_0.85` (0.8% → 3.3%). Augmentation is what closes that back up. Framing
  (`zoom_0.85`, 2.5%) is the largest remaining weakness.

Class imbalance and error profile are unchanged: both models are strongest at
*not* false-alarming on healthy leaves, and the few errors are missed
bacterial-spot leaves. For a screening tool you likely want the opposite bias —
see "Tuning" below.

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

Every image goes through the same four preprocessing steps before any feature is
computed. The order matters and is enforced in one place (`preprocess()`) so
training and inference cannot drift apart:

1. **Resize** to 256×256.
2. **Segment the leaf** — Otsu's threshold on the *saturation* channel. Studio
   backdrops are near-neutral (low saturation) while leaf tissue — green,
   chlorotic yellow or necrotic brown alike — is saturated, so this separates the
   two without a hue window that would exclude diseased tissue. Morphological
   close/open removes speckle, components smaller than 15% of the largest are
   dropped, and interior holes are filled so lesion centres stay inside the mask.
3. **White balance** using the *background* as the neutral reference. Plain
   grey-world assumes the whole frame averages to grey, which a leaf-filled frame
   violates — it bleeds the leaf's own greenness into the correction. The
   backdrop is genuinely neutral, so it makes a much better white reference.
4. **Normalise exposure** so mean brightness *inside the mask* is fixed. Measuring
   over leaf pixels only keeps a bright background from dragging the correction
   around.

Steps 2–4 must happen in that order: white-balancing first shifts the neutral
backdrop off neutral, which hands the background saturation and shreds the mask.

The masked image is then reduced to one 1-D feature vector:

1. **HSV color histogram** (16×16×16 bins) — HSV separates hue from brightness,
   which makes the yellow-brown chlorotic halos of bacterial spot far more
   separable than RGB. L1-normalised to a distribution so the vector does not
   depend on how many pixels the mask kept — a tight crop and a wide shot of the
   same leaf land in the same place.
2. **Local Binary Pattern histogram** (P=24, R=3, uniform) — captures the
   surface-roughness change caused by lesions and necrotic tissue.

Both histograms count **only leaf pixels**. This is the main defence against the
model keying on the backdrop rather than the leaf.

Both are concatenated and fed to the classifiers. The SVM is wrapped in a
`Pipeline` with `StandardScaler` so scaling is persisted with the model.

Class imbalance is handled with `class_weight="balanced"`. Model selection uses
**5-fold cross-validated macro-F1** rather than a single split, because one
80/20 partition is a sample of size one — rerun it with another seed and the
headline number moves. The winner is then refit on all 2,475 images, since the
split existed only to estimate performance.

### Descriptor versioning

The bundle records a `feature_version`. A model trained on an older descriptor
would still *run* — the vector lengths can coincide — while silently scoring
nonsense, because the bins no longer mean what it learned. `load_bundle()`
refuses the mismatch instead, turning a silent accuracy collapse into a clear
"retrain this" error.

---

## Caveats

- On PlantVillage alone, "Damaged" effectively means *bacterial spot*, since
  that is the only pepper disease class present. Add pest/deficiency folders
  (see `LABELS.md`) to broaden class 1 — no code change required.
- **The 99% above is still not field accuracy.** Segmentation removes the
  *background* shortcut, and the consistency numbers show the verdict now
  survives lighting, colour and compression changes. Neither fact makes this a
  field benchmark: every training image is a single detached leaf, centred and
  in focus. Real greenhouse photos add soil, overlapping leaves, occlusion and
  motion blur, and the saturation-Otsu mask in particular assumes a
  low-saturation backdrop — on soil or foliage it will degrade. Expect a real
  drop, just not the *background-shortcut* portion of it.
- **Consistency is not correctness.** A model that answers "healthy" every time
  would score 100% stable. Read the stability figure alongside macro-F1, never
  instead of it.
- **Bell pepper only.** There is no "not a pepper" class. Feed it a tomato,
  potato, or non-leaf image and it will still emit HEALTHY or DAMAGED with high
  apparent confidence. The output is meaningless outside *Capsicum annuum*.
- This is a screening baseline, not an agronomic diagnosis.
