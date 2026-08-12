# Bell Pepper Leaf Labeling Standard

**Scope:** Bell pepper (*Capsicum annuum*) leaves only. Shared labeling
convention for the whole group so every dataset merges cleanly and the
`bell_pepper_pipeline.py` parser assigns the correct class automatically.

---

## 1. Folder naming convention

Every image folder MUST follow this exact pattern:

```
Pepper_bell___<condition_slug>
```

Rules:

1. Prefix is always `Pepper_bell` — required so the parser recognizes it as a
   bell-pepper folder.
2. Use a **triple underscore** `___` between the prefix and the condition.
3. `condition_slug` is **lowercase snake_case** and MUST come from the
   controlled vocabulary in section 2 — no free text, no abbreviations.
4. The word `healthy` may appear ONLY in the healthy class. It is the token the
   parser uses to assign class 0.
5. Do not put commas, spaces, or capital letters in the slug.

---

## 2. Controlled vocabulary

The granular slug is what you label with. The **Binary** column is what the
current pipeline collapses each slug into (0 = Undamaged, 1 = Damaged).

### Healthy

| Canonical folder | Binary | Look for |
|---|:---:|---|
| `Pepper_bell___healthy` | 0 | Uniform green, no lesions/spots/curl |

### Diseases

| Canonical folder | Binary | Category | Look for |
|---|:---:|---|---|
| `Pepper_bell___bacterial_spot` | 1 | Bacterial | Water-soaked spots → brown w/ yellow halo (*Xanthomonas*) |
| `Pepper_bell___cercospora_leaf_spot` | 1 | Fungal | Frogeye: circular tan centers, dark margins |
| `Pepper_bell___powdery_mildew` | 1 | Fungal | White powder, esp. leaf underside |
| `Pepper_bell___phytophthora_blight` | 1 | Oomycete | Dark necrotic lesions, wilting |
| `Pepper_bell___anthracnose` | 1 | Fungal | Sunken lesions with concentric rings |
| `Pepper_bell___mosaic_virus` | 1 | Viral | Mottled green/yellow mosaic (CMV/TMV/PepMV) |
| `Pepper_bell___leaf_curl_virus` | 1 | Viral | Upward curling, stunting (begomovirus) |

### Pests

| Canonical folder | Binary | Look for |
|---|:---:|---|
| `Pepper_bell___aphid_damage` | 1 | Curling, honeydew, sooty mold |
| `Pepper_bell___thrips_damage` | 1 | Silvery stippling, black frass specks |
| `Pepper_bell___spider_mite_damage` | 1 | Fine stippling, webbing |
| `Pepper_bell___whitefly_damage` | 1 | Yellowing, honeydew |
| `Pepper_bell___leaf_miner` | 1 | Serpentine white trails |

### Nutrient deficiency

| Canonical folder | Binary | Look for |
|---|:---:|---|
| `Pepper_bell___nitrogen_deficiency` | 1 | Uniform pale/yellow, older leaves first |
| `Pepper_bell___potassium_deficiency` | 1 | Marginal scorch / browning |
| `Pepper_bell___magnesium_deficiency` | 1 | Interveinal chlorosis, veins stay green |
| `Pepper_bell___calcium_deficiency` | 1 | Distorted young leaves |

### Abiotic / environmental

| Canonical folder | Binary | Look for |
|---|:---:|---|
| `Pepper_bell___sunscald` | 1 | Bleached / papery patches |
| `Pepper_bell___physical_damage` | 1 | Tears, wind/hail, mechanical |
| `Pepper_bell___herbicide_injury` | 1 | Distortion, cupping, chlorosis |

---

## 3. Alias map (incoming dataset name → canonical)

When importing from PlantVillage / Kaggle / Mendeley / OLID-I, **rename** to the
canonical slug. Do not invent new variants.

| You might see | Rename to |
|---|---|
| `Bacterial_spot`, `Pepper,_bell___Bacterial_spot`, `bell_pepper_bacterial` | `Pepper_bell___bacterial_spot` |
| `Healthy`, `normal`, `undamaged`, `Pepper_bell___Healthy` | `Pepper_bell___healthy` |
| `leaf_curl`, `curl_virus`, `TYLCV_like` | `Pepper_bell___leaf_curl_virus` |
| `mosaic`, `CMV`, `TMV` | `Pepper_bell___mosaic_virus` |
| `mite`, `red_spider` | `Pepper_bell___spider_mite_damage` |
| `powdery`, `mildew` | `Pepper_bell___powdery_mildew` |

---

## 4. Binary vs. multiclass

- **Today:** the pipeline is binary. `healthy` → class 0; every other slug → class 1.
- **Later:** because folders keep the granular slug, you can switch to
  multiclass (per-condition) with no re-labeling — only the label-mapping code
  changes.

Keeping the fine-grained slug now costs nothing and preserves that option.

---

## 5. Labeling checklist for contributors

- [ ] Folder name starts with `Pepper_bell___`
- [ ] Condition slug is from section 2 (exact spelling)
- [ ] Only bell pepper leaves inside (no tomato/potato/other species)
- [ ] Ambiguous / multi-symptom images: label by the **dominant** visible symptom
- [ ] Unsure? Park it in `Pepper_bell___unsorted/` for group review — do NOT guess
