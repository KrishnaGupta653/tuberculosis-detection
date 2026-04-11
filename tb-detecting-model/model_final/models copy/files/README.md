# TB Detection System — Complete Pipeline
## Chest X-Ray Tuberculosis Classifier

---

## System Architecture

```
Image → Segmentation → Feature Extraction → Feature Selection → Optimisation → Classification
```

The system is split into **5 production-ready Python modules** plus one shared
utility library:

| File | Role |
|------|------|
| `pipeline_utils.py` | **Shared** — segmentation, feature extraction, dataset loading |
| `svm_model.py` | Module 1 — SVM classifier (≥85% target) |
| `random_forest_model.py` | Module 2 — Random Forest classifier (≥85% target) |
| `gradient_boosting_model.py` | Module 3 — Gradient Boosting classifier (≥85% target) |
| `ensemble_model.py` | Module 4 — Weighted soft-voting ensemble (≥95% target) |
| `predict.py` | Module 5 — CLI inference for all models |
| `train_all.py` | Optional — trains all 4 models in sequence |

---

## Installation

```bash
pip install torch torchvision scikit-learn opencv-python scikit-image \
            scipy numpy matplotlib seaborn pandas tqdm
```

> GPU (CUDA) is used automatically if available.

---

## Dataset Layout

```
dataset/
├── Normal/
│   ├── image001.jpg
│   └── ...
└── TB/
    ├── image001.jpg
    └── ...
```

---

## Training

### Option A — Train all models in sequence (recommended)

```bash
python train_all.py --data_dir ./dataset
```

This trains SVM → RF → GB (with shared feature cache) → Ensemble.

### Option B — Train each model independently

```bash
python svm_model.py               --data_dir ./dataset
python random_forest_model.py     --data_dir ./dataset
python gradient_boosting_model.py --data_dir ./dataset
python ensemble_model.py          --data_dir ./dataset   # run last
```

### Optional feature cache (avoids re-extracting features)

```bash
python svm_model.py --data_dir ./dataset --cache ./features.npz
```

---

## Prediction (Inference)

```bash
# Single image — any model
python predict.py --model svm      --image test.jpg
python predict.py --model rf       --image test.jpg
python predict.py --model gb       --image test.jpg
python predict.py --model ensemble --image test.jpg

# Batch folder
python predict.py --model ensemble --folder test_images/

# Batch + save CSV
python predict.py --model ensemble --folder test_images/ --save_csv results.csv
```

### Output per image

```
============================================================
  TB DETECTION RESULT  [ENSEMBLE]
============================================================
  Image      : patient_042.jpg
  Prediction : TB
  Confidence : 94.7%
  Risk Level : High
────────────────────────────────────────────────────────────
  Probability breakdown:
    Normal    5.3%  ████
    TB       94.7%  ██████████████████████████
============================================================
```

**Risk levels:**

| Level  | Condition |
|--------|-----------|
| Low    | Prediction = Normal, OR TB confidence < 70% |
| Medium | TB confidence 70–90% |
| High   | TB confidence ≥ 90% |

---

## Pipeline Details

### 1 — Segmentation (`LungSegmentor`)
- Multi-scale CLAHE enhancement (3 grid sizes, weighted fusion)
- Local entropy map → foreground mask
- Otsu thresholding (inverted, for dark lungs)
- Morphological close + open cleanup
- Contour filtering by area + circularity
- Fallback: full image used if no valid contour is found
- **No heavy pre-trained segmentation models required**

### 2 — Feature Extraction (Hybrid)
- **Deep features**: DenseNet121 (ImageNet pre-trained, frozen) → 1024-dim
- **GLCM radiomics**: 6 properties × 4 stats × (3 distances × 4 angles averaged) → 24-dim
- Total feature vector: **1048 dimensions**

### 3 — Feature Selection
- `SelectKBest(f_classif, k=200)` — keeps top 200 features
- Fitted on training split only → no data leakage
- Selector saved as `.pkl` and reused identically at inference

### 4 — Optimisation
- `RandomizedSearchCV` with `StratifiedKFold(5)` for each model
- `StandardScaler` (fit on train only)
- `class_weight='balanced'` for SVM and RF

### 5 — Classification
- SVM (RBF kernel), Random Forest, Gradient Boosting
- Ensemble: weighted soft-voting, weights optimised on validation accuracy

---

## Saved Artefacts

```
models/
├── svm_model.pkl
├── svm_scaler.pkl
├── svm_selector.pkl
├── svm_metadata.json
├── random_forest_model.pkl
├── rf_scaler.pkl
├── rf_selector.pkl
├── rf_metadata.json
├── gradient_boosting_model.pkl
├── gb_scaler.pkl
├── gb_selector.pkl
├── gb_metadata.json
├── ensemble_model.pkl          ← self-contained: includes all pipelines
├── ensemble_metadata.json
└── *.png                       ← confusion matrices, ROC curves, comparisons
```

---

## Data Split

| Split      | Size |
|------------|------|
| Train      | 70%  |
| Validation | 15%  |
| Test       | 15%  |

All splits are stratified to maintain class balance.  
Feature selection and scaling are fit **exclusively** on the training split.

---

## Robustness

- Invalid image paths → graceful error with message
- Empty folders → clear exit with explanation
- Feature dimension mismatches → impossible by design (shared selector/scaler)
- Missing model files → `FileNotFoundError` with helpful message
- Corrupt/unreadable images → logged, skipped in batch mode
- Batch CSV export includes error column for failed images
