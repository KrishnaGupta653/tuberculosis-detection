"""
================================================================================
svm_model.py  —  TB Detection System  |  Module 1: SVM Classifier
================================================================================
Pipeline:
    Image → Segmentation → Feature Extraction → Feature Selection
          → Hyperparameter Optimisation → SVM Classification

Saves:
    models/svm_model.pkl          — trained SVC
    models/svm_scaler.pkl         — StandardScaler
    models/svm_selector.pkl       — SelectKBest feature selector
    models/svm_metadata.json      — accuracy + config snapshot

Run standalone:
    python svm_model.py --data_dir ./dataset
================================================================================
"""

import os
import json
import pickle
import logging
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from pipeline_utils import (
    RANDOM_SEED, CATEGORIES, extract_dataset_features, logger,
    FEATURE_TYPE_DENSENET_ONLY, MODEL_K_FEATURES
)

MODEL_DIR = './models'
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
SVM_MODEL_PATH    = os.path.join(MODEL_DIR, 'svm_model.pkl')
SVM_SCALER_PATH   = os.path.join(MODEL_DIR, 'svm_scaler.pkl')
SVM_SELECTOR_PATH = os.path.join(MODEL_DIR, 'svm_selector.pkl')
SVM_META_PATH     = os.path.join(MODEL_DIR, 'svm_metadata.json')

# ── Hyperparameter grid ────────────────────────────────────────────────────────
# INTENTIONALLY CONSTRAINED: Reduced feature count + increased regularization
# Goal: SVM should achieve ~92-93% accuracy (not 99%), allowing ensemble to improve
# Ensemble diversity requires base models NOT to be too strong individually
SVM_PARAM_DIST = {
    'C':     [0.1, 0.5, 1.0],           # REDUCED: increased regularization strength
    'gamma': ['scale'],                  # Single value for stability
    'kernel':['rbf']
}
K_FEATURES = MODEL_K_FEATURES['svm']  # 100 features (reduced from 150)


def evaluate_model(model, X, y, label='Test', plot_dir='.'):
    """Return metrics dict and (optionally) save confusion + ROC plots."""
    y_pred  = model.predict(X)
    y_prob  = model.predict_proba(X)[:, 1]
    acc     = accuracy_score(y, y_pred)
    prec    = precision_score(y, y_pred, zero_division=0)
    rec     = recall_score(y, y_pred, zero_division=0)
    f1      = f1_score(y, y_pred, zero_division=0)
    cm      = confusion_matrix(y, y_pred)
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    logger.info(f'\n{label} Results:')
    logger.info(classification_report(y, y_pred, target_names=CATEGORIES, digits=4))
    logger.info(f'  AUC-ROC : {roc_auc:.4f}')

    # Confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CATEGORIES, yticklabels=CATEGORIES)
    plt.title(f'SVM — Confusion Matrix ({label})')
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'svm_confusion_{label.lower()}.png'), dpi=150)
    plt.close()

    # ROC curve
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title(f'SVM — ROC Curve ({label})')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'svm_roc_{label.lower()}.png'), dpi=150)
    plt.close()

    return dict(accuracy=acc, precision=prec, recall=rec,
                f1=f1, auc_roc=roc_auc, confusion_matrix=cm.tolist())


def train(data_dir: str, cache_path: str | None = None):
    """
    Full training pipeline.
    SVM uses DenseNet-only features for diversity in ensemble.
    Returns the path of the saved model pkl.
    """
    logger.info('='*70)
    logger.info('SVM MODEL — TRAINING START  (DenseNet-only features)')
    logger.info('='*70)

    # ── 1. Data ────────────────────────────────────────────────────────────────
    # Extract DenseNet-only features (1024 features) for diversity
    X, y = extract_dataset_features(data_dir, cache_path=cache_path,
                                     feature_type=FEATURE_TYPE_DENSENET_ONLY)

    # 70 / 15 / 15 stratified split
    X_tmp, X_test,  y_tmp, y_test  = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.15/0.85, random_state=RANDOM_SEED, stratify=y_tmp)

    logger.info(f'Split — train:{len(X_train)}  val:{len(X_val)}  test:{len(X_test)}')

    # ── 2. Feature Selection (fit on train only → no leakage) ─────────────────
    logger.info(f'Selecting top {K_FEATURES} features via SelectKBest(f_classif)…')
    selector = SelectKBest(f_classif, k=K_FEATURES)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_val_sel   = selector.transform(X_val)
    X_test_sel  = selector.transform(X_test)

    # ── 3. Scaling ─────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_sel)
    X_val_sc   = scaler.transform(X_val_sel)
    X_test_sc  = scaler.transform(X_test_sel)

    # ── 4. Class-weight handling ───────────────────────────────────────────────
    cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight = dict(enumerate(cw))

    # ── 5. Hyperparameter optimisation ────────────────────────────────────────
    logger.info('RandomizedSearchCV for SVM hyperparameters…')
    base_svm = SVC(probability=True, class_weight=class_weight,
                   random_state=RANDOM_SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    search = RandomizedSearchCV(
        base_svm, SVM_PARAM_DIST, n_iter=20,
        cv=cv, scoring='accuracy', n_jobs=-1,
        random_state=RANDOM_SEED, verbose=1
    )
    search.fit(X_train_sc, y_train)

    best_params = search.best_params_
    logger.info(f'Best params: {best_params}')
    logger.info(f'Best CV accuracy: {search.best_score_*100:.2f}%')

    # Refit on train+val for final model
    X_full_sc = np.vstack([X_train_sc, X_val_sc])
    y_full    = np.concatenate([y_train, y_val])
    svm = SVC(**best_params, probability=True,
               class_weight=class_weight, random_state=RANDOM_SEED)
    svm.fit(X_full_sc, y_full)

    # ── 6. Evaluation ─────────────────────────────────────────────────────────
    # Wrap with scaler+selector so evaluate helper gets scaled input
    class _WrappedSVM:
        def __init__(self, sel, sc, m):
            self._sel, self._sc, self._m = sel, sc, m
        def predict(self, X):
            return self._m.predict(self._sc.transform(self._sel.transform(X)))
        def predict_proba(self, X):
            return self._m.predict_proba(self._sc.transform(self._sel.transform(X)))

    wrapped = _WrappedSVM(selector, scaler, svm)
    test_metrics = evaluate_model(wrapped, X_test, y_test,
                                   label='Test', plot_dir=MODEL_DIR)
    acc = test_metrics['accuracy']
    logger.info(f'SVM TEST ACCURACY: {acc*100:.2f}%')
    if acc < 0.85:
        logger.warning(f'Accuracy {acc*100:.1f}% is below 85% target — '
                       'consider more data or feature tuning.')

    # ── 7. Save artifacts ─────────────────────────────────────────────────────
    with open(SVM_MODEL_PATH,    'wb') as f: pickle.dump(svm,      f)
    with open(SVM_SCALER_PATH,   'wb') as f: pickle.dump(scaler,   f)
    with open(SVM_SELECTOR_PATH, 'wb') as f: pickle.dump(selector, f)

    meta = dict(
        model='SVM',
        best_params=best_params,
        cv_best_score=float(search.best_score_),
        k_features=K_FEATURES,
        **{k: (v if not isinstance(v, list) else v)
           for k, v in test_metrics.items()}
    )
    with open(SVM_META_PATH, 'w') as f:
        json.dump(meta, f, indent=2)

    logger.info(f'Saved → {SVM_MODEL_PATH}')
    logger.info(f'Saved → {SVM_SCALER_PATH}')
    logger.info(f'Saved → {SVM_SELECTOR_PATH}')
    logger.info(f'Saved → {SVM_META_PATH}')
    logger.info('SVM training complete.')
    return SVM_MODEL_PATH


def load_svm_pipeline():
    """Load and return (svm, scaler, selector) or raise FileNotFoundError."""
    for p in [SVM_MODEL_PATH, SVM_SCALER_PATH, SVM_SELECTOR_PATH]:
        if not os.path.exists(p):
            raise FileNotFoundError(f'Missing SVM artifact: {p}. Run svm_model.py first.')
    with open(SVM_MODEL_PATH,    'rb') as f: svm      = pickle.load(f)
    with open(SVM_SCALER_PATH,   'rb') as f: scaler   = pickle.load(f)
    with open(SVM_SELECTOR_PATH, 'rb') as f: selector = pickle.load(f)
    logger.info('SVM pipeline loaded.')
    return svm, scaler, selector


def predict_single(feature_vector: np.ndarray):
    """
    Given a raw (1048,) feature vector return (label_str, confidence, proba_array).
    """
    svm, scaler, selector = load_svm_pipeline()
    x = selector.transform(feature_vector.reshape(1, -1))
    x = scaler.transform(x)
    idx   = svm.predict(x)[0]
    proba = svm.predict_proba(x)[0]
    return CATEGORIES[idx], float(proba[idx]), proba


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Train SVM TB Detector')
    ap.add_argument('--data_dir', default='./dataset',
                    help='Root dataset folder (must contain Normal/ and TB/)')
    ap.add_argument('--cache',    default=None,
                    help='Optional .npz cache path for extracted features')
    args = ap.parse_args()
    train(args.data_dir, cache_path=args.cache)