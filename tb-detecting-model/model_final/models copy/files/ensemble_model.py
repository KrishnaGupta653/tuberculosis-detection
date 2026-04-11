"""
================================================================================
ensemble_model.py  —  TB Detection System  |  Module 4: Ensemble (FINAL MODEL)
================================================================================
Loads SVM, Random Forest, and Gradient Boosting pipelines, then builds a
weighted soft-voting ensemble whose weights are optimised on a held-out
validation set via grid search over weight combinations.

All three base models share the same feature space (raw 1048-d vectors)
and each internally applies its own selector + scaler, so there is zero
dimension-mismatch risk.

Saves:
    models/ensemble_model.pkl     — EnsembleModel instance
    models/ensemble_metadata.json — weights + per-model + ensemble metrics

Run standalone:
    python ensemble_model.py --data_dir ./dataset
================================================================================
"""

import os
import json
import pickle
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from itertools import product as iterproduct
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_curve, auc
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from pipeline_utils import RANDOM_SEED, CATEGORIES, extract_dataset_features, logger
import svm_model          as svm_mod
import random_forest_model as rf_mod
import gradient_boosting_model as gb_mod

MODEL_DIR = './models'
os.makedirs(MODEL_DIR, exist_ok=True)

ENSEMBLE_MODEL_PATH = os.path.join(MODEL_DIR, 'ensemble_model.pkl')
ENSEMBLE_META_PATH  = os.path.join(MODEL_DIR, 'ensemble_metadata.json')


# ══════════════════════════════════════════════════════════════════════════════
# EnsembleModel
# ══════════════════════════════════════════════════════════════════════════════

class EnsembleModel:
    """
    Weighted soft-voting ensemble over SVM, Random Forest, Gradient Boosting.
    Each base pipeline (model + scaler + selector) is stored internally so the
    object is self-contained for inference.
    """

    def __init__(self):
        # Will hold tuples: (model, scaler, selector, weight)
        self._pipelines: list[tuple] = []
        self._names: list[str] = []
        self._weights: list[float] = []
        self._trained = False

    # ── loading base models ───────────────────────────────────────────────────

    def load_base_models(self):
        """Load all three individually trained pipelines."""
        logger.info('Loading base model pipelines…')
        svm, svm_sc, svm_sel = svm_mod.load_svm_pipeline()
        rf,  rf_sc,  rf_sel  = rf_mod.load_rf_pipeline()
        gb,  gb_sc,  gb_sel  = gb_mod.load_gb_pipeline()

        self._pipelines = [
            (svm, svm_sc, svm_sel),
            (rf,  rf_sc,  rf_sel),
            (gb,  gb_sc,  gb_sel),
        ]
        self._names = ['SVM', 'RandomForest', 'GradientBoosting']
        logger.info('All base models loaded.')

    # ── probability inference per pipeline ───────────────────────────────────

    def _proba_single_pipeline(self, pipeline_idx: int, X: np.ndarray) -> np.ndarray:
        model, scaler, selector = self._pipelines[pipeline_idx]
        Xs = scaler.transform(selector.transform(X))
        return model.predict_proba(Xs)          # (n, 2)

    # ── weight optimisation ───────────────────────────────────────────────────

    def optimise_weights(self, X_val: np.ndarray, y_val: np.ndarray):
        """
        Grid search over weight triples (w_svm, w_rf, w_gb) ∈ {0.1, 0.2, …, 0.8}³
        that sum to 1.0.  Picks the combination maximising validation accuracy.
        """
        logger.info('Optimising ensemble weights on validation set…')
        step   = 0.1
        ticks  = np.round(np.arange(0.1, 0.9, step), 2)
        combos = [(w1, w2, round(1 - w1 - w2, 2))
                  for w1 in ticks for w2 in ticks
                  if 0.05 < round(1 - w1 - w2, 2) <= 0.85]

        probas = [self._proba_single_pipeline(i, X_val)
                  for i in range(len(self._pipelines))]

        best_acc, best_w = -1, (1/3, 1/3, 1/3)
        for w in combos:
            combined = sum(wi * p for wi, p in zip(w, probas))
            acc = accuracy_score(y_val, np.argmax(combined, axis=1))
            if acc > best_acc:
                best_acc, best_w = acc, w

        self._weights = list(best_w)
        logger.info(f'Optimal weights: SVM={best_w[0]:.2f}  '
                    f'RF={best_w[1]:.2f}  GB={best_w[2]:.2f}  '
                    f'(val acc={best_acc*100:.2f}%)')
        return best_acc

    # ── ensemble inference ────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._trained:
            raise RuntimeError('Ensemble not assembled. Call assemble() or load().')
        combined = np.zeros((X.shape[0], 2), dtype=np.float64)
        for i, w in enumerate(self._weights):
            combined += w * self._proba_single_pipeline(i, X)
        return combined

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    # ── assembly ──────────────────────────────────────────────────────────────

    def assemble(self, X_val: np.ndarray, y_val: np.ndarray):
        """Optimise weights; mark as trained."""
        val_acc = self.optimise_weights(X_val, y_val)
        self._trained = True
        return val_acc

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = ENSEMBLE_MODEL_PATH):
        if not self._trained:
            raise RuntimeError('Cannot save untrained ensemble.')
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f'Ensemble saved → {path}')

    @classmethod
    def load(cls, path: str = ENSEMBLE_MODEL_PATH) -> 'EnsembleModel':
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'Ensemble model not found: {path}. Run ensemble_model.py first.')
        with open(path, 'rb') as f:
            instance = pickle.load(f)
        logger.info(f'Ensemble loaded from {path}')
        return instance


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation helper
# ══════════════════════════════════════════════════════════════════════════════

def _evaluate(name: str, predict_fn, proba_fn, X, y, plot_dir: str) -> dict:
    y_pred = predict_fn(X)
    y_prob = proba_fn(X)[:, 1]
    acc  = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec  = recall_score(y, y_pred, zero_division=0)
    f1   = f1_score(y, y_pred, zero_division=0)
    cm   = confusion_matrix(y, y_pred)
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    logger.info(f'\n── {name} ──')
    logger.info(classification_report(y, y_pred, target_names=CATEGORIES, digits=4))

    tag = name.lower().replace(' ', '_')
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=CATEGORIES, yticklabels=CATEGORIES)
    plt.title(f'{name} — Confusion Matrix'); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{tag}_confusion.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f'AUC={roc_auc:.3f}')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title(f'{name} — ROC'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{tag}_roc.png'), dpi=150)
    plt.close()

    return dict(accuracy=acc, precision=prec, recall=rec,
                f1=f1, auc_roc=roc_auc, confusion_matrix=cm.tolist())


# ══════════════════════════════════════════════════════════════════════════════
# Training entry point
# ══════════════════════════════════════════════════════════════════════════════

def train(data_dir: str, cache_path: str | None = None):
    logger.info('='*70)
    logger.info('ENSEMBLE MODEL — TRAINING START')
    logger.info('='*70)

    X, y = extract_dataset_features(data_dir, cache_path=cache_path)

    # 70 / 15 / 15 split
    X_tmp, X_test,  y_tmp, y_test  = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.15/0.85, random_state=RANDOM_SEED, stratify=y_tmp)

    # Build ensemble
    ensemble = EnsembleModel()
    ensemble.load_base_models()
    val_acc = ensemble.assemble(X_val, y_val)
    ensemble._trained = True

    # ── Compare individual models vs ensemble on test set ──────────────────────
    comparison: dict[str, dict] = {}

    for i, name in enumerate(ensemble._names):
        model, scaler, selector = ensemble._pipelines[i]
        pfn   = lambda X, m=model, sc=scaler, sel=selector: m.predict(sc.transform(sel.transform(X)))
        probfn= lambda X, m=model, sc=scaler, sel=selector: m.predict_proba(sc.transform(sel.transform(X)))
        comparison[name] = _evaluate(name, pfn, probfn, X_test, y_test, MODEL_DIR)
        logger.info(f'{name} test accuracy: {comparison[name]["accuracy"]*100:.2f}%')

    comparison['Ensemble'] = _evaluate(
        'Ensemble', ensemble.predict, ensemble.predict_proba,
        X_test, y_test, MODEL_DIR)
    ens_acc = comparison['Ensemble']['accuracy']
    logger.info(f'\nENSEMBLE TEST ACCURACY: {ens_acc*100:.2f}%')
    if ens_acc < 0.95:
        logger.warning('Ensemble accuracy below 95% target — '
                       'ensure base models were trained on sufficient data.')

    # ── Comparison bar chart ───────────────────────────────────────────────────
    names_plt = list(comparison.keys())
    accs_plt  = [comparison[n]['accuracy']*100 for n in names_plt]
    colours   = ['#4c72b0','#55a868','#c44e52','#8172b2']
    plt.figure(figsize=(8, 5))
    bars = plt.bar(names_plt, accs_plt, color=colours)
    plt.bar_label(bars, fmt='%.2f%%')
    plt.ylim([max(0, min(accs_plt)-5), 101])
    plt.ylabel('Accuracy (%)')
    plt.title('Model Comparison — Test Set Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'model_comparison.png'), dpi=150)
    plt.close()
    logger.info(f'Comparison chart → {MODEL_DIR}/model_comparison.png')

    # Save
    ensemble.save(ENSEMBLE_MODEL_PATH)

    meta = {
        'ensemble_weights': dict(zip(ensemble._names, ensemble._weights)),
        'val_accuracy':     float(val_acc),
        'test_comparison':  comparison,
    }
    # Convert numpy types
    def _conv(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    with open(ENSEMBLE_META_PATH, 'w') as f:
        import json as _json
        _json.dump(meta, f, indent=2, default=_conv)

    logger.info(f'Metadata → {ENSEMBLE_META_PATH}')
    logger.info('Ensemble training complete.')
    return ENSEMBLE_MODEL_PATH


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Build Ensemble TB Detector')
    ap.add_argument('--data_dir', default='./dataset')
    ap.add_argument('--cache',    default=None)
    args = ap.parse_args()
    train(args.data_dir, cache_path=args.cache)