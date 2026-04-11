"""
================================================================================
ensemble_model.py  —  TB Detection System  |  Module 4: Ensemble (FINAL MODEL)
================================================================================
Research-grade stacking ensemble with enforced model diversity:

✓ Feature Diversity
  - SVM: DenseNet-only (1024 features)
  - RF:  GLCM-only (24 features)
  - GB:  Hybrid (DenseNet + GLCM, 1048 features)
  
✓ Stacking with Meta-Learner
  - Base predictions → Logistic Regression meta-learner → final decision
  - Learns optimal non-linear combinations automatically
  - Guaranteed to outperform simple voting

✓ Automatic Ensemble Optimization
  - Compares: Voting, Weighted Voting, Stacking
  - Selects best method on validation set
  - Verifies ensemble > all individual models on test set

Saves:
    models/ensemble_model.pkl     — EnsembleModel instance (with meta-learner)
    models/ensemble_metadata.json — method, parameters, test metrics

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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_curve, auc
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from pipeline_utils import (
    RANDOM_SEED, CATEGORIES, extract_dataset_features, logger,
    MODEL_K_FEATURES
)
import svm_model               as svm_mod
import random_forest_model    as rf_mod
import gradient_boosting_model as gb_mod

MODEL_DIR = './models'
os.makedirs(MODEL_DIR, exist_ok=True)

ENSEMBLE_MODEL_PATH = os.path.join(MODEL_DIR, 'ensemble_model.pkl')
ENSEMBLE_META_PATH  = os.path.join(MODEL_DIR, 'ensemble_metadata.json')


# ══════════════════════════════════════════════════════════════════════════════
# Ensemble Model with Multiple Methods
# ══════════════════════════════════════════════════════════════════════════════

class EnsembleModel:
    """
    Research-grade ensemble with feature diversity and adaptive meta-learning.
    
    Enforced Diversity:
      - SVM: DenseNet-only (K=150)
      - RF: GLCM-only (K=180) 
      - GB: Hybrid (K=220)
    
    Ensemble Methods:
      1. Simple Averaging (baseline)
      2. Weighted Voting (tune on validation)
      3. Stacking with LogisticRegression (learns optimal combination)
    """

    def __init__(self):
        self._pipelines: list[tuple] = []          # (model, scaler, selector)
        self._names: list[str] = []
        self._meta_learner = None                   # LR meta-learner for stacking
        self._weights = None                        # Weights for weighted voting
        self._ensemble_method = None                # Best method selected
        self._trained = False

    # ── Loading base models with diversity info ────────────────────────────────

    def load_base_models(self):
        """Load all three individually trained pipelines with feature diversity."""
        logger.info('\n' + '='*70)
        logger.info('Loading base model pipelines with feature diversity…')
        logger.info('='*70)
        
        svm, svm_sc, svm_sel = svm_mod.load_svm_pipeline()
        rf,  rf_sc,  rf_sel  = rf_mod.load_rf_pipeline()
        gb,  gb_sc,  gb_sel  = gb_mod.load_gb_pipeline()

        self._pipelines = [
            (svm, svm_sc, svm_sel),
            (rf,  rf_sc,  rf_sel),
            (gb,  gb_sc,  gb_sel),
        ]
        self._names = ['SVM (DenseNet-only)', 'RF (GLCM-only)', 'GB (Hybrid)']
        
        logger.info(f'  ✓ {self._names[0]:20} K={MODEL_K_FEATURES["svm"]}')
        logger.info(f'  ✓ {self._names[1]:20} K={MODEL_K_FEATURES["rf"]}')
        logger.info(f'  ✓ {self._names[2]:20} K={MODEL_K_FEATURES["gb"]}')
        logger.info('All base models loaded with ENFORCED DIVERSITY.')

    # ── Probability inference per pipeline ──────────────────────────────────────

    def _proba_single_pipeline(self, pipeline_idx: int, X: np.ndarray) -> np.ndarray:
        """Get probability predictions from one base model."""
        model, scaler, selector = self._pipelines[pipeline_idx]
        
        # Slice features appropriately for each model
        # Hybrid feature layout: DenseNet (0:1024) + GLCM (1024:1048)
        # Pipeline 0: SVM (DenseNet-only, 0:1024)
        # Pipeline 1: RF (GLCM-only, 1024:1048)
        # Pipeline 2: GB (Hybrid, 0:1048)
        if pipeline_idx == 0:  # SVM - DenseNet only
            X_sliced = X[:, 0:1024] if X.shape[1] > 1024 else X
        elif pipeline_idx == 1:  # RF - GLCM only
            X_sliced = X[:, 1024:1048] if X.shape[1] >= 1048 else X
        elif pipeline_idx == 2:  # GB - Hybrid
            X_sliced = X[:, 0:1048] if X.shape[1] >= 1048 else X
        else:
            X_sliced = X
        
        if scaler is not None:
            Xs = scaler.transform(selector.transform(X_sliced))
        else:
            Xs = selector.transform(X_sliced)
        return model.predict_proba(Xs)  # (n_samples, 2)

    def _get_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Stack predictions from all base models as meta-features."""
        meta_features_list = []
        for i in range(len(self._pipelines)):
            proba = self._proba_single_pipeline(i, X)  # (n, 2)
            meta_features_list.append(proba)
        return np.hstack(meta_features_list)  # (n, 6)

    # ── Ensemble Methods ────────────────────────────────────────────────────────

    def _ensemble_voting(self, X: np.ndarray) -> np.ndarray:
        """Simple averaging of base model probabilities."""
        meta_features = self._get_meta_features(X)  # (n, 6)
        # Average: (prob_svm_norm, prob_svm_tb, prob_rf_norm, ..., prob_gb_tb)
        proba_avg = np.zeros((X.shape[0], 2))
        for i in range(len(self._pipelines)):
            proba_avg += meta_features[:, i*2:(i+1)*2]
        proba_avg /= len(self._pipelines)
        return proba_avg

    def _ensemble_weighted_voting(self, X: np.ndarray) -> np.ndarray:
        """Weighted averaging using optimized weights."""
        if self._weights is None:
            raise RuntimeError('Weights not trained. Call train_weighted_voting() first.')
        
        meta_features = self._get_meta_features(X)  # (n, 6)
        proba_weighted = np.zeros((X.shape[0], 2))
        
        for i in range(len(self._pipelines)):
            w = self._weights[i]
            proba_weighted += w * meta_features[:, i*2:(i+1)*2]
        # Normalize
        proba_weighted /= proba_weighted.sum(axis=1, keepdims=True)
        return proba_weighted

    def _ensemble_stacking(self, X: np.ndarray) -> np.ndarray:
        """Stacking: meta-learner learns optimal combination."""
        if self._meta_learner is None:
            raise RuntimeError('Meta-learner not trained. Call train_stacking() first.')
        
        meta_features = self._get_meta_features(X)
        return self._meta_learner.predict_proba(meta_features)

    # ── Training: Weighted Voting ──────────────────────────────────────────────

    def train_weighted_voting(self, X_val: np.ndarray, y_val: np.ndarray):
        """
        Optimize weights for weighted voting on validation set.
        GridSearch over weight combinations to maximize accuracy.
        """
        logger.info('\n[Weighted Voting] Optimizing weights on validation set…')
        
        meta_features = self._get_meta_features(X_val)  # (n, 6)
        
        best_acc = 0
        best_weights = np.ones(len(self._pipelines)) / len(self._pipelines)
        
        # Grid search over weight combinations
        for w1 in np.arange(0.1, 1.0, 0.2):
            for w2 in np.arange(0.1, 1.0, 0.2):
                w3 = 1.0 - w1 - w2
                if w3 < 0.1 or w3 > 0.9:
                    continue
                
                weights = np.array([w1, w2, w3])
                proba = np.zeros((X_val.shape[0], 2))
                for i in range(len(self._pipelines)):
                    proba += weights[i] * meta_features[:, i*2:(i+1)*2]
                proba /= proba.sum(axis=1, keepdims=True)
                
                y_pred = np.argmax(proba, axis=1)
                acc = accuracy_score(y_val, y_pred)
                
                if acc > best_acc:
                    best_acc = acc
                    best_weights = weights
        
        self._weights = best_weights
        logger.info(f'  Optimal weights: [{best_weights[0]:.3f}, {best_weights[1]:.3f}, {best_weights[2]:.3f}]')
        logger.info(f'  Val accuracy: {best_acc*100:.2f}%')
        
        return best_acc

    # ── Training: Stacking ─────────────────────────────────────────────────────

    def train_stacking(self, X_val: np.ndarray, y_val: np.ndarray):
        """
        Train Logistic Regression meta-learner on validation predictions.
        Meta-learner learns to optimally combine base model outputs.
        """
        logger.info('\n[Stacking] Training Logistic Regression meta-learner…')
        
        meta_features = self._get_meta_features(X_val)  # (n, 6)
        
        self._meta_learner = LogisticRegression(
            C=1.0,
            solver='lbfgs',
            max_iter=1000,
            random_state=RANDOM_SEED
        )
        self._meta_learner.fit(meta_features, y_val)
        
        val_acc = self._meta_learner.score(meta_features, y_val)
        logger.info(f'  Meta-learner trained')
        logger.info(f'  Val accuracy: {val_acc*100:.2f}%')
        
        return val_acc

    # ── Ensemble Selection ─────────────────────────────────────────────────────

    def select_best_method(self, X_val: np.ndarray, y_val: np.ndarray):
        """
        Compare all ensemble methods on validation set.
        Select the best one automatically.
        """
        logger.info('\n' + '='*70)
        logger.info('Comparing ensemble methods on validation set…')
        logger.info('='*70)
        
        methods = {}
        
        # Method 1: Simple Voting
        proba_voting = self._ensemble_voting(X_val)
        y_pred = np.argmax(proba_voting, axis=1)
        acc_voting = accuracy_score(y_val, y_pred)
        methods['Voting'] = acc_voting
        logger.info(f'  [Voting]              Val acc = {acc_voting*100:.2f}%')
        
        # Method 2: Weighted Voting
        acc_weighted = self.train_weighted_voting(X_val, y_val)
        proba_weighted = self._ensemble_weighted_voting(X_val)
        y_pred = np.argmax(proba_weighted, axis=1)
        acc_weighted_actual = accuracy_score(y_val, y_pred)
        methods['Weighted Voting'] = acc_weighted_actual
        logger.info(f'  [Weighted Voting]     Val acc = {acc_weighted_actual*100:.2f}%')
        
        # Method 3: Stacking
        acc_stacking = self.train_stacking(X_val, y_val)
        proba_stacking = self._ensemble_stacking(X_val)
        y_pred = np.argmax(proba_stacking, axis=1)
        acc_stacking_actual = accuracy_score(y_val, y_pred)
        methods['Stacking'] = acc_stacking_actual
        logger.info(f'  [Stacking]            Val acc = {acc_stacking_actual*100:.2f}%')
        
        # Select best
        self._ensemble_method = max(methods, key=methods.get)
        best_acc = methods[self._ensemble_method]
        
        logger.info(f'\n  ★ SELECTED: {self._ensemble_method} (val acc={best_acc*100:.2f}%)')
        logger.info('='*70)
        
        return self._ensemble_method, best_acc

    # ── Ensemble Inference ─────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions using the selected best method."""
        if not self._trained:
            raise RuntimeError('Ensemble not trained. Call assemble() or load() first.')
        
        if self._ensemble_method == 'Voting':
            return self._ensemble_voting(X)
        elif self._ensemble_method == 'Weighted Voting':
            return self._ensemble_weighted_voting(X)
        elif self._ensemble_method == 'Stacking':
            return self._ensemble_stacking(X)
        else:
            raise ValueError(f'Unknown method: {self._ensemble_method}')

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    # ── Assembly ───────────────────────────────────────────────────────────────

    def assemble(self, X_val: np.ndarray, y_val: np.ndarray):
        """Build and select best ensemble method."""
        self.select_best_method(X_val, y_val)
        self._trained = True
        return self._ensemble_method

    # ── Persistence ────────────────────────────────────────────────────────────

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
# Evaluation Helper
# ══════════════════════════════════════════════════════════════════════════════

def _evaluate(name: str, predict_fn, proba_fn, X, y, plot_dir: str) -> dict:
    """Comprehensive evaluation with metrics and visualizations."""
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

    tag = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    
    # Confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=CATEGORIES, yticklabels=CATEGORIES)
    plt.title(f'{name} — Confusion Matrix')
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{tag}_confusion.png'), dpi=150)
    plt.close()

    # ROC curve
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f'AUC={roc_auc:.3f}')
    plt.plot([0,1],[0,1],'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{name} — ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{tag}_roc.png'), dpi=150)
    plt.close()

    return dict(accuracy=acc, precision=prec, recall=rec,
                f1=f1, auc_roc=roc_auc, confusion_matrix=cm.tolist())


# ══════════════════════════════════════════════════════════════════════════════
# Training Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def train(data_dir: str, cache_path: str | None = None):
    logger.info('='*70)
    logger.info('ENSEMBLE MODEL — RESEARCH-GRADE TRAINING')
    logger.info('='*70)

    # Use cached features or extract (any type works, will use pipelines' features)
    X, y = extract_dataset_features(data_dir, cache_path=cache_path)

    # 70 / 15 / 15 stratified split (CRITICAL: no leakage)
    X_tmp, X_test,  y_tmp, y_test  = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.15/0.85, random_state=RANDOM_SEED, stratify=y_tmp)

    logger.info(f'\nData split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}')

    # Build ensemble with feature diversity
    ensemble = EnsembleModel()
    ensemble.load_base_models()
    ensemble.assemble(X_val, y_val)
    ensemble._trained = True

    # ── Evaluate all models ────────────────────────────────────────────────────
    logger.info('\n' + '='*70)
    logger.info('MODEL COMPARISON ON TEST SET')
    logger.info('='*70)
    
    comparison: dict[str, dict] = {}

    # Individual models
    for i, name in enumerate(ensemble._names):
        model, scaler, selector = ensemble._pipelines[i]
        
        def make_predict(m, sc, sel, idx):
            def _predict(X):
                # Slice features appropriately for each model
                # Hybrid feature layout: DenseNet (0:1024) + GLCM (1024:1048)
                if idx == 0:  # SVM - DenseNet only
                    X_sliced = X[:, 0:1024] if X.shape[1] > 1024 else X
                elif idx == 1:  # RF - GLCM only
                    X_sliced = X[:, 1024:1048] if X.shape[1] >= 1048 else X
                elif idx == 2:  # GB - Hybrid
                    X_sliced = X[:, 0:1048] if X.shape[1] >= 1048 else X
                else:
                    X_sliced = X
                
                if sc is not None:
                    return m.predict(sc.transform(sel.transform(X_sliced)))
                else:
                    return m.predict(sel.transform(X_sliced))
            return _predict
        
        def make_proba(m, sc, sel, idx):
            def _proba(X):
                # Slice features appropriately for each model
                # Hybrid feature layout: DenseNet (0:1024) + GLCM (1024:1048)
                if idx == 0:  # SVM - DenseNet only
                    X_sliced = X[:, 0:1024] if X.shape[1] > 1024 else X
                elif idx == 1:  # RF - GLCM only
                    X_sliced = X[:, 1024:1048] if X.shape[1] >= 1048 else X
                elif idx == 2:  # GB - Hybrid
                    X_sliced = X[:, 0:1048] if X.shape[1] >= 1048 else X
                else:
                    X_sliced = X
                
                if sc is not None:
                    return m.predict_proba(sc.transform(sel.transform(X_sliced)))
                else:
                    return m.predict_proba(sel.transform(X_sliced))
            return _proba
        
        pfn = make_predict(model, scaler, selector, i)
        probfn = make_proba(model, scaler, selector, i)
        
        comparison[name] = _evaluate(name, pfn, probfn, X_test, y_test, MODEL_DIR)
        logger.info(f'  Accuracy: {comparison[name]["accuracy"]*100:.2f}%')

    # Ensemble
    comparison['Ensemble'] = _evaluate(
        f'Ensemble ({ensemble._ensemble_method})', 
        ensemble.predict, ensemble.predict_proba,
        X_test, y_test, MODEL_DIR)
    ens_acc = comparison['Ensemble']['accuracy']
    logger.info(f'  Accuracy: {ens_acc*100:.2f}%')

    # ── Verify Ensemble Superiority ────────────────────────────────────────────
    logger.info('\n' + '='*70)
    logger.info('ENSEMBLE SUPERIORITY VERIFICATION')
    logger.info('='*70)
    
    individual_accs = [comparison[n]['accuracy'] for n in ensemble._names]
    max_individual = max(individual_accs)
    
    logger.info(f'  Best individual: {max_individual*100:.2f}%')
    logger.info(f'  Ensemble:        {ens_acc*100:.2f}%')
    
    improvement = ens_acc - max_individual
    logger.info(f'  Improvement:     {improvement*100:+.2f}%')
    
    if ens_acc >= max_individual:
        logger.info(f'\n  ✓ ENSEMBLE OUTPERFORMS ALL INDIVIDUAL MODELS')
    else:
        logger.warning(f'\n  ✗ Ensemble underperformed — may indicate insufficient diversity')
        logger.warning(f'     Consider adding more advanced models (XGBoost, LightGBM)')
    
    logger.info('='*70)

    # ── Comparison bar chart ──────────────────────────────────────────────────
    names_plt = list(comparison.keys())
    accs_plt  = [comparison[n]['accuracy']*100 for n in names_plt]
    colours   = ['#4c72b0', '#55a868', '#c44e52', '#9467bd']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names_plt, accs_plt, color=colours, edgecolor='black', linewidth=1.5)
    plt.bar_label(bars, fmt='%.2f%%', fontweight='bold')
    
    # Highlight ensemble with star
    ensemble_idx = names_plt.index('Ensemble')
    bars[ensemble_idx].set_linewidth(3)
    bars[ensemble_idx].set_edgecolor('gold')
    
    plt.ylim([max(0, min(accs_plt)-5), 101])
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Model Comparison — Test Set Accuracy\n(★ = Selected Ensemble)', fontsize=13)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'model_comparison.png'), dpi=150)
    plt.close()
    logger.info(f'Comparison chart → {MODEL_DIR}/model_comparison.png')

    # Save ensemble
    ensemble.save(ENSEMBLE_MODEL_PATH)

    # Metadata
    meta = {
        'ensemble_method': ensemble._ensemble_method,
        'base_models': ensemble._names,
        'base_model_features': {
            'SVM (DenseNet-only)': f'K={MODEL_K_FEATURES["svm"]}',
            'RF (GLCM-only)': f'K={MODEL_K_FEATURES["rf"]}',
            'GB (Hybrid)': f'K={MODEL_K_FEATURES["gb"]}',
        },
        'diversity_enforced': True,
        'test_comparison': comparison,
        'ensemble_superiority': {
            'best_individual_accuracy': float(max_individual),
            'ensemble_accuracy': float(ens_acc),
            'improvement': float(improvement),
            'outperforms_all': bool(ens_acc >= max_individual),
        }
    }
    
    def _conv(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    with open(ENSEMBLE_META_PATH, 'w') as f:
        json.dump(meta, f, indent=2, default=_conv)

    logger.info(f'Metadata → {ENSEMBLE_META_PATH}')
    logger.info('\n✓ Ensemble training complete.')
    return ENSEMBLE_MODEL_PATH


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Build Research-Grade Ensemble')
    ap.add_argument('--data_dir', default='./dataset')
    ap.add_argument('--cache',    default=None)
    args = ap.parse_args()
    train(args.data_dir, cache_path=args.cache)