"""
================================================================================
random_forest_model.py  —  TB Detection System  |  Module 2: Random Forest
================================================================================
Pipeline:
    Image → Segmentation → Feature Extraction → Feature Selection
          → Hyperparameter Optimisation → Random Forest Classification

Saves:
    models/random_forest_model.pkl
    models/rf_scaler.pkl
    models/rf_selector.pkl
    models/rf_metadata.json

Run standalone:
    python random_forest_model.py --data_dir ./dataset
================================================================================
"""

import os
import json
import pickle
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.ensemble import RandomForestClassifier
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
    RANDOM_SEED, CATEGORIES, extract_dataset_features, logger
)

MODEL_DIR = './models'
os.makedirs(MODEL_DIR, exist_ok=True)

RF_MODEL_PATH    = os.path.join(MODEL_DIR, 'random_forest_model.pkl')
RF_SCALER_PATH   = os.path.join(MODEL_DIR, 'rf_scaler.pkl')
RF_SELECTOR_PATH = os.path.join(MODEL_DIR, 'rf_selector.pkl')
RF_META_PATH     = os.path.join(MODEL_DIR, 'rf_metadata.json')

RF_PARAM_DIST = {
    'n_estimators':      [100, 200, 300, 500],
    'max_depth':         [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 4],
    'max_features':      ['sqrt', 'log2', 0.3],
}
K_FEATURES = 200


def evaluate_model(model, X, y, label='Test', plot_dir='.'):
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

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=CATEGORIES, yticklabels=CATEGORIES)
    plt.title(f'Random Forest — Confusion Matrix ({label})')
    plt.ylabel('True'); plt.xlabel('Predicted'); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'rf_confusion_{label.lower()}.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title(f'Random Forest — ROC Curve ({label})')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'rf_roc_{label.lower()}.png'), dpi=150)
    plt.close()

    return dict(accuracy=acc, precision=prec, recall=rec,
                f1=f1, auc_roc=roc_auc, confusion_matrix=cm.tolist())


def train(data_dir: str, cache_path: str | None = None):
    logger.info('='*70)
    logger.info('RANDOM FOREST MODEL — TRAINING START')
    logger.info('='*70)

    X, y = extract_dataset_features(data_dir, cache_path=cache_path)

    X_tmp, X_test,  y_tmp, y_test  = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.15/0.85, random_state=RANDOM_SEED, stratify=y_tmp)

    logger.info(f'Split — train:{len(X_train)}  val:{len(X_val)}  test:{len(X_test)}')

    # Feature selection — fit on train only
    selector = SelectKBest(f_classif, k=K_FEATURES)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_val_sel   = selector.transform(X_val)
    X_test_sel  = selector.transform(X_test)

    # Scaling
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_sel)
    X_val_sc   = scaler.transform(X_val_sel)
    X_test_sc  = scaler.transform(X_test_sel)

    # Class weights
    cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight = dict(enumerate(cw))

    # Hyperparameter search
    logger.info('RandomizedSearchCV for Random Forest…')
    base_rf = RandomForestClassifier(
        class_weight=class_weight, random_state=RANDOM_SEED, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    search = RandomizedSearchCV(
        base_rf, RF_PARAM_DIST, n_iter=20,
        cv=cv, scoring='accuracy', n_jobs=-1,
        random_state=RANDOM_SEED, verbose=1
    )
    search.fit(X_train_sc, y_train)

    best_params = search.best_params_
    logger.info(f'Best params: {best_params}')
    logger.info(f'Best CV accuracy: {search.best_score_*100:.2f}%')

    # Refit on train+val
    X_full_sc = np.vstack([X_train_sc, X_val_sc])
    y_full    = np.concatenate([y_train, y_val])
    rf = RandomForestClassifier(
        **best_params, class_weight=class_weight,
        random_state=RANDOM_SEED, n_jobs=-1
    )
    rf.fit(X_full_sc, y_full)

    # Evaluation via wrapper
    class _WrappedRF:
        def __init__(self, sel, sc, m):
            self._sel, self._sc, self._m = sel, sc, m
        def predict(self, X):
            return self._m.predict(self._sc.transform(self._sel.transform(X)))
        def predict_proba(self, X):
            return self._m.predict_proba(self._sc.transform(self._sel.transform(X)))

    test_metrics = evaluate_model(
        _WrappedRF(selector, scaler, rf), X_test, y_test,
        label='Test', plot_dir=MODEL_DIR)
    acc = test_metrics['accuracy']
    logger.info(f'RF TEST ACCURACY: {acc*100:.2f}%')
    if acc < 0.85:
        logger.warning(f'Accuracy {acc*100:.1f}% below 85% target.')

    # Feature importance plot
    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:20]
    plt.figure(figsize=(10, 5))
    plt.bar(range(20), importances[top_idx])
    plt.title('Random Forest — Top 20 Feature Importances')
    plt.xlabel('Feature index'); plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'rf_feature_importance.png'), dpi=150)
    plt.close()

    # Save artifacts
    with open(RF_MODEL_PATH,    'wb') as f: pickle.dump(rf,       f)
    with open(RF_SCALER_PATH,   'wb') as f: pickle.dump(scaler,   f)
    with open(RF_SELECTOR_PATH, 'wb') as f: pickle.dump(selector, f)

    meta = dict(model='RandomForest', best_params=best_params,
                cv_best_score=float(search.best_score_),
                k_features=K_FEATURES, **test_metrics)
    with open(RF_META_PATH, 'w') as f:
        json.dump(meta, f, indent=2)

    logger.info(f'Saved → {RF_MODEL_PATH}')
    logger.info('Random Forest training complete.')
    return RF_MODEL_PATH


def load_rf_pipeline():
    for p in [RF_MODEL_PATH, RF_SCALER_PATH, RF_SELECTOR_PATH]:
        if not os.path.exists(p):
            raise FileNotFoundError(f'Missing RF artifact: {p}. Run random_forest_model.py first.')
    with open(RF_MODEL_PATH,    'rb') as f: rf       = pickle.load(f)
    with open(RF_SCALER_PATH,   'rb') as f: scaler   = pickle.load(f)
    with open(RF_SELECTOR_PATH, 'rb') as f: selector = pickle.load(f)
    logger.info('Random Forest pipeline loaded.')
    return rf, scaler, selector


def predict_single(feature_vector: np.ndarray):
    rf, scaler, selector = load_rf_pipeline()
    x = selector.transform(feature_vector.reshape(1, -1))
    x = scaler.transform(x)
    idx   = rf.predict(x)[0]
    proba = rf.predict_proba(x)[0]
    return CATEGORIES[idx], float(proba[idx]), proba


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Train Random Forest TB Detector')
    ap.add_argument('--data_dir', default='./dataset')
    ap.add_argument('--cache',    default=None)
    args = ap.parse_args()
    train(args.data_dir, cache_path=args.cache)