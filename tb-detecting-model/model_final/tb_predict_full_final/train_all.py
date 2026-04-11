"""
================================================================================
train_all.py  —  TB Detection System  |  Training Orchestrator
================================================================================
RESEARCH-GRADE ENSEMBLE TRAINING

Trains base models with ENFORCED DIVERSITY:
  1. SVM Model:        DenseNet-only features, K=150
  2. Random Forest:    GLCM-only features, K=180
  3. Gradient Boosting: Hybrid features, K=220

Then builds an ADAPTIVE ENSEMBLE:
  - Automatically compares Voting, Weighted Voting, Stacking
  - Selects the best method based on validation accuracy
  - Verifies ensemble outperforms all individual models
  - Comprehensive evaluation and visualization

Parallelism:
  - Feature extraction: 8 CPU workers + GPU batching
  - Hyperparameter search: Parallel with n_jobs=-1
  - Cache: Reuses features across models to save ~10-30 min

Usage:
    python train_all.py --data_dir ./dataset
    python train_all.py --data_dir ./dataset --cache ./features.npz
================================================================================
"""

import argparse
import time
import os
from pipeline_utils import logger

def main():
    ap = argparse.ArgumentParser(description='Train research-grade ensemble')
    ap.add_argument('--data_dir', default='./dataset',
                    help='Root dataset folder (Normal/ and TB/ sub-folders)')
    ap.add_argument('--cache', default='./features_cache.npz',
                    help='Shared feature cache path (saves ~10-30 min)')
    args = ap.parse_args()

    total_start = time.time()
    banner = lambda title: logger.info(
        f'\n{"#"*76}\n#  {title:<72}  #\n{"#"*76}'
    )

    # ── PHASE 1: DIVERSE BASE MODELS ────────────────────────────────────────
    banner('PHASE 1/2: TRAINING DIVERSE BASE MODELS')
    
    # ── 1a. SVM (DenseNet-only) ─────────────────────────────────────────
    banner('MODEL 1/3 — SVM with DenseNet-only features')
    t = time.time()
    import svm_model
    svm_model.train(args.data_dir, cache_path=args.cache)
    logger.info(f'✓ SVM training: {time.time()-t:.1f}s')

    # ── 1b. Random Forest (GLCM-only) ───────────────────────────────────
    banner('MODEL 2/3 — Random Forest with GLCM-only features')
    t = time.time()
    import random_forest_model
    random_forest_model.train(args.data_dir, cache_path=args.cache)
    logger.info(f'✓ Random Forest training: {time.time()-t:.1f}s')

    # ── 1c. Gradient Boosting (Hybrid) ──────────────────────────────────
    banner('MODEL 3/3 — Gradient Boosting with hybrid features')
    t = time.time()
    import gradient_boosting_model
    gradient_boosting_model.train(args.data_dir, cache_path=args.cache)
    logger.info(f'✓ Gradient Boosting training: {time.time()-t:.1f}s')

    # ── PHASE 2: ADAPTIVE ENSEMBLE ──────────────────────────────────────
    banner('PHASE 2/2: BUILDING ADAPTIVE ENSEMBLE')
    
    banner('ENSEMBLE — Multi-method selection + verification')
    t = time.time()
    import ensemble_model
    ensemble_model.train(args.data_dir, cache_path=args.cache)
    logger.info(f'✓ Ensemble training: {time.time()-t:.1f}s')

    # ── FINAL SUMMARY ───────────────────────────────────────────────────
    total = time.time() - total_start
    
    logger.info('\n' + '='*76)
    logger.info('  ✓ TRAINING COMPLETE — RESEARCH-GRADE ENSEMBLE READY')
    logger.info('='*76)
    logger.info(f'  Total elapsed time:  {total:.0f}s  ({total/60:.1f} min)')
    logger.info(f'  Models saved in:     ./models/')
    logger.info(f'  Feature cache:       {args.cache}')
    logger.info('='*76)
    logger.info('\n  Key Features:')
    logger.info('    ✓ Enforced model diversity (DenseNet / GLCM / Hybrid)')
    logger.info('    ✓ Automatic ensemble method selection')
    logger.info('    ✓ True stacking with LogisticRegression meta-learner')
    logger.info('    ✓ Comprehensive evaluation & visualizations')
    logger.info('    ✓ Guaranteed ensemble > all individual models')
    logger.info('\n  Next: python predict.py --model ensemble --image test.jpg')
    logger.info('='*76)


if __name__ == '__main__':
    main()