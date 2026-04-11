"""
================================================================================
train_all.py  —  TB Detection System  |  Training Orchestrator
================================================================================
Trains all four models in the correct order with a shared feature cache to
avoid redundant DenseNet forward passes (the most expensive step).

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
    ap = argparse.ArgumentParser(description='Train all TB detection models')
    ap.add_argument('--data_dir', default='./dataset',
                    help='Root dataset folder (Normal/ and TB/ sub-folders)')
    ap.add_argument('--cache', default='./features_cache.npz',
                    help='Shared feature cache path (saves ~10-30 min)')
    args = ap.parse_args()

    total_start = time.time()
    banner = lambda title: logger.info(f'\n{"#"*70}\n#  {title}\n{"#"*70}')

    # ── 1. SVM ─────────────────────────────────────────────────────────────────
    banner('STEP 1/4 — SVM Model')
    t = time.time()
    import svm_model
    svm_model.train(args.data_dir, cache_path=args.cache)
    logger.info(f'SVM done in {time.time()-t:.1f}s')

    # ── 2. Random Forest ───────────────────────────────────────────────────────
    banner('STEP 2/4 — Random Forest Model')
    t = time.time()
    import random_forest_model
    random_forest_model.train(args.data_dir, cache_path=args.cache)
    logger.info(f'Random Forest done in {time.time()-t:.1f}s')

    # ── 3. Gradient Boosting ───────────────────────────────────────────────────
    banner('STEP 3/4 — Gradient Boosting Model')
    t = time.time()
    import gradient_boosting_model
    gradient_boosting_model.train(args.data_dir, cache_path=args.cache)
    logger.info(f'Gradient Boosting done in {time.time()-t:.1f}s')

    # ── 4. Ensemble ────────────────────────────────────────────────────────────
    banner('STEP 4/4 — Ensemble Model')
    t = time.time()
    import ensemble_model
    ensemble_model.train(args.data_dir, cache_path=args.cache)
    logger.info(f'Ensemble done in {time.time()-t:.1f}s')

    total = time.time() - total_start
    logger.info(f'\n{"="*70}')
    logger.info(f'  ALL MODELS TRAINED SUCCESSFULLY')
    logger.info(f'  Total time: {total:.0f}s ({total/60:.1f} minutes)')
    logger.info(f'  Models saved in: ./models/')
    logger.info(f'{"="*70}')


if __name__ == '__main__':
    main()