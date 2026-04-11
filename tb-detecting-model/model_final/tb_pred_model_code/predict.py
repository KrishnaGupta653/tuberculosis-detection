"""
================================================================================
predict.py  —  TB Detection System  |  Module 5: CLI Inference
================================================================================
Usage:
    python predict.py --model svm       --image  test.jpg
    python predict.py --model rf        --image  test.jpg
    python predict.py --model gb        --image  test.jpg
    python predict.py --model ensemble  --image  test.jpg
    python predict.py --model ensemble  --folder test_images/
    python predict.py --model rf        --folder test_images/ --save_csv results.csv

Outputs (per image):
    Predicted class   : Normal | TB
    Confidence score  : 0–100 %
    Risk level        : Low | Medium | High
================================================================================
"""

import os
import sys
import csv
import logging
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import cv2

# ── shared pipeline ────────────────────────────────────────────────────────────
from pipeline_utils import (
    CATEGORIES, LungSegmentor, extract_features, logger
)

# ── model loaders ─────────────────────────────────────────────────────────────
import svm_model               as svm_mod
import random_forest_model     as rf_mod
import gradient_boosting_model as gb_mod
from ensemble_model import EnsembleModel

# ══════════════════════════════════════════════════════════════════════════════
# Model registry
# ══════════════════════════════════════════════════════════════════════════════

_SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


def _load_model(model_key: str):
    """
    Return a callable that accepts a raw feature vector (1048,) and returns
    (label_str, confidence_float, proba_array).
    """
    key = model_key.lower()
    if key == 'svm':
        model, scaler, selector = svm_mod.load_svm_pipeline()
        def _predict(feat):
            x = scaler.transform(selector.transform(feat.reshape(1, -1)))
            idx   = model.predict(x)[0]
            proba = model.predict_proba(x)[0]
            return CATEGORIES[idx], float(proba[idx]), proba
        return _predict

    elif key in ('rf', 'random_forest', 'randomforest'):
        model, scaler, selector = rf_mod.load_rf_pipeline()
        def _predict(feat):
            x = scaler.transform(selector.transform(feat.reshape(1, -1)))
            idx   = model.predict(x)[0]
            proba = model.predict_proba(x)[0]
            return CATEGORIES[idx], float(proba[idx]), proba
        return _predict

    elif key in ('gb', 'gradient_boosting', 'gradientboosting'):
        model, scaler, selector = gb_mod.load_gb_pipeline()
        def _predict(feat):
            x = scaler.transform(selector.transform(feat.reshape(1, -1)))
            idx   = model.predict(x)[0]
            proba = model.predict_proba(x)[0]
            return CATEGORIES[idx], float(proba[idx]), proba
        return _predict

    elif key == 'ensemble':
        ensemble = EnsembleModel.load()
        def _predict(feat):
            proba = ensemble.predict_proba(feat.reshape(1, -1))[0]
            idx   = int(np.argmax(proba))
            return CATEGORIES[idx], float(proba[idx]), proba
        return _predict

    else:
        raise ValueError(
            f'Unknown model "{model_key}". '
            'Choose from: svm | rf | gb | ensemble'
        )


# ══════════════════════════════════════════════════════════════════════════════
# Single-image prediction
# ══════════════════════════════════════════════════════════════════════════════

_segmentor = LungSegmentor()   # singleton — avoids repeated init


def _risk_level(confidence: float, label: str) -> str:
    """
    Risk level based on TB confidence (i.e. probability of TB class).
    If the prediction is Normal, risk is Low regardless of confidence.
    """
    if label == 'Normal':
        return 'Low'
    if confidence >= 0.90:
        return 'High'
    if confidence >= 0.70:
        return 'Medium'
    return 'Low'


def predict_image(image_path: str, predict_fn) -> dict:
    """
    Run full pipeline on a single image file.

    Parameters
    ----------
    image_path : str — path to chest X-ray image
    predict_fn : callable returned by _load_model()

    Returns
    -------
    dict with keys: image, prediction, confidence, risk_level,
                    prob_normal, prob_tb, error
    """
    result = {
        'image':      image_path,
        'prediction': None,
        'confidence': None,
        'risk_level': None,
        'prob_normal': None,
        'prob_tb':     None,
        'error':       None,
    }

    # ── validation ────────────────────────────────────────────────────────────
    if not os.path.isfile(image_path):
        result['error'] = f'File not found: {image_path}'
        logger.error(result['error'])
        return result

    ext = os.path.splitext(image_path)[1].lower()
    if ext not in _SUPPORTED_EXTENSIONS:
        result['error'] = f'Unsupported format "{ext}".'
        logger.error(result['error'])
        return result

    try:
        # Step 1: Segmentation
        logger.info(f'[1/4] Segmenting: {os.path.basename(image_path)}')
        seg, mask = _segmentor.segment(image_path)
        if seg is None:
            raise ValueError('Segmentation returned None — possibly corrupt image.')

        # Step 2: Feature extraction
        logger.info('[2/4] Extracting hybrid features…')
        feats = extract_features(seg, mask)
        if feats is None:
            raise ValueError('Feature extraction failed.')

        # Step 3 + 4: Selection + Scaling + Prediction (inside predict_fn)
        logger.info('[3/4] Selecting features & scaling…')
        logger.info('[4/4] Classifying…')
        label, confidence, proba = predict_fn(feats)

        risk = _risk_level(confidence, label)

        result.update({
            'prediction':  label,
            'confidence':  round(float(confidence), 4),
            'risk_level':  risk,
            'prob_normal': round(float(proba[0]), 4),
            'prob_tb':     round(float(proba[1]), 4),
        })

    except Exception as exc:
        result['error'] = str(exc)
        logger.error(f'Prediction failed for {image_path}: {exc}')

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Pretty printing
# ══════════════════════════════════════════════════════════════════════════════

def _print_result(r: dict, model_name: str):
    sep = '─' * 60
    if r['error']:
        print(f'\n{sep}')
        print(f'  ERROR: {r["error"]}')
        print(f'{sep}\n')
        return

    tb_bar_len = int(r['prob_tb'] * 30)
    nm_bar_len = 30 - tb_bar_len

    print(f'\n{"="*60}')
    print(f'  TB DETECTION RESULT  [{model_name.upper()}]')
    print(f'{"="*60}')
    print(f'  Image      : {os.path.basename(r["image"])}')
    print(f'  Prediction : {r["prediction"]}')
    print(f'  Confidence : {r["confidence"]*100:.1f}%')
    print(f'  Risk Level : {r["risk_level"]}')
    print(f'{sep}')
    print(f'  Probability breakdown:')
    print(f'    Normal  {r["prob_normal"]*100:5.1f}%  {"█"*nm_bar_len}')
    print(f'    TB      {r["prob_tb"]*100:5.1f}%  {"█"*tb_bar_len}')
    print(f'{"="*60}\n')


def _print_batch_summary(results: list[dict], model_name: str):
    total = len(results)
    errors = sum(1 for r in results if r['error'])
    tb_cases = sum(1 for r in results if r['prediction'] == 'TB')
    normal_cases = total - errors - tb_cases

    print(f'\n{"="*60}')
    print(f'  BATCH SUMMARY  [{model_name.upper()}]')
    print(f'{"="*60}')
    print(f'  Total images  : {total}')
    print(f'  Errors        : {errors}')
    print(f'  Normal        : {normal_cases}')
    print(f'  TB (positive) : {tb_cases}')
    if total - errors > 0:
        high_risk = sum(1 for r in results if r['risk_level'] == 'High')
        print(f'  High Risk     : {high_risk}')
    print(f'{"="*60}\n')


# ══════════════════════════════════════════════════════════════════════════════
# CSV export
# ══════════════════════════════════════════════════════════════════════════════

def _save_csv(results: list[dict], csv_path: str):
    fieldnames = ['image', 'prediction', 'confidence', 'risk_level',
                  'prob_normal', 'prob_tb', 'error']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logger.info(f'Results saved to CSV: {csv_path}')
    print(f'  Results saved → {csv_path}')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='TB Detection Inference — predict.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py --model svm      --image  chest.jpg
  python predict.py --model rf       --image  chest.png
  python predict.py --model gb       --image  chest.jpg
  python predict.py --model ensemble --image  chest.jpg
  python predict.py --model ensemble --folder test_images/
  python predict.py --model ensemble --folder test_images/ --save_csv out.csv
        """
    )
    ap.add_argument('--model', required=True,
                    choices=['svm', 'rf', 'gb', 'ensemble'],
                    help='Which model to use for prediction')
    ap.add_argument('--image',  default=None,
                    help='Path to a single chest X-ray image')
    ap.add_argument('--folder', default=None,
                    help='Path to folder of X-ray images (batch mode)')
    ap.add_argument('--save_csv', default=None,
                    help='Optional CSV path to save batch results')
    ap.add_argument('--verbose', action='store_true',
                    help='Enable DEBUG logging')
    args = ap.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.image is None and args.folder is None:
        ap.error('Provide at least one of --image or --folder.')

    # ── Load model ────────────────────────────────────────────────────────────
    logger.info(f'Loading model: {args.model}')
    try:
        predict_fn = _load_model(args.model)
    except FileNotFoundError as e:
        print(f'\nERROR: {e}\n')
        sys.exit(1)
    except Exception as e:
        print(f'\nUnexpected error loading model: {e}\n')
        sys.exit(1)

    all_results: list[dict] = []

    # ── Single image ──────────────────────────────────────────────────────────
    if args.image:
        r = predict_image(args.image, predict_fn)
        _print_result(r, args.model)
        all_results.append(r)

    # ── Batch folder ──────────────────────────────────────────────────────────
    if args.folder:
        if not os.path.isdir(args.folder):
            print(f'\nERROR: Folder not found: {args.folder}\n')
            sys.exit(1)

        files = sorted([
            os.path.join(args.folder, f)
            for f in os.listdir(args.folder)
            if os.path.splitext(f)[1].lower() in _SUPPORTED_EXTENSIONS
        ])

        if not files:
            print(f'\nERROR: No supported images found in {args.folder}\n')
            sys.exit(1)

        logger.info(f'Batch mode: {len(files)} images in {args.folder}')
        for img_path in files:
            r = predict_image(img_path, predict_fn)
            _print_result(r, args.model)
            all_results.append(r)

        _print_batch_summary(all_results, args.model)

    # ── Optional CSV export ───────────────────────────────────────────────────
    if args.save_csv and all_results:
        _save_csv(all_results, args.save_csv)


if __name__ == '__main__':
    main()