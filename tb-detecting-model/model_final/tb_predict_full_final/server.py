"""
================================================================================
server.py  —  TB Detection System  |  HTTP Server for Image Inference
================================================================================
Usage:
    python server.py                    # Runs on http://localhost:5000
    python server.py --host 0.0.0.0 --port 8000

Endpoints:
    POST /predict
        - model: svm | rf | gb | ensemble (required)
        - image: binary file upload (required)
        - format: json (default) | html (optional)
        Returns: JSON or HTML with prediction, confidence, risk_level, probabilities

    POST /predict_url
        - model: svm | rf | gb | ensemble (required)
        - image_url: URL to image (required)
        - format: json (default) | html (optional)
        Returns: JSON or HTML with prediction, confidence, risk_level, probabilities

    GET /health
        Returns: {"status": "ok"}

    GET /models
        Returns: list of available models

Examples:
    # Single image upload (JSON response)
    curl -X POST -F "image=@test.jpg" -F "model=ensemble" \\
         http://localhost:5000/predict

    # Single image upload (HTML response - view in browser)
    curl -X POST -F "image=@test.jpg" -F "model=ensemble" -F "format=html" \\
         http://localhost:5000/predict

    # Via ngrok
    curl -X POST -F "image=@test.jpg" -F "model=ensemble" -F "format=html" \\
         https://YOUR-NGROK-URL/predict

    # From URL (JSON)
    curl -X POST -d '{"image_url":"https://...", "model":"svm"}' \\
         -H "Content-Type: application/json" \\
         http://localhost:5000/predict_url
    
    # From URL (HTML)
    curl -X POST -d '{"image_url":"https://...", "model":"svm", "format":"html"}' \\
         -H "Content-Type: application/json" \\
         http://localhost:5000/predict_url
================================================================================
"""

import os
import sys
import logging
import argparse
import warnings
import tempfile
import io
from werkzeug.utils import secure_filename

import numpy as np
import cv2
warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify, render_template_string
import requests

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
# Flask App Setup
# ══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB max file size
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif'}

# Global model cache
_model_cache = {}
_segmentor = LungSegmentor()


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _load_model(model_key: str):
    """
    Return a callable that accepts a raw feature vector (1048,) and returns
    (label_str, confidence_float, proba_array).
    Uses cache to avoid reloading.
    """
    key = model_key.lower()
    
    if key in _model_cache:
        return _model_cache[key]
    
    if key == 'svm':
        model, scaler, selector = svm_mod.load_svm_pipeline()
        def _predict(feat):
            # Selector was trained on 1024 DenseNet features
            x = scaler.transform(selector.transform(feat[0:1024].reshape(1, -1)))
            idx   = model.predict(x)[0]
            proba = model.predict_proba(x)[0]
            return CATEGORIES[idx], float(proba[idx]), proba
        _model_cache[key] = _predict
        return _predict

    elif key in ('rf', 'random_forest', 'randomforest'):
        model, scaler, selector = rf_mod.load_rf_pipeline()
        def _predict(feat):
            # Selector was trained on 1024 DenseNet features
            x = scaler.transform(selector.transform(feat[0:1024].reshape(1, -1)))
            idx   = model.predict(x)[0]
            proba = model.predict_proba(x)[0]
            return CATEGORIES[idx], float(proba[idx]), proba
        _model_cache[key] = _predict
        return _predict

    elif key in ('gb', 'gradient_boosting', 'gradientboosting'):
        model, scaler, selector = gb_mod.load_gb_pipeline()
        def _predict(feat):
            # Selector was trained on 1024 DenseNet features
            x = scaler.transform(selector.transform(feat[0:1024].reshape(1, -1)))
            idx   = model.predict(x)[0]
            proba = model.predict_proba(x)[0]
            return CATEGORIES[idx], float(proba[idx]), proba
        _model_cache[key] = _predict
        return _predict

    elif key == 'ensemble':
        ensemble = EnsembleModel.load()
        def _predict(feat):
            # Manual ensemble voting with proper feature slicing per model
            # Hybrid feature layout: DenseNet (0:1024) + GLCM (1024:1048)
            svm, svm_sc, svm_sel = svm_mod.load_svm_pipeline()
            rf,  rf_sc,  rf_sel  = rf_mod.load_rf_pipeline()
            gb,  gb_sc,  gb_sel  = gb_mod.load_gb_pipeline()
            
            # All selectors expect 1024 DenseNet features (what they were trained on)
            proba_svm = svm.predict_proba(
                svm_sc.transform(svm_sel.transform(feat[0:1024].reshape(1, -1)))
            )[0]  # SVM: SelectKBest on DenseNet
            proba_rf = rf.predict_proba(
                rf_sc.transform(rf_sel.transform(feat[0:1024].reshape(1, -1)))
            )[0]  # RF: SelectKBest on DenseNet
            proba_gb = gb.predict_proba(
                gb_sc.transform(gb_sel.transform(feat[0:1024].reshape(1, -1)))
            )[0]  # GB: SelectKBest on DenseNet
            
            # Average for ensemble voting
            proba_ens = (proba_svm + proba_rf + proba_gb) / 3.0
            idx = int(np.argmax(proba_ens))
            return CATEGORIES[idx], float(proba_ens[idx]), proba_ens
        _model_cache[key] = _predict
        return _predict

    else:
        raise ValueError(
            f'Unknown model "{model_key}". '
            'Choose from: svm | rf | gb | ensemble'
        )


def _risk_level(confidence: float, label: str) -> str:
    """Risk level based on TB confidence."""
    if label == 'Normal':
        return 'Low'
    if confidence >= 0.90:
        return 'High'
    if confidence >= 0.70:
        return 'Medium'
    return 'Low'


def _format_result_html(result: dict, model_name: str, filename: str = "") -> str:
    """Format prediction result as HTML."""
    if result.get('error'):
        return f"""
        <html>
        <head>
            <style>
                body {{ font-family: monospace; background: #f5f5f5; padding: 20px; }}
                .error {{ background: #ffcccc; border: 2px solid #cc0000; padding: 20px; 
                         border-radius: 8px; color: #660000; font-size: 18px; }}
            </style>
        </head>
        <body>
            <pre class="error">
════════════════════════════════════════════════════════════
  ERROR: {result['error']}
════════════════════════════════════════════════════════════
            </pre>
        </body>
        </html>
        """
    
    conf_pct = result.get('confidence', 0) * 100
    prob_normal_pct = result.get('prob_normal', 0) * 100
    prob_tb_pct = result.get('prob_tb', 0) * 100
    
    # Bar visualization (30 chars wide)
    tb_bar_len = int(prob_tb_pct / 100 * 30)
    nm_bar_len = 30 - tb_bar_len
    tb_bar = "█" * tb_bar_len
    nm_bar = "█" * nm_bar_len
    
    # Color coding
    prediction = result.get('prediction', 'Unknown')
    risk = result.get('risk_level', 'Unknown')
    
    prediction_color = '#cc0000' if prediction == 'TB' else '#00aa00'
    risk_color_map = {'High': '#ff0000', 'Medium': '#ff9900', 'Low': '#00aa00'}
    risk_color = risk_color_map.get(risk, '#999999')
    
    html = f"""
    <html>
    <head>
        <style>
            body {{ 
                font-family: monospace; 
                background: #1e1e1e; 
                color: #d4d4d4;
                padding: 30px;
                margin: 0;
            }}
            .container {{
                background: #252526;
                border: 2px solid #007acc;
                border-radius: 8px;
                padding: 20px;
                max-width: 700px;
                margin: 0 auto;
            }}
            .header {{
                text-align: center;
                border-bottom: 2px solid #007acc;
                padding-bottom: 15px;
                margin-bottom: 20px;
                font-weight: bold;
                font-size: 18px;
            }}
            .result-row {{
                display: flex;
                justify-content: space-between;
                padding: 10px 0;
                border-bottom: 1px solid #3e3e42;
            }}
            .result-row:last-child {{
                border-bottom: none;
            }}
            .label {{
                font-weight: bold;
                color: #9cdcfe;
                min-width: 150px;
            }}
            .value {{
                color: #ce9178;
            }}
            .prediction-value {{
                color: {prediction_color};
                font-weight: bold;
                font-size: 16px;
            }}
            .risk-value {{
                color: {risk_color};
                font-weight: bold;
            }}
            .prob-section {{
                margin-top: 20px;
                padding-top: 15px;
                border-top: 2px solid #007acc;
            }}
            .prob-row {{
                display: flex;
                padding: 10px 0;
                align-items: center;
            }}
            .prob-label {{
                width: 100px;
                color: #9cdcfe;
            }}
            .prob-pct {{
                width: 60px;
                text-align: right;
                color: #ce9178;
                font-weight: bold;
            }}
            .prob-bar {{
                flex: 1;
                background: #3e3e42;
                border-radius: 3px;
                margin-left: 15px;
                overflow: hidden;
                height: 20px;
            }}
            .bar-fill {{
                height: 100%;
                background: linear-gradient(90deg, #007acc, #005a9e);
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-size: 12px;
                font-weight: bold;
            }}
            .divider {{
                text-align: center;
                color: #666;
                padding: 10px 0;
                font-size: 12px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                ═══════════════════════════════════════════════════════════<br>
                TB DETECTION RESULT [{model_name.upper()}]<br>
                ═══════════════════════════════════════════════════════════
            </div>
            
            <div class="result-row">
                <span class="label">Image</span>
                <span class="value">{filename}</span>
            </div>
            
            <div class="result-row">
                <span class="label">Prediction</span>
                <span class="prediction-value">{prediction}</span>
            </div>
            
            <div class="result-row">
                <span class="label">Confidence</span>
                <span class="value">{conf_pct:.1f}%</span>
            </div>
            
            <div class="result-row">
                <span class="label">Risk Level</span>
                <span class="risk-value">{risk}</span>
            </div>
            
            <div class="divider">───────────────────────────────────────────────────────────</div>
            
            <div class="prob-section">
                <div style="margin-bottom: 15px; color: #9cdcfe; font-weight: bold;">
                    Probability Breakdown:
                </div>
                
                <div class="prob-row">
                    <span class="prob-label">Normal</span>
                    <span class="prob-pct">{prob_normal_pct:5.1f}%</span>
                    <div class="prob-bar">
                        <div class="bar-fill" style="width: {prob_normal_pct}%; background: linear-gradient(90deg, #107c10, #107c10);">
                            {'█' * nm_bar_len if nm_bar_len > 5 else ''}
                        </div>
                    </div>
                </div>
                
                <div class="prob-row">
                    <span class="prob-label">TB</span>
                    <span class="prob-pct">{prob_tb_pct:5.1f}%</span>
                    <div class="prob-bar">
                        <div class="bar-fill" style="width: {prob_tb_pct}%; background: linear-gradient(90deg, #f44747, #d32f2f);">
                            {'█' * tb_bar_len if tb_bar_len > 5 else ''}
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="divider">═══════════════════════════════════════════════════════════</div>
        </div>
    </body>
    </html>
    """
    return html


def _process_image_file(file_obj) -> dict:
    """
    Process an image file object.
    Returns dict with keys: image_data (numpy BGR), filename, error
    """
    try:
        # Read file into memory
        file_bytes = file_obj.read()
        if not file_bytes:
            return {'error': 'Empty file uploaded'}
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(file_bytes, np.uint8)
        
        # Decode image in BGR color (required by LungSegmentor)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {'error': 'Failed to decode image. Check format.'}
        
        return {
            'image_data': img,
            'filename': getattr(file_obj, 'filename', 'unknown'),
            'error': None
        }
    except Exception as e:
        return {'error': f'Image processing error: {str(e)}'}


def _predict_from_image(image_data, predict_fn, model_key: str) -> dict:
    """
    Run full pipeline on image data (numpy BGR array).
    """
    result = {
        'prediction': None,
        'confidence': None,
        'risk_level': None,
        'prob_normal': None,
        'prob_tb': None,
        'error': None,
    }
    
    try:
        # Segmentation (accepts BGR ndarray)
        seg, mask = _segmentor.segment(image_data)
        if seg is None:
            raise ValueError('Segmentation failed — possibly corrupt image.')
        
        # Feature extraction (hybrid: 1024 DenseNet + 24 GLCM = 1048 total)
        feats = extract_features(seg, mask)
        if feats is None:
            raise ValueError('Feature extraction failed.')
        
        # Prediction (feature slicing handled inside predict_fn for each model)
        label, confidence, proba = predict_fn(feats)
        risk = _risk_level(confidence, label)
        
        result.update({
            'prediction': label,
            'confidence': round(float(confidence), 4),
            'risk_level': risk,
            'prob_normal': round(float(proba[0]), 4),
            'prob_tb': round(float(proba[1]), 4),
        })
    
    except Exception as e:
        result['error'] = str(e)
        logger.error(f'Prediction error: {e}')
    
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'service': 'TB Detection Server'})


@app.route('/models', methods=['GET'])
def models():
    """List available models."""
    return jsonify({
        'available_models': ['svm', 'rf', 'gb', 'ensemble'],
        'aliases': {
            'rf': ['random_forest', 'randomforest'],
            'gb': ['gradient_boosting', 'gradientboosting']
        }
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict TB from uploaded image.
    
    Required params:
        - model: svm | rf | gb | ensemble
        - image: binary file upload
    
    Optional params:
        - format: json (default) | html
    
    Returns: JSON or HTML with prediction results
    """
    try:
        # Get format preference
        response_format = request.form.get('format', 'json').lower()
        if response_format not in ('json', 'html'):
            response_format = 'json'
        
        # Validate model
        model_name = request.form.get('model', '').lower()
        if not model_name:
            if response_format == 'html':
                return _format_result_html({'error': 'Missing "model" parameter'}, 'unknown'), 400
            return jsonify({'error': 'Missing "model" parameter'}), 400
        
        # Validate image
        if 'image' not in request.files:
            if response_format == 'html':
                return _format_result_html({'error': 'Missing "image" file'}, model_name), 400
            return jsonify({'error': 'Missing "image" file'}), 400
        
        file = request.files['image']
        if file.filename == '':
            if response_format == 'html':
                return _format_result_html({'error': 'No file selected'}, model_name), 400
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            error_msg = f'Unsupported file format. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
            if response_format == 'html':
                return _format_result_html({'error': error_msg}, model_name), 400
            return jsonify({'error': error_msg}), 400
        
        # Load model
        try:
            predict_fn = _load_model(model_name)
        except ValueError as e:
            if response_format == 'html':
                return _format_result_html({'error': str(e)}, model_name), 400
            return jsonify({'error': str(e)}), 400
        except FileNotFoundError as e:
            if response_format == 'html':
                return _format_result_html({'error': f'Model files not found: {e}'}, model_name), 500
            return jsonify({'error': f'Model files not found: {e}'}), 500
        
        # Process image
        img_result = _process_image_file(file)
        if img_result['error']:
            if response_format == 'html':
                return _format_result_html({'error': img_result['error']}, model_name), 400
            return jsonify({'error': img_result['error']}), 400
        
        # Run prediction
        result = _predict_from_image(img_result['image_data'], predict_fn, model_name)
        
        if result['error']:
            if response_format == 'html':
                return _format_result_html({'error': result['error']}, model_name), 400
            return jsonify({'error': result['error']}), 400
        
        result['model'] = model_name
        result['filename'] = img_result['filename']
        
        if response_format == 'html':
            html_response = _format_result_html(result, model_name, img_result['filename'])
            return html_response, 200
        
        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f'Unexpected error in /predict: {e}')
        if response_format == 'html':
            return _format_result_html({'error': f'Server error: {str(e)}'}, 'unknown'), 500
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/predict_url', methods=['POST'])
def predict_url():
    """
    Predict TB from image URL.
    
    Required JSON params:
        - model: svm | rf | gb | ensemble
        - image_url: URL to image file
    
    Optional JSON params:
        - format: json (default) | html
    
    Returns: JSON or HTML with prediction results
    """
    try:
        data = request.get_json() or {}
        
        # Get format preference
        response_format = data.get('format', 'json').lower()
        if response_format not in ('json', 'html'):
            response_format = 'json'
        
        # Validate model
        model_name = data.get('model', '').lower()
        if not model_name:
            if response_format == 'html':
                return _format_result_html({'error': 'Missing "model" parameter'}, 'unknown'), 400
            return jsonify({'error': 'Missing "model" parameter'}), 400
        
        # Validate URL
        image_url = data.get('image_url', '').strip()
        if not image_url:
            if response_format == 'html':
                return _format_result_html({'error': 'Missing "image_url" parameter'}, model_name), 400
            return jsonify({'error': 'Missing "image_url" parameter'}), 400
        
        # Load model
        try:
            predict_fn = _load_model(model_name)
        except ValueError as e:
            if response_format == 'html':
                return _format_result_html({'error': str(e)}, model_name), 400
            return jsonify({'error': str(e)}), 400
        except FileNotFoundError as e:
            if response_format == 'html':
                return _format_result_html({'error': f'Model files not found: {e}'}, model_name), 500
            return jsonify({'error': f'Model files not found: {e}'}), 500
        
        # Download image
        try:
            logger.info(f'Downloading image from: {image_url}')
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            
            nparr = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                error_msg = 'Failed to decode image from URL'
                if response_format == 'html':
                    return _format_result_html({'error': error_msg}, model_name), 400
                return jsonify({'error': error_msg}), 400
        
        except requests.RequestException as e:
            error_msg = f'Failed to download image: {str(e)}'
            if response_format == 'html':
                return _format_result_html({'error': error_msg}, model_name), 400
            return jsonify({'error': error_msg}), 400
        
        # Run prediction
        result = _predict_from_image(img, predict_fn, model_name)
        
        if result['error']:
            if response_format == 'html':
                return _format_result_html({'error': result['error']}, model_name), 400
            return jsonify({'error': result['error']}), 400
        
        result['model'] = model_name
        result['image_url'] = image_url
        
        if response_format == 'html':
            html_response = _format_result_html(result, model_name, image_url.split('/')[-1])
            return html_response, 200
        
        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f'Unexpected error in /predict_url: {e}')
        try:
            response_format = request.get_json().get('format', 'json').lower()
        except:
            response_format = 'json'
        if response_format == 'html':
            return _format_result_html({'error': f'Server error: {str(e)}'}, 'unknown'), 500
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large."""
    return jsonify({'error': 'File too large. Max 50 MB.'}), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404."""
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': ['/health', '/models', '/predict', '/predict_url']
    }), 404


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='TB Detection HTTP Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python server.py                          # localhost:5000
  python server.py --host 0.0.0.0 --port 8000
  python server.py --debug                  # Enables auto-reload
        """
    )
    ap.add_argument('--host', default='127.0.0.1',
                    help='Host to bind to (default: 127.0.0.1)')
    ap.add_argument('--port', type=int, default=5000,
                    help='Port to bind to (default: 5000)')
    ap.add_argument('--debug', action='store_true',
                    help='Enable Flask debug mode')
    ap.add_argument('--verbose', action='store_true',
                    help='Enable DEBUG logging')
    
    args = ap.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f'Starting TB Detection Server on {args.host}:{args.port}')
    
    # Pre-load models for faster first request
    logger.info('Pre-loading models...')
    try:
        for model_name in ['svm', 'rf', 'gb', 'ensemble']:
            _load_model(model_name)
            logger.info(f'  ✓ {model_name} loaded')
    except Exception as e:
        logger.warning(f'Failed to pre-load some models: {e}')
    
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )


if __name__ == '__main__':
    main()
