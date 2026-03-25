"""
================================================================================
INFER SVM ON SINGLE IMAGE
================================================================================
Loads trained SVM model and predicts on a single X-ray image
Usage: python infer_svm_on_image.py --image path/to/image.png
"""

import os
import cv2
import numpy as np
import pickle
import argparse
from glob import glob

from pipeline_shared import Config, EntropyGuidedSegmentor, FractalWaveletExtractor

def infer_svm_on_single_image(image_path, model_path=None, verbose=True):
    """Predict TB on a single image using SVM"""
    
    # Load model if not provided
    if model_path is None:
        model_files = glob(os.path.join(Config.MODEL_DIR, "svm_model_*.pkl"))
        if not model_files:
            print("❌ No SVM models found!")
            return None
        model_path = sorted(model_files)[-1]
    
    with open(model_path, 'rb') as f:
        model_pkg = pickle.load(f)
    
    model = model_pkg['model']
    scaler = model_pkg['scaler']
    
    # Load and segment image
    if verbose:
        print(f"\n{'='*80}")
        print("🔍 SVM INFERENCE ON SINGLE IMAGE")
        print(f"{'='*80}\n")
        print(f"📷 Image: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return None
    
    # Segment
    if verbose:
        print("🔬 Segmenting lung ROI...")
    segmentor = EntropyGuidedSegmentor()
    segmented_img, mask, _ = segmentor.segment_lung_roi(image_path)
    
    if segmented_img is None:
        print("❌ Could not segment lung ROI")
        return None
    
    if verbose:
        print("✓ Lung ROI segmented successfully")
    
    # Extract features
    if verbose:
        print("📊 Extracting radiomics features...")
    extractor = FractalWaveletExtractor()
    features = extractor.extract_all_features(segmented_img, mask)
    
    if features is None:
        print("❌ Could not extract features")
        return None
    
    features = features.reshape(1, -1)
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    label_name = Config.CATEGORIES[prediction]
    confidence = probabilities[prediction] * 100
    
    if verbose:
        print(f"\n{'='*80}")
        print("🎯 PREDICTION RESULT")
        print(f"{'='*80}\n")
        print(f"Predicted Class: {label_name}")
        print(f"Confidence:      {confidence:.2f}%")
        print(f"\nProbabilities:")
        for i, cat in enumerate(Config.CATEGORIES):
            print(f"  {cat:10s}: {probabilities[i]*100:6.2f}%")
        print(f"{'='*80}\n")
    
    return {
        'image': image_path,
        'prediction': label_name,
        'prediction_idx': int(prediction),
        'confidence': float(confidence),
        'probabilities': {
            Config.CATEGORIES[i]: float(probabilities[i])
            for i in range(len(Config.CATEGORIES))
        },
        'model_path': model_path
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict TB on single image using SVM")
    parser.add_argument('--image', type=str, required=True, help='Path to X-ray image')
    parser.add_argument('--model', type=str, default=None, help='Path to SVM model')
    
    args = parser.parse_args()
    
    result = infer_svm_on_single_image(args.image, args.model, verbose=True)
    
    if result:
        print(f"✅ Inference successful!")
