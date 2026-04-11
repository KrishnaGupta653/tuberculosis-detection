"""
================================================================================
ADVANCED ENSEMBLE INFERENCE - MAKE PREDICTIONS
================================================================================
Load trained advanced ensemble model and make predictions on new images

USAGE:
    python infer_advanced_ensemble.py --image path/to/image.png
    python infer_advanced_ensemble.py --image path/to/image.jpg [--threshold 0.5]
    python infer_advanced_ensemble.py --batch path/to/image/folder/

================================================================================
"""

import os
import cv2
import numpy as np
import torch
import pickle
import argparse
from glob import glob
from pathlib import Path
from torchvision import models, transforms
import sys

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
from pipeline_shared import EntropyGuidedSegmentor, FractalWaveletExtractor

# ═══════════════════════════════════════════════════════════════════════════
# INFERENCE ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class AdvancedEnsembleInference:
    """Load and use trained advanced ensemble model"""
    
    def __init__(self, model_path):
        print(f"\n{'='*80}")
        print("[LOAD] Loading Advanced Ensemble Model")
        print(f"{'='*80}\n")
        
        # Load model
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.ensemble = model_data['ensemble']
        self.scaler = model_data['scaler']
        self.best_models = model_data['best_models']
        self.optimal_threshold = model_data.get('optimal_threshold', 0.5)
        
        print(f"[OK] Model loaded from {model_path}")
        print(f"     Optimal threshold: {self.optimal_threshold:.4f}")
        
        # Setup feature extractors
        self.segmentor = EntropyGuidedSegmentor()
        self.extractor = FractalWaveletExtractor()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # ResNet50 for deep features
        try:
            self.resnet50 = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V1
            )
            self.resnet50.eval()
            self.resnet50.to(self.device)
            self.features_model = torch.nn.Sequential(*list(self.resnet50.children())[:-1])
            
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            self.use_resnet = True
            print(f"[OK] ResNet50 features available on {self.device}")
        except Exception as e:
            self.use_resnet = False
            print(f"[WARNING] ResNet50 not available: {e}")
    
    def extract_dual_features(self, image_path):
        """Extract classical + deep features from image"""
        
        # Load and segment
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        img = cv2.resize(img, (224, 224))
        segmented_img, mask, _ = self.segmentor.segment_lung_roi(img)
        
        if segmented_img is None:
            return None
        
        # Classical radiomics
        classical_features = self.extractor.extract_all_features(segmented_img, mask)
        if classical_features is None:
            return None
        
        features_list = [classical_features]
        
        # Deep features (ResNet50)
        if self.use_resnet:
            try:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    deep_features = self.features_model(img_tensor)
                    deep_features = deep_features.view(deep_features.size(0), -1)
                    deep_features = deep_features.cpu().numpy().flatten()
                
                features_list.append(deep_features)
            except Exception as e:
                print(f"[WARNING] Failed to extract ResNet50 features: {e}")
        
        # Combine features
        combined_features = np.hstack(features_list)
        return combined_features
    
    def predict_image(self, image_path, threshold=None):
        """Predict on single image"""
        
        if threshold is None:
            threshold = self.optimal_threshold
        
        # Extract features
        features = self.extract_dual_features(image_path)
        if features is None:
            return {
                'status': 'ERROR',
                'image': image_path,
                'prediction': 'FAILED',
                'reason': 'Could not extract features'
            }
        
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Get probabilities
        proba = self.ensemble.predict_proba(features_scaled)[0]
        
        # Make prediction with threshold
        prediction = 'TB' if proba[1] >= threshold else 'Normal'
        confidence = max(proba[0], proba[1])
        
        return {
            'status': 'OK',
            'image': image_path,
            'prediction': prediction,
            'confidence': confidence,
            'prob_normal': proba[0],
            'prob_tb': proba[1],
            'threshold': threshold
        }
    
    def predict_batch(self, image_dir, threshold=None):
        """Predict on multiple images in directory"""
        
        image_files = glob(os.path.join(image_dir, '*.png'))
        image_files += glob(os.path.join(image_dir, '*.jpg'))
        image_files += glob(os.path.join(image_dir, '*.jpeg'))
        
        results = []
        
        print(f"\n{'='*80}")
        print(f"[PREDICT] Batch Prediction on {len(image_files)} images")
        print(f"{'='*80}\n")
        
        for i, img_path in enumerate(image_files, 1):
            result = self.predict_image(img_path, threshold)
            results.append(result)
            
            status = result['status']
            pred = result['prediction']
            conf = result.get('confidence', 0)
            
            print(f"[{i:3d}/{len(image_files)}] {status:6s} {os.path.basename(img_path):30s} "
                  f"→ {pred:8s} ({conf*100:5.1f}%)")
        
        return results


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Advanced TB Detection Ensemble - Inference'
    )
    parser.add_argument('--model', type=str, default=None,
                       help='Model path (auto-detect if not specified)')
    parser.add_argument('--image', type=str, help='Single image path for prediction')
    parser.add_argument('--batch', type=str, help='Directory with images for batch prediction')
    parser.add_argument('--threshold', type=float, default=None,
                       help='Decision threshold (default: optimal from training)')
    args = parser.parse_args()
    
    # Auto-detect model if not specified
    model_path = args.model
    if model_path is None:
        import glob as glob_module
        models = glob_module.glob('./saved_models/advanced/advanced_ensemble_*.pkl')
        if not models:
            print("[ERROR] No trained models found in saved_models/advanced/")
            print("        First run: python models/train_advanced_ensemble.py --mode train")
            return
        
        # Use most recent model
        model_path = max(models, key=os.path.getctime)
        print(f"[AUTO] Using latest model: {model_path}")
    
    # Load inference engine
    engine = AdvancedEnsembleInference(model_path)
    
    # Single image prediction
    if args.image:
        result = engine.predict_image(args.image, args.threshold)
        
        print(f"\n{'='*80}")
        print("[RESULT] SINGLE IMAGE PREDICTION")
        print(f"{'='*80}\n")
        
        if result['status'] == 'OK':
            print(f"Image:              {result['image']}")
            print(f"Prediction:         {result['prediction']}")
            print(f"Confidence:         {result['confidence']*100:.2f}%")
            print(f"Probability Normal: {result['prob_normal']*100:.2f}%")
            print(f"Probability TB:     {result['prob_tb']*100:.2f}%")
            print(f"Threshold:          {result['threshold']:.4f}")
        else:
            print(f"ERROR: {result.get('reason', 'Unknown error')}")
    
    # Batch prediction
    elif args.batch:
        results = engine.predict_batch(args.batch, args.threshold)
        
        # Summary statistics
        predictions = [r['prediction'] for r in results if r['status'] == 'OK']
        if predictions:
            tb_count = sum(1 for p in predictions if p == 'TB')
            normal_count = sum(1 for p in predictions if p == 'Normal')
            
            print(f"\n{'='*80}")
            print("[SUMMARY] BATCH PREDICTION RESULTS")
            print(f"{'='*80}\n")
            
            print(f"Total images:    {len(image_files)}")
            print(f"TB detected:     {tb_count} ({tb_count/len(predictions)*100:.1f}%)")
            print(f"Normal detected: {normal_count} ({normal_count/len(predictions)*100:.1f}%)")
            print(f"Failed:          {len(results) - len(predictions)}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
