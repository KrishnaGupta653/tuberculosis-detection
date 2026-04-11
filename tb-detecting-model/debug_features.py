"""
✨ DIAGNOSTIC: Compare feature extraction methods
individual trainers vs model_final_fast
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path

# Add parent for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from pipeline_shared import Config, load_image_paths, extract_features_parallel
from models.model_final_fast import (
    EntropyGuidedSegmentor,
    FractalWaveletExtractor,
    HybridFeatureFusion,
    process_single_image_worker
)

print("="*80)
print("FEATURE EXTRACTION COMPARISON")
print("="*80)

# Load a sample image
data_dir = './dataset'
normal_dir = os.path.join(data_dir, 'Normal')
tb_dir = os.path.join(data_dir, 'TB')

# Get one normal and one TB image
normal_img = os.path.join(normal_dir, os.listdir(normal_dir)[0])
tb_img = os.path.join(tb_dir, os.listdir(tb_dir)[0])

print(f"\nSample images:")
print(f"  Normal: {normal_img}")
print(f"  TB: {tb_img}")

# ─── METHOD 1: Individual Trainer Features (Radiomics Only) ───
print(f"\n{'='*80}")
print("METHOD 1: Individual Trainer (Radiomics Only)")
print(f"{'='*80}")

try:
    X_individual, _ = extract_features_parallel(
        [normal_img, tb_img], 
        [0, 1],
        "Sample"
    )
    print(f"✓ Shape: {X_individual.shape}")
    print(f"✓ Feature count per image: {X_individual.shape[1]}")
    print(f"✓ Sample values (first 10): {X_individual[0][:10]}")
    print(f"✓ Data type: {X_individual.dtype}")
    print(f"✓ Feature range: [{X_individual.min():.4f}, {X_individual.max():.4f}]")
except Exception as e:
    print(f"✗ Error: {e}")

# ─── METHOD 2: Hybrid (Radiomics + Deep Learning) ───
print(f"\n{'='*80}")
print("METHOD 2: model_final_fast.py (Radiomics + Deep Learning)")
print(f"{'='*80}")

try:
    # Process images using model_final_fast pipeline
    processed_data = []
    for img_path, label in [(normal_img, 0), (tb_img, 1)]:
        result = process_single_image_worker(img_path, label)
        if result:
            processed_data.append(result)
    
    # Extract hybrid features
    feature_fusion = HybridFeatureFusion()
    hybrid_features = feature_fusion.extract_hybrid_batch(processed_data)
    X_hybrid = np.array(hybrid_features)
    
    print(f"✓ Shape: {X_hybrid.shape}")
    print(f"✓ Feature count per image: {X_hybrid.shape[1]}")
    print(f"✓ Sample values (first 10): {X_hybrid[0][:10]}")
    print(f"✓ Data type: {X_hybrid.dtype}")
    print(f"✓ Feature range: [{X_hybrid.min():.4f}, {X_hybrid.max():.4f}]")
    
    # Break down the components
    print(f"\n📊 Feature breakdown:")
    print(f"  Radiomics features: {processed_data[0]['radiomic_features'].shape[0]}")
    print(f"  Deep learning features: {X_hybrid.shape[1] - processed_data[0]['radiomic_features'].shape[0]}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print("Individual trainers use RADIOMICS ONLY (~97 features) = 66% accuracy")
print("model_final_fast uses HYBRID (radiomics + DenseNet ~1121 features) = Should be 97%+")
print(f"{'='*80}\n")
