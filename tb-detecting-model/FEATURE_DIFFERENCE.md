"""
✨ KEY DIFFERENCE BETWEEN SYSTEMS
"""

# ═══════════════════════════════════════════════════════════════════════════

# INDIVIDUAL TRAINERS (SVM, RandomForest, GradientBoosting)

# ═══════════════════════════════════════════════════════════════════════════

"""
File: models/train_svm_model.py (and similar for RF, GB)

Pipeline:

1. Load images from dataset/Normal and dataset/TB
2. Extract RADIOMICS FEATURES ONLY using pipeline_shared.extract_features_parallel()
   - Fractal dimension
   - Wavelet features
   - GLCM (Gray-Level Co-occurrence Matrix)
   - Gabor filters
     = ~97 features per image
3. Train single model (SVM, RF, or GB) on these 97 features
4. Result: 65-66% accuracy

Code:
X_train, y_train = extract_features_parallel(X_paths_train, y_train) # X_train shape: (N_samples, 97)

    svm = SVC(...)
    svm.fit(X_train, y_train)

"""

# ═══════════════════════════════════════════════════════════════════════════

# model_final_fast.py (FULL SYSTEM)

# ═══════════════════════════════════════════════════════════════════════════

"""
File: models/model_final_fast.py

Pipeline:

1. Load images from dataset
2. SEGMENT using entropy-guided lung segmentation
3. Extract HYBRID FEATURES using HybridFeatureFusion:

   A) RADIOMICS (same as individual trainers):
   - Fractal dimension
   - Wavelet features
   - GLCM
   - Gabor filters
     = ~97 features

   B) DEEP LEARNING (NEW - not in individual trainers):
   - Pass segmented image through DenseNet121 (pre-trained)
   - Extract features from dense layers
     = ~1024 features

   C) HYBRID VECTOR = CONCATENATE(Radiomics + Deep Learning)
   = 97 + 1024 = ~1121 features per image

4. Optional: Use QUANTUM FEATURE SELECTION to reduce features

5. Train ENSEMBLE of multiple models with STACKING + CALIBRATION
   - Base models: SVM (RBF), SVM (Poly), RF, GB (2 versions)
   - Meta-learner: LogisticRegression
   - Calibration: CalibratedClassifierCV for probability calibration

6. Optimize decision threshold for medical accuracy

7. Result: Should be 97%+ accuracy (IF deep learning is working)

Code:
processed_data = process_single_image_worker(img_path, label)
hybrid_features = feature_fusion.extract_hybrid_batch(processed_data) # hybrid_features shape: (N_samples, 1121)

    ensemble = EnsembleClassifier()
    ensemble.train(X_train, y_train, X_val, y_val)
    ensemble.optimize_threshold(X_val, y_val)
    predictions = ensemble.predict(X_test)

"""

# ═══════════════════════════════════════════════════════════════════════════

# FEATURE DIMENSION COMPARISON

# ═══════════════════════════════════════════════════════════════════════════

print("""
┌─────────────────────────────────────────────────────────────────────────┐
│ SYSTEM COMPARISON │
├──────────────────────┬──────────────────┬──────────────┬────────────────┤
│ System │ Features │ Models │ Accuracy │
├──────────────────────┼──────────────────┼──────────────┼────────────────┤
│ train_svm_model.py │ 97 (radiomics) │ SVM only │ ~65% │
│ train_rf_model.py │ 97 (radiomics) │ RF only │ ~65% │
│ train_gb_model.py │ 97 (radiomics) │ GB only │ ~65% │
├──────────────────────┼──────────────────┼──────────────┼────────────────┤
│ model_final_fast.py │ 1121 (hybrid) │ Ensemble+ │ Should be 97%+ │
│ │ Radiomics+DNN │ Stacking+Cal │ (if working) │
└──────────────────────┴──────────────────┴──────────────┴────────────────┘

[97 radiomics features] vs [97 radiomics + 1024 DNN features]

The ~10x difference in features (97 vs 1121) is why model_final_fast
should be 97%+ while individuals are only 66%.
""")

# ═══════════════════════════════════════════════════════════════════════════

# WHAT COULD BE WRONG

# ═══════════════════════════════════════════════════════════════════════════

print("""
✗ If model_final_fast.py is only getting 55% (current state):

Possible issues:

1. Deep learning features are NaN or all zeros → Check DenseNet extraction
2. Deep learning features are being dropped in feature selection
3. Feature normalization is wrong (radiomics vs DNN on different scales)
4. Dataset class imbalance is severe (e.g., 97% TB, 3% Normal)
5. Deep learning model weights not loaded properly
6. GPU extraction is failing silently

✓ To verify they're ACTUALLY different:

- Run debug_features.py to see actual feature shapes
- Should see: (N, 97) for individual vs (N, 1121) for model_final_fast
  """)
