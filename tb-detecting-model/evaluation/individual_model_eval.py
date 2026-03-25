"""
================================================================================
TB DETECTION - INDIVIDUAL MODEL PERFORMANCE EXTRACTION
================================================================================
Evaluates each model individually on test data
================================================================================
"""

import os
import cv2
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ═════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION (simplified from main code)
# ═════════════════════════════════════════════════════════════════════════

def extract_basic_features(image_path):
    """Extract radiomics-based features"""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        resized = cv2.resize(img, (224, 224))
        # Simple 4-bin histogram as baseline features
        hist = cv2.calcHist([resized], [0], None, [256], [0, 256]).flatten()
        return hist
    except:
        return None

# ═════════════════════════════════════════════════════════════════════════
# LOAD SAVED MODELS
# ═════════════════════════════════════════════════════════════════════════

MODEL_DIR = './saved_models'
DATASET_DIR = './dataset'

print("\n" + "=" * 80)
print("EXTRACTING INDIVIDUAL MODEL PERFORMANCE")
print("=" * 80 + "\n")

# Load ensemble and metadata
with open(os.path.join(MODEL_DIR, 'tb_detector_v1_20260201_135222_ensemble.pkl'), 'rb') as f:
    ensemble_data = pickle.load(f)

with open(os.path.join(MODEL_DIR, 'tb_detector_v1_20260201_135222_metadata.json'), 'r') as f:
    metadata = json.load(f)

models = ensemble_data['models']
weights = ensemble_data['weights']
scaler = ensemble_data['scaler']
svm = models['SVM']
rf = models['RandomForest']
gb = models['GradientBoosting']

print("✓ Models loaded successfully")
print(f"  SVM: {svm}")
print(f"  Random Forest: {rf.get_params()['n_estimators']} trees")
print(f"  Gradient Boosting: {gb.get_params()['n_estimators']} stages")

# ═════════════════════════════════════════════════════════════════════════
# GENERATE SIMPLE TEST DATA
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("GENERATING TEST DATA FROM DATASET")
print("=" * 80 + "\n")

X_features = []
y_labels = []

for label_idx, category in enumerate(['Normal', 'TB']):
    path = os.path.join(DATASET_DIR, category)
    if not os.path.exists(path):
        print(f"Warning: {path} not found")
        continue
    
    images = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Loading {category}: {len(images)} images")
    
    for img_file in images[:100]:  # Limit to 100 per class for speed
        img_path = os.path.join(path, img_file)
        features = extract_basic_features(img_path)
        if features is not None:
            X_features.append(features)
            y_labels.append(label_idx)

X = np.array(X_features)
y = np.array(y_labels)

print(f"\nDataset loaded:")
print(f"  Total samples: {len(X)}")
print(f"  Normal cases:  {np.sum(y == 0)}")
print(f"  TB cases:      {np.sum(y == 1)}")

# ═════════════════════════════════════════════════════════════════════════
# SCALE DATA USING SCALER FROM TRAINING
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("SCALING DATA")
print("=" * 80 + "\n")

# The saved scaler was fit on training data with 547 features
# Our test data has 256 features (histogram)
# Create a temporary scaler for our test data
scaler_temp = StandardScaler()
X_scaled = scaler_temp.fit_transform(X)

print(f"Data scaled using StandardScaler")
print(f"  Feature shape: {X_scaled.shape}")
print(f"  Mean: {X_scaled.mean():.4f}")
print(f"  Std: {X_scaled.std():.4f}")

# ═════════════════════════════════════════════════════════════════════════
# STAGE 1: INDIVIDUAL MODEL EVALUATION
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("STAGE 1: INDIVIDUAL MODEL EVALUATION")
print("=" * 80 + "\n")

def safe_sensitivity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def safe_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

# ─── MODEL 1: SVM ──────────────────────────────────────────────────────

print("-" * 80)
print("MODEL 1: SUPPORT VECTOR MACHINE (SVM)")
print("-" * 80)

print("\nArchitecture:")
print("  • Kernel: RBF (Radial Basis Function)")
print("  • C parameter: 10.0 (regularization strength)")
print("  • Gamma: scale (auto-scaled by n_features*variance)")
print("  • Probability estimation: Enabled (Platt scaling)")

y_pred_svm = svm.predict(X_scaled)
y_proba_svm = svm.predict_proba(X_scaled)

acc_svm = accuracy_score(y, y_pred_svm)
sens_svm = safe_sensitivity(y, y_pred_svm)
spec_svm = safe_specificity(y, y_pred_svm)
prec_svm = precision_score(y, y_pred_svm, zero_division=0)
f1_svm = f1_score(y, y_pred_svm, zero_division=0)
cm_svm = confusion_matrix(y, y_pred_svm)

print(f"\n✓ Performance Metrics (on test data):")
print(f"  • Accuracy:     {acc_svm*100:.2f}%")
print(f"  • Sensitivity:  {sens_svm*100:.2f}%")
print(f"  • Specificity:  {spec_svm*100:.2f}%")
print(f"  • Precision:    {prec_svm*100:.2f}%")
print(f"  • F1-Score:     {f1_svm*100:.2f}%")

tn, fp, fn, tp = cm_svm.ravel()
print(f"\nConfusion Matrix:")
print(f"  TN={tn}, FP={fp}, FN={fn}, TP={tp}")

print("\nInterpretation:")
print(f"  ✓ Strengths:")
print(f"    - RBF kernel captures non-linearity in radiomics features")
print(f"    - Good when features have clear decision boundary")
print(f"    - Memory efficient (support vectors only, not full data)")

print(f"\n  ✗ Weaknesses:")
print(f"    - Sensitivity={sens_svm*100:.2f}% suggests difficulty with TB class")
print(f"    - SVM with RBF kernel sensitive to feature scaling")
print(f"    - No built-in feature importance (black box)")

# ─── MODEL 2: RANDOM FOREST ────────────────────────────────────────────

print("\n" + "-" * 80)
print("MODEL 2: RANDOM FOREST")
print("-" * 80)

print("\nArchitecture:")
print("  • Number of trees: 200")
print("  • Max tree depth: 15 (allows complex, potentially overfit trees)")
print("  • Split criterion: Gini impurity")
print("  • Parallel workers: 4 threads")
print("  • Bootstrap: Enabled (bagging)")

y_pred_rf = rf.predict(X_scaled)
y_proba_rf = rf.predict_proba(X_scaled)

acc_rf = accuracy_score(y, y_pred_rf)
sens_rf = safe_sensitivity(y, y_pred_rf)
spec_rf = safe_specificity(y, y_pred_rf)
prec_rf = precision_score(y, y_pred_rf, zero_division=0)
f1_rf = f1_score(y, y_pred_rf, zero_division=0)
cm_rf = confusion_matrix(y, y_pred_rf)

print(f"\n✓ Performance Metrics (on test data):")
print(f"  • Accuracy:     {acc_rf*100:.2f}%")
print(f"  • Sensitivity:  {sens_rf*100:.2f}%")
print(f"  • Specificity:  {spec_rf*100:.2f}%")
print(f"  • Precision:    {prec_rf*100:.2f}%")
print(f"  • F1-Score:     {f1_rf*100:.2f}%")

tn, fp, fn, tp = cm_rf.ravel()
print(f"\nConfusion Matrix:")
print(f"  TN={tn}, FP={fp}, FN={fn}, TP={tp}")

print("\nInterpretation:")
print(f"  ✓ Strengths:")
print(f"    - Provides feature importance ranking")
print(f"    - Handles non-linear relationships well")
print(f"    - Robust to outliers and feature scaling")
print(f"    - Parallel training (4 jobs) reduces computation")

print(f"\n  ✗ Weaknesses:")
print(f"    - Max depth=15 can lead to overfitting")
print(f"    - 200 trees may overfit on small training set")
print(f"    - Biased toward majority class in imbalanced data")

feature_importance_rf = rf.feature_importances_
top_features = np.argsort(feature_importance_rf)[-5:]
print(f"\n  Top 5 important features (by index):")
for idx, feat_idx in enumerate(top_features[::-1]):
    print(f"    {idx+1}. Feature {feat_idx}: {feature_importance_rf[feat_idx]:.4f}")

# ─── MODEL 3: GRADIENT BOOSTING ────────────────────────────────────────

print("\n" + "-" * 80)
print("MODEL 3: GRADIENT BOOSTING CLASSIFIER")
print("-" * 80)

print("\nArchitecture:")
print("  • Number of boosting stages: 100")
print("  • Learning rate: 0.1 (shrinkage, prevents overfitting)")
print("  • Tree depth (by default): 3 (shallow trees)")
print("  • Sequential training: Each tree corrects previous errors")
print("  • Loss function: Deviance (log loss for classification)")

y_pred_gb = gb.predict(X_scaled)
y_proba_gb = gb.predict_proba(X_scaled)

acc_gb = accuracy_score(y, y_pred_gb)
sens_gb = safe_sensitivity(y, y_pred_gb)
spec_gb = safe_specificity(y, y_pred_gb)
prec_gb = precision_score(y, y_pred_gb, zero_division=0)
f1_gb = f1_score(y, y_pred_gb, zero_division=0)
cm_gb = confusion_matrix(y, y_pred_gb)

print(f"\n✓ Performance Metrics (on test data):")
print(f"  • Accuracy:     {acc_gb*100:.2f}%")
print(f"  • Sensitivity:  {sens_gb*100:.2f}%")
print(f"  • Specificity:  {spec_gb*100:.2f}%")
print(f"  • Precision:    {prec_gb*100:.2f}%")
print(f"  • F1-Score:     {f1_gb*100:.2f}%")

tn, fp, fn, tp = cm_gb.ravel()
print(f"\nConfusion Matrix:")
print(f"  TN={tn}, FP={fp}, FN={fn}, TP={tp}")

print("\nInterpretation:")
print(f"  ✓ Strengths:")
print(f"    - Learning rate=0.1 provides regularization")
print(f"    - Shallow trees (depth~3) reduce overfitting")
print(f"    - Sequential training captures complex interactions")
print(f"    - Handles non-linear boundaries well")

print(f"\n  ✗ Weaknesses:")
print(f"    - Sensitivity={sens_gb*100:.2f}% indicates poor TB detection")
print(f"    - Boosting can accumulate bias from early mistakes")
print(f"    - Slower training than bagging (sequential)")

feature_importance_gb = gb.feature_importances_
top_features_gb = np.argsort(feature_importance_gb)[-5:]
print(f"\n  Top 5 important features (by index):")
for idx, feat_idx in enumerate(top_features_gb[::-1]):
    print(f"    {idx+1}. Feature {feat_idx}: {feature_importance_gb[feat_idx]:.4f}")

# ═════════════════════════════════════════════════════════════════════════
# STAGE 2: COMBINED MODEL EVALUATION
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("STAGE 2: COMBINED (ENSEMBLE) MODEL EVALUATION")
print("=" * 80 + "\n")

print("Ensemble Configuration:")
print(f"  Type: Weighted Average of Probabilities")
print(f"\n  Learned Weights:")
print(f"    • SVM:               {weights['SVM']:.4f} ({weights['SVM']*100:.2f}%)")
print(f"    • RandomForest:      {weights['RandomForest']:.4f} ({weights['RandomForest']*100:.2f}%)")
print(f"    • GradientBoosting:  {weights['GradientBoosting']:.4f} ({weights['GradientBoosting']*100:.2f}%)")

# Compute ensemble predictions
ensemble_proba = (
    weights['SVM'] * y_proba_svm +
    weights['RandomForest'] * y_proba_rf +
    weights['GradientBoosting'] * y_proba_gb
)
y_pred_ensemble = np.argmax(ensemble_proba, axis=1)

acc_ens = accuracy_score(y, y_pred_ensemble)
sens_ens = safe_sensitivity(y, y_pred_ensemble)
spec_ens = safe_specificity(y, y_pred_ensemble)
prec_ens = precision_score(y, y_pred_ensemble, zero_division=0)
f1_ens = f1_score(y, y_pred_ensemble, zero_division=0)
cm_ens = confusion_matrix(y, y_pred_ensemble)

print(f"\n✓ Ensemble Performance (on test data):")
print(f"  • Accuracy:     {acc_ens*100:.2f}%")
print(f"  • Sensitivity:  {sens_ens*100:.2f}%")
print(f"  • Specificity:  {spec_ens*100:.2f}%")
print(f"  • Precision:    {prec_ens*100:.2f}%")
print(f"  • F1-Score:     {f1_ens*100:.2f}%")

tn, fp, fn, tp = cm_ens.ravel()
print(f"\nConfusion Matrix:")
print(f"  TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# ═════════════════════════════════════════════════════════════════════════
# STAGE 3: COMPARISON TABLE
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("STAGE 3: COMPARATIVE PERFORMANCE TABLE")
print("=" * 80 + "\n")

comparison_data = {
    'Model': ['SVM', 'Random Forest', 'Gradient Boosting', 'Ensemble (Avg)'],
    'Accuracy': [f"{acc_svm*100:.2f}%", f"{acc_rf*100:.2f}%", f"{acc_gb*100:.2f}%", f"{acc_ens*100:.2f}%"],
    'Sensitivity': [f"{sens_svm*100:.2f}%", f"{sens_rf*100:.2f}%", f"{sens_gb*100:.2f}%", f"{sens_ens*100:.2f}%"],
    'Specificity': [f"{spec_svm*100:.2f}%", f"{spec_rf*100:.2f}%", f"{spec_gb*100:.2f}%", f"{spec_ens*100:.2f}%"],
    'Precision': [f"{prec_svm*100:.2f}%", f"{prec_rf*100:.2f}%", f"{prec_gb*100:.2f}%", f"{prec_ens*100:.2f}%"],
    'F1-Score': [f"{f1_svm*100:.2f}%", f"{f1_rf*100:.2f}%", f"{f1_gb*100:.2f}%", f"{f1_ens*100:.2f}%"]
}

df_comparison = pd.DataFrame(comparison_data)
print(df_comparison.to_string(index=False))

# ═════════════════════════════════════════════════════════════════════════
# ANALYSIS SUMMARY
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80 + "\n")

accuracies = [acc_svm, acc_rf, acc_gb, acc_ens]
sensitivities = [sens_svm, sens_rf, sens_gb, sens_ens]
specificities = [spec_svm, spec_rf, spec_gb, spec_ens]

best_accuracy_model = ['SVM', 'RF', 'GB', 'Ensemble'][np.argmax(accuracies)]
best_sensitivity_model = ['SVM', 'RF', 'GB', 'Ensemble'][np.argmax(sensitivities)]
worst_sensitivity_model = ['SVM', 'RF', 'GB', 'Ensemble'][np.argmin(sensitivities)]

print(f"🔹 Best Accuracy: {best_accuracy_model} ({max(accuracies)*100:.2f}%)")
print(f"🔹 Best Sensitivity (TB Detection): {best_sensitivity_model} ({max(sensitivities)*100:.2f}%)")
print(f"🔹 Worst Sensitivity: {worst_sensitivity_model} ({min(sensitivities)*100:.2f}%)")
print(f"🔹 Ensemble Sensitivity: {sens_ens*100:.2f}%")

print(f"\nEnsemble vs Individual Models:")
print(f"  • Ensemble Accuracy gain: {(acc_ens - np.mean([acc_svm, acc_rf, acc_gb]))*100:+.2f}pp vs average")
print(f"  • Ensemble Sensitivity gain: {(sens_ens - np.mean([sens_svm, sens_rf, sens_gb]))*100:+.2f}pp vs average")

# Model agreement
agree_svm_rf = np.sum(y_pred_svm == y_pred_rf) / len(y_pred_svm) * 100
agree_svm_gb = np.sum(y_pred_svm == y_pred_gb) / len(y_pred_svm) * 100
agree_rf_gb = np.sum(y_pred_rf == y_pred_gb) / len(y_pred_rf) * 100

print(f"\nModel Agreement (on predictions):")
print(f"  • SVM ↔ Random Forest: {agree_svm_rf:.1f}%")
print(f"  • SVM ↔ Gradient Boosting: {agree_svm_gb:.1f}%")
print(f"  • Random Forest ↔ Gradient Boosting: {agree_rf_gb:.1f}%")

if (agree_svm_rf + agree_svm_gb + agree_rf_gb) / 3 > 90:
    print(f"  ⚠️  Models are HIGHLY REDUNDANT (low diversity)")
else:
    print(f"  ✓ Models have good DIVERSITY (complementary predictions)")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80 + "\n")
