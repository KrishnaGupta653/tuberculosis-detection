"""
================================================================================
MODEL COMPARISON SCRIPT
================================================================================
Compares accuracy of:
1. SVM (individual)
2. Random Forest (individual)  
3. Gradient Boosting (individual)
4. Ensemble (all 3 combined)

USAGE:
    cd <project_root>
    python test/compare_all_models.py
================================================================================
"""

import os
import sys
import pickle
from glob import glob
from datetime import datetime
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline_shared import Config, load_image_paths, extract_features_parallel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report, roc_auc_score)

# ═══════════════════════════════════════════════════════════════════════════
# LOAD TEST DATA
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("LOADING TEST DATA...")
print("="*80 + "\n")

# Load test images
test_img_dir = os.path.join(Config.DATASET_DIR, 'TB')
normal_img_dir = os.path.join(Config.DATASET_DIR, 'Normal')

print(f"📂 Loading images from: {Config.DATASET_DIR}")
X_test, y_test, test_files = load_image_paths(
    test_img_dir, 
    normal_img_dir, 
    dataset_type='test'
)

print(f"✓ Test set: {len(X_test)} images")
print(f"✓ Class distribution: {np.bincount(y_test)}")

# Extract features from test set
print(f"\n🔬 Extracting test features...")
X_test_features = extract_features_parallel(X_test, n_jobs=Config.N_JOBS)
print(f"✓ Test features shape: {X_test_features.shape}")

# Normalize test features
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test_features)

# ═══════════════════════════════════════════════════════════════════════════
# LOAD INDIVIDUAL MODELS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("LOADING TRAINED MODELS...")
print("="*80 + "\n")

results = {}

# Find latest SVM model
svm_files = glob(os.path.join(Config.MODEL_DIR, 'svm_model_*.pkl'))
if svm_files:
    svm_path = sorted(svm_files)[-1]
    print(f"📦 Loading SVM: {os.path.basename(svm_path)}")
    with open(svm_path, 'rb') as f:
        svm_model = pickle.load(f)
    results['SVM'] = {'model': svm_model, 'path': svm_path}
else:
    print("⚠️  No SVM model found")

# Find latest Random Forest model
rf_files = glob(os.path.join(Config.MODEL_DIR, 'randomforest_model_*.pkl'))
if rf_files:
    rf_path = sorted(rf_files)[-1]
    print(f"📦 Loading Random Forest: {os.path.basename(rf_path)}")
    with open(rf_path, 'rb') as f:
        rf_model = pickle.load(f)
    results['Random Forest'] = {'model': rf_model, 'path': rf_path}
else:
    print("⚠️  No Random Forest model found")

# Find latest Gradient Boosting model
gb_files = glob(os.path.join(Config.MODEL_DIR, 'gradientboosting_model_*.pkl'))
if gb_files:
    gb_path = sorted(gb_files)[-1]
    print(f"📦 Loading Gradient Boosting: {os.path.basename(gb_path)}")
    with open(gb_path, 'rb') as f:
        gb_model = pickle.load(f)
    results['Gradient Boosting'] = {'model': gb_model, 'path': gb_path}
else:
    print("⚠️  No Gradient Boosting model found")

# Find latest Ensemble model
ensemble_files = glob(os.path.join(Config.MODEL_DIR, 'enhanced_ensemble_*.pkl'))
if ensemble_files:
    ensemble_path = sorted(ensemble_files)[-1]
    print(f"📦 Loading Ensemble: {os.path.basename(ensemble_path)}")
    with open(ensemble_path, 'rb') as f:
        ensemble_data = pickle.load(f)
    results['Ensemble'] = {'model': ensemble_data, 'path': ensemble_path}
else:
    print("⚠️  No Ensemble model found")

# ═══════════════════════════════════════════════════════════════════════════
# EVALUATE INDIVIDUAL MODELS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("EVALUATING INDIVIDUAL MODELS ON TEST SET...")
print("="*80 + "\n")

model_metrics = {}

for model_name, model_info in results.items():
    if model_name == 'Ensemble':
        continue
    
    print(f"\n📊 {model_name}:")
    print("-" * 40)
    
    try:
        model = model_info['model']
        
        # Get predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # ROC-AUC if probability predictions available
        try:
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
        except:
            roc_auc = None
        
        model_metrics[model_name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'specificity': specificity,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion': (tn, fp, fn, tp)
        }
        
        print(f"  Accuracy:    {acc*100:6.2f}%")
        print(f"  Sensitivity: {rec*100:6.2f}% (Recall)")
        print(f"  Specificity: {specificity*100:6.2f}%")
        print(f"  Precision:   {prec*100:6.2f}%")
        print(f"  F1-Score:    {f1:.4f}")
        if roc_auc:
            print(f"  ROC-AUC:     {roc_auc:.4f}")
        print(f"  Confusion:   TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# EVALUATE ENSEMBLE MODEL
# ═══════════════════════════════════════════════════════════════════════════

if 'Ensemble' in results:
    print("\n" + "="*80)
    print("EVALUATING ENSEMBLE (COMBINED) MODEL...")
    print("="*80 + "\n")
    
    print(f"📊 Ensemble (SVM + RF + GB):")
    print("-" * 40)
    
    try:
        ensemble_data = results['Ensemble']['model']
        
        # Ensemble should have individual models inside
        if isinstance(ensemble_data, dict) and 'models' in ensemble_data:
            models_dict = ensemble_data['models']
            
            # Get predictions from each model
            svm_pred = models_dict['SVM'].predict(X_test_scaled)
            rf_pred = models_dict['Random Forest'].predict(X_test_scaled)
            gb_pred = models_dict['Gradient Boosting'].predict(X_test_scaled)
            
            # Ensemble voting
            ensemble_pred = (svm_pred + rf_pred + gb_pred) >= 2
            ensemble_pred = ensemble_pred.astype(int)
        else:
            ensemble_pred = ensemble_data.predict(X_test_scaled)
        
        # Calculate metrics
        acc = accuracy_score(y_test, ensemble_pred)
        prec = precision_score(y_test, ensemble_pred)
        rec = recall_score(y_test, ensemble_pred)
        f1 = f1_score(y_test, ensemble_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, ensemble_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        model_metrics['Ensemble'] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'specificity': specificity,
            'f1': f1,
            'confusion': (tn, fp, fn, tp)
        }
        
        print(f"  Accuracy:    {acc*100:6.2f}%")
        print(f"  Sensitivity: {rec*100:6.2f}% (Recall)")
        print(f"  Specificity: {specificity*100:6.2f}%")
        print(f"  Precision:   {prec*100:6.2f}%")
        print(f"  F1-Score:    {f1:.4f}")
        print(f"  Confusion:   TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════════════

if model_metrics:
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80 + "\n")
    
    # Create comparison
    comparison_data = []
    for model_name in ['SVM', 'Random Forest', 'Gradient Boosting', 'Ensemble']:
        if model_name in model_metrics:
            m = model_metrics[model_name]
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{m['accuracy']*100:6.2f}%",
                'Sensitivity': f"{m['recall']*100:6.2f}%",
                'Specificity': f"{m['specificity']*100:6.2f}%",
                'Precision': f"{m['precision']*100:6.2f}%",
                'F1': f"{m['f1']:.4f}"
            })
    
    # Print table
    if comparison_data:
        print(f"{'Model':<20} {'Accuracy':<12} {'Sensitivity':<12} {'Specificity':<12} {'Precision':<12} {'F1':<10}")
        print("-" * 80)
        for row in comparison_data:
            print(f"{row['Model']:<20} {row['Accuracy']:<12} {row['Sensitivity']:<12} {row['Specificity']:<12} {row['Precision']:<12} {row['F1']:<10}")
    
    # Find best model
    print("\n" + "="*80)
    best_model = max(model_metrics.items(), key=lambda x: x[1]['accuracy'])
    print(f"🏆 BEST MODEL: {best_model[0]} ({best_model[1]['accuracy']*100:.2f}% accuracy)")
    print("="*80)

print("\n✅ Comparison complete!\n")
