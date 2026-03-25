"""
================================================================================
TEST SVM MODEL ACCURACY
================================================================================
Loads trained SVM model and reports comprehensive test metrics
"""

import os
import pickle
import json
import numpy as np
from datetime import datetime
from glob import glob

from pipeline_shared import Config, load_image_paths, extract_features_parallel

def test_svm_model(model_path=None):
    """Test SVM model on unseen test data"""
    
    print(f"\n{'='*80}")
    print("🧪 TESTING SVM MODEL ACCURACY")
    print(f"{'='*80}\n")
    
    # Find latest model if not specified
    if model_path is None:
        model_files = glob(os.path.join(Config.MODEL_DIR, "svm_model_*.pkl"))
        if not model_files:
            print("❌ No SVM models found!")
            return
        model_path = sorted(model_files)[-1]
    
    print(f"📂 Loading model from: {model_path}")
    
    # Load model package
    with open(model_path, 'rb') as f:
        model_pkg = pickle.load(f)
    
    model = model_pkg['model']
    scaler = model_pkg['scaler']
    
    print(f"✓ Model loaded successfully (trained: {model_pkg['timestamp']})")
    
    # Load test data
    print("\n📂 Loading test image paths...")
    from sklearn.model_selection import train_test_split
    X_paths_all, y_all = load_image_paths(Config.DATA_DIR)
    
    _, X_paths_test, _, y_test = train_test_split(
        X_paths_all, y_all, test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_SEED, stratify=y_all
    )
    
    print(f"✓ Test set: {len(X_paths_test)} images")
    
    # Extract features
    print("\n🔬 Extracting features from test set...")
    X_test, y_test = extract_features_parallel(X_paths_test, y_test, "Test features")
    X_test_scaled = scaler.transform(X_test)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Metrics
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                f1_score, confusion_matrix, classification_report, roc_auc_score)
    
    accuracy = accuracy_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred, pos_label=1)
    specificity = recall_score(y_test, y_pred, pos_label=0)
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n{'='*80}")
    print("📊 SVM MODEL TEST RESULTS")
    print(f"{'='*80}\n")
    
    print(f"✓ Accuracy:    {accuracy*100:6.2f}%")
    print(f"✓ Sensitivity: {sensitivity*100:6.2f}%  (TP detection rate)")
    print(f"✓ Specificity: {specificity*100:6.2f}%  (TN detection rate)")
    print(f"✓ Precision:   {precision*100:6.2f}%")
    print(f"✓ F1-Score:    {f1:6.4f}")
    print(f"✓ ROC-AUC:     {roc_auc:6.4f}")
    
    print(f"\n📋 Confusion Matrix:")
    print(f"   TN (Correct Normal):  {tn:4d}")
    print(f"   FP (False alarm):     {fp:4d}")
    print(f"   FN (Missed TB):       {fn:4d}")
    print(f"   TP (Detected TB):     {tp:4d}")
    
    miss_rate = fn / (fn + tp) * 100 if (fn + tp) > 0 else 0
    false_alarm_rate = fp / (tn + fp) * 100 if (tn + fp) > 0 else 0
    
    print(f"\n⚠️  TB Miss Rate:     {miss_rate:5.1f}% ({fn} TB cases missed)")
    print(f"⚠️  False Alarm Rate: {false_alarm_rate:5.1f}% ({fp} normal cases flagged)")
    
    print(f"\n{classification_report(y_test, y_pred, target_names=Config.CATEGORIES)}")
    
    # Save results
    results = {
        'model_path': model_path,
        'test_timestamp': datetime.now().isoformat(),
        'metrics': {
            'accuracy': float(accuracy),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'precision': float(precision),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc)
        },
        'confusion_matrix': {
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn),
            'TP': int(tp)
        },
        'rates': {
            'miss_rate_percent': float(miss_rate),
            'false_alarm_rate_percent': float(false_alarm_rate)
        },
        'data_info': {
            'n_test': len(X_test),
            'class_distribution': np.bincount(y_test).tolist()
        }
    }
    
    results_path = model_path.replace('.pkl', '_test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Test results saved to: {results_path}")
    
    return results

if __name__ == "__main__":
    test_svm_model()
    print(f"\n{'='*80}\n")
