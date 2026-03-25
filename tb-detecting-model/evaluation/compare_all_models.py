"""
================================================================================
COMPARE ALL INDIVIDUAL MODELS — SIDE-BY-SIDE ANALYSIS
================================================================================
Loads all trained models and compares predictions on test set
Shows individual vs. ensemble performance
"""

import os
import pickle
import json
import numpy as np
from glob import glob
from datetime import datetime

from pipeline_shared import Config, load_image_paths, extract_features_parallel

def compare_all_models():
    """Compare SVM, RF, GB on test set"""
    
    print(f"\n{'='*80}")
    print("📊 COMPARING ALL INDIVIDUAL MODELS")
    print(f"{'='*80}\n")
    
    # Load all models
    models_dict = {}
    model_types = {
        'SVM': 'svm_model_*.pkl',
        'Random Forest': 'randomforest_model_*.pkl',
        'Gradient Boosting': 'gradientboosting_model_*.pkl'
    }
    
    for model_type, pattern in model_types.items():
        model_files = glob(os.path.join(Config.MODEL_DIR, pattern))
        if not model_files:
            print(f"⚠️  {model_type}: No models found")
            continue
        
        model_path = sorted(model_files)[-1]
        print(f"📂 Loading {model_type:20s}: {os.path.basename(model_path)}")
        
        with open(model_path, 'rb') as f:
            models_dict[model_type] = {
                'path': model_path,
                'data': pickle.load(f)
            }
    
    if len(models_dict) < 3:
        print("\n❌ Not all models found! Please train all models first.")
        return
    
    print(f"\n✓ Loaded {len(models_dict)} models\n")
    
    # Load test data
    print("📂 Loading test image paths...")
    from sklearn.model_selection import train_test_split
    X_paths_all, y_all = load_image_paths(Config.DATA_DIR)
    
    _, X_paths_test, _, y_test = train_test_split(
        X_paths_all, y_all, test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_SEED, stratify=y_all
    )
    
    print(f"✓ Test set: {len(X_paths_test)} images\n")
    
    # Extract features once
    print("🔬 Extracting features from test set...")
    X_test, y_test = extract_features_parallel(X_paths_test, y_test, "Test features")
    
    # Predictions from all models
    print("\n🎯 Getting predictions from all models...\n")
    
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                f1_score, confusion_matrix, classification_report, roc_auc_score)
    
    all_predictions = {}
    all_metrics = {}
    
    for model_type, model_info in models_dict.items():
        model = model_info['data']['model']
        scaler = model_info['data']['scaler']
        
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)
        
        all_predictions[model_type] = {
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        sensitivity = recall_score(y_test, y_pred, pos_label=1)
        specificity = recall_score(y_test, y_pred, pos_label=0)
        precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        all_metrics[model_type] = {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
        }
    
    # Ensemble by averaging probabilities
    print("\n🔗 Creating ensemble by averaging predictions...\n")
    
    ensemble_proba_sum = np.zeros_like(all_predictions['SVM']['y_pred_proba'])
    for model_type in models_dict.keys():
        ensemble_proba_sum += all_predictions[model_type]['y_pred_proba']
    
    ensemble_proba = ensemble_proba_sum / len(models_dict)
    ensemble_pred = np.argmax(ensemble_proba, axis=1)
    
    accuracy_ens = accuracy_score(y_test, ensemble_pred)
    sensitivity_ens = recall_score(y_test, ensemble_pred, pos_label=1)
    specificity_ens = recall_score(y_test, ensemble_pred, pos_label=0)
    precision_ens = precision_score(y_test, ensemble_pred, pos_label=1, zero_division=0)
    f1_ens = f1_score(y_test, ensemble_pred)
    roc_auc_ens = roc_auc_score(y_test, ensemble_proba[:, 1])
    
    cm_ens = confusion_matrix(y_test, ensemble_pred)
    tn_ens, fp_ens, fn_ens, tp_ens = cm_ens.ravel()
    
    all_metrics['Ensemble (Average)'] = {
        'accuracy': accuracy_ens,
        'sensitivity': sensitivity_ens,
        'specificity': specificity_ens,
        'precision': precision_ens,
        'f1_score': f1_ens,
        'roc_auc': roc_auc_ens,
        'tn': tn_ens, 'fp': fp_ens, 'fn': fn_ens, 'tp': tp_ens
    }
    
    # Print comparison table
    print(f"{'='*80}")
    print("📊 DETAILED COMPARISON — SIDE-BY-SIDE ANALYSIS")
    print(f"{'='*80}\n")
    
    print(f"{'Model':<25} | {'Accuracy':<8} | {'Sensitiv.':<8} | {'Specific.':<8} | {'Precision':<8} | {'F1':<6}")
    print(f"{'-'*25}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}")
    
    for model_type in list(models_dict.keys()) + ['Ensemble (Average)']:
        m = all_metrics[model_type]
        print(f"{model_type:<25} | {m['accuracy']*100:7.2f}% | {m['sensitivity']*100:7.2f}% | {m['specificity']*100:7.2f}% | {m['precision']*100:7.2f}% | {m['f1_score']:.4f}")
    
    print(f"\n{'='*80}")
    print("📋 CONFUSION MATRICES")
    print(f"{'='*80}\n")
    
    for model_type in list(models_dict.keys()) + ['Ensemble (Average)']:
        m = all_metrics[model_type]
        miss_rate = m['fn'] / (m['fn'] + m['tp']) * 100 if (m['fn'] + m['tp']) > 0 else 0
        false_alarm = m['fp'] / (m['tn'] + m['fp']) * 100 if (m['tn'] + m['fp']) > 0 else 0
        
        print(f"{model_type}:")
        print(f"  TN={m['tn']:3d}  FP={m['fp']:3d}   |   TB Miss Rate: {miss_rate:5.1f}%")
        print(f"  FN={m['fn']:3d}  TP={m['tp']:3d}   |   False Alarm:  {false_alarm:5.1f}%")
        print()
    
    # Best performing model
    print(f"{'='*80}")
    print("🏆 PERFORMANCE COMPARISON")
    print(f"{'='*80}\n")
    
    best_accuracy = max(all_metrics.items(), key=lambda x: x[1]['accuracy'])
    best_sensitivity = max(all_metrics.items(), key=lambda x: x[1]['sensitivity'])
    best_specificity = max(all_metrics.items(), key=lambda x: x[1]['specificity'])
    best_f1 = max(all_metrics.items(), key=lambda x: x[1]['f1_score'])
    best_roc = max(all_metrics.items(), key=lambda x: x[1]['roc_auc'])
    
    print(f"🥇 Best Accuracy:    {best_accuracy[0]:25s} ({best_accuracy[1]['accuracy']*100:.2f}%)")
    print(f"🥇 Best Sensitivity: {best_sensitivity[0]:25s} ({best_sensitivity[1]['sensitivity']*100:.2f}%)")
    print(f"🥇 Best Specificity: {best_specificity[0]:25s} ({best_specificity[1]['specificity']*100:.2f}%)")
    print(f"🥇 Best F1-Score:    {best_f1[0]:25s} ({best_f1[1]['f1_score']:.4f})")
    print(f"🥇 Best ROC-AUC:     {best_roc[0]:25s} ({best_roc[1]['roc_auc']:.4f})")
    
    # Insights
    print(f"\n{'='*80}")
    print("💡 KEY INSIGHTS")
    print(f"{'='*80}\n")
    
    # Check if ensemble helps
    ens_vs_models = []
    for model_type in models_dict.keys():
        diff = all_metrics['Ensemble (Average)']['accuracy'] - all_metrics[model_type]['accuracy']
        ens_vs_models.append((model_type, diff))
    
    avg_improvement = np.mean([abs(d) for _, d in ens_vs_models])
    
    if avg_improvement < 0.02:
        print("⚠️  Ensemble provides minimal improvement over individual models")
        print("    → Models may be too similar / highly correlated")
    else:
        print("✓ Ensemble shows meaningful improvement")
    
    # Sensitivity analysis
    print("\n⚠️  Sensitivity Analysis (TB Detection Rate):")
    for model_type in list(models_dict.keys()) + ['Ensemble (Average)']:
        sens = all_metrics[model_type]['sensitivity']
        print(f"    {model_type:25s}: {sens*100:.1f}% (miss {100-sens*100:.1f}% of TB cases)")
    
    # Save comparison report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(Config.MODEL_DIR, f"model_comparison_{timestamp}.json")
    
    report = {
        'timestamp': timestamp,
        'test_set_size': len(X_test),
        'models_compared': list(models_dict.keys()),
        'metrics': all_metrics,
        'comparison_table': {
            model_type: {
                'accuracy_percent': m['accuracy'] * 100,
                'sensitivity_percent': m['sensitivity'] * 100,
                'specificity_percent': m['specificity'] * 100,
                'roc_auc': m['roc_auc']
            }
            for model_type, m in all_metrics.items()
        }
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Comparison report saved to: {report_path}")
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    compare_all_models()
