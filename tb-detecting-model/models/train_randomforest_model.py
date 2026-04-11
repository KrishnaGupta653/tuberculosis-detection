"""
================================================================================
RANDOM FOREST MODEL TRAINER — INDIVIDUAL
================================================================================
Trains Random Forest independently with class weighting.
Saves: randomforest_model_TIMESTAMP.pkl with metrics
"""

import numpy as np
import pickle
import os
import sys
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, accuracy_score, confusion_matrix,
                            precision_score, recall_score, f1_score)
from tqdm import tqdm

# Add parent directory to path so pipeline_shared can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from pipeline_shared import (
    Config, load_image_paths, extract_features_parallel, 
    get_sample_weights
)

# ═══════════════════════════════════════════════════════════════════════════
# RANDOM FOREST MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_randomforest_model():
    """Train Random Forest with class balancing"""
    
    print(f"\n{'='*80}")
    print("🎯 TRAINING RANDOM FOREST MODEL INDEPENDENTLY")
    print(f"{'='*80}\n")
    
    # Step 1: Load data
    print("📂 Loading image paths...")
    X_paths_train, y_train = load_image_paths(os.path.join(Config.DATA_DIR))
    
    from sklearn.model_selection import train_test_split
    X_paths_train, X_paths_test, y_train, y_test = train_test_split(
        X_paths_train, y_train, test_size=Config.TEST_SIZE, 
        random_state=Config.RANDOM_SEED, stratify=y_train
    )
    
    print(f"✓ Train set: {len(X_paths_train)} images")
    print(f"✓ Test set: {len(X_paths_test)} images")
    print(f"✓ Class distribution (train): {np.bincount(y_train)}")
    print(f"✓ Class distribution (test): {np.bincount(y_test)}")
    
    # Step 2: Extract features
    print("\n🔬 Extracting radiomics features from training images...")
    X_train, y_train = extract_features_parallel(X_paths_train, y_train, 
                                                  "Training features")
    
    print(f"\n✓ Features extracted: shape {X_train.shape}")
    print(f"✓ Training samples: {len(X_train)}")
    
    # Step 3: Normalize (for consistency)
    print("\n📊 Normalizing features...")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Step 4: Extract test features
    print("\n🔬 Extracting radiomics features from test images...")
    X_test, y_test = extract_features_parallel(X_paths_test, y_test, 
                                               "Test features")
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✓ Test features shape: {X_test.shape}")
    
    # Step 5: Compute sample weights
    print("\n⚖️  Computing class weights...")
    sample_weights_train = get_sample_weights(y_train)
    class_weights_dict = {}
    for cls in np.unique(y_train):
        class_weights_dict[cls] = sample_weights_train[y_train == cls][0]
    print(f"✓ Class weights: {class_weights_dict}")
    
    # Step 6: Train Random Forest
    print("\n🤖 Training Random Forest with 200 trees...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight=class_weights_dict,
        random_state=Config.RANDOM_SEED,
        n_jobs=Config.N_JOBS,
        verbose=1
    )
    
    rf_model.fit(X_train_scaled, y_train, sample_weight=sample_weights_train)
    print("✓ Random Forest training complete!")
    
    # Step 7: Cross-validation
    print(f"\n📈 Running {Config.CROSS_VAL_FOLDS}-fold cross-validation...")
    cv = StratifiedKFold(n_splits=Config.CROSS_VAL_FOLDS, 
                        shuffle=True, random_state=Config.RANDOM_SEED)
    cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, 
                               cv=cv, scoring='accuracy')
    print(f"✓ CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Step 8: Test evaluation
    print("\n🎯 Evaluating on test set...")
    y_pred = rf_model.predict(X_test_scaled)
    y_pred_proba = rf_model.predict_proba(X_test_scaled)
    
    test_accuracy = accuracy_score(y_test, y_pred)
    test_sensitivity = recall_score(y_test, y_pred, pos_label=1)
    test_specificity = recall_score(y_test, y_pred, pos_label=0)
    test_precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    test_f1 = f1_score(y_test, y_pred)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n📊 TEST SET METRICS:")
    print(f"  Accuracy:    {test_accuracy*100:.2f}%")
    print(f"  Sensitivity: {test_sensitivity*100:.2f}%")
    print(f"  Specificity: {test_specificity*100:.2f}%")
    print(f"  Precision:   {test_precision*100:.2f}%")
    print(f"  F1-Score:    {test_f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={tn}, FP={fp}")
    print(f"    FN={fn}, TP={tp}")
    
    print("\n" + classification_report(y_test, y_pred, 
                                       target_names=Config.CATEGORIES))
    
    # Step 9: Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(Config.MODEL_DIR, f"randomforest_model_{timestamp}.pkl")
    
    # Feature importance
    feature_importance = rf_model.feature_importances_
    top_features_idx = np.argsort(feature_importance)[-10:][::-1]
    
    model_package = {
        'model': rf_model,
        'scaler': scaler,
        'timestamp': timestamp,
        'config': {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'class_weight': class_weights_dict,
            'n_features': X_train.shape[1]
        },
        'feature_importance': {
            'all': feature_importance.tolist(),
            'top_10_indices': top_features_idx.tolist(),
            'top_10_scores': feature_importance[top_features_idx].tolist()
        },
        'training_metrics': {
            'cv_mean_accuracy': float(cv_scores.mean()),
            'cv_std_accuracy': float(cv_scores.std()),
            'cv_scores': cv_scores.tolist()
        },
        'test_metrics': {
            'accuracy': float(test_accuracy),
            'sensitivity': float(test_sensitivity),
            'specificity': float(test_specificity),
            'precision': float(test_precision),
            'f1_score': float(test_f1),
            'confusion_matrix': cm.tolist(),
            'y_pred': y_pred.tolist(),
            'y_pred_proba': y_pred_proba.tolist(),
            'y_true': y_test.tolist()
        },
        'data_info': {
            'n_train': len(X_train),
            'n_test': len(X_test),
            'class_distribution_train': np.bincount(y_train).tolist(),
            'class_distribution_test': np.bincount(y_test).tolist()
        }
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_package, f)
    
    print(f"\n✅ Random Forest model saved to: {model_path}")
    print(f"   Total parameters stored: {len(model_package)} items")
    
    return model_path, model_package


if __name__ == "__main__":
    model_path, model_pkg = train_randomforest_model()
    print(f"\n{'='*80}")
    print("✅ RANDOM FOREST TRAINING COMPLETE")
    print(f"{'='*80}\n")
