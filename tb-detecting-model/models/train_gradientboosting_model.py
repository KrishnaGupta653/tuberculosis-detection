"""
================================================================================
GRADIENT BOOSTING MODEL TRAINER — INDIVIDUAL
================================================================================
Trains Gradient Boosting independently with class weighting.
Saves: gradientboosting_model_TIMESTAMP.pkl with metrics
"""

import numpy as np
import pickle
import os
import sys
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier
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
# GRADIENT BOOSTING MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_gradientboosting_model():
    """Train Gradient Boosting with class balancing"""
    
    print(f"\n{'='*80}")
    print("🎯 TRAINING GRADIENT BOOSTING MODEL INDEPENDENTLY")
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
    
    # Step 3: Normalize
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
    
    # Step 6: Train Gradient Boosting
    print("\n🤖 Training Gradient Boosting with 100 stages...")
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=Config.RANDOM_SEED,
        verbose=1
    )
    
    gb_model.fit(X_train_scaled, y_train, sample_weight=sample_weights_train)
    print("✓ Gradient Boosting training complete!")
    
    # Step 7: Cross-validation
    print(f"\n📈 Running {Config.CROSS_VAL_FOLDS}-fold cross-validation...")
    cv = StratifiedKFold(n_splits=Config.CROSS_VAL_FOLDS, 
                        shuffle=True, random_state=Config.RANDOM_SEED)
    cv_scores = cross_val_score(gb_model, X_train_scaled, y_train, 
                               cv=cv, scoring='accuracy')
    print(f"✓ CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Step 8: Test evaluation
    print("\n🎯 Evaluating on test set...")
    y_pred = gb_model.predict(X_test_scaled)
    y_pred_proba = gb_model.predict_proba(X_test_scaled)
    
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
    model_path = os.path.join(Config.MODEL_DIR, f"gradientboosting_model_{timestamp}.pkl")
    
    # Feature importance
    feature_importance = gb_model.feature_importances_
    top_features_idx = np.argsort(feature_importance)[-10:][::-1]
    
    model_package = {
        'model': gb_model,
        'scaler': scaler,
        'timestamp': timestamp,
        'config': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'subsample': 0.8,
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
    
    print(f"\n✅ Gradient Boosting model saved to: {model_path}")
    print(f"   Total parameters stored: {len(model_package)} items")
    
    return model_path, model_package


if __name__ == "__main__":
    model_path, model_pkg = train_gradientboosting_model()
    print(f"\n{'='*80}")
    print("✅ GRADIENT BOOSTING TRAINING COMPLETE")
    print(f"{'='*80}\n")
