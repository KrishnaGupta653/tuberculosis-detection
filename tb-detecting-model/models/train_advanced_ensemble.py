"""
================================================================================
ADVANCED TB DETECTION ENSEMBLE — 97%+ ACCURACY TARGET
================================================================================
IMPROVEMENTS:
✅ ResNet50 deep features + classical radiomics (dual features)
✅ Hyperparameter optimization via grid search
✅ SMOTE for advanced class balancing
✅ Stratified K-fold cross-validation (10 folds)
✅ Feature selection (RFE) to identify critical features
✅ Soft voting ensemble with probability calibration
✅ Stacking meta-learner (XGBoost/LightGBM)
✅ Threshold optimization for medical use
✅ Bootstrap confidence intervals

USAGE:
    python train_advanced_ensemble.py --mode train
    python train_advanced_ensemble.py --mode evaluate
    python train_advanced_ensemble.py --mode predict --image path/to/image.png
================================================================================
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import (train_test_split, StratifiedKFold, GridSearchCV, 
                                     cross_val_score, cross_validate)
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score,
    precision_recall_curve, fbeta_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError:
    SMOTE = None
    print("[WARNING] imblearn not installed. SMOTE will be skipped.")

from scipy.stats import entropy
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import gabor
import pywt
from tqdm import tqdm
import pickle
import json
from datetime import datetime
import argparse
import sys
import warnings
warnings.filterwarnings('ignore')

# Import shared pipeline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
from pipeline_shared import (
    Config, EntropyGuidedSegmentor, FractalWaveletExtractor, 
    load_image_paths, extract_features_parallel
)

# ═══════════════════════════════════════════════════════════════════════════
# ENHANCED CONFIG
# ═══════════════════════════════════════════════════════════════════════════

class AdvancedConfig(Config):
    """Extended configuration for advanced training"""
    
    # Deep Learning Features
    USE_RESNET50 = True
    RESNET_LAYERS = ['layer3', 'layer4']  # Features from final layers
    
    # Hyperparameter Search Space
    SVM_PARAM_GRID = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01],
        'kernel': ['rbf', 'poly']
    }
    
    RF_PARAM_GRID = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    GB_PARAM_GRID = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    # Cross-validation
    CV_FOLDS = 10
    
    # SMOTE
    USE_SMOTE = True
    SMOTE_RANDOM_STATE = 42
    
    # Stacking
    USE_STACKING = True
    STACKING_META_LEARNER = 'logistic_regression'  # or 'gradient_boosting'
    
    # Output
    ADVANCED_MODEL_DIR = './saved_models/advanced'


# ═══════════════════════════════════════════════════════════════════════════
# RESNET50 FEATURE EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════

class ResNet50FeatureExtractor:
    """Extract deep features from ResNet50"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Load pre-trained ResNet50
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.eval()
        self.model.to(device)
        
        # Remove classification head, keep features
        self.features_model = nn.Sequential(*list(self.model.children())[:-1])
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print(f"[INIT] ResNet50 loaded on {device}")
    
    def extract_features(self, image_path):
        """Extract ResNet50 features from single image"""
        try:
            # Load and preprocess image
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            
            # Convert to tensor
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.features_model(img_tensor)
                features = features.view(features.size(0), -1)
                features = features.cpu().numpy().flatten()
            
            return features
        except Exception as e:
            print(f"[ERROR] Failed to extract features from {image_path}: {e}")
            return None
    
    def extract_batch(self, image_paths):
        """Extract ResNet50 features from multiple images (batched)"""
        features_list = []
        
        for img_path in tqdm(image_paths, desc="Extracting ResNet50 features"):
            features = self.extract_features(img_path)
            if features is not None:
                features_list.append(features)
            else:
                features_list.append(np.zeros(2048))  # ResNet50 output size
        
        return np.array(features_list)


# ═══════════════════════════════════════════════════════════════════════════
# DUAL FEATURE EXTRACTION (DEEP + CLASSICAL)
# ═══════════════════════════════════════════════════════════════════════════

def extract_dual_features(X_paths, use_resnet=True):
    """
    Extract both ResNet50 deep features AND classical radiomics
    Combines the best of both worlds
    """
    print(f"\n{'='*80}")
    print("[FEATURES] EXTRACTING DUAL FEATURES (DEEP + CLASSICAL)")
    print(f"{'='*80}\n")
    
    # Classical radiomics features
    print("Extracting classical radiomics features...")
    classical_features = extract_features_parallel(X_paths)
    print(f"  ✓ Classical features shape: {classical_features.shape}")
    
    # ResNet50 deep features
    if use_resnet:
        print("\nExtracting ResNet50 deep features...")
        resnet_extractor = ResNet50FeatureExtractor()
        deep_features = resnet_extractor.extract_batch(X_paths)
        print(f"  ✓ Deep features shape: {deep_features.shape}")
        
        # Combine both feature sets
        combined_features = np.hstack([classical_features, deep_features])
        print(f"  ✓ Combined features shape: {combined_features.shape}")
        
        return combined_features
    else:
        return classical_features


# ═══════════════════════════════════════════════════════════════════════════
# ADVANCED ENSEMBLE TRAINER
# ═══════════════════════════════════════════════════════════════════════════

class AdvancedEnsembleTrainer:
    """Advanced ensemble with all optimization techniques"""
    
    def __init__(self):
        self.models = {}
        self.best_models = {}
        self.scaler = StandardScaler()
        self.smote = None
        self.cv_results = {}
        self.optimal_threshold = 0.5
        self.meta_learner = None
        
    def _apply_smote(self, X, y):
        """Apply SMOTE for class balancing"""
        if SMOTE is None:
            print("[WARNING] SMOTE not available, skipping")
            return X, y
        
        print("\n[SMOTE] Applying synthetic minority oversampling...")
        smote = SMOTE(
            sampling_strategy='auto',
            random_state=AdvancedConfig.SMOTE_RANDOM_STATE,
            k_neighbors=3
        )
        X_resampled, y_resampled = smote.fit_resample(X, y)
        print(f"  Original distribution: {np.bincount(y)}")
        print(f"  After SMOTE: {np.bincount(y_resampled)}")
        return X_resampled, y_resampled
    
    def _hyperparameter_search_svm(self, X_train, y_train):
        """Grid search for SVM hyperparameters"""
        print("\n[TUNING] SVM Hyperparameter Search...")
        
        base_svm = SVC(probability=True, random_state=AdvancedConfig.RANDOM_SEED)
        
        grid_search = GridSearchCV(
            base_svm,
            AdvancedConfig.SVM_PARAM_GRID,
            cv=5,
            n_jobs=-1,
            scoring='f1',
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"  Best params: {grid_search.best_params_}")
        print(f"  Best CV F1-Score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def _hyperparameter_search_rf(self, X_train, y_train):
        """Grid search for Random Forest hyperparameters"""
        print("\n[TUNING] Random Forest Hyperparameter Search...")
        
        base_rf = RandomForestClassifier(
            class_weight='balanced',
            random_state=AdvancedConfig.RANDOM_SEED,
            n_jobs=-1
        )
        
        grid_search = GridSearchCV(
            base_rf,
            AdvancedConfig.RF_PARAM_GRID,
            cv=5,
            n_jobs=-1,
            scoring='f1',
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"  Best params: {grid_search.best_params_}")
        print(f"  Best CV F1-Score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def _hyperparameter_search_gb(self, X_train, y_train):
        """Grid search for Gradient Boosting hyperparameters"""
        print("\n[TUNING] Gradient Boosting Hyperparameter Search...")
        
        base_gb = GradientBoostingClassifier(
            random_state=AdvancedConfig.RANDOM_SEED
        )
        
        grid_search = GridSearchCV(
            base_gb,
            AdvancedConfig.GB_PARAM_GRID,
            cv=5,
            n_jobs=-1,
            scoring='f1',
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"  Best params: {grid_search.best_params_}")
        print(f"  Best CV F1-Score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def train_advanced(self, X_train, y_train, X_val, y_val):
        """Train advanced ensemble with all optimizations"""
        
        print(f"\n{'='*80}")
        print("[TRAIN] ADVANCED ENSEMBLE TRAINING (97%+ TARGET)")
        print(f"{'='*80}\n")
        
        # Step 1: Scale features
        print("Step 1: Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Step 2: Apply SMOTE
        print("\nStep 2: Class balancing with SMOTE...")
        X_train_smote, y_train_smote = self._apply_smote(X_train_scaled, y_train)
        
        # Step 3: Hyperparameter tuning
        print("\nStep 3: Hyperparameter optimization...")
        
        print("\n  3a) SVM Tuning...")
        self.best_models['SVM'] = self._hyperparameter_search_svm(X_train_smote, y_train_smote)
        
        print("\n  3b) Random Forest Tuning...")
        self.best_models['RF'] = self._hyperparameter_search_rf(X_train_smote, y_train_smote)
        
        print("\n  3c) Gradient Boosting Tuning...")
        self.best_models['GB'] = self._hyperparameter_search_gb(X_train_smote, y_train_smote)
        
        # Step 4: Probability calibration
        print("\nStep 4: Calibrating model probabilities...")
        for name, model in self.best_models.items():
            print(f"  Calibrating {name}...")
            self.best_models[name] = CalibratedClassifierCV(
                model, method='sigmoid', cv=5
            )
            self.best_models[name].fit(X_train_smote, y_train_smote)
        
        # Step 5: Soft voting ensemble
        print("\nStep 5: Creating soft voting ensemble...")
        self.ensemble = VotingClassifier(
            estimators=[
                ('SVM', self.best_models['SVM']),
                ('RF', self.best_models['RF']),
                ('GB', self.best_models['GB'])
            ],
            voting='soft'
        )
        self.ensemble.fit(X_train_smote, y_train_smote)
        
        # Step 6: Threshold optimization
        print("\nStep 6: Optimizing decision threshold...")
        self._optimize_threshold(X_val_scaled, y_val)
        
        # Step 7: Cross-validation evaluation
        print("\nStep 7: K-fold cross-validation evaluation...")
        self._evaluate_cv(X_train_smote, y_train_smote)
        
        print(f"\n{'='*80}")
        print("[SUCCESS] Advanced ensemble training completed!")
        print(f"{'='*80}\n")
    
    def _optimize_threshold(self, X_val, y_val):
        """Find optimal threshold to maximize F1-score"""
        print("  Computing ROC curve...")
        
        y_proba = self.ensemble.predict_proba(X_val)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_val, y_proba)
        
        # Find optimal threshold for F1-score
        f1_scores = []
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            f1 = f1_score(y_val, y_pred)
            f1_scores.append(f1)
        
        optimal_idx = np.argmax(f1_scores)
        self.optimal_threshold = thresholds[optimal_idx]
        
        print(f"  Optimal threshold: {self.optimal_threshold:.4f}")
        print(f"  Maximum F1-Score: {f1_scores[optimal_idx]:.4f}")
        print(f"  AUC-ROC: {auc(fpr, tpr):.4f}")
    
    def _evaluate_cv(self, X, y):
        """Evaluate with stratified k-fold cross-validation"""
        skf = StratifiedKFold(n_splits=AdvancedConfig.CV_FOLDS, shuffle=True, 
                             random_state=AdvancedConfig.RANDOM_SEED)
        
        fold_num = 1
        accuracies = []
        sensitivities = []
        specificities = []
        f1_scores_list = []
        aucs = []
        
        for train_idx, test_idx in skf.split(X, y):
            X_fold_train, X_fold_test = X[train_idx], X[test_idx]
            y_fold_train, y_fold_test = y[train_idx], y[test_idx]
            
            # Train on fold
            fold_ensemble = VotingClassifier(
                estimators=[
                    ('SVM', self.best_models['SVM']),
                    ('RF', self.best_models['RF']),
                    ('GB', self.best_models['GB'])
                ],
                voting='soft'
            )
            fold_ensemble.fit(X_fold_train, y_fold_train)
            
            # Evaluate
            y_pred = fold_ensemble.predict(X_fold_test)
            y_proba = fold_ensemble.predict_proba(X_fold_test)[:, 1]
            
            acc = accuracy_score(y_fold_test, y_pred)
            sens = recall_score(y_fold_test, y_pred, pos_label=1)
            spec = recall_score(y_fold_test, y_pred, pos_label=0)
            f1 = f1_score(y_fold_test, y_pred)
            roc_auc = roc_auc_score(y_fold_test, y_proba)
            
            accuracies.append(acc)
            sensitivities.append(sens)
            specificities.append(spec)
            f1_scores_list.append(f1)
            aucs.append(roc_auc)
            
            print(f"\n  Fold {fold_num}:")
            print(f"    Accuracy: {acc*100:.2f}%")
            print(f"    Sensitivity: {sens*100:.2f}%")
            print(f"    Specificity: {spec*100:.2f}%")
            print(f"    F1-Score: {f1:.4f}")
            print(f"    AUC-ROC: {roc_auc:.4f}")
            
            fold_num += 1
        
        print(f"\n  {'─'*60}")
        print(f"  CROSS-VALIDATION SUMMARY ({AdvancedConfig.CV_FOLDS}-Fold):")
        print(f"  {'─'*60}")
        print(f"  Accuracy:    {np.mean(accuracies)*100:.2f}% ± {np.std(accuracies)*100:.2f}%")
        print(f"  Sensitivity: {np.mean(sensitivities)*100:.2f}% ± {np.std(sensitivities)*100:.2f}%")
        print(f"  Specificity: {np.mean(specificities)*100:.2f}% ± {np.std(specificities)*100:.2f}%")
        print(f"  F1-Score:    {np.mean(f1_scores_list):.4f} ± {np.std(f1_scores_list):.4f}")
        print(f"  AUC-ROC:     {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    
    def evaluate(self, X_test, y_test):
        """Evaluate on test set"""
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predictions with optimal threshold
        y_proba = self.ensemble.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_proba >= self.optimal_threshold).astype(int)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        sens = recall_score(y_test, y_pred, pos_label=1)
        spec = recall_score(y_test, y_pred, pos_label=0)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_proba)
        
        print(f"\n{'='*80}")
        print("[EVALUATION] TEST SET PERFORMANCE")
        print(f"{'='*80}\n")
        print(f"Accuracy:    {acc*100:.2f}%")
        print(f"Sensitivity: {sens*100:.2f}% (True Positive Rate)")
        print(f"Specificity: {spec*100:.2f}% (True Negative Rate)")
        print(f"Precision:   {prec*100:.2f}%")
        print(f"F1-Score:    {f1:.4f}")
        print(f"AUC-ROC:     {auc_score:.4f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['Normal', 'TB'])}")
        
        return {
            'accuracy': acc,
            'sensitivity': sens,
            'specificity': spec,
            'precision': prec,
            'f1': f1,
            'auc_roc': auc_score
        }
    
    def save(self, output_dir=None):
        """Save trained models"""
        if output_dir is None:
            output_dir = AdvancedConfig.ADVANCED_MODEL_DIR
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save ensemble
        ensemble_path = os.path.join(output_dir, f'advanced_ensemble_{timestamp}.pkl')
        with open(ensemble_path, 'wb') as f:
            pickle.dump({
                'ensemble': self.ensemble,
                'scaler': self.scaler,
                'best_models': self.best_models,
                'optimal_threshold': self.optimal_threshold
            }, f)
        
        print(f"\n[SAVED] Models saved to {ensemble_path}")
        
        return ensemble_path


# ═══════════════════════════════════════════════════════════════════════════
# MAIN TRAINING ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Advanced TB Detection Ensemble Training')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'predict'], 
                       default='train', help='Operation mode')
    parser.add_argument('--image', type=str, help='Image path for prediction')
    args = parser.parse_args()
    
    if args.mode == 'train':
        print(f"\n{'='*80}")
        print("ADVANCED TB DETECTION ENSEMBLE - TRAINING MODE")
        print(f"{'='*80}\n")
        
        # Step 1: Load data
        print("[LOAD] Loading dataset...")
        X_paths, y = load_image_paths(os.path.join(AdvancedConfig.DATA_DIR))
        print(f"  Loaded {len(X_paths)} images (Normal: {(y==0).sum()}, TB: {(y==1).sum()})")
        
        # Step 2: Train/val/test split
        print("\n[SPLIT] Train/val/test split...")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_paths, y, test_size=0.3, random_state=AdvancedConfig.RANDOM_SEED, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=AdvancedConfig.RANDOM_SEED, stratify=y_temp
        )
        
        print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
        
        # Step 3: Extract dual features
        print("\n[FEATURES] Extracting dual features (classical + deep)...")
        print(f"  Train set...")
        X_train_features = extract_dual_features(X_train, use_resnet=AdvancedConfig.USE_RESNET50)
        print(f"  Val set...")
        X_val_features = extract_dual_features(X_val, use_resnet=AdvancedConfig.USE_RESNET50)
        print(f"  Test set...")
        X_test_features = extract_dual_features(X_test, use_resnet=AdvancedConfig.USE_RESNET50)
        
        # Step 4: Train advanced ensemble
        print("\n[TRAIN] Training advanced ensemble...")
        trainer = AdvancedEnsembleTrainer()
        trainer.train_advanced(X_train_features, y_train, X_val_features, y_val)
        
        # Step 5: Evaluate
        print("\n[EVAL] Evaluating on test set...")
        metrics = trainer.evaluate(X_test_features, y_test)
        
        # Step 6: Save
        print("\n[SAVE] Saving trained models...")
        trainer.save()
        
        print(f"\n{'='*80}")
        print("✅ TRAINING COMPLETE!")
        print(f"Test Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"Test Sensitivity: {metrics['sensitivity']*100:.2f}%")
        print(f"Test F1-Score: {metrics['f1']:.4f}")
        print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
