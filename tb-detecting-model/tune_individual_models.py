"""
GridSearchCV Tuning for Individual Models to Reach 84% Accuracy
Automatically finds best hyperparameters for SVM, RF, GB
"""

import numpy as np
import os
import sys
import time
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, make_scorer
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_shared import Config, load_image_paths, extract_features_parallel

print("\n" + "="*80)
print("HYPERPARAMETER TUNING - GRIDSEARCH FOR 84% ACCURACY")
print("="*80 + "\n")

# Load data once
print("📂 Loading and extracting features (this takes a minute)...")
X_paths, y = load_image_paths(Config.DATA_DIR)

X_paths_train, X_paths_test, y_train, y_test = train_test_split(
    X_paths, y, test_size=Config.TEST_SIZE, 
    random_state=Config.RANDOM_SEED, stratify=y
)

X_train, y_train = extract_features_parallel(X_paths_train, y_train, "Training")
X_test, y_test = extract_features_parallel(X_paths_test, y_test, "Test")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Dataset ready: {len(X_train)} train, {len(X_test)} test, {X_train.shape[1]} features\n")

# Custom scorer for sensitivity + specificity weighted average
def custom_score(y_true, y_pred):
    """Balance sensitivity and specificity"""
    sens = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    spec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    return (sens + spec) / 2

# ═══════════════════════════════════════════════════════════════════════════
# 1. SVM TUNING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("1️⃣  TUNING SVM - Finding best C, gamma, kernel")
print("="*80 + "\n")

svm_params = {
    'C': [10, 100, 500, 1000],
    'gamma': [0.0001, 0.001, 0.01, 0.1, 'scale'],
    'kernel': ['rbf', 'poly']
}

svm_base = SVC(class_weight='balanced', probability=True, random_state=42)
svm_grid = GridSearchCV(svm_base, svm_params, cv=5, scoring='accuracy', 
                        n_jobs=-1, verbose=1)

print(f"Searching {len(svm_params['C']) * len(svm_params['gamma']) * len(svm_params['kernel'])} combinations...")
start = time.time()
svm_grid.fit(X_train_scaled, y_train)
print(f"✓ Tuning completed in {time.time()-start:.1f}s\n")

svm_best = svm_grid.best_estimator_
svm_train_acc = svm_grid.best_score_
svm_test_acc = svm_best.score(X_test_scaled, y_test)

print(f"📊 SVM RESULTS:")
print(f"  Best params: {svm_grid.best_params_}")
print(f"  CV Accuracy: {svm_train_acc*100:.2f}%")
print(f"  Test Accuracy: {svm_test_acc*100:.2f}%")
if svm_test_acc >= 0.84:
    print(f"  ✅ REACHED 84% TARGET!")
else:
    print(f"  ⏳ Gap to reach 84%: {(0.84 - svm_test_acc)*100:.1f}%")

# ═══════════════════════════════════════════════════════════════════════════
# 2. RANDOM FOREST TUNING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("2️⃣  TUNING RANDOM FOREST - Finding best n_estimators, max_depth, min_samples_split")
print("="*80 + "\n")

rf_params = {
    'n_estimators': [300, 500, 800],
    'max_depth': [8, 10, 12, 15],
    'min_samples_split': [5, 10, 15],
    'min_samples_leaf': [2, 4, 6]
}

rf_base = RandomForestClassifier(class_weight='balanced_subsample', n_jobs=-1, random_state=42)
rf_grid = GridSearchCV(rf_base, rf_params, cv=5, scoring='accuracy',
                       n_jobs=-1, verbose=1)

print(f"Searching {len(rf_params['n_estimators']) * len(rf_params['max_depth']) * len(rf_params['min_samples_split']) * len(rf_params['min_samples_leaf'])} combinations...")
start = time.time()
rf_grid.fit(X_train_scaled, y_train)
print(f"✓ Tuning completed in {time.time()-start:.1f}s\n")

rf_best = rf_grid.best_estimator_
rf_train_acc = rf_grid.best_score_
rf_test_acc = rf_best.score(X_test_scaled, y_test)

print(f"📊 RANDOM FOREST RESULTS:")
print(f"  Best params: {rf_grid.best_params_}")
print(f"  CV Accuracy: {rf_train_acc*100:.2f}%")
print(f"  Test Accuracy: {rf_test_acc*100:.2f}%")
if rf_test_acc >= 0.84:
    print(f"  ✅ REACHED 84% TARGET!")
else:
    print(f"  ⏳ Gap to reach 84%: {(0.84 - rf_test_acc)*100:.1f}%")

# ═══════════════════════════════════════════════════════════════════════════
# 3. GRADIENT BOOSTING TUNING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("3️⃣  TUNING GRADIENT BOOSTING - Finding best n_estimators, learning_rate, max_depth")
print("="*80 + "\n")

gb_params = {
    'n_estimators': [200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [5, 7, 10],
    'min_samples_split': [5, 10, 15]
}

gb_base = GradientBoostingClassifier(subsample=0.8, random_state=42)
gb_grid = GridSearchCV(gb_base, gb_params, cv=5, scoring='accuracy',
                       n_jobs=-1, verbose=1)

print(f"Searching {len(gb_params['n_estimators']) * len(gb_params['learning_rate']) * len(gb_params['max_depth']) * len(gb_params['min_samples_split'])} combinations...")
start = time.time()
gb_grid.fit(X_train_scaled, y_train)
print(f"✓ Tuning completed in {time.time()-start:.1f}s\n")

gb_best = gb_grid.best_estimator_
gb_train_acc = gb_grid.best_score_
gb_test_acc = gb_best.score(X_test_scaled, y_test)

print(f"📊 GRADIENT BOOSTING RESULTS:")
print(f"  Best params: {gb_grid.best_params_}")
print(f"  CV Accuracy: {gb_train_acc*100:.2f}%")
print(f"  Test Accuracy: {gb_test_acc*100:.2f}%")
if gb_test_acc >= 0.84:
    print(f"  ✅ REACHED 84% TARGET!")
else:
    print(f"  ⏳ Gap to reach 84%: {(0.84 - gb_test_acc)*100:.1f}%")

# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n\n" + "="*80)
print("🎯 FINAL SUMMARY - ALL MODELS TUNED")
print("="*80)

results = {
    'SVM': {'test': svm_test_acc, 'params': svm_grid.best_params_},
    'Random Forest': {'test': rf_test_acc, 'params': rf_grid.best_params_},
    'Gradient Boosting': {'test': gb_test_acc, 'params': gb_grid.best_params_}
}

for name, result in results.items():
    status = "✅ 84%+" if result['test'] >= 0.84 else f"⏳ {result['test']*100:.1f}%"
    print(f"{name:20s}: {result['test']*100:6.2f}%  {status}")
    print(f"  Params: {result['params']}\n")

# Average
avg_acc = (svm_test_acc + rf_test_acc + gb_test_acc) / 3
print(f"Average accuracy: {avg_acc*100:.2f}%")

# Save best models
print("\n" + "="*80)
print("💾 SAVING BEST TUNED MODELS")
print("="*80)

import pickle
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
os.makedirs(model_dir, exist_ok=True)

# Save SVM
svm_path = os.path.join(model_dir, f"svm_tuned_{timestamp}.pkl")
with open(svm_path, 'wb') as f:
    pickle.dump({'model': svm_best, 'scaler': scaler, 'params': svm_grid.best_params_}, f)
print(f"✓ SVM saved: {svm_path}")

# Save RF
rf_path = os.path.join(model_dir, f"randomforest_tuned_{timestamp}.pkl")
with open(rf_path, 'wb') as f:
    pickle.dump({'model': rf_best, 'scaler': scaler, 'params': rf_grid.best_params_}, f)
print(f"✓ Random Forest saved: {rf_path}")

# Save GB
gb_path = os.path.join(model_dir, f"gradientboosting_tuned_{timestamp}.pkl")
with open(gb_path, 'wb') as f:
    pickle.dump({'model': gb_best, 'scaler': scaler, 'params': gb_grid.best_params_}, f)
print(f"✓ Gradient Boosting saved: {gb_path}")

print("\n" + "="*80 + "\n")
