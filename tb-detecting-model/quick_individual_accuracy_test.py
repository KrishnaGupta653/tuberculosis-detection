"""
Quick test to show current individual model accuracies
and provide tuning recommendations to reach 84%
"""

import numpy as np
import os
import sys
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_shared import Config, load_image_paths, extract_features_parallel

print("\n" + "="*80)
print("INDIVIDUAL MODEL ACCURACY TEST")
print("="*80 + "\n")

# Load data
print("📂 Loading image paths...")
X_paths, y = load_image_paths(Config.DATA_DIR)

X_paths_train, X_paths_test, y_train, y_test = train_test_split(
    X_paths, y, test_size=Config.TEST_SIZE, 
    random_state=Config.RANDOM_SEED, stratify=y
)

# Extract features
print("🔬 Extracting features...")
X_train, y_train = extract_features_parallel(X_paths_train, y_train, "Training")
X_test, y_test = extract_features_parallel(X_paths_test, y_test, "Test")

# Normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n✓ Dataset: {len(X_train)} train, {len(X_test)} test")
print(f"✓ Features: {X_train.shape[1]} dimensions\n")

# ═══════════════════════════════════════════════════════════════════════════
# TEST 1: SVM
# ═══════════════════════════════════════════════════════════════════════════
print("1️⃣  SVM MODEL")
print("-" * 80)

svm = SVC(kernel='rbf', C=10.0, gamma='scale', class_weight='balanced', probability=True, random_state=42)
svm.fit(X_train_scaled, y_train)

svm_pred = svm.predict(X_test_scaled)
svm_acc = accuracy_score(y_test, svm_pred)
svm_sens = recall_score(y_test, svm_pred, pos_label=1)
svm_spec = recall_score(y_test, svm_pred, pos_label=0)

print(f"Current SVM Accuracy: {svm_acc*100:.2f}%")
print(f"  Sensitivity: {svm_sens*100:.2f}%  |  Specificity: {svm_spec*100:.2f}%")
print(f"\n🎯 TO REACH 84% - Increase SVM accuracy by {max(0, 84-svm_acc*100):.1f}%:")
print(f"  1. Increase C value: Try C=[100, 500, 1000] (stronger regularization)")
print(f"  2. Try different kernels: 'poly' (degree=3-5) or 'sigmoid'")
print(f"  3. Tune gamma: Try [0.0001, 0.001, 0.01] (lower = more influence)")
print(f"  4. Add feature scaling: Already done ✓")
print(f"  5. GridSearchCV params: C=[10,100,1000], gamma=[1e-4,1e-3,1e-2,0.1], kernel=['rbf','poly']")

# ═══════════════════════════════════════════════════════════════════════════
# TEST 2: Random Forest
# ═══════════════════════════════════════════════════════════════════════════
print("\n\n2️⃣  RANDOM FOREST MODEL")
print("-" * 80)

rf = RandomForestClassifier(
    n_estimators=200, max_depth=15, min_samples_split=5,
    class_weight='balanced_subsample', n_jobs=-1, random_state=42
)
rf.fit(X_train_scaled, y_train)

rf_pred = rf.predict(X_test_scaled)
rf_acc = accuracy_score(y_test, rf_pred)
rf_sens = recall_score(y_test, rf_pred, pos_label=1)
rf_spec = recall_score(y_test, rf_pred, pos_label=0)

print(f"Current RF Accuracy: {rf_acc*100:.2f}%")
print(f"  Sensitivity: {rf_sens*100:.2f}%  |  Specificity: {rf_spec*100:.2f}%")
print(f"\n🎯 TO REACH 84% - Increase RF accuracy by {max(0, 84-rf_acc*100):.1f}%:")
print(f"  1. Increase n_estimators: Try [300, 500, 1000] (more trees = better)")
print(f"  2. Reduce max_depth: Try [8, 10, 12] (prevent overfitting)")
print(f"  3. Increase min_samples_split: Try [8, 10, 15] (smoother splits)")
print(f"  4. Tune min_samples_leaf: Try [4, 5, 8] (minimum samples per leaf)")
print(f"  5. GridSearchCV params: n_estimators=[200,500,1000], max_depth=[8,10,12], min_samples_split=[5,10,15]")

# ═══════════════════════════════════════════════════════════════════════════
# TEST 3: Gradient Boosting
# ═══════════════════════════════════════════════════════════════════════════
print("\n\n3️⃣  GRADIENT BOOSTING MODEL")
print("-" * 80)

gb = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=3,
    min_samples_split=5, subsample=0.8, random_state=42
)
gb.fit(X_train_scaled, y_train)

gb_pred = gb.predict(X_test_scaled)
gb_acc = accuracy_score(y_test, gb_pred)
gb_sens = recall_score(y_test, gb_pred, pos_label=1)
gb_spec = recall_score(y_test, gb_pred, pos_label=0)

print(f"Current GB Accuracy: {gb_acc*100:.2f}%")
print(f"  Sensitivity: {gb_sens*100:.2f}%  |  Specificity: {gb_spec*100:.2f}%")
print(f"\n🎯 TO REACH 84% - Increase GB accuracy by {max(0, 84-gb_acc*100):.1f}%:")
print(f"  1. Increase n_estimators: Try [200, 300, 500] (more boosting rounds)")
print(f"  2. Lower learning_rate: Try [0.01, 0.05] (slower learning = better)")
print(f"  3. Increase max_depth: Try [5, 7, 10] (deeper trees capture patterns)")
print(f"  4. Reduce subsample: Try [0.6, 0.7, 0.8] (stochastic sampling)")
print(f"  5. GridSearchCV params: n_estimators=[100,200,300], learning_rate=[0.01,0.05,0.1], max_depth=[5,7,10]")

# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n\n" + "="*80)
print("SUMMARY - CURRENT ACCURACIES")
print("="*80)
print(f"SVM:              {svm_acc*100:6.2f}%  (target: 84% | gap: {84-svm_acc*100:+.1f}%)")
print(f"Random Forest:    {rf_acc*100:6.2f}%  (target: 84% | gap: {84-rf_acc*100:+.1f}%)")
print(f"Gradient Boosting:{gb_acc*100:6.2f}%  (target: 84% | gap: {84-gb_acc*100:+.1f}%)")
print(f"Ensemble Average: {(svm_acc+rf_acc+gb_acc)*100/3:6.2f}%")

print("\n" + "="*80)
print("QUICK TUNING STRATEGY (Try these in order)")
print("="*80)
print("""
FOR EACH MODEL, do GridSearchCV with these parameter grids:

SVM TUNING:
  svm_params = {
      'C': [10, 100, 500, 1000],
      'gamma': [0.0001, 0.001, 0.01, 0.1],
      'kernel': ['rbf', 'poly']
  }

RANDOM FOREST TUNING:
  rf_params = {
      'n_estimators': [300, 500, 800],
      'max_depth': [8, 10, 12],
      'min_samples_split': [5, 10, 15]
  }

GRADIENT BOOSTING TUNING:
  gb_params = {
      'n_estimators': [200, 300, 500],
      'learning_rate': [0.01, 0.05, 0.1],
      'max_depth': [5, 7, 10]
  }

Run: python tune_individual_models.py (next file)
""")

print("="*80 + "\n")
