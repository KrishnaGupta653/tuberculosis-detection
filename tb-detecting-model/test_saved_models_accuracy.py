"""
Load and Test Your Already-Trained Individual Models
Shows actual accuracy on test set
"""

import pickle
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, recall_score, precision_score, 
                            f1_score, confusion_matrix, classification_report)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_shared import Config, load_image_paths, extract_features_parallel

print("\n" + "="*80)
print("📊 TESTING YOUR SAVED INDIVIDUAL MODELS")
print("="*80 + "\n")

# Load and prepare test data
print("📂 Loading images...")
X_paths, y = load_image_paths(Config.DATA_DIR)

X_paths_train, X_paths_test, y_train, y_test = train_test_split(
    X_paths, y, test_size=Config.TEST_SIZE, 
    random_state=Config.RANDOM_SEED, stratify=y
)

print("🔬 Extracting features from test images...")
X_test, y_test = extract_features_parallel(X_paths_test, y_test, "Test features")

print(f"✓ Test set: {len(X_test)} images, {X_test.shape[1]} features\n")

# Model directory
model_dir = os.path.join(os.path.dirname(__file__), 'saved_models')

# Load saved models
models_to_test = {
    'SVM': 'svm_model_20260326_040306.pkl',
    'Random Forest': 'randomforest_model_20260326_030855.pkl',
    'Gradient Boosting': 'gradientboosting_model_20260326_030907.pkl'
}

results = {}

for model_name, filename in models_to_test.items():
    filepath = os.path.join(model_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"❌ {model_name}: File not found at {filepath}")
        continue
    
    print(f"\n{'='*80}")
    print(f"🤖 TESTING: {model_name.upper()}")
    print(f"{'='*80}\n")
    
    try:
        # Load model
        with open(filepath, 'rb') as f:
            model_package = pickle.load(f)
        
        model = model_package['model']
        scaler = model_package['scaler']
        
        print(f"✓ Model loaded from: {filename}\n")
        
        # Scale test data
        X_test_scaled = scaler.transform(X_test)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        sensitivity = recall_score(y_test, y_pred, pos_label=1)  # TB detection rate
        specificity = recall_score(y_test, y_pred, pos_label=0)  # Normal detection rate
        precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, y_pred)
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Print results
        print(f"📊 METRICS:")
        print(f"  ┌─ Accuracy:    {accuracy*100:6.2f}%  (overall correctness)")
        print(f"  ├─ Sensitivity: {sensitivity*100:6.2f}%  (TB detection rate - IMPORTANT!)")
        print(f"  ├─ Specificity: {specificity*100:6.2f}%  (Normal detection rate)")
        print(f"  ├─ Precision:   {precision*100:6.2f}%  (when it says TB, is it right?)")
        print(f"  └─ F1-Score:    {f1:.4f}    (balance of precision & recall)")
        
        print(f"\n📈 CONFUSION MATRIX:")
        print(f"  ┌─ True Negatives:  {tn:4d}  (correctly identified Normal)")
        print(f"  ├─ False Positives:{fp:4d}  (wrongly called TB)")
        print(f"  ├─ False Negatives:{fn:4d}  (wrongly called Normal)")
        print(f"  └─ True Positives: {tp:4d}  (correctly identified TB)")
        
        print(f"\n🎯 GOAL STATUS:")
        if accuracy >= 0.84:
            print(f"  ✅ REACHED 84% TARGET! ({accuracy*100:.1f}%)")
        else:
            gap = (0.84 - accuracy) * 100
            print(f"  ⏳ Gap to 84%: {gap:.1f}% (current: {accuracy*100:.1f}%)")
        
        results[model_name] = {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1': f1
        }
        
    except Exception as e:
        print(f"❌ Error testing {model_name}: {str(e)}")

# Summary
print(f"\n\n{'='*80}")
print("🎯 SUMMARY - ALL MODELS")
print(f"{'='*80}\n")

if results:
    print(f"{'Model':<20} {'Accuracy':<12} {'Sensitivity':<15} {'Status':<20}")
    print("-" * 70)
    
    for model_name, metrics in results.items():
        acc = metrics['accuracy'] * 100
        sens = metrics['sensitivity'] * 100
        status = "✅ 84%+" if acc >= 84 else f"⏳ {84-acc:.1f}% to go"
        print(f"{model_name:<20} {acc:6.2f}%     {sens:6.2f}%         {status:<20}")
    
    avg_acc = np.mean([m['accuracy'] for m in results.values()]) * 100
    print("-" * 70)
    print(f"{'Ensemble Average':<20} {avg_acc:6.2f}%")
    
    print(f"\n{'='*80}")
    print("📝 INTERPRETATION:")
    print(f"{'='*80}")
    print(f"""
✓ ACCURACY: How many predictions are correct overall
  - Your goal: 84%+ for individual models

✓ SENSITIVITY: Can the model catch TB cases?
  - Critical for medical: You want high sensitivity (minimize false negatives)
  - Missing a TB case is dangerous!

✓ SPECIFICITY: Can the model identify Normal cases correctly?
  - High = few false alarms

✓ PRECISION: When model says "TB", is it actually TB?
  - High precision = fewer false positives

✓ F1-SCORE: Balance between precision & recall
  - Best for medical use when classes are imbalanced

🎯 FOR MEDICAL IMAGING:
  Sensitivity > Specificity (better to have false alarm than miss TB)
    """)

print(f"{'='*80}\n")
