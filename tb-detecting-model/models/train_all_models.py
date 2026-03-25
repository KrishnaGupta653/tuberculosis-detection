"""
================================================================================
MASTER TRAINING ORCHESTRATOR
================================================================================
Runs all individual model trainers sequentially or in parallel
Usage: python train_all_models.py [--sequential] | python train_all_models.py --parallel
"""

import subprocess
import sys
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

TRAINERS = [
    ('train_svm_model.py', 'SVM'),
    ('train_randomforest_model.py', 'Random Forest'),
    ('train_gradientboosting_model.py', 'Gradient Boosting')
]

def run_trainer_sequential():
    """Run all trainers one after another"""
    
    print(f"\n{'='*80}")
    print("[TRAIN] TRAINING ALL MODELS — SEQUENTIAL MODE")
    print(f"{'='*80}\n")
    
    results = {}
    start_time = time.time()
    
    for trainer_file, model_name in TRAINERS:
        print(f"\n{'='*80}")
        print(f"[TIME] Training {model_name}...")
        print(f"{'='*80}\n")
        
        model_start = time.time()
        try:
            result = subprocess.run(
                [sys.executable, trainer_file],
                cwd='.',
                check=True,
                capture_output=False
            )
            model_time = time.time() - model_start
            results[model_name] = {
                'status': 'SUCCESS [OK]',
                'time_seconds': model_time
            }
            print(f"\n[OK] {model_name} completed in {model_time/60:.1f} minutes")
        except subprocess.CalledProcessError as e:
            model_time = time.time() - model_start
            results[model_name] = {
                'status': 'FAILED [FAIL]',
                'time_seconds': model_time,
                'error': str(e)
            }
            print(f"\n[FAIL] {model_name} failed after {model_time/60:.1f} minutes")
    
    total_time = time.time() - start_time
    
    # Summary
    print(f"\n{'='*80}")
    print("[SUMMARY] TRAINING SUMMARY")
    print(f"{'='*80}\n")
    
    for model_name, result in results.items():
        status = result['status']
        time_min = result['time_seconds'] / 60
        print(f"  {status:30s} {model_name:25s} ({time_min:6.1f} min)")
    
    print(f"\n{'='*80}")
    print(f"Total Training Time: {total_time/60/60:.1f} hours ({total_time/60:.0f} minutes)")
    print(f"{'='*80}\n")
    
    return all(r['status'].startswith('SUCCESS') for r in results.values())

def run_trainer_parallel():
    """Run all trainers in parallel using thread pool"""
    
    print(f"\n{'='*80}")
    print("[TRAIN] TRAINING ALL MODELS — PARALLEL MODE")
    print(f"[INFO] Starting {len(TRAINERS)} trainers concurrently...")
    print(f"{'='*80}\n")
    
    results = {}
    start_time = time.time()
    
    def train_single_model(trainer_file, model_name):
        """Helper to run a single trainer and capture result"""
        print(f"\n[TIME] Starting {model_name} training...")
        model_start = time.time()
        try:
            result = subprocess.run(
                [sys.executable, trainer_file],
                cwd='.',
                check=True,
                capture_output=False
            )
            model_time = time.time() - model_start
            print(f"\n[OK] {model_name} completed in {model_time/60:.1f} minutes")
            return (model_name, 'SUCCESS [OK]', model_time, None)
        except subprocess.CalledProcessError as e:
            model_time = time.time() - model_start
            print(f"\n[FAIL] {model_name} failed after {model_time/60:.1f} minutes")
            return (model_name, 'FAILED [FAIL]', model_time, str(e))
    
    # Launch all trainers concurrently
    with ThreadPoolExecutor(max_workers=len(TRAINERS)) as executor:
        futures = {
            executor.submit(train_single_model, trainer_file, model_name): model_name
            for trainer_file, model_name in TRAINERS
        }
        
        # Collect results as they complete
        for future in as_completed(futures):
            model_name, status, model_time, error = future.result()
            results[model_name] = {
                'status': status,
                'time_seconds': model_time,
                'error': error
            }
    
    total_time = time.time() - start_time
    
    # Summary
    print(f"\n{'='*80}")
    print("[SUMMARY] TRAINING SUMMARY (PARALLEL MODE)")
    print(f"{'='*80}\n")
    
    for model_name, result in results.items():
        status = result['status']
        time_min = result['time_seconds'] / 60
        print(f"  {status:30s} {model_name:25s} ({time_min:6.1f} min)")
    
    print(f"\n{'='*80}")
    print(f"Total Wall-Clock Time: {total_time/60:.0f} minutes ({total_time/60/60:.1f} hours)")
    print(f"{'='*80}\n")
    
    return all(r['status'].startswith('SUCCESS') for r in results.values())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequential', action='store_true', default=True,
                       help='Run trainers sequentially (default)')
    parser.add_argument('--parallel', action='store_true',
                       help='Run trainers in parallel')
    args = parser.parse_args()
    
    if args.parallel:
        success = run_trainer_parallel()
    else:
        success = run_trainer_sequential()
    
    if success:
        print(f"[SUCCESS] ALL MODELS TRAINED SUCCESSFULLY!")
        print(f"\n[INFO] Models saved in: ./saved_models/")
        print(f"[INFO] Next: Test individual models with test_*_accuracy.py scripts")
        sys.exit(0)
    else:
        print(f"[FAILED] SOME MODELS FAILED TO TRAIN")
        sys.exit(1)
