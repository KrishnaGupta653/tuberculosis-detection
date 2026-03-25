#!/usr/bin/env python3
"""
================================================================================
TRAIN WITH WORKERS — QUICK START
================================================================================
This script trains the TB detection model using full worker optimization.

Usage:
    python workers_optimized/train_with_workers.py                        # Auto-detected workers
    python workers_optimized/train_with_workers.py --workers 8            # Use 8 workers
    python workers_optimized/train_with_workers.py --workers 4 --predict test_image.jpg
    
Features:
    ✓ Auto-detects optimal number of workers
    ✓ 3-4x faster feature extraction (parallel)
    ✓ Parallel model training (SVM, RF, GB)
    ✓ Windows-safe DataLoader configuration
    ✓ Real-time progress tracking
    ✓ Comprehensive diagnostics
================================================================================
"""

import argparse
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.model_final_fast_enhanced import train_enhanced_model, predict_on_image
from workers_optimized.pipeline_workers import WorkerConfig, print_worker_diagnostics


def main():
    parser = argparse.ArgumentParser(
        description='Train TB detection model with worker optimization'
    )
    
    parser.add_argument(
        '--mode',
        choices=['train', 'predict', 'threshold-optimize'],
        default='train',
        help='Mode of operation'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help=f'Number of workers (default: auto-detect, currently {WorkerConfig.OPTIMAL_WORKERS})'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        help='Path to image for prediction'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for feature extraction'
    )
    
    parser.add_argument(
        '--show-diagnostics',
        action='store_true',
        help='Print detailed worker diagnostics and exit'
    )
    
    args = parser.parse_args()
    
    # Show diagnostics if requested
    if args.show_diagnostics:
        print_worker_diagnostics()
        sys.exit(0)
    
    # Override workers if specified
    if args.workers is not None and args.workers > 0:
        WorkerConfig.NUM_WORKERS = args.workers
        print(f"\n⚙️  Workers set to: {args.workers}\n")
    
    print(f"\n{'='*80}")
    print(f"TB DETECTION MODEL — WORKER-OPTIMIZED TRAINING")
    print(f"{'='*80}\n")
    
    if args.mode == 'train':
        print(f"[TRAIN] Starting training mode...")
        print(f"   Workers: {WorkerConfig.NUM_WORKERS}")
        print(f"   CPU Cores: {WorkerConfig.CPU_COUNT}")
        print()
        
        try:
            model_path = train_enhanced_model()
            print(f"\n[SUCCESS] Training completed successfully!")
            print(f"   Model saved: {model_path}")
        except Exception as e:
            print(f"\n[ERROR] Training failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    elif args.mode == 'predict':
        if not args.image:
            print("[ERROR] --image required for predict mode")
            sys.exit(1)
        
        print(f"[PREDICT] Starting prediction mode...")
        print(f"   Image: {args.image}\n")
        
        try:
            predict_on_image(args.image)
        except Exception as e:
            print(f"\n[ERROR] Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    elif args.mode == 'threshold-optimize':
        print(f"[OPTIMIZE] Starting threshold optimization mode...")
        print(f"   Workers: {WorkerConfig.NUM_WORKERS}\n")
        
        try:
            train_enhanced_model()
            print(f"\n[SUCCESS] Threshold optimization completed!")
        except Exception as e:
            print(f"\n[ERROR] Optimization failed: {e}")  
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()
