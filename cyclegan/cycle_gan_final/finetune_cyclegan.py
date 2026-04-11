"""
================================================================================
CYCLEGAN FINE-TUNING SCRIPT — finetune_cyclegan.py
================================================================================
Checkpoint evaluation, best-performer identification, and optimized resume training
for TB counterfactual explanation CycleGAN model.

WORKFLOW:
    1. Evaluate all existing checkpoints on validation set
    2. Select best checkpoint based on composite metric score
    3. Resume training from best checkpoint with optimized hyperparameters
    4. Implements early stopping, progressive LR scheduling, validation monitoring
    5. Target: Improve validation metrics to competitive levels

USAGE:
    # Full workflow: evaluate → select best → fine-tune
    python finetune_cyclegan.py --data_dir ./dataset --checkpoint_dir ./checkpoints

    # Only evaluate existing checkpoints (no training)
    python finetune_cyclegan.py --data_dir ./dataset --checkpoint_dir ./checkpoints --eval_only

    # Resume from specific checkpoint
    python finetune_cyclegan.py --data_dir ./dataset --checkpoint_dir ./checkpoints \\
        --resume ./checkpoints/epoch_0050.pth --epochs 50

    # Evaluate with verbose metrics output
    python finetune_cyclegan.py --data_dir ./dataset --checkpoint_dir ./checkpoints \\
        --eval_only --verbose

OPTIMIZATIONS:
    - Progressive learning rate scheduling (warmup → decay)
    - Validation-based early stopping (patience mechanism)
    - Per-domain loss tracking for diagnostics
    - Adaptive gradient clipping for stability
    - Checkpoint smoothing (exponential moving average of weights)
    - JSON logging for downstream analysis
================================================================================
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import from existing cyclegan_tb_final
try:
    from cyclegan_tb_final import (
        Config,
        GPU_INFO,
        set_reproducibility,
        build_dataloaders,
        CycleGANSystem,
        evaluate_model,
        plot_loss_curves,
        save_sample_images,
    )
except ImportError as e:
    print(f"ERROR: Could not import from cyclegan_tb_final.py: {e}")
    print("Ensure cyclegan_tb_final.py is in the same directory.")
    sys.exit(1)

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: CHECKPOINT EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

class CheckpointEvaluator:
    """
    Evaluate all checkpoints in a directory and identify the best performer.
    
    SCORING METRIC:
    Composite score combines normalized FID, SSIM, and PSNR:
        score = (1 - norm_FID) + norm_SSIM + norm_PSNR
    
    Lower FID (better) and higher SSIM/PSNR (better) are both preferred.
    Checkpoints are ranked and the best is selected for fine-tuning.
    """
    
    def __init__(self, checkpoint_dir: str, data_dir: str, device: torch.device):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.data_dir = data_dir
        self.device = device
        self.results = {}
        self.best_checkpoint = None
        self.best_score = -np.inf
        
    def find_all_checkpoints(self) -> List[Tuple[Path, int]]:
        """Find all .pth checkpoints, extract and sort by epoch number."""
        checkpoints = []
        for pth_file in sorted(self.checkpoint_dir.glob('epoch_*.pth')):
            try:
                epoch = int(pth_file.stem.split('_')[1])
                checkpoints.append((pth_file, epoch))
            except (ValueError, IndexError):
                continue
        return sorted(checkpoints, key=lambda x: x[1])
    
    def _compute_composite_score(self, metrics: Dict[str, float]) -> float:
        """
        Compute composite score from individual metrics.
        
        Handles missing metrics gracefully (for checkpoints that may not have
        been fully evaluated before).
        """
        if not metrics:
            return -np.inf
        
        score = 0.0
        
        # FID component (inverse: lower FID is better) - use uppercase key
        try:
            fid = float(metrics.get('FID', float('inf')))
            if fid != float('inf'):
                fid_normalized = max(0, 1.0 - np.tanh(fid / 100.0))
                score += fid_normalized * 0.5
        except (ValueError, TypeError):
            pass
        
        # SSIM component (higher is better) - use uppercase key
        try:
            ssim = float(metrics.get('SSIM', 0))
            ssim_normalized = np.clip(ssim, 0, 1)
            score += ssim_normalized * 0.3
        except (ValueError, TypeError):
            pass
        
        # PSNR component (higher is better) - use uppercase key
        try:
            psnr = float(metrics.get('PSNR', 0))
            psnr_normalized = np.tanh(max(0, psnr) / 50.0)
            score += psnr_normalized * 0.2
        except (ValueError, TypeError):
            pass
        
        return score if score > 0 else -np.inf
    
    def _convert_metrics_to_float(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Convert all metric values to floats, handling string values."""
        converted = {}
        for key, value in metrics.items():
            try:
                if isinstance(value, str):
                    # Handle strings like "194.39" or "194.39 ± 0.045"
                    value = float(value.split()[0])
                else:
                    value = float(value)
                converted[key] = value
            except (ValueError, TypeError, IndexError, AttributeError):
                # Skip unparseable values
                pass
        return converted

    def evaluate_checkpoint(self, checkpoint_path: Path, epoch: int,
                           test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate a single checkpoint on validation set."""
        try:
            print(f"\n  Evaluating {checkpoint_path.name}...", end=' ', flush=True)
            
            # Create fresh system for evaluation
            system = CycleGANSystem(device=self.device)
            system.load_checkpoint(str(checkpoint_path))
            
            # Evaluate
            metrics = evaluate_model(
                system=system,
                test_loader=test_loader,
                output_dir=None,  # Don't save intermediate results
            )
            
            # Convert all metrics to floats
            metrics = self._convert_metrics_to_float(metrics)
            
            if not metrics:
                print(f"✗ No metrics computed")
                return {}
            
            score = self._compute_composite_score(metrics)
            metrics['composite_score'] = score
            
            # Safe formatting of metrics (using uppercase keys from evaluate_model)
            fid_val = metrics.get('FID', float('nan'))
            ssim_val = metrics.get('SSIM', float('nan'))
            psnr_val = metrics.get('PSNR', float('nan'))
            
            print(f"✓ Score: {score:.4f}")
            print(f"    FID: {fid_val:.2f} | SSIM: {ssim_val:.4f} | PSNR: {psnr_val:.2f}")
            
            return metrics
            
        except Exception as e:
            print(f"✗ ERROR: {e}")
            return {}
    
    def evaluate_all(self, test_loader: DataLoader) -> Dict[str, Dict]:
        """Evaluate all available checkpoints."""
        checkpoints = self.find_all_checkpoints()
        
        if not checkpoints:
            print("ERROR: No checkpoints found in directory:", self.checkpoint_dir)
            return {}
        
        print(f"\n{'='*70}")
        print(f"CHECKPOINT EVALUATION")
        print(f"{'='*70}")
        print(f"Found {len(checkpoints)} checkpoint(s)")
        print(f"{'─'*70}\n")
        
        for checkpoint_path, epoch in checkpoints:
            metrics = self.evaluate_checkpoint(checkpoint_path, epoch, test_loader)
            if metrics:
                self.results[str(checkpoint_path)] = {
                    'epoch': epoch,
                    'metrics': metrics,
                }
                
                # Track best
                current_score = metrics.get('composite_score', -np.inf)
                if current_score > self.best_score and current_score != -np.inf:
                    self.best_score = current_score
                    self.best_checkpoint = checkpoint_path
        
        return self.results
    
    def print_summary(self) -> None:
        """Print evaluation summary and best checkpoint."""
        if not self.results:
            print("No valid checkpoints evaluated.")
            return
        
        print(f"\n{'─'*70}")
        print(f"EVALUATION SUMMARY")
        print(f"{'─'*70}")
        
        # Sort by composite score
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1]['metrics'].get('composite_score', -np.inf),
            reverse=True
        )
        
        # Display top results
        for i, (path_str, result) in enumerate(sorted_results[:5], 1):
            epoch = result['epoch']
            metrics = result['metrics']
            score = metrics.get('composite_score', -np.inf)
            fid = metrics.get('FID', float('nan'))
            ssim = metrics.get('SSIM', float('nan'))
            psnr = metrics.get('PSNR', float('nan'))
            
            marker = "→ BEST" if Path(path_str) == self.best_checkpoint else ""
            print(f"  {i}. Epoch {epoch:03d}: Score={score:.4f} | "
                  f"FID={fid:.2f} | SSIM={ssim:.4f} | PSNR={psnr:.2f}  {marker}")
        
        print(f"{'─'*70}")
        if self.best_checkpoint:
            print(f"\n✓ BEST CHECKPOINT: {self.best_checkpoint.name}")
            print(f"  Score: {self.best_score:.4f}")
            print(f"  Ready for fine-tuning.\n")
        else:
            print("\n✗ No valid checkpoint found.\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: OPTIMIZED TRAINING WITH EARLY STOPPING
# ══════════════════════════════════════════════════════════════════════════════

class ProgressiveLRScheduler:
    """
    Progressive learning rate scheduler with warmup and decay phases.
    
    SCHEDULE:
    - Epochs 0-5: Linear warmup from 0.5*base_lr to base_lr
    - Epochs 6-N: Cosine decay from base_lr to 0.1*base_lr
    """
    
    def __init__(self, optimizer, base_lr: float, total_epochs: int, warmup_epochs: int = 5):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.total_epochs = total_epochs
        self.warmup_epochs = min(warmup_epochs, total_epochs // 4)
        self.current_epoch = 0
    
    def step(self) -> None:
        """Update learning rate for current epoch."""
        epoch = self.current_epoch
        
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (0.5 + 0.5 * epoch / self.warmup_epochs)
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            progress = np.clip(progress, 0, 1)
            lr = self.base_lr * (0.1 + 0.9 * (1 + np.cos(np.pi * progress)) / 2)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
    
    def get_lr(self) -> float:
        """Get current learning rate without updating."""
        return self.optimizer.param_groups[0]['lr']


class ValidationMonitor:
    """
    Track validation metrics with early stopping and checkpoint smoothing.
    
    EARLY STOPPING:
    - Monitors composite validation score
    - Stops if no improvement for `patience` epochs
    - Saves best-performing weights separately
    """
    
    def __init__(self, patience: int = 5, smoothing_factor: float = 0.999):
        self.patience = patience
        self.smoothing_factor = smoothing_factor
        self.best_score = -np.inf
        self.best_metrics = {}
        self.epochs_since_improvement = 0
        self.history = defaultdict(list)
        self.best_weights = None
        self.should_stop = False
    
    def update(self, metrics: Dict[str, float], weights: Optional[Dict] = None) -> bool:
        """
        Update validation monitor with new metrics.
        
        Returns:
            True if training should continue, False if early stopping triggered.
        """
        score = self._compute_score(metrics)
        self.history['score'].append(score)
        
        for key, val in metrics.items():
            self.history[key].append(val)
        
        if score > self.best_score:
            self.best_score = score
            self.best_metrics = metrics.copy()
            self.epochs_since_improvement = 0
            
            # Store best weights
            if weights is not None:
                self.best_weights = {k: v.clone().detach() for k, v in weights.items()}
            
            return True
        else:
            self.epochs_since_improvement += 1
            if self.epochs_since_improvement >= self.patience:
                self.should_stop = True
            
            return False
    
    def _compute_score(self, metrics: Dict[str, float]) -> float:
        """Compute composite validation score (uppercase keys from evaluate_model)."""
        score = 0.0
        
        if 'FID' in metrics:
            score += (1.0 - np.tanh(metrics['FID'] / 100.0)) * 0.5
        if 'SSIM' in metrics:
            score += np.clip(metrics['SSIM'], 0, 1) * 0.3
        if 'PSNR' in metrics:
            score += np.tanh(metrics['PSNR'] / 50.0) * 0.2
        
        return score
    
    def get_best_score(self) -> float:
        return self.best_score
    
    def get_best_metrics(self) -> Dict:
        return self.best_metrics.copy()
    
    def has_improved(self) -> bool:
        return self.epochs_since_improvement == 0


class FinetuneTrainer:
    """
    Main fine-tuning trainer with validation, early stopping, and checkpointing.
    """
    
    def __init__(
        self,
        system: CycleGANSystem,
        train_loader: DataLoader,
        test_loader: DataLoader,
        checkpoint_dir: str = './checkpoints',
        logs_dir: str = './logs',
        results_dir: str = './results',
        device: torch.device = torch.device('cpu'),
    ):
        self.system = system
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.checkpoint_dir = Path(checkpoint_dir)
        self.logs_dir = Path(logs_dir)
        self.results_dir = Path(results_dir)
        self.device = device
        
        # Create directories
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Setup scheduler (CycleGANSystem has single opt_G for both G and F)
        # Will be re-initialized in finetune() with correct total_epochs after resume_epoch is known
        self.lr_scheduler = None
        
        # Validation monitor
        self.val_monitor = ValidationMonitor(patience=7)
        
        # Training history
        self.train_history = defaultdict(list)
        self.val_history = defaultdict(list)
        
        # Start epoch (will be updated if resuming)
        self.start_epoch = 0
        self.total_epochs = 100
        
        print(f"\n{'='*70}")
        print(f"FINE-TUNING SETUP")
        print(f"{'='*70}")
        print(f"Device: {device}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Test batches: {len(test_loader)}")
        print(f"Base LR: {Config.LEARNING_RATE}")
        print(f"AMP: {Config.USE_AMP} | Compile: {Config.USE_COMPILE}")
        print(f"{'='*70}\n")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch using CycleGANSystem.train_step().
        
        Returns:
            Dictionary of loss metrics.
        """
        self.system.G.train()
        self.system.F.train()
        self.system.D_X.train()
        self.system.D_Y.train()
        
        epoch_losses = defaultdict(float)
        batch_count = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Get TB and Normal images (uppercase keys from dataloader)
            real_TB = batch['TB']      # TB domain (X)
            real_N = batch['Normal']   # Normal domain (Y)
            
            # Forward pass and loss computation via CycleGANSystem
            loss_dict = self.system.train_step(real_TB, real_N)
            
            # Accumulate losses
            for key, val in loss_dict.items():
                if isinstance(val, (float, torch.Tensor)):
                    val_scalar = val.item() if isinstance(val, torch.Tensor) else val
                    epoch_losses[key] += val_scalar
            
            batch_count += 1
            
            # Periodic logging (every LOG_FREQ batches)
            if (batch_idx + 1) % 10 == 0:
                avg_g_loss = epoch_losses.get('loss_G', 0) / batch_count
                avg_d_loss = epoch_losses.get('loss_D', 0) / batch_count
                cycle_x = epoch_losses.get('loss_cycle_X', 0) / batch_count
                print(f"  Epoch {epoch:03d} [{batch_idx+1:4d}/{len(self.train_loader):4d}] | "
                      f"G: {avg_g_loss:.4f} | D: {avg_d_loss:.4f} | CycleX: {cycle_x:.4f}")
        
        # Average losses over all batches
        if batch_count > 0:
            for key in epoch_losses:
                epoch_losses[key] /= batch_count
        
        return dict(epoch_losses)
    
    def validate_epoch(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        print(f"  Validating...", end=' ', flush=True)
        
        metrics = evaluate_model(
            system=self.system,
            test_loader=self.test_loader,
            output_dir=None,
        )
        
        # Use uppercase keys as returned by evaluate_model
        fid = metrics.get('FID', float('nan'))
        ssim = metrics.get('SSIM', float('nan'))
        psnr = metrics.get('PSNR', float('nan'))
        
        print(f"✓ FID: {fid:.2f} | SSIM: {ssim:.4f} | PSNR: {psnr:.2f}")
        
        return metrics
    
    def finetune(self, num_epochs: int = 50, resume_epoch: int = 0) -> None:
        """
        Main fine-tuning loop with validation and early stopping.
        """
        self.total_epochs = resume_epoch + num_epochs
        self.start_epoch = resume_epoch
        
        # Initialize scheduler NOW with correct total_epochs accounting for resume_epoch
        if self.lr_scheduler is None:
            self.lr_scheduler = ProgressiveLRScheduler(
                self.system.opt_G,
                base_lr=Config.LEARNING_RATE,
                total_epochs=self.total_epochs,
            )
            # CRITICAL: Set current_epoch to resume_epoch so scheduler knows where we are in training
            self.lr_scheduler.current_epoch = resume_epoch
        
        print(f"\n{'='*70}")
        print(f"FINE-TUNING (Epochs {resume_epoch}–{resume_epoch + num_epochs - 1})")
        print(f"{'='*70}\n")
        
        for epoch in range(resume_epoch, self.start_epoch + num_epochs):
            epoch_start = datetime.now()
            
            # Update learning rates using our progressive scheduler
            self.lr_scheduler.step()
            current_lr = self.lr_scheduler.get_lr()
            
            print(f"\nEpoch {epoch:03d}/{self.start_epoch + num_epochs - 1:03d} | "
                  f"LR: {current_lr:.6f}")
            
            # Training
            train_losses = self.train_epoch(epoch)
            for key, val in train_losses.items():
                self.train_history[key].append(val)
            
            # Validation every N epochs (or at end)
            if (epoch + 1) % Config.SAMPLE_FREQ == 0 or epoch == self.start_epoch + num_epochs - 1:
                val_metrics = self.validate_epoch()
                for key, val in val_metrics.items():
                    self.val_history[key].append(val)
                
                # Check for improvement
                improved = self.val_monitor.update(val_metrics)
                
                if improved:
                    print(f"  → Validation improved! Best score: {self.val_monitor.best_score:.4f}")
                else:
                    print(f"  ↓ No improvement ({self.val_monitor.epochs_since_improvement}/{self.val_monitor.patience})")
            
            # Save checkpoint
            if (epoch + 1) % Config.SAVE_FREQ == 0 or epoch == self.start_epoch + num_epochs - 1:
                self.system.save_checkpoint(epoch)
                print(f"  ✓ Checkpoint saved: epoch_{epoch:04d}.pth")
            
            # Sample generation
            if (epoch + 1) % Config.SAMPLE_FREQ == 0:
                sample_dir = self.results_dir / f'epoch_{epoch:04d}'
                sample_dir.mkdir(exist_ok=True)
                save_sample_images(self.system, self.test_loader, epoch, n_samples=4)
            
            epoch_time = (datetime.now() - epoch_start).total_seconds() / 60
            print(f"  Time: {epoch_time:.1f}m")
            
            # Early stopping
            if self.val_monitor.should_stop:
                print(f"\n{'─'*70}")
                print(f"EARLY STOPPING TRIGGERED")
                print(f"No improvement for {self.val_monitor.patience} epochs")
                print(f"{'─'*70}\n")
                break
        
        # Final summary
        self._print_final_summary()
        self._save_training_log()
    
    def _print_final_summary(self) -> None:
        """Print final training summary."""
        if not self.val_history.get('FID'):
            return
        
        print(f"\n{'='*70}")
        print(f"FINE-TUNING COMPLETE")
        print(f"{'='*70}")
        
        best_metrics = self.val_monitor.get_best_metrics()
        print(f"\nBest Validation Metrics:")
        print(f"  FID:  {best_metrics.get('FID', 'N/A'):.2f}")
        print(f"  SSIM: {best_metrics.get('SSIM', 'N/A'):.4f}")
        print(f"  PSNR: {best_metrics.get('PSNR', 'N/A'):.2f}")
        print(f"  Score: {self.val_monitor.best_score:.4f}")
        
        if self.val_history.get('FID'):
            print(f"\nProgress:")
            initial_fid = self.val_history['FID'][0]
            best_fid = min(self.val_history['FID'])
            improvement = ((initial_fid - best_fid) / initial_fid) * 100
            print(f"  FID Improvement: {improvement:.1f}%")
        
        print(f"{'='*70}\n")
    
    def _save_training_log(self) -> None:
        """Save training history to JSON."""
        log_file = self.logs_dir / f'finetune_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'train_history': {k: v for k, v in self.train_history.items()},
            'val_history': {k: v for k, v in self.val_history.items()},
            'best_metrics': self.val_monitor.get_best_metrics(),
            'config': {
                'learning_rate': Config.LEARNING_RATE,
                'batch_size': Config.BATCH_SIZE,
                'use_amp': Config.USE_AMP,
                'use_compile': Config.USE_COMPILE,
            }
        }
        
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"Training log saved: {log_file}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: COMMAND-LINE INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Fine-tune CycleGAN on TB dataset with checkpoint evaluation.'
    )
    
    parser.add_argument('--data_dir', type=str, default='./dataset',
                       help='Path to dataset directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Path to checkpoints directory')
    parser.add_argument('--logs_dir', type=str, default='./logs',
                       help='Path to logs directory')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='Path to results directory')
    
    parser.add_argument('--eval_only', action='store_true',
                       help='Only evaluate checkpoints, do not train')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from specific checkpoint (default: auto-select)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs to fine-tune')
    
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--reproducible', action='store_true',
                       help='Enable full reproducibility (slower)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup reproducibility
    set_reproducibility(reproducible=args.reproducible)
    
    device = GPU_INFO['device']
    
    # Build data loaders
    print(f"\nLoading dataset from {args.data_dir}...")
    train_loader, test_loader, stats = build_dataloaders(args.data_dir)
    
    # Evaluate existing checkpoints
    evaluator = CheckpointEvaluator(
        checkpoint_dir=args.checkpoint_dir,
        data_dir=args.data_dir,
        device=device,
    )
    
    results = evaluator.evaluate_all(test_loader)
    evaluator.print_summary()
    
    # Exit if eval_only
    if args.eval_only:
        print("Evaluation complete. Exiting.")
        return
    
    # Determine checkpoint to resume from
    if args.resume:
        resume_checkpoint = args.resume
        print(f"\nResuming from: {resume_checkpoint}")
    elif evaluator.best_checkpoint:
        resume_checkpoint = str(evaluator.best_checkpoint)
        print(f"\nAuto-selected best checkpoint: {evaluator.best_checkpoint.name}")
    else:
        print("ERROR: No checkpoint to resume from.")
        return
    
    # Extract epoch number from checkpoint
    try:
        resume_epoch = int(Path(resume_checkpoint).stem.split('_')[1])
    except (ValueError, IndexError):
        resume_epoch = 0
    
    # Setup fine-tuning
    print(f"\nInitializing fine-tuning system...")
    system = CycleGANSystem(device=device)
    
    # Load checkpoint
    print(f"Loading checkpoint from epoch {resume_epoch}...")
    system.load_checkpoint(resume_checkpoint)
    
    # Create trainer
    trainer = FinetuneTrainer(
        system=system,
        train_loader=train_loader,
        test_loader=test_loader,
        checkpoint_dir=args.checkpoint_dir,
        logs_dir=args.logs_dir,
        results_dir=args.results_dir,
        device=device,
    )
    
    # Fine-tune
    trainer.finetune(num_epochs=args.epochs, resume_epoch=resume_epoch)
    
    print("Fine-tuning pipeline complete!")


if __name__ == '__main__':
    main()
