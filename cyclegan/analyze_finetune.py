"""
================================================================================
FINE-TUNING ANALYSIS SCRIPT — analyze_finetune.py
================================================================================
Post-training analysis: visualizes metrics, computes improvements, generates summary.

USAGE:
    python analyze_finetune.py logs/finetune_log_20260401_153045.json

OUTPUT:
    - FID/SSIM/PSNR curves over training epochs
    - Best metrics summary
    - Improvement percentages
    - Training/validation loss divergence analysis
================================================================================
"""

import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path
from datetime import datetime


def load_finetune_log(log_file: str) -> dict:
    """Load fine-tuning log JSON."""
    with open(log_file, 'r') as f:
        return json.load(f)


def compute_improvements(log_data: dict) -> dict:
    """Compute improvement statistics."""
    val_hist = log_data.get('val_history', {})
    train_hist = log_data.get('train_history', {})
    
    improvements = {}
    
    # Per-metric improvements
    for metric in ['fid', 'ssim', 'psnr']:
        if metric in val_hist and len(val_hist[metric]) >= 2:
            initial = val_hist[metric][0]
            best = val_hist[metric][-1]
            
            if metric == 'fid':
                # Lower is better
                improvement = ((initial - best) / initial) * 100
                improvements[metric] = {
                    'initial': initial,
                    'best': best,
                    'improvement_pct': improvement,
                }
            else:
                # Higher is better (SSIM, PSNR)
                improvement = ((best - initial) / initial) * 100
                improvements[metric] = {
                    'initial': initial,
                    'best': best,
                    'improvement_pct': improvement,
                }
    
    # Training stability
    if 'loss_G' in train_hist and 'loss_D' in train_hist:
        g_losses = train_hist['loss_G']
        d_losses = train_hist['loss_D']
        
        if g_losses and d_losses:
            improvements['training_stability'] = {
                'G_loss_final': g_losses[-1],
                'D_loss_final': d_losses[-1],
                'G_loss_reduction': ((g_losses[0] - g_losses[-1]) / g_losses[0]) * 100,
                'D_loss_reduction': ((d_losses[0] - d_losses[-1]) / d_losses[0]) * 100,
            }
    
    return improvements


def print_summary(log_data: dict, improvements: dict) -> None:
    """Print training summary to console."""
    print("\n" + "="*70)
    print("FINE-TUNING ANALYSIS SUMMARY")
    print("="*70)
    
    # Timestamp
    if 'timestamp' in log_data:
        print(f"\nTimestamp: {log_data['timestamp']}")
    
    # Config
    if 'config' in log_data:
        cfg = log_data['config']
        print(f"\nConfiguration:")
        print(f"  Learning Rate: {cfg.get('learning_rate', 'N/A')}")
        print(f"  Batch Size: {cfg.get('batch_size', 'N/A')}")
        print(f"  AMP: {cfg.get('use_amp', 'N/A')}")
        print(f"  Compile: {cfg.get('use_compile', 'N/A')}")
    
    # Validation metrics improvements
    print(f"\nValidation Metrics Improvements:")
    print(f"{'─'*70}")
    
    if 'fid' in improvements:
        fid = improvements['fid']
        print(f"  FID (Fréchet Inception Distance):")
        print(f"    Initial:  {fid['initial']:.2f}")
        print(f"    Best:     {fid['best']:.2f}")
        print(f"    ↓ {fid['improvement_pct']:.1f}% improvement")
    
    if 'ssim' in improvements:
        ssim = improvements['ssim']
        print(f"\n  SSIM (Structural Similarity):")
        print(f"    Initial:  {ssim['initial']:.4f}")
        print(f"    Best:     {ssim['best']:.4f}")
        print(f"    ↑ {ssim['improvement_pct']:.1f}% improvement")
    
    if 'psnr' in improvements:
        psnr = improvements['psnr']
        print(f"\n  PSNR (Peak Signal-to-Noise Ratio):")
        print(f"    Initial:  {psnr['initial']:.2f} dB")
        print(f"    Best:     {psnr['best']:.2f} dB")
        print(f"    ↑ {psnr['improvement_pct']:.1f}% improvement")
    
    # Training stability
    if 'training_stability' in improvements:
        ts = improvements['training_stability']
        print(f"\nTraining Stability:")
        print(f"{'─'*70}")
        print(f"  Generator Loss:")
        print(f"    Final:  {ts['G_loss_final']:.6f}")
        print(f"    ↓ {ts['G_loss_reduction']:.1f}% reduction from start")
        print(f"\n  Discriminator Loss:")
        print(f"    Final:  {ts['D_loss_final']:.6f}")
        print(f"    ↓ {ts['D_loss_reduction']:.1f}% reduction from start")
    
    # Best metrics
    if 'best_metrics' in log_data:
        best = log_data['best_metrics']
        print(f"\nBest Epoch Metrics:")
        print(f"{'─'*70}")
        if 'fid' in best:
            print(f"  FID:  {best['fid']:.2f}")
        if 'ssim' in best:
            print(f"  SSIM: {best['ssim']:.4f}")
        if 'psnr' in best:
            print(f"  PSNR: {best['psnr']:.2f} dB")
    
    print(f"{'='*70}\n")


def plot_metrics(log_data: dict, output_file: str = None) -> None:
    """Generate comprehensive metrics visualization."""
    train_hist = log_data.get('train_history', {})
    val_hist = log_data.get('val_history', {})
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. FID Curve
    ax1 = fig.add_subplot(gs[0, 0])
    if 'fid' in val_hist:
        ax1.plot(val_hist['fid'], 'b-o', linewidth=2, markersize=4)
        ax1.set_xlabel('Validation Checkpoint')
        ax1.set_ylabel('FID Score (lower is better)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True, alpha=0.3)
        ax1.set_title('FID (Fréchet Inception Distance) Over Training', fontweight='bold')
        
        # Add best value annotation
        best_idx = np.argmin(val_hist['fid'])
        best_val = val_hist['fid'][best_idx]
        ax1.plot(best_idx, best_val, 'r*', markersize=15, label=f'Best: {best_val:.2f}')
        ax1.legend()
    
    # 2. SSIM Curve
    ax2 = fig.add_subplot(gs[0, 1])
    if 'ssim' in val_hist:
        ax2.plot(val_hist['ssim'], 'g-o', linewidth=2, markersize=4)
        ax2.set_xlabel('Validation Checkpoint')
        ax2.set_ylabel('SSIM (higher is better)', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        ax2.grid(True, alpha=0.3)
        ax2.set_title('SSIM (Structural Similarity) Over Training', fontweight='bold')
        ax2.set_ylim([0, 1])
        
        # Add best value annotation
        best_idx = np.argmax(val_hist['ssim'])
        best_val = val_hist['ssim'][best_idx]
        ax2.plot(best_idx, best_val, 'r*', markersize=15, label=f'Best: {best_val:.4f}')
        ax2.legend()
    
    # 3. PSNR Curve
    ax3 = fig.add_subplot(gs[1, 0])
    if 'psnr' in val_hist:
        ax3.plot(val_hist['psnr'], 'purple', marker='o', linewidth=2, markersize=4)
        ax3.set_xlabel('Validation Checkpoint')
        ax3.set_ylabel('PSNR (dB, higher is better)', color='purple')
        ax3.tick_params(axis='y', labelcolor='purple')
        ax3.grid(True, alpha=0.3)
        ax3.set_title('PSNR (Peak Signal-to-Noise Ratio) Over Training', fontweight='bold')
        
        # Add best value annotation
        best_idx = np.argmax(val_hist['psnr'])
        best_val = val_hist['psnr'][best_idx]
        ax3.plot(best_idx, best_val, 'r*', markersize=15, label=f'Best: {best_val:.2f}')
        ax3.legend()
    
    # 4. Training Losses
    ax4 = fig.add_subplot(gs[1, 1])
    if 'loss_G' in train_hist and 'loss_D' in train_hist:
        epochs = range(len(train_hist['loss_G']))
        ax4.plot(epochs, train_hist['loss_G'], 'b-', linewidth=1.5, label='Generator', alpha=0.8)
        ax4.plot(epochs, train_hist['loss_D'], 'r-', linewidth=1.5, label='Discriminator', alpha=0.8)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4.set_title('Training Losses Over Epochs', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # 5. Cycle Loss Contributions
    ax5 = fig.add_subplot(gs[2, 0])
    if 'loss_cycle' in train_hist:
        ax5.plot(train_hist['loss_cycle'], 'orange', linewidth=1.5)
        ax5.fill_between(range(len(train_hist['loss_cycle'])), 
                         train_hist['loss_cycle'], alpha=0.3, color='orange')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Cycle Loss (lower is better)')
        ax5.set_title('Cycle-Consistency Loss Over Training', fontweight='bold')
        ax5.grid(True, alpha=0.3)
    
    # 6. Composite Validation Score
    ax6 = fig.add_subplot(gs[2, 1])
    if 'fid' in val_hist and 'ssim' in val_hist and 'psnr' in val_hist:
        # Compute composite score like in the trainer
        fid_norm = [1.0 - np.tanh(f / 100.0) for f in val_hist['fid']]
        ssim_norm = [np.clip(s, 0, 1) for s in val_hist['ssim']]
        psnr_norm = [np.tanh(p / 50.0) for p in val_hist['psnr']]
        
        composite = [0.5*f + 0.3*s + 0.2*p 
                    for f, s, p in zip(fid_norm, ssim_norm, psnr_norm)]
        
        ax6.plot(composite, 'g-o', linewidth=2, markersize=5)
        ax6.set_xlabel('Validation Checkpoint')
        ax6.set_ylabel('Composite Score (higher is better)')
        ax6.set_title('Composite Validation Score', fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        # Add best value
        best_idx = np.argmax(composite)
        best_val = composite[best_idx]
        ax6.plot(best_idx, best_val, 'r*', markersize=15)
    
    plt.suptitle('CycleGAN Fine-Tuning Metrics Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Metrics plot saved: {output_file}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze fine-tuning results from JSON log.'
    )
    parser.add_argument('log_file', type=str, help='Path to finetune_log_*.json')
    parser.add_argument('--output', type=str, default=None, 
                       help='Output path for plots (default: display)')
    args = parser.parse_args()
    
    # Load log
    if not Path(args.log_file).exists():
        print(f"ERROR: Log file not found: {args.log_file}")
        return
    
    log_data = load_finetune_log(args.log_file)
    
    # Compute improvements
    improvements = compute_improvements(log_data)
    
    # Print summary
    print_summary(log_data, improvements)
    
    # Generate plots
    output_file = args.output or f"{Path(args.log_file).stem}_plots.png"
    plot_metrics(log_data, output_file)


if __name__ == '__main__':
    main()
