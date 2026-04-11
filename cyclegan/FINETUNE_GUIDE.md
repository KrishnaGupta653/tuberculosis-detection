# CycleGAN Fine-Tuning Guide

## Overview

The fine-tuning pipeline (`finetune_cyclegan.py`) provides an automated workflow for improving your TB CycleGAN model:

1. **Checkpoint Evaluation** — Assesses all existing checkpoints on validation metrics (FID, SSIM, PSNR)
2. **Best Selection** — Identifies the best-performing checkpoint using a composite scoring metric
3. **Optimized Resume** — Continues training from the best checkpoint with improved hyperparameters
4. **Progress Tracking** — Monitors validation metrics, implements early stopping, and logs results

---

## Quick Start

### 1. Evaluate All Checkpoints (No Training)

```bash
python finetune_cyclegan.py \
  --data_dir ./dataset \
  --checkpoint_dir ./checkpoints \
  --eval_only
```

**Output:**

- Evaluates each checkpoint on validation set
- Computes composite scores (FID, SSIM, PSNR)
- Displays top-5 performers
- Recommends best checkpoint for fine-tuning

### 2. Full Pipeline (Evaluate + Fine-Tune)

```bash
python finetune_cyclegan.py \
  --data_dir ./dataset \
  --checkpoint_dir ./checkpoints \
  --epochs 50
```

**Workflow:**

1. Evaluates all checkpoints
2. Auto-selects best performer
3. Resumes training from best checkpoint
4. Fine-tunes for 50 additional epochs
5. Saves improved checkpoints and logs

### 3. Resume from Specific Checkpoint

```bash
python finetune_cyclegan.py \
  --data_dir ./dataset \
  --checkpoint_dir ./checkpoints \
  --resume ./checkpoints/epoch_0050.pth \
  --epochs 50
```

---

## Advanced Options

### Performance & Reproducibility

```bash
# Verbose output with detailed metrics
python finetune_cyclegan.py --eval_only --verbose

# Enable strict reproducibility (slower but deterministic)
python finetune_cyclegan.py --reproducible --epochs 50

# Mixed precision disabled (all float32)
# Modify Config.USE_AMP = False in cyclegan_tb_final.py
```

### Using Helper Scripts (Windows)

```powershell
# Evaluate only
powershell -ExecutionPolicy Bypass -File run_finetune.ps1 -Mode eval

# Full pipeline with 60 epochs
powershell -ExecutionPolicy Bypass -File run_finetune.ps1 -Mode full -Epochs 60

# Resume with verbose output
powershell -ExecutionPolicy Bypass -File run_finetune.ps1 -Mode resume `
  -ResumeCheckpoint "./checkpoints/epoch_0050.pth" -Verbose
```

---

## Key Features Explained

### 1. Checkpoint Evaluation

**Composite Scoring Metric:**

```
Score = 0.5 × (1 - tanh(FID/100))
      + 0.3 × SSIM
      + 0.2 × tanh(PSNR/50)
```

- **FID (Fréchet Inception Distance):** Lower is better (~0 = perfect)
- **SSIM (Structural Similarity):** Higher is better (0–1 range)
- **PSNR (Peak Signal-to-Noise Ratio):** Higher is better (dB scale)

**Why Composite Score?**

- Single metrics can be misleading (e.g., high FID but good perceptual quality)
- Balances generation quality, structure preservation, and noise characteristics
- Works with partial metrics (gracefully handles missing evaluations)

### 2. Progressive Learning Rate Scheduling

**Schedule:**

- **Epochs 0–5:** Linear warmup from 0.5× base_lr to base_lr
- **Epochs 6–N:** Cosine decay from base_lr to 0.1× base_lr

**Benefits:**

- Prevents catastrophic forgetting from prior training
- Warmup stabilizes gradients
- Gradual decay reduces overfitting
- Adaptive adjustment for different training phases

### 3. Validation Monitoring & Early Stopping

Tracks validation metrics across training:

- **Patience:** 7 epochs without improvement triggers early stop
- **Best Weights:** Stores best-performing model separately
- **History:** Logs all metrics for analysis

**Example:**

```
Epoch 003/052 | LR: 0.000195
  ...training...
  Validating... ✓ FID: 45.32 | SSIM: 0.658 | PSNR: 28.12
  → Validation improved! Best score: 0.6234
  ✓ Checkpoint saved: epoch_0003.pth
  Time: 12.4m

Epoch 004/052 | LR: 0.000197
  ...training...
  Validating... ✓ FID: 46.18 | SSIM: 0.655 | PSNR: 27.95
  ↓ No improvement (1/7)
```

### 4. Gradient Clipping & Stability

Training applies gradient clipping to prevent explosion:

```python
torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
```

Especially important during:

- Transition from discriminator to generator training
- Early warmup phase
- Sudden metric changes

### 5. Comprehensive Logging

Creates `logs/finetune_log_YYYYMMDD_HHMMSS.json`:

```json
{
  "timestamp": "2026-04-01T15:30:45.123456",
  "train_history": {
    "loss_G": [0.452, 0.421, ...],
    "loss_D": [0.523, 0.498, ...],
    "loss_cycle": [0.234, 0.221, ...]
  },
  "val_history": {
    "fid": [52.1, 48.3, 45.2, ...],
    "ssim": [0.642, 0.658, ...],
    "psnr": [27.5, 28.1, ...]
  },
  "best_metrics": {
    "fid": 42.1,
    "ssim": 0.671,
    "psnr": 28.9
  }
}
```

---

## Configuration Parameters

Edit these in `cyclegan_tb_final.py` Config class:

| Parameter           | Default | Notes                                                |
| ------------------- | ------- | ---------------------------------------------------- |
| `BATCH_SIZE`        | 1       | Must be 1 (Instance Norm requirement)                |
| `LEARNING_RATE`     | 0.0002  | Starting LR (scheduler modifies)                     |
| `LAMBDA_CYCLE`      | 10.0    | Cycle-consistency weight (higher = preserve anatomy) |
| `LAMBDA_IDENTITY`   | 5.0     | Identity loss weight (higher = preserve intensity)   |
| `LAMBDA_STRUCTURAL` | 1.0     | VGG perceptual loss weight                           |
| `USE_AMP`           | True    | Mixed precision training (faster on RTX30xx+)        |
| `USE_COMPILE`       | True    | PyTorch 2.0 compilation (needs torch≥2.0)            |
| `REPRODUCIBLE_MODE` | False   | Deterministic=True, benchmark=False (slower)         |
| `LOG_FREQ`          | 100     | Print loss every N batches                           |
| `SAVE_FREQ`         | 25      | Save checkpoint every N epochs                       |
| `SAMPLE_FREQ`       | 20      | Generate sample images every N epochs                |

---

## Performance Targets & Realistic Expectations

### Medical Image Generation Metrics

For TB chest X-ray translation, realistic target ranges are:

| Metric   | Poor  | Acceptable | Good      | Excellent |
| -------- | ----- | ---------- | --------- | --------- |
| **FID**  | >100  | 60–100     | 40–60     | <40       |
| **SSIM** | <0.60 | 0.60–0.70  | 0.70–0.80 | >0.80     |
| **PSNR** | <20   | 20–25      | 25–30     | >30       |

### Why "95–97% Accuracy" Doesn't Apply to CycleGAN

Your request mentioned "increase accuracy to 95–97%", but CycleGAN doesn't have a traditional accuracy metric because:

1. **Unsupervised Learning:** No ground truth pseudo-normal images exist
2. **Generative Quality:** Assessed via FID/SSIM/PSNR, not classification accuracy
3. **Clinical Validity:** Evaluated by radiologists, not algorithms

**What Fine-Tuning Targets:**

- Reduce FID by 10–20% (generate more realistic counterfactuals)
- Increase SSIM by 5–10% (preserve anatomical structure)
- Increase PSNR by 1–3 dB (reduce pixel noise)
- Achieve visual quality comparable to peer-reviewed CycleGAN papers

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'cyclegan_tb_final'"

**Solution:**

```bash
# Ensure both scripts are in the same directory
ls -la finetune_cyclegan.py cyclegan_tb_final.py

# Or specify Python path
PYTHONPATH=. python finetune_cyclegan.py --data_dir ./dataset
```

### Issue: "No checkpoints found"

**Solution:**

```bash
# Verify checkpoint directory structure
ls -la ./checkpoints/epoch_*.pth

# Check file permissions
chmod +r ./checkpoints/epoch_*.pth
```

### Issue: "CUDA out of memory"

**Solutions:**

1. Reduce batch size (currently 1 — already minimal)
2. Reduce image size in Config: `IMG_SIZE = 64` (from 128)
3. Disable AMP: Set `Config.USE_AMP = False`
4. Use gradient accumulation (modify trainer)

### Issue: Training is very slow

**Check:**

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Compile:', hasattr(torch, 'compile'))"
```

**Solutions:**

1. Enable AMP: `Config.USE_AMP = True`
2. Enable Compile: `Config.USE_COMPILE = True`
3. Increase NUM_WORKERS: `Config.NUM_WORKERS = 8` (if not on Windows)

### Issue: Validation doesn't improve

**Likely causes:**

1. Starting from near-optimal checkpoint (already well-trained)
2. Learning rate too low (check LR scheduler output)
3. Dataset too small or imbalanced
4. Medical features already at realistic limits

**Next steps:**

- Inspect generated counterfactuals manually
- Lower SAVE_FREQ to see sample generation every epoch
- Try longer training patience: modify `patience=7` in ValidationMonitor

---

## Example: Complete Fine-Tuning Workflow

```bash
# Step 1: Evaluate all checkpoints
python finetune_cyclegan.py --data_dir ./dataset --checkpoint_dir ./checkpoints --eval_only --verbose

# Output:
# ======================================================================
# CHECKPOINT EVALUATION
# ======================================================================
# Found 5 checkpoint(s)
#
#   Evaluating epoch_0000.pth... ✓ Score: 0.4521
#   Evaluating epoch_0025.pth... ✓ Score: 0.5832
#   Evaluating epoch_0050.pth... ✓ Score: 0.6174 ← BEST
#   Evaluating epoch_0075.pth... ✓ Score: 0.6098
#   Evaluating epoch_0100.pth... ✓ Score: 0.5921
#
# ======================================================================
# EVALUATION SUMMARY
# ======================================================================
#   Epoch 050: Score=0.6174 | FID=42.32 | SSIM=0.6741 | PSNR=28.54 → BEST
#   Epoch 025: Score=0.5832 | FID=48.21 | SSIM=0.6412 | PSNR=27.82
#   ...
# ======================================================================
#
# ✓ BEST CHECKPOINT: epoch_0050.pth
#   Ready for fine-tuning.

# Step 2: Fine-tune from best checkpoint (auto-select)
python finetune_cyclegan.py --data_dir ./dataset --checkpoint_dir ./checkpoints --epochs 50

# Output:
# Training starts from epoch 50, runs 50 additional epochs (50–99)
# Logs saved to: logs/finetune_log_20260401_153045.json
# Improved checkpoints saved to: checkpoints/epoch_0050.pth through epoch_0099.pth

# Step 3: Analyze results
python analyze_finetune.py logs/finetune_log_20260401_153045.json
# (see accompanying analyze_finetune.py script)
```

---

## Post-Fine-Tuning Analysis

After fine-tuning completes, analyze results with the companion script:

```bash
# Generate plots and summary statistics
python analyze_finetune.py logs/finetune_log_20260401_153045.json

# Output:
# - FID, SSIM, PSNR curves over time
# - Best metrics summary table
# - Improvement statistics
# - Training/validation loss divergence plot
```

---

## Next Steps & Best Practices

1. **Manual Inspection:** Review generated counterfactual images in `results/` to ensure clinical plausibility
2. **Radiologist Review:** Have domain expert evaluate if counterfactuals highlight realistic TB features
3. **Deployment:** Package best checkpoint with inference script (see `cyclegan_inference.py`)
4. **Reproducibility:** Save hyperparameters: `Config.LEARNING_RATE`, scheduler params, etc.
5. **Monitoring:** Log all metrics (already done) for publication

---

## References

- **CycleGAN:** Zhu et al. (2017) "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks"
- **Medical Imaging:** Singla et al. (2022) "GANterfactual — Counterfactual Explanations for Medical Non-experts"
- **Mixed Precision:** NVIDIA Apex AMP + PyTorch 2.0 torch.amp documentation
- **Metrics:** FID (Heusel et al., 2017), SSIM (Wang et al., 2004), PSNR (standard)

---

## Support & Debugging

For detailed technical logs, enable verbose output:

```bash
python -u finetune_cyclegan.py --eval_only --verbose 2>&1 | tee debug.log
```

This will display:

- GPU device name and VRAM
- CUDA availability and PyTorch compile support
- Estimated memory per batch
- Per-layer activation ranges (for debugging NaN/gradient issues)

---

**Happy fine-tuning! 🚀**
