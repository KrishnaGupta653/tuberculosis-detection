# Quick Start Reference — Fine-Tuning Pipeline

## 5-Minute Setup

### 1. Evaluate Existing Checkpoints (No Training)

```bash
python finetune_cyclegan.py \
  --data_dir ./dataset \
  --checkpoint_dir ./checkpoints \
  --eval_only
```

**What this does:**

- Loads all epoch\_\*.pth checkpoints
- Tests each on validation set
- Computes FID, SSIM, PSNR for each
- Ranks by composite score
- **Output:** Displays best checkpoint to resume from

**Expected output:**

```
======================================================================
CHECKPOINT EVALUATION
======================================================================
Found 5 checkpoint(s)
──────────────────────────────────────────────────────────────────────

  Evaluating epoch_0000.pth... ✓ Score: 0.4521
      FID: 52.12 | SSIM: 0.6412 | PSNR: 27.45

  Evaluating epoch_0025.pth... ✓ Score: 0.5832
      FID: 48.21 | SSIM: 0.6588 | PSNR: 28.12

  Evaluating epoch_0050.pth... ✓ Score: 0.6174
      FID: 42.32 | SSIM: 0.6741 | PSNR: 28.54

  Evaluating epoch_0075.pth... ✓ Score: 0.6098
      FID: 43.18 | SSIM: 0.6698 | PSNR: 28.31

  Evaluating epoch_0100.pth... ✓ Score: 0.5921
      FID: 45.67 | SSIM: 0.6521 | PSNR: 28.02

──────────────────────────────────────────────────────────────────────
EVALUATION SUMMARY
──────────────────────────────────────────────────────────────────────
  Epoch 050: Score=0.6174 | FID=42.32 | SSIM=0.6741 | PSNR=28.54 → BEST
  Epoch 025: Score=0.5832 | FID=48.21 | SSIM=0.6588 | PSNR=28.12
  Epoch 075: Score=0.6098 | FID=43.18 | SSIM=0.6698 | PSNR=28.31
  ...

✓ BEST CHECKPOINT: ./checkpoints/epoch_0050.pth
  Ready for fine-tuning.
```

---

### 2. Fine-Tune from Best Checkpoint (Full Pipeline)

```bash
python finetune_cyclegan.py \
  --data_dir ./dataset \
  --checkpoint_dir ./checkpoints \
  --epochs 50
```

**What this does:**

1. Evaluates all checkpoints (same as step 1)
2. Auto-selects best checkpoint (epoch_0050.pth in example above)
3. Resumes training from epoch 50
4. Trains for 50 more epochs (total epochs 50–99)
5. Every 20 epochs: validates and saves checkpoint
6. Early stops if no validation improvement for 7 epochs
7. Saves all logs to `logs/finetune_log_YYYYMMDD_HHMMSS.json`

**Expected output:**

```
============================================================
FINE-TUNING (Epochs 50–99)
============================================================

Epoch 050/099 | LR: 0.000195
  Epoch 050 [  100/  200] | G: 0.452 | D: 0.523 | Cycle: 0.234
  Epoch 050 [  200/  200] | G: 0.441 | D: 0.515 | Cycle: 0.228
  ✓ Checkpoint saved: epoch_0050.pth
  Time: 12.4m

Epoch 051/099 | LR: 0.000196
  ...training...
  Time: 12.3m

Epoch 052/099 | LR: 0.000197
  ...training...
  Validating... ✓ FID: 41.52 | SSIM: 0.6758 | PSNR: 28.71
  → Validation improved! Best score: 0.6201
  Time: 12.5m

...more epochs...

Epoch 070/099 | LR: 0.000185
  Validating... ✓ FID: 38.92 | SSIM: 0.6912 | PSNR: 29.18
  ↓ No improvement (1/7)
  Time: 12.4m

...final epochs...

============================================================
FINE-TUNING COMPLETE
============================================================

Best Validation Metrics:
  FID:  38.92
  SSIM: 0.6912
  PSNR: 29.18
  Score: 0.6456

Progress:
  FID Improvement: 8.0%

============================================================

Training log saved: logs/finetune_log_20260401_153045.json
```

---

### 3. Analyze Fine-Tuning Results

```bash
python analyze_finetune.py logs/finetune_log_20260401_153045.json
```

**What this does:**

- Loads training log
- Computes improvement %
- Prints detailed summary
- Generates multi-panel metrics plot
- Saves plot as PNG

**Output:**

```
======================================================================
FINE-TUNING ANALYSIS SUMMARY
======================================================================

Timestamp: 2026-04-01T15:30:45.123456

Configuration:
  Learning Rate: 0.0002
  Batch Size: 1
  AMP: True
  Compile: True

Validation Metrics Improvements:
──────────────────────────────────────────────────────────────────────
  FID (Fréchet Inception Distance):
    Initial:  42.32
    Best:     38.92
    ↓ 8.0% improvement

  SSIM (Structural Similarity):
    Initial:  0.6741
    Best:     0.6912
    ↑ 2.5% improvement

  PSNR (Peak Signal-to-Noise Ratio):
    Initial:  28.54 dB
    Best:     29.18 dB
    ↑ 2.2% improvement

Training Stability:
──────────────────────────────────────────────────────────────────────
  Generator Loss:
    Final:  0.435621
    ↓ 3.6% reduction from start

  Discriminator Loss:
    Final:  0.502134
    ↓ 4.2% reduction from start

Best Epoch Metrics:
──────────────────────────────────────────────────────────────────────
  FID:  38.92
  SSIM: 0.6912
  PSNR: 29.18 dB

✓ Metrics plot saved: logs/finetune_log_20260401_153045_plots.png

[Opens PNG with 6-panel visualization]
```

---

## Common Workflows

### Workflow A: Identify Best Checkpoint Only

```bash
python finetune_cyclegan.py --data_dir ./dataset --checkpoint_dir ./checkpoints --eval_only --verbose
```

**Use when:** You want to see which checkpoint is best before committing to training.

---

### Workflow B: Resume from Specific Old Checkpoint

```bash
python finetune_cyclegan.py \
  --data_dir ./dataset \
  --resume ./checkpoints/epoch_0025.pth \
  --epochs 75
```

**Use when:** You want to compare fine-tuning from different starting points.

---

### Workflow C: Reproducible Training (Deterministic)

```bash
python finetune_cyclegan.py \
  --data_dir ./dataset \
  --checkpoint_dir ./checkpoints \
  --epochs 50 \
  --reproducible
```

**Use when:** You need exactly reproducible results for paper submission.

**Cost:** ~15% slower training (disables CUDA benchmark).

---

### Workflow D: Help & Debugging

```bash
# See all options
python finetune_cyclegan.py --help

# Full verbose output to log file
python -u finetune_cyclegan.py \
  --data_dir ./dataset \
  --checkpoint_dir ./checkpoints \
  --epochs 50 \
  --verbose 2>&1 | tee training_debug.log
```

---

## Windows PowerShell Quick Commands

### Evaluate Checkpoints

```powershell
powershell -ExecutionPolicy Bypass -File run_finetune.ps1 -Mode eval
```

### Full Pipeline (50 epochs)

```powershell
powershell -ExecutionPolicy Bypass -File run_finetune.ps1 -Mode full -Epochs 50
```

### Resume from Specific Checkpoint

```powershell
powershell -ExecutionPolicy Bypass -File run_finetune.ps1 `
  -Mode resume `
  -ResumeCheckpoint "./checkpoints/epoch_0050.pth" `
  -Epochs 50 `
  -Verbose
```

---

## File Organization After Running

```
cyclegan/
├── checkpoints/
│   ├── epoch_0000.pth
│   ├── epoch_0025.pth
│   ├── epoch_0050.pth        ← Best (from eval)
│   ├── epoch_0050.pth         ← Same, used for resume
│   ├── epoch_0055.pth         ← New (from fine-tuning)
│   ├── ... more epoch_xxxx.pth files ...
│   └── epoch_0099.pth         ← Final checkpoint
│
├── results/
│   ├── epoch_0050/            (original generation)
│   ├── epoch_0055/
│   ├── epoch_0060/
│   ├── ... continues every 20 epochs ...
│   └── epoch_0099/
│
├── logs/
│   ├── training_log.json      (original)
│   └── finetune_log_20260401_153045.json  ← NEW
│
├── finetune_cyclegan.py       ← Your new script
├── analyze_finetune.py        ← Your analysis script
├── FINETUNE_GUIDE.md          ← Full documentation
└── ... (existing files)
```

---

## Key Metrics Explained

| Metric   | Range  | Good  | Why It Matters                             |
| -------- | ------ | ----- | ------------------------------------------ |
| **FID**  | 0–∞    | <40   | Measures realism of generated images       |
| **SSIM** | 0–1    | >0.70 | Measures structural similarity to original |
| **PSNR** | 0–∞ dB | >25   | Measures pixel-level accuracy              |

**Goal:** FID ↓, SSIM ↑, PSNR ↑

---

## Troubleshooting Quick Links

- **"No module named cyclegan_tb_final"** → Ensure both .py files in same directory
- **"CUDA out of memory"** → Already at minimal batch_size=1; try IMG_SIZE=64
- **"Training very slow"** → Enable AMP (Config.USE_AMP=True) and Compile (Config.USE_COMPILE=True)
- **"Validation doesn't improve"** → Starting from near-optimal; expected behavior

See **FINETUNE_GUIDE.md** for full troubleshooting section.

---

## Next Steps After Fine-Tuning

1. **Review Generated Counterfactuals** → Open `results/epoch_00XX/` folders
2. **Run Medical Domain Expert Review** → Have radiologist check plausibility
3. **Deploy Best Model** → Use `cyclegan_inference.py` with best checkpoint
4. **Publish Results** → Include metrics plots and logs in paper appendix

---

## Example: Full End-to-End Session

```bash
# Terminal 1: Evaluate
python finetune_cyclegan.py --data_dir ./dataset --checkpoint_dir ./checkpoints --eval_only

# Output tells you best checkpoint is epoch_0050.pth

# Terminal 2: Fine-tune (runs ~10 hours for 50 epochs on RTX 3080)
python finetune_cyclegan.py --data_dir ./dataset --checkpoint_dir ./checkpoints --epochs 50

# While training, in browser/file explorer:
# → Watch results/epoch_00XX/ fill with new counterfactual images

# When done, analyze:
python analyze_finetune.py logs/finetune_log_20260401_153045.json

# Output: Summary table + 6-panel metrics visualization
```

---

**For detailed documentation, see:** [FINETUNE_GUIDE.md](FINETUNE_GUIDE.md)

**Questions?** Check traces in `logs/` or enable `--verbose` for debug output.

Good luck with your fine-tuning! 🚀
