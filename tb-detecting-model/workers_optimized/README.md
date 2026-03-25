================================================================================
WORKER OPTIMIZATION GUIDE — COMPREHENSIVE REFERENCE
================================================================================

This guide explains the new worker-based parallel training system and how to
use it effectively for 3-4x faster feature extraction and model training.

================================================================================
QUICK START
================================================================================

1. AUTOMATIC (RECOMMENDED)
   cd workers_optimized
   python train_with_workers.py
   - Auto-detects CPU cores
   - Uses 75% of cores for workers
   - Trains with parallel feature extraction

2. MANUAL WORKER COUNT
   python workers_optimized/train_with_workers.py --workers 8

3. SHOW DIAGNOSTICS
   python workers_optimized/train_with_workers.py --show-diagnostics

================================================================================
WHAT'S NEW?
================================================================================

PARALLEL FEATURE EXTRACTION
✅ extract_features_with_workers() - True multiprocessing (3-4x faster)
✅ Works cross-platform (Windows/Mac/Linux)
✅ Auto-detects optimal worker count

WORKER-AWARE MODEL TRAINING
✅ SVM: Full parallelization
✅ Random Forest: Full parallelization
✅ Gradient Boosting: Balanced parallelization

AUTO-DETECTED CONFIGURATION
✅ WorkerConfig.CPU_COUNT - Auto-detects available cores
✅ WorkerConfig.OPTIMAL_WORKERS - Uses 75% of cores
✅ Sensible defaults for all parameters

================================================================================
PERFORMANCE IMPROVEMENTS
================================================================================

FEATURE EXTRACTION:
Sequential (old): ~8-12 images/second
Parallel (new): ~30-40 images/second
Speedup: 3-4x faster

FULL TRAINING (1000 images):
Sequential: ~120-150 seconds
Parallel (4 workers): ~35-50 seconds
Speedup: 2.5-3x faster

================================================================================
HOW TO USE
================================================================================

FROM COMMAND LINE:
cd workers_optimized
python train_with_workers.py --show-diagnostics

FROM PYTHON CODE:
from workers_optimized.pipeline_workers import extract_features_with_workers

X_train, y_train = extract_features_with_workers(
image_paths,
labels
)

================================================================================
FILES IN THIS FOLDER
================================================================================

pipeline_workers.py

- Core parallel processing module
- WorkerConfig class for configuration
- extract_features_with_workers() function
- extract_features_batched() for memory efficiency
- configure_sklearn_workers() for model optimization

train_with_workers.py

- Command-line training script
- Can be run standalone
- Handles train/predict/threshold-optimize modes
- Options: --workers, --image, --show-diagnostics

WORKERS_GUIDE.md

- This file - comprehensive reference

================================================================================
CONFIGURATION
================================================================================

Modify WorkerConfig for custom settings:

from workers_optimized.pipeline_workers import WorkerConfig

WorkerConfig.NUM_WORKERS = 4
WorkerConfig.BATCH_SIZE = 32
WorkerConfig.CHUNK_SIZE = 4

================================================================================
TROUBLESHOOTING
================================================================================

Issue: RuntimeError on Windows
Solution: Use spawn method (already default) - ensure code in if **name** == '**main**'

Issue: Out of memory
Solution: Use extract_features_batched() instead of extract_features_with_workers()

Issue: Slower than sequential
Solution: Overhead only pays off for large datasets (>500 images)

Issue: BrokenPipeError
Solution: Reduce workers or batch size

================================================================================
BEST PRACTICES
================================================================================

✓ Use auto-detected workers (WorkerConfig.OPTIMAL_WORKERS)
✓ For >1000 images: use parallel (3-4x faster)
✓ For <100 images: use sequential (overhead > benefit)
✓ Monitor memory during first run
✓ Always wrap main code in if **name** == '**main**'

================================================================================
IMPORT FROM THIS FOLDER
================================================================================

From parent directory:
from workers_optimized.pipeline_workers import extract_features_with_workers
from workers_optimized.train_with_workers import main

From workers_optimized directory:
from pipeline_workers import extract_features_with_workers
from train_with_workers import main

================================================================================
