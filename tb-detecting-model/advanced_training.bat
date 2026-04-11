@echo off
REM ============================================================================
REM TB DETECTION ADVANCED ENSEMBLE - QUICK START
REM ============================================================================
REM
REM This script sets up and runs the advanced training system to achieve >97% accuracy
REM
REM Usage: 
REM   double-click this .bat file or run: advanced_training.bat
REM
REM REQUIREMENTS:
REM   - Python 3.8+
REM   - GPU (NVIDIA CUDA) - optional but recommended
REM   - 16GB+ RAM
REM   - Dataset: dataset/Normal/ and dataset/TB/ folders
REM
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ============================================================================
echo  TB DETECTION SYSTEM - ADVANCED ENSEMBLE (97%+ ACCURACY)
echo ============================================================================
echo.

REM Check Python
echo [CHECK] Verifying Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Please install Python 3.8+
    pause
    exit /b 1
)
echo [OK] Python found

REM Check GPU
echo.
echo [CHECK] Checking GPU availability...
python -c "import torch; cuda_available = torch.cuda.is_available(); print('[OK] GPU available: ' + str(cuda_available)); exit(0)"

REM Install dependencies
echo.
echo [INSTALL] Installing required packages...
echo   Installing: torch, torchvision, scikit-learn, imbalanced-learn...
pip install --upgrade torch torchvision
pip install --upgrade scikit-learn
pip install --upgrade imbalanced-learn
pip install tqdm scipy scikit-image matplotlib seaborn pandas
echo [OK] Dependencies installed

REM Verify dataset
echo.
echo [DATASET] Verifying dataset structure...
if not exist "dataset\Normal" (
    echo [ERROR] dataset/Normal folder not found!
    pause
    exit /b 1
)
if not exist "dataset\TB" (
    echo [ERROR] dataset/TB folder not found!
    pause
    exit /b 1
)
echo [OK] Dataset structure verified

REM Create output directory
echo.
echo [SETUP] Creating output directories...
if not exist "saved_models\advanced" (
    mkdir saved_models\advanced
    echo [OK] Created saved_models/advanced
)

REM Run advanced training
echo.
echo ============================================================================
echo [TRAIN] STARTING ADVANCED ENSEMBLE TRAINING
echo ============================================================================
echo.
echo Target Accuracy: >97%%
echo Expected Time: 30-120 minutes (depending on GPU)
echo Output Location: saved_models/advanced/
echo.
echo ============================================================================
echo.

cd models
python train_advanced_ensemble.py --mode train

if errorlevel 1 (
    echo.
    echo [ERROR] Training failed! Check the output above for details.
    pause
    exit /b 1
)

cd ..

echo.
echo ============================================================================
echo [SUCCESS] TRAINING COMPLETED!
echo ============================================================================
echo.
echo Next steps:
echo   1. Check training results above
echo   2. Test model: python models/train_advanced_ensemble.py --mode evaluate
echo   3. Make predictions: python models/train_advanced_ensemble.py --mode predict --image path/to/image.png
echo.
echo Results saved to: saved_models/advanced/
echo.
pause
