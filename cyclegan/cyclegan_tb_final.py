"""
================================================================================
TB COUNTERFACTUAL EXPLANATION SYSTEM — cyclegan_tb_final.py
================================================================================
Publication-quality CycleGAN pipeline for generating counterfactual explanations
in tuberculosis detection from chest X-ray images.

OBJECTIVE:
    Transform TB-positive chest X-rays into their "healthy" counterparts using
    unpaired image-to-image translation (CycleGAN). The pixel-wise difference
    between original and generated image serves as the counterfactual explanation,
    highlighting the specific morphological features that drove the AI's diagnosis.

DESIGN PHILOSOPHY:
    - Medical validity is prioritized over raw GAN performance metrics.
    - Cycle-consistency and identity losses are weighted heavily to prevent
      hallucination of non-existent anatomy (a critical safety concern).
    - LSGAN (least-squares adversarial loss) is used for training stability
      over standard binary cross-entropy GAN loss.
    - All architectural choices are documented with clinical rationale.

GPU OPTIMIZATIONS (v2 additions):
    - torch.compile()         : ~20–30% speedup via Triton/CUDA kernels (PyTorch 2.0+)
    - Mixed Precision (AMP)   : ~50–100% speedup via float16 forward pass
    - GradScaler              : Prevents float16 underflow during AMP backward pass
    - cudnn.benchmark         : ~10–15% speedup (auto-selects fastest conv algorithm)
    - PIN_MEMORY + NUM_WORKERS: Faster CPU→GPU data transfers
    - GPU info logging        : Prints device name, VRAM, CUDA version at startup
    NOTE: cudnn.benchmark=True trades exact reproducibility for speed.
          Set REPRODUCIBLE_MODE=True in Config to restore full determinism for
          final paper runs.

USAGE:
    # Train from scratch
    python cyclegan_tb_final.py --mode train --data_dir ./dataset

    # Resume training from checkpoint
    python cyclegan_tb_final.py --mode train --data_dir ./dataset --resume ./checkpoints/epoch_0050.pth

    # Generate counterfactuals for all TB images
    python cyclegan_tb_final.py --mode generate --data_dir ./dataset --checkpoint ./checkpoints/epoch_0200.pth

    # Evaluate a trained model
    python cyclegan_tb_final.py --mode evaluate --data_dir ./dataset --checkpoint ./checkpoints/epoch_0200.pth

    # Train with strict reproducibility (disables benchmark, enables deterministic)
    python cyclegan_tb_final.py --mode train --data_dir ./dataset --reproducible

REFERENCE ARCHITECTURE:
    Based on: Zhu et al. "Unpaired Image-to-Image Translation using
    Cycle-Consistent Adversarial Networks" (ICCV 2017), adapted for
    medical imaging with hallucination mitigation strategies from:
    Singla et al. "GANterfactual — Counterfactual Explanations for Medical
    Non-experts" (Frontiers in AI, 2022).
    Mixed precision strategy from: NVIDIA Apex AMP documentation and
    PyTorch 2.0 torch.amp API.
================================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import sys
import json
import time
import random
import argparse
import warnings
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from itertools import chain

# ── Scientific computing ───────────────────────────────────────────────────────
import numpy as np
from PIL import Image

# ── PyTorch ────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.utils import save_image
# ── Evaluation metrics ─────────────────────────────────────────────────────────
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    warnings.warn(
        "torchmetrics not found. FID/SSIM/PSNR evaluation will be skipped. "
        "Install with: pip install torchmetrics"
    )

# ── Visualization ──────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: GPU DETECTION & DIAGNOSTICS
# Purpose: Identify and report hardware at startup. Determines which GPU
#          optimizations to enable. Runs before Config so every subsequent
#          module can branch on CUDA availability.
# ══════════════════════════════════════════════════════════════════════════════

def _has_working_triton() -> bool:
    """Best-effort Triton availability check for torch.compile on CUDA."""
    try:
        import triton  # noqa: F401
        return True
    except Exception:
        return False


def detect_and_report_gpu(verbose: bool = True) -> dict:
    """
    Detect GPU availability and print a diagnostic report at startup.

    Returns a dict with hardware info used throughout the pipeline:
    {
        'cuda_available': bool,
        'device': torch.device,
        'device_name': str,
        'vram_gb': float,       # Total VRAM in GB (0 if CPU)
        'cuda_version': str,
        'torch_version': str,
        'amp_supported': bool,  # True for CUDA + PyTorch ≥ 1.6
        'compile_supported': bool,  # True for PyTorch ≥ 2.0 + Triton on CUDA
        'triton_available': bool,
    }
    """
    cuda_available      = torch.cuda.is_available()
    device              = torch.device('cuda' if cuda_available else 'cpu')
    amp_supported       = cuda_available  # torch.amp.autocast works on CUDA
    triton_available    = _has_working_triton()
    compile_supported   = cuda_available and hasattr(torch, 'compile') and triton_available

    info = {
        'cuda_available'   : cuda_available,
        'device'           : device,
        'device_name'      : 'CPU',
        'vram_gb'          : 0.0,
        'cuda_version'     : 'N/A',
        'torch_version'    : torch.__version__,
        'amp_supported'    : amp_supported,
        'triton_available' : triton_available,
        'compile_supported': compile_supported,
    }

    if verbose:
        sep = '═' * 70
        print(f'\n{sep}')
        print('  GPU DIAGNOSTIC REPORT')
        print(sep)

    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        vram  = props.total_memory / 1e9

        info['device_name'] = props.name
        info['vram_gb']     = vram
        info['cuda_version'] = torch.version.cuda

        if verbose:
            print(f'  Status        : ✓ CUDA AVAILABLE — GPU TRAINING ENABLED')
            print(f'  Device        : {props.name}')
            print(f'  VRAM          : {vram:.2f} GB')
            print(f'  CUDA Version  : {torch.version.cuda}')
            print(f'  PyTorch       : {torch.__version__}')
            print(f'  AMP (float16) : {"✓ Enabled" if amp_supported else "✗ Not supported"}')
            if compile_supported:
                print('  torch.compile : ✓ Enabled')
            elif not hasattr(torch, 'compile'):
                print('  torch.compile : ✗ PyTorch < 2.0')
            else:
                print('  torch.compile : ✗ Triton not available')

        # VRAM-based batch-size recommendation
        if verbose:
            if vram >= 16:
                print(f'  VRAM Tier     : High (≥16 GB) — batch_size=1 is conservative; '
                      f'can try 2–4 for faster training')
            elif vram >= 8:
                print(f'  VRAM Tier     : Mid (8–16 GB) — batch_size=1 recommended')
            else:
                print(f'  VRAM Tier     : Low (<8 GB) — batch_size=1 required; '
                      f'reduce IMG_SIZE to 128 if OOM')
    else:
        if verbose:
            print(f'  Status        : ✗ CUDA NOT AVAILABLE — CPU TRAINING')
            print(f'  PyTorch       : {torch.__version__}')
            print(f'  Tip           : Install CUDA-enabled PyTorch:')
            print(f'                  pip install torch torchvision '
                  f'--index-url https://download.pytorch.org/whl/cu121')

    if verbose:
        print(sep + '\n')
    return info


# Run GPU detection once at module load. On Windows DataLoader workers import
# this module too, so keep verbose output only in the main process.
_IS_MAIN_PROCESS = mp.current_process().name == 'MainProcess'
GPU_INFO = detect_and_report_gpu(verbose=_IS_MAIN_PROCESS)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: CONFIGURATION
# Purpose: Central, reproducible configuration object. All hyperparameters
#          are documented with their clinical or technical rationale.
#          GPU-specific settings are derived from GPU_INFO above.
# ══════════════════════════════════════════════════════════════════════════════

class Config:
    """
    Central configuration for the CycleGAN training pipeline.

    GPU OPTIMIZATION SETTINGS:
    - USE_AMP: Mixed precision training. Enabled automatically when CUDA is
      available. ~50–100% speedup on Ampere+ GPUs (RTX 30xx, A100, etc.)
      with no loss in model quality.
    - USE_COMPILE: torch.compile() using TorchInductor/Triton backend.
      Enabled for PyTorch 2.0+ on CUDA. First epoch is slower (compilation),
      subsequent epochs ~20–30% faster.
    - REPRODUCIBLE_MODE: When True, sets cudnn.deterministic=True and
      benchmark=False. Guarantees bit-exact reproducibility across runs
      at the cost of ~10–15% speed. Use True for final paper submissions,
      False for exploratory training.

    MEDICAL DESIGN DECISIONS:
    - LAMBDA_CYCLE=10.0 : Prevents hallucination of non-existent anatomy.
    - LAMBDA_IDENTITY=5.0: Preserves bone density / lung margin signals.
    - LAMBDA_STRUCTURAL=1.0: VGG perceptual loss for texture fidelity.
    - BATCH_SIZE=1: Required for Instance Normalization stability.
    """

    # ── Dataset ────────────────────────────────────────────────────────────────
    DATA_DIR        = './dataset'
    TB_DIR          = 'TB'
    NORMAL_DIR      = 'Normal'
    IMG_SIZE        = 128  # 256 is ideal for texture detail but may require batch_size=1 on 8 GB VRAM
    VALID_EXTS      = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

    # ── Output Directories ─────────────────────────────────────────────────────
    CHECKPOINT_DIR  = './checkpoints'
    RESULTS_DIR     = './results'
    LOG_DIR         = './logs'

    # ── Hardware (auto-configured from GPU_INFO) ───────────────────────────────
    DEVICE          = GPU_INFO['device']
    # PIN_MEMORY: Pins CPU tensors in page-locked memory for faster GPU transfer.
    # Only beneficial with CUDA — adds overhead on CPU-only systems.
    PIN_MEMORY      = GPU_INFO['cuda_available']
    # NUM_WORKERS: Parallel data loading subprocesses. 0 on CPU avoids
    # multiprocessing overhead. On GPU, 4 workers keep the GPU fed.
    NUM_WORKERS     = 4 if GPU_INFO['cuda_available'] else 0

    # ── GPU Optimization Flags ─────────────────────────────────────────────────
    # AMP: Mixed precision — float16 for forward pass, float32 for weights.
    USE_AMP         = GPU_INFO['amp_supported']
    # torch.compile: JIT-compiles the model for Triton/CUDA kernel fusion.
    USE_COMPILE     = GPU_INFO['compile_supported']
    # REPRODUCIBLE_MODE: True = deterministic (paper), False = fast (experiments).
    # Can be overridden via --reproducible CLI flag.
    REPRODUCIBLE_MODE = False

    # ── Architecture ───────────────────────────────────────────────────────────
    # 9 residual blocks for 256×256+ (6 blocks sufficient for 128×128).
    N_RESIDUAL_BLOCKS = 6 
    N_DISC_LAYERS     = 3   # 3 layers → 70×70 PatchGAN receptive field
    N_GEN_FILTERS     = 64
    N_DISC_FILTERS    = 64

    # ── Loss Weights (CRITICAL for medical validity) ───────────────────────────
    LAMBDA_CYCLE      = 10.0
    LAMBDA_IDENTITY   = 5.0
    LAMBDA_STRUCTURAL = 1.0

    # ── Optimization ───────────────────────────────────────────────────────────
    LEARNING_RATE   = 0.0002
    BETA1           = 0.5    # GAN-standard Adam beta1 (not default 0.9)
    BETA2           = 0.999
    BATCH_SIZE      = 1      # Must be 1 — Instance Norm requires it
    N_EPOCHS        = 100
    DECAY_EPOCH     = 50    # Linear LR decay begins at this epoch

    # ── Image History Buffer ───────────────────────────────────────────────────
    BUFFER_SIZE     = 50     # Prevents discriminator oscillation

    # ── Logging & Checkpointing ────────────────────────────────────────────────
    LOG_FREQ        = 100    # Print loss every N batches
    SAVE_FREQ       = 25     # Save checkpoint every N epochs
    SAMPLE_FREQ     = 20      # Save sample images every N epochs

    # ── Reproducibility ────────────────────────────────────────────────────────
    SEED            = 42

    # ── Evaluation ─────────────────────────────────────────────────────────────
    N_EVAL_IMAGES   = 50


def set_reproducibility(seed: int = Config.SEED, reproducible: bool = Config.REPRODUCIBLE_MODE) -> None:
    """
    Configure all random seeds and CUDA determinism settings.

    Args:
        seed: Random seed for Python, NumPy, and PyTorch.
        reproducible: If True, enables full determinism (slower but exact).
                      If False, enables cudnn.benchmark for speed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if reproducible:
            # Full determinism: every run produces identical outputs.
            # ~10–15% slower than benchmark mode.
            # USE THIS for final paper experiments.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark     = False
            print(f'[Reproducibility] STRICT mode — deterministic=True, benchmark=False, seed={seed}')
        else:
            # benchmark=True: cuDNN auto-selects the fastest conv algorithm
            # for the fixed 256×256 input size. ~10–15% faster than strict mode.
            # NOTE: Results may differ by floating-point epsilon between runs.
            # USE THIS for exploratory training and hyperparameter sweeps.
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark     = True
            print(f'[Reproducibility] FAST mode — benchmark=True, seed={seed}')
    else:
        print(f'[Reproducibility] CPU mode — seed={seed}')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: DATA PIPELINE
# Purpose: Robust loading with medical-grade preprocessing. Augmentations
#          are restricted to clinically plausible transformations only.
# ══════════════════════════════════════════════════════════════════════════════

def validate_dataset(data_dir: str) -> dict:
    """
    Validate dataset directory structure and report statistics.
    Raises informative errors before any GPU memory is allocated.
    """
    data_dir   = Path(data_dir)
    tb_dir     = data_dir / Config.TB_DIR
    normal_dir = data_dir / Config.NORMAL_DIR

    if not data_dir.exists():
        raise FileNotFoundError(f'Dataset directory not found: {data_dir}')
    if not tb_dir.exists():
        raise FileNotFoundError(
            f'TB subdirectory not found: {tb_dir}\n'
            f'Expected: {data_dir}/TB/ and {data_dir}/Normal/'
        )
    if not normal_dir.exists():
        raise FileNotFoundError(f'Normal subdirectory not found: {normal_dir}')

    def collect_images(directory: Path) -> list:
        return sorted([
            p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in Config.VALID_EXTS
        ])

    tb_paths     = collect_images(tb_dir)
    normal_paths = collect_images(normal_dir)

    if not tb_paths:
        raise ValueError(f'No valid images found in {tb_dir}')
    if not normal_paths:
        raise ValueError(f'No valid images found in {normal_dir}')

    ratio = len(tb_paths) / len(normal_paths)
    imbalance_warning = ''
    if ratio > 3.0 or ratio < 0.33:
        imbalance_warning = (
            f'\n  ⚠ WARNING: Severe class imbalance (TB/Normal={ratio:.2f}). '
            'May cause generator to favor one domain.'
        )

    print(f'\n[Dataset Validation] {"─"*47}')
    print(f'  Root     : {data_dir.resolve()}')
    print(f'  TB       : {len(tb_paths)} images')
    print(f'  Normal   : {len(normal_paths)} images')
    print(f'  Ratio    : {ratio:.2f}{imbalance_warning}')
    print(f'{"─"*60}\n')

    return {'tb_paths': tb_paths, 'normal_paths': normal_paths}


def build_transforms(mode: str = 'train', img_size: int = Config.IMG_SIZE) -> transforms.Compose:
    """
    Build image preprocessing transforms.

    MEDICAL AUGMENTATION POLICY:
    Allowed  — rotation ±10° (patient positioning), H-flip (texture learning),
               minimal brightness/contrast ±5% (scanner variance).
    Forbidden — vertical flip (anatomically impossible), strong color jitter
               (X-ray intensity is clinically meaningful), random erasing
               (destroys pathology signals).
    """
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize(int(img_size * 1.12), Image.BICUBIC),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10, fill=0),
            transforms.ColorJitter(brightness=0.05, contrast=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])


class UnpairedXRayDataset(Dataset):
    """
    Unpaired TB ↔ Normal chest X-ray dataset.

    Uses wrap-around indexing so both domains are covered each epoch even
    when their sizes differ, without dropping samples from either domain.
    Images are loaded as RGB — correct for grayscale X-rays because
    ImageNet-pretrained VGG16 (used in structural loss) expects 3 channels.
    """

    def __init__(self, tb_paths: list, normal_paths: list, transform=None):
        self.tb_paths     = tb_paths
        self.normal_paths = normal_paths
        self.transform    = transform
        self._len         = max(len(tb_paths), len(normal_paths))

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict:
        tb_path     = self.tb_paths[idx % len(self.tb_paths)]
        normal_path = self.normal_paths[idx % len(self.normal_paths)]

        tb_img     = self._load_image(tb_path)
        normal_img = self._load_image(normal_path)

        if self.transform:
            tb_img     = self.transform(tb_img)
            normal_img = self.transform(normal_img)

        return {
            'TB'          : tb_img,
            'Normal'      : normal_img,
            'TB_path'     : str(tb_path),
            'Normal_path' : str(normal_path),
        }

    @staticmethod
    def _load_image(path: Path) -> Image.Image:
        try:
            return Image.open(path).convert('RGB')
        except Exception as e:
            raise IOError(f'Failed to load image {path}: {e}')


def build_dataloaders(data_dir: str) -> tuple:
    """
    Build train and test DataLoaders with independent per-domain 80/20 split.

    Independent splitting prevents cross-domain data leakage (where a test
    TB image could implicitly be seen alongside a training Normal image).

    Returns:
        (train_loader, test_loader, dataset_stats_dict)
    """
    info         = validate_dataset(data_dir)
    tb_paths     = info['tb_paths']
    normal_paths = info['normal_paths']

    rng = random.Random(Config.SEED)
    rng.shuffle(tb_paths)
    rng.shuffle(normal_paths)

    def split80(paths):
        n = int(len(paths) * 0.8)
        return paths[:n], paths[n:]

    tb_train,     tb_test     = split80(tb_paths)
    normal_train, normal_test = split80(normal_paths)

    train_dataset = UnpairedXRayDataset(tb_train, normal_train, build_transforms('train'))
    test_dataset  = UnpairedXRayDataset(tb_test,  normal_test,  build_transforms('test'))

    train_loader = DataLoader(
        train_dataset,
        batch_size  = Config.BATCH_SIZE,
        shuffle     = True,
        num_workers = Config.NUM_WORKERS,
        pin_memory  = Config.PIN_MEMORY,
        drop_last   = True,
        # persistent_workers avoids re-spawning worker processes each epoch.
        # Only enable when num_workers > 0 (i.e., on GPU).
        persistent_workers = (Config.NUM_WORKERS > 0),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = 1,
        shuffle     = False,
        num_workers = Config.NUM_WORKERS,
        pin_memory  = Config.PIN_MEMORY,
        persistent_workers = (Config.NUM_WORKERS > 0),
    )

    print(f'[DataLoader] Train: {len(train_dataset)} | Test: {len(test_dataset)} | '
          f'Workers: {Config.NUM_WORKERS} | pin_memory: {Config.PIN_MEMORY}')
    return train_loader, test_loader, {
        'tb_train': len(tb_train), 'tb_test': len(tb_test),
        'normal_train': len(normal_train), 'normal_test': len(normal_test),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: ARCHITECTURE
# Purpose: ResNet generator and PatchGAN discriminator with Instance
#          Normalization. Architecture is compatible with AMP (float16).
# ══════════════════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """
    Residual block with reflection padding and Instance Normalization.

    REFLECTION PADDING: Prevents artificial black-border artifacts that
    discriminators exploit, especially at lung-boundary edges.

    INSTANCE NORM: Normalizes per-image statistics. Required for batch_size=1
    and enables style-transfer (pathology texture ↔ normal texture) while
    preserving content (anatomy structure).

    SKIP CONNECTION: The additive shortcut ensures that unmodified anatomy
    passes through unchanged — the block only learns the pathology delta.

    AMP COMPATIBILITY: All operations (Conv2d, InstanceNorm2d, ReLU) are
    float16-compatible. No explicit dtype casting needed here.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0, bias=False),
            nn.InstanceNorm2d(channels, affine=False),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0, bias=False),
            nn.InstanceNorm2d(channels, affine=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    """
    ResNet-9 generator: Encoder → 9 Residual Blocks → Decoder.

    WHY RESNET OVER U-NET: U-Net skip connections allow the generator to
    pass pixel values directly from input to output, bypassing the bottleneck.
    In a CycleGAN, this creates an identity shortcut — the network can
    output the original image unmodified and still minimize reconstruction
    loss. ResNet forces all information through the bottleneck, preventing
    this. Medical anatomy is preserved via identity/cycle losses, not skip
    connections.

    WHY TANH OUTPUT: Normalizes output to [-1, 1], matching the input
    normalization. Visualize as (output + 1) / 2 → [0, 1].

    GPU NOTE: torch.compile() wraps the entire model including this forward
    pass. Triton generates fused kernels for conv+norm+relu sequences,
    reducing kernel launch overhead and memory bandwidth.
    """

    def __init__(
        self,
        in_channels  : int = 3,
        out_channels : int = 3,
        n_filters    : int = Config.N_GEN_FILTERS,
        n_blocks     : int = Config.N_RESIDUAL_BLOCKS,
    ):
        super().__init__()
        assert n_blocks > 0

        # Initial 7×7 convolution
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, n_filters, kernel_size=7, padding=0, bias=False),
            nn.InstanceNorm2d(n_filters, affine=False),
            nn.ReLU(inplace=True),
        ]

        # Encoder: 2× stride-2 downsampling
        for i in range(2):
            mult = 2 ** i
            model += [
                nn.Conv2d(n_filters * mult, n_filters * mult * 2,
                          kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(n_filters * mult * 2, affine=False),
                nn.ReLU(inplace=True),
            ]

        # Residual bottleneck (operates at 64×64 for 256 input)
        mult = 4
        for _ in range(n_blocks):
            model.append(ResidualBlock(n_filters * mult))

        # Decoder: 2× stride-2 upsampling via transposed conv
        for i in range(2):
            mult = 2 ** (2 - i)
            model += [
                nn.ConvTranspose2d(n_filters * mult, n_filters * mult // 2,
                                   kernel_size=3, stride=2, padding=1,
                                   output_padding=1, bias=False),
                nn.InstanceNorm2d(n_filters * mult // 2, affine=False),
                nn.ReLU(inplace=True),
            ]

        # Output 7×7 convolution + Tanh
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(n_filters, out_channels, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)
        self._init_weights()

    def _init_weights(self) -> None:
        """N(0, 0.02) weight init — standard for GANs (Radford et al., 2015)."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class PatchGANDiscriminator(nn.Module):
    """
    70×70 PatchGAN discriminator.

    WHY PATCHGAN: Evaluates ~70×70 pixel patches rather than the whole image.
    For TB detection, this is critical: the discriminator must assess whether
    local pathological textures (fibrotic streaks, cavity walls, consolidations)
    have been plausibly translated — not just whether the global composition
    looks right. A global discriminator would memorize patient-level features
    (heart size, rib geometry) in a small dataset.

    WHY NO SIGMOID: LSGAN loss uses MSELoss, which does not require [0,1]
    output. Removing sigmoid improves gradient flow in early training.

    GPU NOTE: LeakyReLU with inplace=True reduces memory allocation by
    ~1 tensor per layer — meaningful at batch_size=1 with many patches.
    """

    def __init__(
        self,
        in_channels : int = 3,
        n_filters   : int = Config.N_DISC_FILTERS,
        n_layers    : int = Config.N_DISC_LAYERS,
    ):
        super().__init__()

        # First layer: no normalization (per PatchGAN paper)
        layers = [
            nn.Conv2d(in_channels, n_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf = n_filters
        for _ in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            layers += [
                nn.Conv2d(nf_prev, nf, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(nf, affine=False),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        # Stride-1 penultimate layer
        nf_prev = nf
        nf = min(nf * 2, 512)
        layers += [
            nn.Conv2d(nf_prev, nf, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(nf, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, 1, kernel_size=4, stride=1, padding=1),
            # No sigmoid — LSGAN uses raw logits
        ]

        self.model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

class LSGANLoss(nn.Module):
    """
    Least-Squares GAN adversarial loss (Mao et al., 2017).

    WHY LSGAN OVER BCE-GAN: BCE saturates when D is confident (loss→0,
    gradient→0). With a small medical dataset, D becomes confident quickly.
    LSGAN's MSE loss penalizes generated samples proportional to their
    distance from the decision boundary, maintaining non-zero gradients
    throughout training. This reduces mode collapse — critical here because
    TB presentations vary (apical, miliary, pleural) and the generator must
    learn all modes.

    AMP COMPATIBILITY: nn.MSELoss is float16-safe.
    """

    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def discriminator_loss(self, real_pred: torch.Tensor,
                           fake_pred: torch.Tensor) -> torch.Tensor:
        real_target = torch.ones_like(real_pred)
        fake_target = torch.zeros_like(fake_pred)
        return 0.5 * (self.criterion(real_pred, real_target) +
                      self.criterion(fake_pred, fake_target))

    def generator_loss(self, fake_pred: torch.Tensor) -> torch.Tensor:
        return self.criterion(fake_pred, torch.ones_like(fake_pred))


class StructuralLoss(nn.Module):
    """
    Multi-scale VGG16 perceptual loss for anatomical texture preservation.

    WHY PERCEPTUAL LOSS: Pixel-L1 loss treats all pixel differences equally.
    Structural fidelity — rib spacing, lung boundary curvature, heart
    silhouette — matters more clinically than absolute pixel values. VGG16
    features encode edges (Block 1), textures (Block 2), and higher-level
    structures (Block 3), providing a richer structural similarity signal.

    WHY FROZEN VGG: If VGG weights were trainable, the definition of
    "structural similarity" would shift during training, creating an unstable
    optimization target. Frozen weights ensure consistent semantics.

    NORMALIZATION: X-rays in [-1,1] are renormalized to ImageNet statistics
    before VGG inference. This is necessary because VGG was trained on
    ImageNet-normalized RGB images.

    GPU / AMP: VGG is frozen and cast to float32 internally by autocast.
    The register_buffer call ensures vgg_mean/std move with .to(device).
    """

    def __init__(self, device: torch.device = Config.DEVICE):
        super().__init__()
        self.device = device

        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.vgg_blocks = nn.ModuleList([
            vgg[:4].eval(),    # Block 1: low-level edges
            vgg[4:9].eval(),   # Block 2: mid-level textures
            vgg[9:16].eval(),  # Block 3: higher-level structures
        ])
        for param in self.parameters():
            param.requires_grad_(False)
        self.to(device)

        self.register_buffer('vgg_mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('vgg_std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.l1 = nn.L1Loss()

    def _normalize_for_vgg(self, x: torch.Tensor) -> torch.Tensor:
        # AMP note: upcast to float32 for VGG inference — VGG is frozen and
        # its weights are float32. autocast handles this automatically.
        x = (x + 1.0) / 2.0
        # Ensure buffers are on the same device as input (handles GPU/CPU mismatch in autocast)
        vgg_mean = self.vgg_mean.to(x.device)
        vgg_std = self.vgg_std.to(x.device)
        return (x - vgg_mean) / vgg_std

    def forward(self, real: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
        real_n = self._normalize_for_vgg(real)
        gen_n  = self._normalize_for_vgg(generated)

        loss   = torch.tensor(0.0, device=self.device)
        r_feat = real_n
        g_feat = gen_n

        for block in self.vgg_blocks:
            r_feat = block(r_feat)
            g_feat = block(g_feat)
            loss  += self.l1(r_feat, g_feat)

        return loss / len(self.vgg_blocks)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: IMAGE HISTORY BUFFER
# Purpose: Reduces discriminator oscillation by exposing it to a mix of
#          recent and historical fake images (Shrivastava et al., CVPR 2017).
# ══════════════════════════════════════════════════════════════════════════════

class ImageHistoryBuffer:
    """
    Rolling buffer of generated images for stable discriminator training.

    Without this buffer, D updates based only on the current G output.
    If G improves suddenly, D over-corrects, causing oscillation.
    The buffer introduces "memory" — D must reject all historical fakes,
    not just the latest ones, smoothing the optimization landscape.

    GPU NOTE: Images are stored as detached CPU tensors to avoid holding
    VRAM across steps. They are moved back to device only when queried.
    """

    def __init__(self, max_size: int = Config.BUFFER_SIZE):
        self.max_size = max_size
        self.buffer   = []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        if self.max_size == 0:
            return images

        return_images = []
        for image in images.unbind(0):
            image = image.unsqueeze(0)
            if len(self.buffer) < self.max_size:
                self.buffer.append(image.detach().cpu())
                return_images.append(image)
            else:
                if random.random() > 0.5:
                    idx = random.randint(0, self.max_size - 1)
                    # Move historical image to the current device before returning
                    hist_img = self.buffer[idx].to(image.device)
                    return_images.append(hist_img)
                    self.buffer[idx] = image.detach().cpu()
                else:
                    return_images.append(image)

        return torch.cat(return_images, dim=0)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: LEARNING RATE SCHEDULER
# Purpose: Linear LR decay starting at DECAY_EPOCH.
#          Canonical CycleGAN schedule (Zhu et al., 2017).
# ══════════════════════════════════════════════════════════════════════════════

class LinearDecayLR:
    """
    Constant LR for first half of training, linear decay to 0 for second half.

    Epochs [0, DECAY_EPOCH)          : LR = LEARNING_RATE
    Epochs [DECAY_EPOCH, N_EPOCHS]   : LR decays linearly → 0

    This schedule prevents large weight updates late in training when the
    generators are already producing medically plausible counterfactuals.
    """

    def __init__(self, n_epochs: int, decay_epoch: int, initial_lr: float):
        self.n_epochs    = n_epochs
        self.decay_epoch = decay_epoch
        self.initial_lr  = initial_lr

    def get_lr_factor(self, epoch: int) -> float:
        if epoch < self.decay_epoch:
            return 1.0
        progress = (epoch - self.decay_epoch) / (self.n_epochs - self.decay_epoch)
        return max(0.0, 1.0 - progress)

    def update(self, optimizer: torch.optim.Optimizer, epoch: int) -> float:
        new_lr = self.initial_lr * self.get_lr_factor(epoch)
        for pg in optimizer.param_groups:
            pg['lr'] = new_lr
        return new_lr


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: CYCLEGAN TRAINING SYSTEM
# Purpose: Encapsulates all 4 networks, losses, optimizers, GradScalers,
#          and the GPU-optimized training step in a single resumable class.
# ══════════════════════════════════════════════════════════════════════════════

class CycleGANSystem:
    """
    Full CycleGAN with dual generators, discriminators, and GPU optimizations.

    NOTATION (Zhu et al., 2017):
        Domain X = TB-positive chest X-rays
        Domain Y = Healthy (Normal) chest X-rays
        G        = Generator X→Y  (TB → Healthy counterfactual)
        F        = Generator Y→X  (Healthy → TB reconstruction)
        D_X      = Discriminator for domain X (real vs fake TB)
        D_Y      = Discriminator for domain Y (real vs fake Healthy)

    GPU OPTIMIZATIONS IN THIS CLASS:
    1. torch.compile()  : Applied to G, F, D_X, D_Y after construction.
                          First epoch is ~2× slower (compilation overhead),
                          all subsequent epochs are ~20–30% faster.
    2. Mixed Precision  : Forward pass runs in float16 via autocast context.
                          Weights remain in float32. ~50–100% speedup on
                          Ampere+ GPUs (RTX 30xx, A100, H100).
    3. GradScaler       : Scales loss before backward() to prevent float16
                          underflow (gradients near zero can round to 0 in
                          float16). Unscaled before gradient clipping.
    4. Separate scalers : G and D use independent GradScalers so their
                          scale factors adapt independently.
    5. Gradient clipping: max_norm=1.0 prevents exploding gradients in early
                          epochs before loss landscape stabilizes.

    LOSS TERMS:
        Total_G = L_GAN(G,D_Y) + L_GAN(F,D_X)          ← Adversarial
                + λ_cycle * [||F(G(x))-x||₁ + ||G(F(y))-y||₁]  ← Cycle
                + λ_ident * [||G(y)-y||₁   + ||F(x)-x||₁  ]    ← Identity
                + λ_struct * L_VGG(real_X, G(real_X))           ← Structural
    """

    def __init__(self, device: torch.device = Config.DEVICE):
        self.device = device

        # ── Build Networks ─────────────────────────────────────────────────────
        self.G   = ResNetGenerator().to(device)
        self.F   = ResNetGenerator().to(device)
        self.D_X = PatchGANDiscriminator().to(device)
        self.D_Y = PatchGANDiscriminator().to(device)

        # ── GPU OPTIMIZATION 1: torch.compile() ───────────────────────────────
        # torch.compile() uses TorchInductor to JIT-compile the model into
        # optimized Triton/CUDA kernels. It fuses conv+norm+relu sequences,
        # reducing kernel launch overhead and memory bandwidth.
        #
        # mode='reduce-overhead': Optimizes for repeated same-shape inputs —
        #   perfect for our fixed 256×256 training images.
        #   Alternative: mode='max-autotune' for maximum speed at the cost of
        #   a longer initial compilation (~5 min) and more VRAM.
        #
        # IMPORTANT: compile() wraps the module. The wrapped object supports
        # .state_dict() / .load_state_dict() normally for checkpointing.
        # However, we save the original (unwrapped) module's state_dict to
        # avoid _orig_mod prefix issues across PyTorch versions.
        if Config.USE_COMPILE:
            print('[GPU] Compiling networks with torch.compile(mode="reduce-overhead")...')
            print('      (First epoch will be slower — this is normal)')
            self.G   = torch.compile(self.G,   mode='reduce-overhead')
            self.F   = torch.compile(self.F,   mode='reduce-overhead')
            self.D_X = torch.compile(self.D_X, mode='reduce-overhead')
            self.D_Y = torch.compile(self.D_Y, mode='reduce-overhead')
            print('[GPU] Compilation registered. Kernels will be built on first forward pass.\n')
        else:
            reason = 'CUDA unavailable or PyTorch < 2.0'
            if GPU_INFO['cuda_available'] and hasattr(torch, 'compile') and not GPU_INFO['triton_available']:
                reason = 'Triton not available'
            print(f'[GPU] torch.compile() skipped ({reason}).')

        self._compile_runtime_fallback_used = False

        # ── Loss Functions ─────────────────────────────────────────────────────
        self.criterion_adv    = LSGANLoss()
        self.criterion_cycle  = nn.L1Loss()
        self.criterion_ident  = nn.L1Loss()
        self.criterion_struct = StructuralLoss(device)

        # ── Optimizers ─────────────────────────────────────────────────────────
        # G and F share one optimizer; D_X and D_Y share another.
        # This allows decoupled LR schedules if needed in future ablations.
        self.opt_G = torch.optim.Adam(
            chain(self.G.parameters(), self.F.parameters()),
            lr=Config.LEARNING_RATE, betas=(Config.BETA1, Config.BETA2),
        )
        self.opt_D = torch.optim.Adam(
            chain(self.D_X.parameters(), self.D_Y.parameters()),
            lr=Config.LEARNING_RATE, betas=(Config.BETA1, Config.BETA2),
        )

        # ── GPU OPTIMIZATION 2 & 3: GradScalers for AMP ───────────────────────
        # GradScaler multiplies the loss by a large scale factor before
        # backward(), then divides gradients after backward() and before
        # the optimizer step. This prevents float16 gradient underflow.
        #
        # enabled=False on CPU: GradScaler becomes a transparent no-op,
        # so the training step code is identical for both CPU and GPU.
        self.scaler_G = torch.amp.GradScaler(enabled=Config.USE_AMP)
        self.scaler_D = torch.amp.GradScaler(enabled=Config.USE_AMP)

        if Config.USE_AMP:
            print('[GPU] Mixed Precision (AMP) ENABLED — float16 forward pass active.')
        else:
            print('[GPU] Mixed Precision disabled (CPU or unsupported device).')

        # ── Learning Rate Scheduler ────────────────────────────────────────────
        self.scheduler = LinearDecayLR(
            n_epochs    = Config.N_EPOCHS,
            decay_epoch = Config.DECAY_EPOCH,
            initial_lr  = Config.LEARNING_RATE,
        )

        # ── Image History Buffers ──────────────────────────────────────────────
        self.fake_X_buffer = ImageHistoryBuffer()
        self.fake_Y_buffer = ImageHistoryBuffer()

        # ── Loss History (for plotting and checkpointing) ──────────────────────
        self.epoch_loss_history = {
            'loss_G': [], 'loss_D': [],
            'loss_cycle_X': [], 'loss_cycle_Y': [],
            'loss_ident_X': [], 'loss_ident_Y': [],
            'loss_struct': [],
        }

        # ── Output Directories ─────────────────────────────────────────────────
        for d in [Config.CHECKPOINT_DIR, Config.RESULTS_DIR, Config.LOG_DIR]:
            os.makedirs(d, exist_ok=True)

    def _set_requires_grad(self, nets: list, requires_grad: bool) -> None:
        """
        Toggle gradient computation for a list of networks.
        Freezing D during G update halves the backward-pass memory and time.
        """
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad

    def _handle_compile_runtime_error(self, err: Exception) -> bool:
        """
        Disable torch.compile at runtime if a Triton backend error appears.
        Returns True if the error was handled and caller should retry forward.
        """
        if not Config.USE_COMPILE:
            return False

        msg = str(err).lower()
        triton_missing = (
            'tritonmissing' in msg
            or 'cannot find a working triton installation' in msg
            or ('triton' in msg and 'missing' in msg)
        )
        if not triton_missing:
            return False

        if not self._compile_runtime_fallback_used:
            print('\n[GPU WARNING] Triton backend unavailable during compiled forward pass.')
            print('              Disabling torch.compile and continuing without compilation.')
            self.G = self._unwrap(self.G)
            self.F = self._unwrap(self.F)
            self.D_X = self._unwrap(self.D_X)
            self.D_Y = self._unwrap(self.D_Y)
            Config.USE_COMPILE = False
            self._compile_runtime_fallback_used = True
        return True

    def train_step(self, real_X: torch.Tensor, real_Y: torch.Tensor) -> dict:
        """
        Single GPU-optimized training step.

        AMP STRATEGY:
        - torch.amp.autocast() wraps the forward pass. Inside the context,
          PyTorch automatically selects float16 or float32 for each op:
          * Conv2d, Linear → float16  (2× memory bandwidth, tensor core speedup)
          * BatchNorm, Softmax → float32  (precision-sensitive ops stay in fp32)
        - GradScaler.scale(loss).backward() scales the loss before autograd.
        - GradScaler.unscale_(optimizer) restores true gradient magnitudes
          before clipping (critical — clip against unscaled gradients).
        - GradScaler.step(optimizer) skips the step if scaled gradients contain
          inf/nan (which can appear transiently in float16 arithmetic).
        - GradScaler.update() adjusts the scale factor for the next step.

        Args:
            real_X: Batch of TB images, shape (1, 3, H, W), on CPU.
            real_Y: Batch of Normal images, shape (1, 3, H, W), on CPU.
        Returns:
            dict of scalar loss values for logging.
        """
        # Non-blocking transfer overlaps GPU computation with the next
        # batch's CPU→GPU transfer when PIN_MEMORY=True.
        real_X = real_X.to(self.device, non_blocking=True)
        real_Y = real_Y.to(self.device, non_blocking=True)

        # ══════════════════════════════════════════════════════════════════════
        # GENERATOR UPDATE
        # ══════════════════════════════════════════════════════════════════════
        self._set_requires_grad([self.D_X, self.D_Y], False)
        self.opt_G.zero_grad(set_to_none=True)  # set_to_none saves a memset call

        # autocast: entire forward pass runs in float16 on GPU
        try:
            with torch.amp.autocast(device_type='cuda' if Config.USE_AMP else 'cpu',
                                     enabled=Config.USE_AMP):

                fake_Y  = self.G(real_X)    # TB → Healthy (PRIMARY COUNTERFACTUAL)
                rec_X   = self.F(fake_Y)    # Healthy → TB  (cycle)
                fake_X  = self.F(real_Y)    # Healthy → TB
                rec_Y   = self.G(fake_X)    # TB → Healthy  (cycle)
                ident_X = self.F(real_X)    # TB → TB  (should be identity)
                ident_Y = self.G(real_Y)    # Normal → Normal  (should be identity)

                # 1. Adversarial losses
                loss_adv_G = self.criterion_adv.generator_loss(self.D_Y(fake_Y))
                loss_adv_F = self.criterion_adv.generator_loss(self.D_X(fake_X))

                # 2. Cycle consistency (primary hallucination guard)
                loss_cycle_X = self.criterion_cycle(rec_X, real_X) * Config.LAMBDA_CYCLE
                loss_cycle_Y = self.criterion_cycle(rec_Y, real_Y) * Config.LAMBDA_CYCLE

                # 3. Identity loss (anatomy preservation)
                loss_ident_X = self.criterion_ident(ident_X, real_X) * Config.LAMBDA_IDENTITY
                loss_ident_Y = self.criterion_ident(ident_Y, real_Y) * Config.LAMBDA_IDENTITY

                # 4. Structural perceptual loss (texture fidelity)
                loss_struct = self.criterion_struct(real_X, fake_Y) * Config.LAMBDA_STRUCTURAL

                loss_G = (loss_adv_G + loss_adv_F +
                          loss_cycle_X + loss_cycle_Y +
                          loss_ident_X + loss_ident_Y +
                          loss_struct)
        except Exception as e:
            if not self._handle_compile_runtime_error(e):
                raise
            with torch.amp.autocast(device_type='cuda' if Config.USE_AMP else 'cpu',
                                     enabled=Config.USE_AMP):

                fake_Y  = self.G(real_X)
                rec_X   = self.F(fake_Y)
                fake_X  = self.F(real_Y)
                rec_Y   = self.G(fake_X)
                ident_X = self.F(real_X)
                ident_Y = self.G(real_Y)

                loss_adv_G = self.criterion_adv.generator_loss(self.D_Y(fake_Y))
                loss_adv_F = self.criterion_adv.generator_loss(self.D_X(fake_X))

                loss_cycle_X = self.criterion_cycle(rec_X, real_X) * Config.LAMBDA_CYCLE
                loss_cycle_Y = self.criterion_cycle(rec_Y, real_Y) * Config.LAMBDA_CYCLE

                loss_ident_X = self.criterion_ident(ident_X, real_X) * Config.LAMBDA_IDENTITY
                loss_ident_Y = self.criterion_ident(ident_Y, real_Y) * Config.LAMBDA_IDENTITY

                loss_struct = self.criterion_struct(real_X, fake_Y) * Config.LAMBDA_STRUCTURAL

                loss_G = (loss_adv_G + loss_adv_F +
                          loss_cycle_X + loss_cycle_Y +
                          loss_ident_X + loss_ident_Y +
                          loss_struct)

        # Scale → backward → unscale → clip → step → update
        self.scaler_G.scale(loss_G).backward()
        self.scaler_G.unscale_(self.opt_G)   # Must unscale before clipping
        nn.utils.clip_grad_norm_(
            chain(self.G.parameters(), self.F.parameters()), max_norm=1.0
        )
        self.scaler_G.step(self.opt_G)
        self.scaler_G.update()

        # ══════════════════════════════════════════════════════════════════════
        # DISCRIMINATOR UPDATE
        # ══════════════════════════════════════════════════════════════════════
        self._set_requires_grad([self.D_X, self.D_Y], True)
        self.opt_D.zero_grad(set_to_none=True)

        try:
            with torch.amp.autocast(device_type='cuda' if Config.USE_AMP else 'cpu',
                                     enabled=Config.USE_AMP):
                # History buffer prevents discriminator memorizing only the latest fakes
                fake_Y_hist = self.fake_Y_buffer.query(fake_Y.detach())
                fake_X_hist = self.fake_X_buffer.query(fake_X.detach())

                loss_D_Y = self.criterion_adv.discriminator_loss(
                    self.D_Y(real_Y), self.D_Y(fake_Y_hist)
                )
                loss_D_X = self.criterion_adv.discriminator_loss(
                    self.D_X(real_X), self.D_X(fake_X_hist)
                )
                loss_D = 0.5 * (loss_D_X + loss_D_Y)
        except Exception as e:
            if not self._handle_compile_runtime_error(e):
                raise
            with torch.amp.autocast(device_type='cuda' if Config.USE_AMP else 'cpu',
                                     enabled=Config.USE_AMP):
                fake_Y_hist = self.fake_Y_buffer.query(fake_Y.detach())
                fake_X_hist = self.fake_X_buffer.query(fake_X.detach())

                loss_D_Y = self.criterion_adv.discriminator_loss(
                    self.D_Y(real_Y), self.D_Y(fake_Y_hist)
                )
                loss_D_X = self.criterion_adv.discriminator_loss(
                    self.D_X(real_X), self.D_X(fake_X_hist)
                )
                loss_D = 0.5 * (loss_D_X + loss_D_Y)

        self.scaler_D.scale(loss_D).backward()
        self.scaler_D.unscale_(self.opt_D)
        nn.utils.clip_grad_norm_(
            chain(self.D_X.parameters(), self.D_Y.parameters()), max_norm=1.0
        )
        self.scaler_D.step(self.opt_D)
        self.scaler_D.update()

        return {
            'loss_G'      : loss_G.item(),
            'loss_D'      : loss_D.item(),
            'loss_cycle_X': loss_cycle_X.item(),
            'loss_cycle_Y': loss_cycle_Y.item(),
            'loss_ident_X': loss_ident_X.item(),
            'loss_ident_Y': loss_ident_Y.item(),
            'loss_struct' : loss_struct.item(),
        }

    def generate_counterfactual(self, real_tb: torch.Tensor) -> dict:
        """
        Generate a healthy counterfactual for a TB X-ray at inference time.

        DIFFERENCE MAP:
        The pixel-wise absolute difference |real_tb - fake_healthy| is the
        primary clinical output. High-intensity regions = features the model
        associates with TB (apical opacities, cavities, consolidations).
        Averaged over 3 channels → single-channel explanation map.

        Args:
            real_tb: TB image tensor (1, 3, H, W) in [-1, 1], on any device.
        Returns:
            dict: 'healthy' (generated image), 'diff_map' (explanation).
        """
        # Use eval mode for inference — disables dropout/batchnorm stochasticity
        self.G.eval()
        with torch.no_grad():
            real_tb = real_tb.to(self.device, non_blocking=True)
            # AMP for inference too — faster generation
            with torch.amp.autocast(device_type='cuda' if Config.USE_AMP else 'cpu',
                                     enabled=Config.USE_AMP):
                fake_healthy = self.G(real_tb)

            diff     = torch.abs(real_tb.float() - fake_healthy.float())
            diff_map = diff.mean(dim=1, keepdim=True)  # (1, 1, H, W)

        self.G.train()
        return {'healthy': fake_healthy, 'diff_map': diff_map}

    def _unwrap(self, model):
        """
        Unwrap a torch.compile()-wrapped model to access its raw state_dict.
        torch.compile wraps the module as _orig_mod — this extracts it safely.
        """
        if hasattr(model, '_orig_mod'):
            return model._orig_mod
        return model

    def save_checkpoint(self, epoch: int, extra_info: dict = None) -> str:
        """
        Save full model state for resumable training.

        COMPILE NOTE: We save the UNWRAPPED module's state_dict.
        This ensures checkpoints are portable across PyTorch versions
        and can be loaded without torch.compile() available.
        """
        checkpoint = {
            'epoch'       : epoch,
            'G_state'     : self._unwrap(self.G).state_dict(),
            'F_state'     : self._unwrap(self.F).state_dict(),
            'D_X_state'   : self._unwrap(self.D_X).state_dict(),
            'D_Y_state'   : self._unwrap(self.D_Y).state_dict(),
            'opt_G_state' : self.opt_G.state_dict(),
            'opt_D_state' : self.opt_D.state_dict(),
            # Save GradScaler state so AMP scale factor resumes correctly
            'scaler_G_state': self.scaler_G.state_dict(),
            'scaler_D_state': self.scaler_D.state_dict(),
            'loss_history': self.epoch_loss_history,
            'config': {
                'n_residual_blocks': Config.N_RESIDUAL_BLOCKS,
                'n_gen_filters'    : Config.N_GEN_FILTERS,
                'lambda_cycle'     : Config.LAMBDA_CYCLE,
                'lambda_identity'  : Config.LAMBDA_IDENTITY,
                'lambda_structural': Config.LAMBDA_STRUCTURAL,
                'use_amp'          : Config.USE_AMP,
                'use_compile'      : Config.USE_COMPILE,
            },
        }
        if extra_info:
            checkpoint.update(extra_info)

        path = os.path.join(Config.CHECKPOINT_DIR, f'epoch_{epoch:04d}.pth')
        torch.save(checkpoint, path)
        print(f'  [Checkpoint] Saved → {path}')
        return path

    def load_checkpoint(self, path: str) -> int:
        """
        Load checkpoint. Returns the resumed epoch number.
        Safely handles checkpoints saved with or without torch.compile().
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f'Checkpoint not found: {path}')

        checkpoint = torch.load(path, map_location=self.device)

        # Load into unwrapped models to handle compile() wrapping
        self._unwrap(self.G).load_state_dict(checkpoint['G_state'])
        self._unwrap(self.F).load_state_dict(checkpoint['F_state'])
        self._unwrap(self.D_X).load_state_dict(checkpoint['D_X_state'])
        self._unwrap(self.D_Y).load_state_dict(checkpoint['D_Y_state'])
        self.opt_G.load_state_dict(checkpoint['opt_G_state'])
        self.opt_D.load_state_dict(checkpoint['opt_D_state'])

        # Restore GradScaler state (scale factor, growth interval, etc.)
        if 'scaler_G_state' in checkpoint:
            self.scaler_G.load_state_dict(checkpoint['scaler_G_state'])
            self.scaler_D.load_state_dict(checkpoint['scaler_D_state'])

        self.epoch_loss_history = checkpoint.get('loss_history', self.epoch_loss_history)
        epoch = checkpoint['epoch']
        print(f'  [Checkpoint] Resumed from epoch {epoch} ← {path}')
        return epoch

    def print_model_summary(self) -> None:
        """Print parameter counts and active GPU optimizations."""
        def count_params(model):
            m = self._unwrap(model)
            total     = sum(p.numel() for p in m.parameters())
            trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
            return total, trainable

        print(f'\n{"="*70}')
        print(f'  CycleGAN Model Summary')
        print(f'{"="*70}')
        total_all = 0
        for name, model in [
            ('G  (TB→Healthy)', self.G),
            ('F  (Healthy→TB)', self.F),
            ('D_X (TB Disc.) ', self.D_X),
            ('D_Y (Norm Disc.)', self.D_Y),
        ]:
            total, trainable = count_params(model)
            total_all += total
            print(f'  {name:<25}: {total/1e6:.2f}M params  ({trainable/1e6:.2f}M trainable)')
        print(f'  {"─"*50}')
        print(f'  {"Total":<25}: {total_all/1e6:.2f}M params')
        print(f'\n  GPU Optimizations Active:')
        print(f'    AMP (float16)  : {"✓" if Config.USE_AMP     else "✗"}')
        print(f'    torch.compile(): {"✓" if Config.USE_COMPILE else "✗"}')
        print(f'    cudnn.benchmark: {"✓" if not Config.REPRODUCIBLE_MODE else "✗ (reproducible mode)"}')
        print(f'    pin_memory     : {"✓" if Config.PIN_MEMORY  else "✗"}')
        print(f'    workers        : {Config.NUM_WORKERS}')
        print(f'{"="*70}\n')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9: TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train(
    system       : CycleGANSystem,
    train_loader : DataLoader,
    test_loader  : DataLoader,
    resume_epoch : int = 0,
) -> None:
    """
    Full training loop with logging, GPU monitoring, and checkpointing.

    VRAM MONITORING: Logs peak VRAM usage after the first batch. If this
    exceeds your GPU's total VRAM, reduce IMG_SIZE to 128 or batch_size to 1.
    """
    print(f'\n{"="*70}')
    print(f'  Training: {Config.N_EPOCHS} epochs | '
          f'Device: {Config.DEVICE} | AMP: {Config.USE_AMP}')
    print(f'  λ_cycle={Config.LAMBDA_CYCLE} | '
          f'λ_identity={Config.LAMBDA_IDENTITY} | '
          f'λ_structural={Config.LAMBDA_STRUCTURAL}')
    print(f'{"="*70}\n')

    log_path     = os.path.join(Config.LOG_DIR, 'training_log.json')
    training_log = []
    first_batch  = True
    training_start_global = time.time()  # Track total training time

    for epoch in range(resume_epoch, Config.N_EPOCHS):
        epoch_start = time.time()

        new_lr = system.scheduler.update(system.opt_G, epoch)
        system.scheduler.update(system.opt_D, epoch)

        epoch_losses = {k: 0.0 for k in system.epoch_loss_history}
        n_batches    = 0

        for batch_idx, batch in enumerate(train_loader):
            losses = system.train_step(batch['TB'], batch['Normal'])

            # ── Log peak VRAM after first batch ────────────────────────────────
            if first_batch and GPU_INFO['cuda_available']:
                peak_vram = torch.cuda.max_memory_allocated() / 1e9
                total_vram = GPU_INFO['vram_gb']
                print(f'  [VRAM] Peak after first batch: '
                      f'{peak_vram:.2f} GB / {total_vram:.2f} GB '
                      f'({100*peak_vram/total_vram:.1f}% utilization)')
                if peak_vram > total_vram * 0.95:
                    print('  ⚠ WARNING: Near VRAM limit. Consider reducing '
                          'IMG_SIZE to 128 if you get OOM errors.')
                first_batch = False

            for k, v in losses.items():
                epoch_losses[k] += v
            n_batches += 1

            if (batch_idx + 1) % Config.LOG_FREQ == 0:
                print(
                    f'  [{epoch+1:03d}/{Config.N_EPOCHS}] '
                    f'[{batch_idx+1:04d}/{len(train_loader)}] '
                    f'LR={new_lr:.6f} | '
                    f'G={losses["loss_G"]:.4f} '
                    f'D={losses["loss_D"]:.4f} '
                    f'CycX={losses["loss_cycle_X"]:.4f} '
                    f'CycY={losses["loss_cycle_Y"]:.4f} '
                    f'Str={losses["loss_struct"]:.4f}'
                )

        # Epoch average
        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)
            system.epoch_loss_history[k].append(epoch_losses[k])

        epoch_time = time.time() - epoch_start
        total_elapsed = time.time() - training_start_global
        batch_time_avg = epoch_time / max(n_batches, 1)
        
        # Estimate remaining time
        epochs_completed = epoch + 1 - resume_epoch
        if epochs_completed > 0:
            avg_time_per_epoch = total_elapsed / epochs_completed
            epochs_remaining = Config.N_EPOCHS - (epoch + 1)
            estimated_time_remaining = avg_time_per_epoch * epochs_remaining
            eta_str = f' | ETA: {estimated_time_remaining/60:.1f}m'
        else:
            eta_str = ''
        
        vram_str   = ''
        if GPU_INFO['cuda_available']:
            peak = torch.cuda.max_memory_allocated() / 1e9
            vram_str = f' | VRAM peak: {peak:.2f}GB'
            torch.cuda.reset_peak_memory_stats()

        print(
            f'\n  ╔{"═"*65}╗'
            f'\n  ║ Epoch [{epoch+1:03d}/{Config.N_EPOCHS}] | '
            f'Time: {epoch_time:.2f}s ({batch_time_avg*1000:.1f}ms/batch) | '
            f'Elapsed: {total_elapsed/60:.1f}m{eta_str}'
            f'\n  ║ LR={new_lr:.6f}{vram_str}'
            f'\n  ║ G={epoch_losses["loss_G"]:.4f} '
            f'D={epoch_losses["loss_D"]:.4f} '
            f'CycX={epoch_losses["loss_cycle_X"]:.4f} '
            f'CycY={epoch_losses["loss_cycle_Y"]:.4f}'
            f'\n  ║ IdentX={epoch_losses["loss_ident_X"]:.4f} '
            f'IdentY={epoch_losses["loss_ident_Y"]:.4f} '
            f'Struct={epoch_losses["loss_struct"]:.4f}'
            f'\n  ╚{"═"*65}╝\n'
        )

        log_entry = {
            'epoch': epoch + 1, 
            'epoch_time_sec': epoch_time,
            'total_elapsed_sec': total_elapsed,
            'batch_time_avg_ms': batch_time_avg * 1000,
            **epoch_losses
        }
        training_log.append(log_entry)
        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=2)

        if (epoch + 1) % Config.SAMPLE_FREQ == 0:
            save_sample_images(system, test_loader, epoch + 1)
            plot_loss_curves(system.epoch_loss_history, epoch + 1)

        if (epoch + 1) % Config.SAVE_FREQ == 0:
            system.save_checkpoint(epoch + 1)

    system.save_checkpoint(Config.N_EPOCHS, {'final': True})
    print(f'\n[Training Complete] Logs → {log_path}')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10: VISUALIZATION & COUNTERFACTUAL OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def tensor_to_numpy_image(t: torch.Tensor) -> np.ndarray:
    """Convert (C,H,W) tensor in [-1,1] to uint8 numpy array."""
    t   = t.detach().cpu().float()
    t   = ((t + 1.0) / 2.0).clamp(0, 1)
    t   = t.permute(1, 2, 0)
    arr = (t.numpy() * 255).astype(np.uint8)
    if arr.shape[2] == 1:
        arr = arr.squeeze(2)
    return arr


def generate_difference_map_visualization(
    original_tensor  : torch.Tensor,
    generated_tensor : torch.Tensor,
    diff_map_tensor  : torch.Tensor,
    save_path        : str,
    title_suffix     : str = '',
) -> None:
    """
    4-panel clinical visualization:
    Panel 1: Original TB X-ray
    Panel 2: Generated healthy counterfactual
    Panel 3: Grayscale difference map
    Panel 4: Inferno colormap difference (primary clinical output)

    COLORMAP CHOICE — INFERNO over JET:
    Inferno is perceptually uniform (equal perceptual distance per unit value)
    and accessible to color-blind observers. Jet is not perceptually uniform
    and creates false boundaries. For clinical output intended for radiologists,
    inferno is the medically appropriate choice.
    """
    original  = tensor_to_numpy_image(original_tensor.squeeze(0))
    generated = tensor_to_numpy_image(generated_tensor.squeeze(0))

    dm    = diff_map_tensor.squeeze().detach().cpu().numpy()
    dm    = (dm - dm.min()) / (dm.max() - dm.min() + 1e-8)
    cmap_rgb = (cm.inferno(dm)[:, :, :3] * 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor='#0d1117')
    titles = [
        'Original TB X-Ray',
        'Generated Healthy\n(Counterfactual)',
        'Difference Map\n(Grayscale)',
        'Difference Map\n(Inferno — Clinical)',
    ]
    images = [original, generated, (dm * 255).astype(np.uint8), cmap_rgb]
    cmaps  = ['gray', 'gray', 'gray', None]

    for ax, img, title, cmap in zip(axes, images, titles, cmaps):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=8)
        ax.axis('off')

    if title_suffix:
        fig.suptitle(f'Counterfactual Explanation — {title_suffix}',
                     color='white', fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='#0d1117', edgecolor='none')
    plt.close()


def save_sample_images(system: CycleGANSystem, test_loader: DataLoader,
                       epoch: int, n_samples: int = 4) -> None:
    """Save counterfactual sample visualizations during training."""
    sample_dir = os.path.join(Config.RESULTS_DIR, f'epoch_{epoch:04d}')
    os.makedirs(sample_dir, exist_ok=True)

    system.G.eval()
    saved = 0

    with torch.no_grad():
        for batch in test_loader:
            if saved >= n_samples:
                break
            real_tb = batch['TB'].to(system.device)
            result  = system.generate_counterfactual(real_tb)
            for i in range(real_tb.shape[0]):
                if saved >= n_samples:
                    break
                generate_difference_map_visualization(
                    original_tensor  = real_tb[i:i+1],
                    generated_tensor = result['healthy'][i:i+1],
                    diff_map_tensor  = result['diff_map'][i:i+1],
                    save_path        = os.path.join(sample_dir, f'sample_{saved:03d}.png'),
                    title_suffix     = f'Epoch {epoch}',
                )
                saved += 1

    system.G.train()
    print(f'  [Samples] {saved} counterfactuals → {sample_dir}')


def plot_loss_curves(loss_history: dict, epoch: int) -> None:
    """Plot and save 6-panel training loss curves."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor='#0d1117')
    axes = axes.flatten()

    plot_cfg = [
        ('loss_G',       'Generator Loss',        '#58a6ff'),
        ('loss_D',       'Discriminator Loss',     '#f85149'),
        ('loss_cycle_X', 'Cycle Loss X (TB→H→TB)', '#3fb950'),
        ('loss_cycle_Y', 'Cycle Loss Y (H→TB→H)',  '#d29922'),
        ('loss_ident_X', 'Identity Loss X',         '#bc8cff'),
        ('loss_struct',  'Structural Loss',          '#ff7b72'),
    ]

    for ax, (key, label, color) in zip(axes, plot_cfg):
        values = loss_history.get(key, [])
        if values:
            ax.plot(values, color=color, linewidth=1.5, alpha=0.9)
            if len(values) > 10:
                w   = min(10, len(values) // 5)
                avg = np.convolve(values, np.ones(w) / w, mode='valid')
                ax.plot(range(w - 1, len(values)), avg, color='white',
                        linewidth=2, alpha=0.7, linestyle='--', label=f'{w}-ep avg')
                ax.legend(facecolor='#21262d', labelcolor='white', fontsize=8)
        ax.set_title(label, color='white', fontsize=10, fontweight='bold')
        ax.set_xlabel('Epoch', color='#8b949e', fontsize=8)
        ax.set_ylabel('Loss',  color='#8b949e', fontsize=8)
        ax.tick_params(colors='#8b949e')
        ax.set_facecolor('#161b22')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')

    fig.suptitle(f'CycleGAN Training Curves — Epoch {epoch}',
                 color='white', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(Config.LOG_DIR, f'loss_curves_epoch_{epoch:04d}.png')
    plt.savefig(save_path, dpi=120, bbox_inches='tight',
                facecolor='#0d1117', edgecolor='none')
    plt.close()
    print(f'  [Plot] Loss curves → {save_path}')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11: EVALUATION METRICS
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(system: CycleGANSystem, test_loader: DataLoader,
                   output_dir: str = None) -> dict:
    """
    Comprehensive quantitative evaluation.

    METRICS:
    - FID  : Statistical distance between generated and real Normal distributions.
             Lower = more realistic counterfactuals.
    - SSIM : Structural similarity between TB image and its counterfactual.
             Target 0.6–0.8: enough similarity to confirm anatomy is preserved,
             enough difference to confirm pathology was removed.
    - PSNR : Pixel-level fidelity. Secondary metric.
    - Cycle L1: Proxy for hallucination. Lower = less invented anatomy.
    - Identity L1: Measures anatomy preservation directly.
    """
    if output_dir is None:
        output_dir = os.path.join(Config.RESULTS_DIR, 'evaluation')
    os.makedirs(output_dir, exist_ok=True)

    system.G.eval()
    system.F.eval()

    metrics = {
        'cycle_reconstruction_L1': [],
        'identity_L1_G': [],
        'diff_map_mean_intensity': [],
        'diff_map_std_intensity': [],
    }

    if TORCHMETRICS_AVAILABLE:
        fid_metric  = FrechetInceptionDistance(normalize=True).to(system.device)
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(system.device)
        psnr_metric = PeakSignalNoiseRatio(data_range=2.0).to(system.device)
        ssim_values, psnr_values = [], []

    print(f'\n[Evaluation] Running on test set...')

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            real_X = batch['TB'].to(system.device, non_blocking=True)
            real_Y = batch['Normal'].to(system.device, non_blocking=True)

            with torch.amp.autocast(device_type='cuda' if Config.USE_AMP else 'cpu',
                                     enabled=Config.USE_AMP):
                result   = system.generate_counterfactual(real_X)
                fake_Y   = result['healthy']
                diff_map = result['diff_map']
                rec_X    = system.F(fake_Y)
                ident_Y  = system.G(real_Y)

            cycle_err = F.l1_loss(rec_X.float(), real_X.float()).item()
            ident_err = F.l1_loss(ident_Y.float(), real_Y.float()).item()
            metrics['cycle_reconstruction_L1'].append(cycle_err)
            metrics['identity_L1_G'].append(ident_err)

            dm = diff_map.squeeze().cpu().float().numpy()
            metrics['diff_map_mean_intensity'].append(float(dm.mean()))
            metrics['diff_map_std_intensity'].append(float(dm.std()))

            if TORCHMETRICS_AVAILABLE:
                real_Y_01 = ((real_Y.float() + 1.0) / 2.0).clamp(0, 1)
                fake_Y_01 = ((fake_Y.float() + 1.0) / 2.0).clamp(0, 1)
                fid_metric.update(real_Y_01, real=True)
                fid_metric.update(fake_Y_01, real=False)
                ssim_values.append(ssim_metric(fake_Y.float(), real_X.float()).item())
                psnr_values.append(psnr_metric(fake_Y.float(), real_X.float()).item())

            if i < 10:
                generate_difference_map_visualization(
                    original_tensor  = real_X,
                    generated_tensor = fake_Y,
                    diff_map_tensor  = diff_map,
                    save_path        = os.path.join(output_dir, f'eval_sample_{i:03d}.png'),
                    title_suffix     = f'Eval Sample {i}',
                )

    results = {
        'cycle_reconstruction_L1_mean': float(np.mean(metrics['cycle_reconstruction_L1'])),
        'cycle_reconstruction_L1_std' : float(np.std(metrics['cycle_reconstruction_L1'])),
        'identity_L1_mean'            : float(np.mean(metrics['identity_L1_G'])),
        'identity_L1_std'             : float(np.std(metrics['identity_L1_G'])),
        'diff_map_mean_intensity'     : float(np.mean(metrics['diff_map_mean_intensity'])),
        'diff_map_std_intensity'      : float(np.mean(metrics['diff_map_std_intensity'])),
    }

    if TORCHMETRICS_AVAILABLE:
        results['FID']  = float(fid_metric.compute().item())
        results['SSIM'] = float(np.mean(ssim_values))
        results['PSNR'] = float(np.mean(psnr_values))

    print(f'\n{"="*70}')
    print(f'  Evaluation Report')
    print(f'{"="*70}')
    print(f'  Cycle L1     : {results["cycle_reconstruction_L1_mean"]:.4f} '
          f'± {results["cycle_reconstruction_L1_std"]:.4f}')
    print(f'    ↳ Lower = less hallucination')
    print(f'  Identity L1  : {results["identity_L1_mean"]:.4f} '
          f'± {results["identity_L1_std"]:.4f}')
    print(f'    ↳ Lower = better anatomy preservation')
    print(f'  Diff Map Mean: {results["diff_map_mean_intensity"]:.4f}')
    if TORCHMETRICS_AVAILABLE:
        print(f'  FID          : {results["FID"]:.2f}  (↓ better)')
        print(f'  SSIM         : {results["SSIM"]:.4f} (target 0.6–0.8)')
        print(f'  PSNR         : {results["PSNR"]:.2f} dB')
    print(f'{"="*70}\n')

    with open(os.path.join(output_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f'  [Evaluation] Metrics → {output_dir}/evaluation_metrics.json')

    system.G.train()
    system.F.train()
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12: BATCH COUNTERFACTUAL GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_all_counterfactuals(system: CycleGANSystem, test_loader: DataLoader,
                                 output_dir: str = None) -> None:
    """
    Generate and save counterfactual explanations for all TB test images.

    Includes per-image hallucination safety check via cycle reconstruction
    error. Images exceeding the threshold are flagged to a JSON report for
    clinical review — they should not be used without radiologist verification.

    OUTPUT STRUCTURE:
    output_dir/
    ├── counterfactuals/      ← Generated healthy images
    ├── difference_maps/      ← Grayscale difference maps
    ├── visualizations/       ← 4-panel clinical panels
    └── hallucination_warnings.json
    """
    if output_dir is None:
        output_dir = os.path.join(Config.RESULTS_DIR, 'counterfactuals_final')

    dirs = {
        'cf'  : os.path.join(output_dir, 'counterfactuals'),
        'diff': os.path.join(output_dir, 'difference_maps'),
        'vis' : os.path.join(output_dir, 'visualizations'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    system.G.eval()
    system.F.eval()

    HALLUCINATION_THRESHOLD = 0.15
    hallucination_warnings  = []

    print(f'\n[Generation] Generating counterfactuals for all TB test images...')

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            real_X  = batch['TB'].to(system.device, non_blocking=True)
            tb_path = Path(batch['TB_path'][0])
            stem    = tb_path.stem

            with torch.amp.autocast(device_type='cuda' if Config.USE_AMP else 'cpu',
                                     enabled=Config.USE_AMP):
                result   = system.generate_counterfactual(real_X)
                fake_Y   = result['healthy']
                diff_map = result['diff_map']
                rec_X    = system.F(fake_Y)

            # Hallucination safety check
            cycle_err = F.l1_loss(rec_X.float(), real_X.float()).item()
            if cycle_err > HALLUCINATION_THRESHOLD:
                hallucination_warnings.append({
                    'image'      : str(tb_path),
                    'cycle_error': cycle_err,
                    'threshold'  : HALLUCINATION_THRESHOLD,
                })
                print(f'  ⚠ [HALLUCINATION RISK] {stem}: '
                      f'cycle_error={cycle_err:.4f} > {HALLUCINATION_THRESHOLD}')

            # Save counterfactual image
            save_image((fake_Y.float() + 1.0) / 2.0,
                       os.path.join(dirs['cf'], f'{stem}_healthy.png'))

            # Save grayscale difference map
            dm_np   = diff_map.squeeze().cpu().float().numpy()
            dm_norm = (dm_np - dm_np.min()) / (dm_np.max() - dm_np.min() + 1e-8)
            Image.fromarray((dm_norm * 255).astype(np.uint8), mode='L').save(
                os.path.join(dirs['diff'], f'{stem}_diffmap.png')
            )

            # Save 4-panel visualization
            generate_difference_map_visualization(
                original_tensor  = real_X,
                generated_tensor = fake_Y,
                diff_map_tensor  = diff_map,
                save_path        = os.path.join(dirs['vis'], f'{stem}_visualization.png'),
                title_suffix     = stem,
            )

            if (batch_idx + 1) % 20 == 0:
                print(f'  Progress: {batch_idx+1}/{len(test_loader)}')

    if hallucination_warnings:
        warn_path = os.path.join(output_dir, 'hallucination_warnings.json')
        with open(warn_path, 'w') as f:
            json.dump(hallucination_warnings, f, indent=2)
        print(f'\n  ⚠ {len(hallucination_warnings)} hallucination warnings → {warn_path}')
    else:
        print(f'\n  ✓ No hallucination warnings. All cycle errors below threshold.')

    print(f'  [Generation] Complete → {output_dir}')
    system.G.train()
    system.F.train()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 13: COMMAND-LINE INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='CycleGAN Counterfactual TB Explanation Pipeline (GPU-optimized)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train from scratch (uses GPU automatically if available)
  python cyclegan_tb_final.py --mode train --data_dir ./dataset

  # Train with full determinism for paper submission
  python cyclegan_tb_final.py --mode train --data_dir ./dataset --reproducible

  # Resume from checkpoint
  python cyclegan_tb_final.py --mode train --data_dir ./dataset --resume ./checkpoints/epoch_0050.pth

  # Generate counterfactuals
  python cyclegan_tb_final.py --mode generate --data_dir ./dataset --checkpoint ./checkpoints/epoch_0200.pth

  # Evaluate
  python cyclegan_tb_final.py --mode evaluate --data_dir ./dataset --checkpoint ./checkpoints/epoch_0200.pth
        """
    )
    parser.add_argument('--mode', choices=['train', 'generate', 'evaluate'],
                        default='train')
    parser.add_argument('--data_dir',    type=str,   default=Config.DATA_DIR)
    parser.add_argument('--checkpoint',  type=str,   default=None)
    parser.add_argument('--resume',      type=str,   default=None)
    parser.add_argument('--epochs',      type=int,   default=Config.N_EPOCHS)
    parser.add_argument('--lambda_cycle',     type=float, default=Config.LAMBDA_CYCLE)
    parser.add_argument('--lambda_identity',  type=float, default=Config.LAMBDA_IDENTITY)
    parser.add_argument('--img_size',    type=int,   default=Config.IMG_SIZE)
    parser.add_argument('--seed',        type=int,   default=Config.SEED)
    parser.add_argument('--output_dir',  type=str,   default=None)
    parser.add_argument(
        '--reproducible', action='store_true',
        help='Enable full determinism (cudnn.deterministic=True, benchmark=False). '
             'Use for final paper runs. Disables cudnn.benchmark speed boost.'
    )
    parser.add_argument(
        '--no_amp', action='store_true',
        help='Disable mixed precision even if CUDA is available. '
             'Use for debugging numerical issues.'
    )
    parser.add_argument(
        '--no_compile', action='store_true',
        help='Disable torch.compile(). Use if compilation causes issues on older GPUs.'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Apply CLI overrides ────────────────────────────────────────────────────
    Config.DATA_DIR          = args.data_dir
    Config.N_EPOCHS          = args.epochs
    Config.LAMBDA_CYCLE      = args.lambda_cycle
    Config.LAMBDA_IDENTITY   = args.lambda_identity
    Config.IMG_SIZE          = args.img_size
    Config.SEED              = args.seed
    Config.REPRODUCIBLE_MODE = args.reproducible

    # Disable GPU optimizations if explicitly requested
    if args.no_amp:
        Config.USE_AMP = False
        print('[CLI] AMP disabled via --no_amp flag.')
    if args.no_compile:
        Config.USE_COMPILE = False
        print('[CLI] torch.compile() disabled via --no_compile flag.')

    # ── Reproducibility ────────────────────────────────────────────────────────
    set_reproducibility(Config.SEED, Config.REPRODUCIBLE_MODE)

    print(f'\n{"="*70}')
    print(f'  TB CycleGAN Counterfactual Pipeline  |  Mode: {args.mode.upper()}')
    print(f'  Device: {Config.DEVICE}  |  AMP: {Config.USE_AMP}  |  '
          f'Compile: {Config.USE_COMPILE}')
    print(f'{"="*70}\n')

    # ── Build DataLoaders ──────────────────────────────────────────────────────
    train_loader, test_loader, _ = build_dataloaders(Config.DATA_DIR)

    # ── Initialize System ──────────────────────────────────────────────────────
    system = CycleGANSystem(Config.DEVICE)
    system.print_model_summary()

    # ── Dispatch ───────────────────────────────────────────────────────────────
    if args.mode == 'train':
        resume_epoch = 0
        if args.resume:
            resume_epoch = system.load_checkpoint(args.resume)
        train(system, train_loader, test_loader, resume_epoch=resume_epoch)

    elif args.mode == 'generate':
        if not args.checkpoint:
            raise ValueError('--checkpoint is required for mode=generate')
        system.load_checkpoint(args.checkpoint)
        generate_all_counterfactuals(
            system, test_loader,
            output_dir=args.output_dir
        )

    elif args.mode == 'evaluate':
        if not args.checkpoint:
            raise ValueError('--checkpoint is required for mode=evaluate')
        system.load_checkpoint(args.checkpoint)
        evaluate_model(
            system, test_loader,
            output_dir=args.output_dir
        )


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    main()
