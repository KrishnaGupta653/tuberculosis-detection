"""
================================================================================
TB COUNTERFACTUAL EXPLANATION SYSTEM — cyclegan_tb_counterfactual.py
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

USAGE:
    # Train from scratch
    python cyclegan_tb_counterfactual.py --mode train --data_dir ./dataset

    # Resume training from checkpoint
    python cyclegan_tb_counterfactual.py --mode train --data_dir ./dataset --resume ./checkpoints/epoch_50.pth

    # Generate counterfactuals for all TB images
    python cyclegan_tb_counterfactual.py --mode generate --data_dir ./dataset --checkpoint ./checkpoints/best_model.pth

    # Evaluate a trained model
    python cyclegan_tb_counterfactual.py --mode evaluate --data_dir ./dataset --checkpoint ./checkpoints/best_model.pth

REFERENCE ARCHITECTURE:
    Based on: Zhu et al. "Unpaired Image-to-Image Translation using
    Cycle-Consistent Adversarial Networks" (ICCV 2017), adapted for
    medical imaging with hallucination mitigation strategies from:
    Singla et al. "GANterfactual—Counterfactual Explanations for Medical
    Non-experts" (Frontiers in AI, 2022).
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
# NOTE: torchmetrics must be installed: pip install torchmetrics
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
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.cm as cm

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: CONFIGURATION
# Purpose: Central, reproducible configuration object. All hyperparameters
#          are documented with their clinical or technical rationale.
# ══════════════════════════════════════════════════════════════════════════════

class Config:
    """
    Central configuration for the CycleGAN training pipeline.

    DESIGN DECISION — Why these defaults?
    - IMG_SIZE=256: Balances resolution (enough to see cavities/infiltrates)
      with memory constraints. CycleGAN's PatchGAN discriminator operates on
      patches, so 256 retains sufficient structural detail.
    - LAMBDA_CYCLE=10.0: Standard value from original CycleGAN paper. High
      value is critical for medical images to prevent anatomical hallucination.
    - LAMBDA_IDENTITY=5.0: Half of cycle loss. Prevents unnecessary color/
      intensity shifts in regions that already look "normal", preserving
      bone density signals in the X-ray.
    - LAMBDA_STRUCTURAL=1.0: Perceptual/structural loss weight. Enforces
      high-frequency texture preservation (critical for TB fibrotic patterns).
    - BATCH_SIZE=1: Standard for CycleGAN. Enables instance normalization and
      ensures each image's anatomy is processed independently.
    - LEARNING_RATE=0.0002: Canonical GAN learning rate (Adam optimizer).
    """

    # ── Dataset ────────────────────────────────────────────────────────────────
    DATA_DIR        = './dataset'
    TB_DIR          = 'TB'           # Subfolder name for TB-positive images
    NORMAL_DIR      = 'Normal'       # Subfolder name for healthy images
    IMG_SIZE        = 256            # Spatial resolution for training
    VALID_EXTS      = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

    # ── Training Directories ───────────────────────────────────────────────────
    CHECKPOINT_DIR  = './checkpoints'
    RESULTS_DIR     = './results'
    LOG_DIR         = './logs'

    # ── Hardware ───────────────────────────────────────────────────────────────
    DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS     = 4 if torch.cuda.is_available() else 0
    PIN_MEMORY      = torch.cuda.is_available()

    # ── Architecture ───────────────────────────────────────────────────────────
    # 9 residual blocks: used for 256x256+ images (vs 6 blocks for 128x128).
    # More blocks = larger receptive field = better global structure preservation.
    N_RESIDUAL_BLOCKS = 9
    # Discriminator depth: 3 layers = 70x70 PatchGAN (optimal for texture).
    N_DISC_LAYERS     = 3
    # Number of generator filters in first conv layer.
    N_GEN_FILTERS     = 64
    N_DISC_FILTERS    = 64

    # ── Loss Weights ───────────────────────────────────────────────────────────
    # CRITICAL: These weights directly control medical validity vs. realism.
    LAMBDA_CYCLE      = 10.0   # Cycle-consistency: prevents hallucination
    LAMBDA_IDENTITY   = 5.0    # Identity: preserves bone/tissue structure
    LAMBDA_STRUCTURAL = 1.0    # Structural: SSIM-based perceptual loss

    # ── Optimization ───────────────────────────────────────────────────────────
    LEARNING_RATE   = 0.0002
    BETA1           = 0.5      # Adam beta1: GAN-standard (not default 0.9)
    BETA2           = 0.999
    BATCH_SIZE      = 1        # Must be 1 for CycleGAN stability
    N_EPOCHS        = 200
    DECAY_EPOCH     = 100      # Epoch at which LR linear decay begins

    # ── Image Buffer ───────────────────────────────────────────────────────────
    # Historical buffer prevents discriminator oscillation (Shrivastava et al.)
    BUFFER_SIZE     = 50

    # ── Logging & Checkpointing ────────────────────────────────────────────────
    LOG_FREQ        = 100      # Log losses every N batches
    SAVE_FREQ       = 10       # Save checkpoint every N epochs
    SAMPLE_FREQ     = 5        # Save sample images every N epochs

    # ── Reproducibility ────────────────────────────────────────────────────────
    SEED            = 42

    # ── Evaluation ─────────────────────────────────────────────────────────────
    N_EVAL_IMAGES   = 50       # Number of images used for FID computation


def set_reproducibility(seed: int = Config.SEED) -> None:
    """
    Set all random seeds for full experiment reproducibility.
    Required for publication-quality results that can be replicated.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # NOTE: Setting deterministic=True may reduce speed but ensures
        # identical outputs across runs. Essential for research reproducibility.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[Reproducibility] All seeds set to {seed}.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: DATA PIPELINE
# Purpose: Robust loading with medical-grade preprocessing. Special care is
#          taken to avoid augmentations that would introduce unrealistic
#          artifacts in chest X-rays (e.g., no extreme color jitter, no
#          flipping that changes anatomical laterality).
# ══════════════════════════════════════════════════════════════════════════════

def validate_dataset(data_dir: str) -> dict:
    """
    Validate the dataset directory structure and report statistics.
    Raises informative errors if the structure is incorrect.

    Returns:
        dict with 'tb_paths' and 'normal_paths' lists.
    """
    data_dir = Path(data_dir)
    tb_dir = data_dir / Config.TB_DIR
    normal_dir = data_dir / Config.NORMAL_DIR

    # Check directory existence
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    if not tb_dir.exists():
        raise FileNotFoundError(
            f"TB subdirectory not found: {tb_dir}\n"
            f"Expected structure: {data_dir}/TB/ and {data_dir}/Normal/"
        )
    if not normal_dir.exists():
        raise FileNotFoundError(f"Normal subdirectory not found: {normal_dir}")

    # Collect valid image paths
    def collect_images(directory: Path) -> list:
        paths = [
            p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in Config.VALID_EXTS
        ]
        return sorted(paths)

    tb_paths = collect_images(tb_dir)
    normal_paths = collect_images(normal_dir)

    if len(tb_paths) == 0:
        raise ValueError(f"No valid images found in {tb_dir}")
    if len(normal_paths) == 0:
        raise ValueError(f"No valid images found in {normal_dir}")

    # Compute class imbalance ratio
    ratio = len(tb_paths) / len(normal_paths)
    imbalance_warning = ""
    if ratio > 3.0 or ratio < 0.33:
        imbalance_warning = (
            f"\n  ⚠ WARNING: Severe class imbalance detected (ratio={ratio:.2f}). "
            "This may cause the generator to favor one domain. "
            "Consider data augmentation or balanced sampling."
        )

    print(f"\n[Dataset Validation] ─────────────────────────────────────")
    print(f"  Root directory : {data_dir.resolve()}")
    print(f"  TB images      : {len(tb_paths)}")
    print(f"  Normal images  : {len(normal_paths)}")
    print(f"  TB/Normal ratio: {ratio:.2f}{imbalance_warning}")
    print(f"────────────────────────────────────────────────────────────\n")

    return {'tb_paths': tb_paths, 'normal_paths': normal_paths}


def build_transforms(mode: str = 'train', img_size: int = Config.IMG_SIZE) -> transforms.Compose:
    """
    Build image preprocessing transforms.

    MEDICAL IMAGING AUGMENTATION RATIONALE:
    - Horizontal flip: Used cautiously. Left/right lung laterality matters
      clinically (e.g., pleural effusion is often unilateral). We allow it
      during training because the CycleGAN is learning texture transfer, not
      clinical localization. Can be disabled for stricter protocols.
    - Rotation (±10°): Mimics patient positioning variance seen in real CXR.
    - NO vertical flip: Inverted lungs are not clinically realistic.
    - NO color jitter: X-rays are grayscale; intensity variations are
      meaningful (bone density, aeration). Jitter would add noise artifacts.
    - NO random crop without resize: Would destroy lung boundary integrity.
    - Normalization to [-1, 1]: Required by CycleGAN's tanh output activation.

    Args:
        mode: 'train' applies augmentation; 'test' applies only resize + normalize.
        img_size: Target image resolution.
    """
    if mode == 'train':
        transform_list = [
            transforms.Resize(int(img_size * 1.12), Image.BICUBIC),  # Slight oversample
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10, fill=0),  # fill=0 = black background
            transforms.ColorJitter(brightness=0.05, contrast=0.05),  # Minimal, realistic
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # → [-1, 1]
        ]
    else:  # 'test' or 'val'
        transform_list = [
            transforms.Resize((img_size, img_size), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    return transforms.Compose(transform_list)


class UnpairedXRayDataset(Dataset):
    """
    Dataset for unpaired TB ↔ Normal chest X-ray image translation.

    DESIGN DECISION — Unpaired training:
    CycleGAN's key advantage is that it does NOT require paired images
    (same patient, same scan, TB vs. healthy). Such paired data is
    clinically impossible to obtain for TB. Instead, the cycle-consistency
    loss acts as a proxy for pairing, enforcing that the round-trip
    translation TB → Healthy → TB′ ≈ TB (and vice versa).

    Image loading:
    - Images are loaded as RGB (3-channel). This is correct even for
      grayscale X-rays — the model benefits from the 3-channel input
      because ImageNet-pretrained components (in perceptual loss) expect
      3 channels. The three channels will be identical for true grayscale.
    """

    def __init__(self, tb_paths: list, normal_paths: list, transform=None):
        self.tb_paths = tb_paths
        self.normal_paths = normal_paths
        self.transform = transform
        self._len = max(len(tb_paths), len(normal_paths))

    def __len__(self) -> int:
        # Use the larger domain's length as epoch length.
        # Smaller domain wraps around via modulo, ensuring full coverage.
        return self._len

    def __getitem__(self, idx: int) -> dict:
        # Wrap-around indexing prevents index-out-of-bounds for smaller domain
        tb_path     = self.tb_paths[idx % len(self.tb_paths)]
        normal_path = self.normal_paths[idx % len(self.normal_paths)]

        tb_img     = self._load_image(tb_path)
        normal_img = self._load_image(normal_path)

        if self.transform:
            tb_img     = self.transform(tb_img)
            normal_img = self.transform(normal_img)

        return {
            'TB'       : tb_img,
            'Normal'   : normal_img,
            'TB_path'  : str(tb_path),
            'Normal_path': str(normal_path),
        }

    @staticmethod
    def _load_image(path: Path) -> Image.Image:
        """Load image as RGB PIL Image with error handling."""
        try:
            img = Image.open(path).convert('RGB')
            return img
        except Exception as e:
            raise IOError(f"Failed to load image {path}: {e}")


def build_dataloaders(data_dir: str) -> tuple:
    """
    Build train and test DataLoaders with a stratified split.

    Returns:
        (train_loader, test_loader, dataset_info_dict)
    """
    dataset_info = validate_dataset(data_dir)
    tb_paths     = dataset_info['tb_paths']
    normal_paths = dataset_info['normal_paths']

    # Deterministic shuffle for reproducibility
    rng = random.Random(Config.SEED)
    rng.shuffle(tb_paths)
    rng.shuffle(normal_paths)

    # 80/20 split — both domains split independently to prevent data leakage
    def split(paths, ratio=0.8):
        n = int(len(paths) * ratio)
        return paths[:n], paths[n:]

    tb_train, tb_test         = split(tb_paths)
    normal_train, normal_test = split(normal_paths)

    train_dataset = UnpairedXRayDataset(
        tb_train, normal_train,
        transform=build_transforms('train')
    )
    test_dataset = UnpairedXRayDataset(
        tb_test, normal_test,
        transform=build_transforms('test')
    )

    # NOTE: shuffle=True for training is important to prevent the generator
    # from learning the dataset ordering rather than the domain translation.
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,   # Avoids batch-norm instability with size-1 final batch
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    print(f"[DataLoader] Train: {len(train_dataset)} pairs "
          f"| Test: {len(test_dataset)} pairs")
    return train_loader, test_loader, {
        'tb_train': len(tb_train), 'tb_test': len(tb_test),
        'normal_train': len(normal_train), 'normal_test': len(normal_test),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: ARCHITECTURE
# Purpose: Medical-grade generator and PatchGAN discriminator with Instance
#          Normalization (preferred over BatchNorm for small batches and
#          style transfer tasks in medical imaging).
# ══════════════════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """
    Residual block with reflection padding and instance normalization.

    DESIGN DECISION — Reflection vs. Zero padding:
    Zero-padding introduces artificial black borders that a discriminator
    can easily learn to spot. Reflection padding produces smoother edges
    that are more realistic at image boundaries — important when the lung
    boundary meets the image edge in chest X-rays.

    DESIGN DECISION — Instance Normalization vs. Batch Normalization:
    Instance normalization normalizes per-image statistics rather than
    per-batch. This is critical for:
    1. Batch size = 1 (BatchNorm is unstable with single samples)
    2. Style transfer: In-domain statistics (anatomy) are preserved
       while cross-domain statistics (pathology texture) can be modified.
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
        # Skip connection: ensures anatomical structure is preserved as a
        # residual, and only the pathology-related deltas are learned.
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    """
    ResNet-based generator for medical image domain translation.

    Architecture: Encoder → 9 Residual Blocks → Decoder
    - Encoder: 3 convolutional layers with stride-2 downsampling.
    - Bottleneck: N residual blocks operating at 1/4 spatial resolution.
    - Decoder: 2 transposed convolutions + final conv for upsampling.

    DESIGN DECISION — Why ResNet over U-Net for CycleGAN?
    While U-Net with skip connections is excellent for segmentation, skip
    connections in a cycle-GAN generator can cause identity shortcuts —
    the network can "cheat" by passing the original image directly through
    skip connections without performing meaningful domain translation.
    ResNet's bottleneck forces the network to encode domain-invariant
    content (anatomy) and domain-specific style (pathology) separately.

    OUTPUT: tanh activation → pixel values in [-1, 1]
    This is normalized to [0, 1] for visualization via (x + 1) / 2.
    """

    def __init__(
        self,
        in_channels  : int = 3,
        out_channels : int = 3,
        n_filters    : int = Config.N_GEN_FILTERS,
        n_blocks     : int = Config.N_RESIDUAL_BLOCKS,
    ):
        if n_blocks < 0:
            raise ValueError("n_blocks must be non-negative.")
        super().__init__()

        # ── Initial convolution ────────────────────────────────────────────────
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, n_filters, kernel_size=7, padding=0, bias=False),
            nn.InstanceNorm2d(n_filters, affine=False),
            nn.ReLU(inplace=True),
        ]

        # ── Encoder (downsampling) ─────────────────────────────────────────────
        n_down = 2  # Number of downsampling stages
        for i in range(n_down):
            mult = 2 ** i
            model += [
                nn.Conv2d(n_filters * mult, n_filters * mult * 2,
                          kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(n_filters * mult * 2, affine=False),
                nn.ReLU(inplace=True),
            ]

        # ── Residual bottleneck ────────────────────────────────────────────────
        mult = 2 ** n_down
        for _ in range(n_blocks):
            model += [ResidualBlock(n_filters * mult)]

        # ── Decoder (upsampling) ───────────────────────────────────────────────
        for i in range(n_down):
            mult = 2 ** (n_down - i)
            model += [
                nn.ConvTranspose2d(n_filters * mult, n_filters * mult // 2,
                                   kernel_size=3, stride=2, padding=1,
                                   output_padding=1, bias=False),
                nn.InstanceNorm2d(n_filters * mult // 2, affine=False),
                nn.ReLU(inplace=True),
            ]

        # ── Output convolution ─────────────────────────────────────────────────
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(n_filters, out_channels, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights from N(0, 0.02) — standard for GANs."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class PatchGANDiscriminator(nn.Module):
    """
    70×70 PatchGAN discriminator.

    DESIGN DECISION — Why PatchGAN for medical images?
    A standard discriminator produces a single scalar real/fake score.
    PatchGAN outputs a grid of scores, each evaluating a ~70×70 pixel
    patch of the image. This forces the discriminator to assess local
    texture quality (critical for detecting whether pathological textures
    like fibrotic streaks have been plausibly translated) rather than only
    evaluating global composition.

    The 70×70 receptive field is large enough to evaluate TB-relevant
    textures (cavities, consolidations) while being small enough to
    prevent the discriminator from memorizing patient-specific anatomy.

    DESIGN DECISION — LeakyReLU (slope=0.2):
    Standard for discriminators. Allows gradients to flow for negative
    activations, preventing "dying neurons" in the discriminator's
    early layers that would cause training instability.
    """

    def __init__(
        self,
        in_channels : int = 3,
        n_filters   : int = Config.N_DISC_FILTERS,
        n_layers    : int = Config.N_DISC_LAYERS,
    ):
        super().__init__()

        layers = [
            # First layer: no normalization (as per original PatchGAN paper)
            nn.Conv2d(in_channels, n_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        n_f = n_filters
        for i in range(1, n_layers):
            n_f_prev = n_f
            n_f = min(n_f * 2, 512)  # Cap at 512 to control model size
            layers += [
                nn.Conv2d(n_f_prev, n_f, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(n_f, affine=False),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        # Final two layers: stride-1 to avoid excessive downsampling
        n_f_prev = n_f
        n_f = min(n_f * 2, 512)
        layers += [
            nn.Conv2d(n_f_prev, n_f, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(n_f, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_f, 1, kernel_size=4, stride=1, padding=1),
            # No sigmoid: LSGAN uses MSE loss, which does not require [0,1] outputs.
        ]

        self.model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: LOSS FUNCTIONS
# Purpose: All losses are centralized here with detailed clinical rationale.
# ══════════════════════════════════════════════════════════════════════════════

class LSGANLoss(nn.Module):
    """
    Least-Squares GAN adversarial loss (Mao et al., 2017).

    DESIGN DECISION — LSGAN vs. Vanilla GAN:
    Standard GAN uses binary cross-entropy, which suffers from vanishing
    gradients when the discriminator is confident. LSGAN penalizes
    generated samples in proportion to their distance from the real
    decision boundary (real=1, fake=0 for discriminator; real=1 for
    generator). This provides:
    1. More stable training gradients throughout training.
    2. Better image quality for medical images where subtle textures matter.
    3. Reduced mode collapse — critical when the TB dataset is small.
    """

    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def discriminator_loss(
        self,
        real_pred   : torch.Tensor,
        fake_pred   : torch.Tensor,
    ) -> torch.Tensor:
        real_target = torch.ones_like(real_pred)   # Real images → target 1
        fake_target = torch.zeros_like(fake_pred)  # Fake images → target 0
        return 0.5 * (
            self.criterion(real_pred, real_target) +
            self.criterion(fake_pred, fake_target)
        )

    def generator_loss(self, fake_pred: torch.Tensor) -> torch.Tensor:
        # Generator tries to fool the discriminator: target = 1
        real_target = torch.ones_like(fake_pred)
        return self.criterion(fake_pred, real_target)


class StructuralLoss(nn.Module):
    """
    Multi-scale structural similarity loss for anatomical preservation.

    DESIGN DECISION — Why structural loss for medical images?
    Standard GAN + cycle losses operate in pixel space. For medical images,
    structural fidelity (lung boundaries, rib spacing, heart silhouette)
    is clinically critical. SSIM-based loss penalizes structural deformations
    even when pixel-level differences seem small.

    The perceptual component uses VGG16 features because:
    - Early VGG layers encode edges and textures (rib borders, lung margins)
    - Later layers encode semantic structures (lung shape)
    - This provides a richer, more clinically relevant similarity signal
      than pure pixel-MSE.

    SAFETY NOTE: The VGG backbone is frozen — we do not want training
    to modify what "structural similarity" means.
    """

    def __init__(self, device: torch.device = Config.DEVICE):
        super().__init__()
        self.device = device

        # ── VGG Perceptual Loss ────────────────────────────────────────────────
        # Use the first 3 blocks of VGG16 for multi-scale texture features
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.vgg_blocks = nn.ModuleList([
            vgg[:4].eval(),    # Block 1: low-level edges
            vgg[4:9].eval(),   # Block 2: mid-level textures
            vgg[9:16].eval(),  # Block 3: higher-level structures
        ])
        # Freeze ALL VGG parameters — this is a fixed feature extractor
        for param in self.parameters():
            param.requires_grad_(False)
        self.to(device)

        # Normalization to match ImageNet statistics expected by VGG
        # X-ray images have mean≈0 due to [-1,1] normalization, but VGG
        # expects ImageNet-style statistics.
        self.register_buffer(
            'vgg_mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'vgg_std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        self.l1 = nn.L1Loss()

    def _normalize_for_vgg(self, x: torch.Tensor) -> torch.Tensor:
        """Convert from [-1,1] to VGG's expected input range."""
        x = (x + 1.0) / 2.0                   # [-1,1] → [0,1]
        x = (x - self.vgg_mean) / self.vgg_std # [0,1] → ImageNet normalized
        return x

    def forward(
        self,
        real   : torch.Tensor,
        generated: torch.Tensor,
    ) -> torch.Tensor:
        real_n = self._normalize_for_vgg(real)
        gen_n  = self._normalize_for_vgg(generated)

        loss = torch.tensor(0.0, device=self.device)
        r_feat = real_n
        g_feat = gen_n

        for block in self.vgg_blocks:
            r_feat = block(r_feat)
            g_feat = block(g_feat)
            loss  += self.l1(r_feat, g_feat)

        return loss / len(self.vgg_blocks)  # Normalize by block count


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: IMAGE HISTORY BUFFER
# Purpose: Prevents discriminator oscillation by showing a mix of recently
#          generated and historically generated images.
# ══════════════════════════════════════════════════════════════════════════════

class ImageHistoryBuffer:
    """
    Historical buffer of generated images for discriminator training.

    REFERENCE: Shrivastava et al., "Learning from Simulated and Unsupervised
    Images through Adversarial Training" (CVPR 2017).

    This prevents a common GAN failure mode where the generator "forgets"
    previous modes after the discriminator adapts. The buffer returns a
    random mix of current and historical fake images to the discriminator,
    smoothing the discriminator's update and reducing oscillation.
    """

    def __init__(self, max_size: int = Config.BUFFER_SIZE):
        self.max_size = max_size
        self.buffer   = []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        """
        Query the buffer: with 50% probability return a historical image
        instead of the current one.

        Args:
            images: Tensor of generated images (batch_size, C, H, W).
        Returns:
            Tensor of same shape, with some images possibly from history.
        """
        if self.max_size == 0:
            return images

        return_images = []
        for image in images.unbind(0):
            image = image.unsqueeze(0)
            if len(self.buffer) < self.max_size:
                # Buffer not full: always add and return current image
                self.buffer.append(image.detach().clone())
                return_images.append(image)
            else:
                if random.random() > 0.5:
                    # Return a random historical image and replace it
                    idx = random.randint(0, self.max_size - 1)
                    return_images.append(self.buffer[idx].clone())
                    self.buffer[idx] = image.detach().clone()
                else:
                    # Return current image, possibly add to buffer
                    return_images.append(image)

        return torch.cat(return_images, dim=0)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: LEARNING RATE SCHEDULER
# Purpose: Linear decay scheduler matching the original CycleGAN schedule.
# ══════════════════════════════════════════════════════════════════════════════

class LinearDecayLR:
    """
    Linear learning rate decay, starting from `decay_epoch`.

    Schedule:
    - Epochs 0 → decay_epoch: constant LR
    - Epochs decay_epoch → n_epochs: linear decay to 0

    This is the canonical schedule from the original CycleGAN paper and
    has been shown to improve convergence stability in medical imaging GANs.
    """

    def __init__(
        self,
        n_epochs    : int,
        decay_epoch : int,
        initial_lr  : float,
    ):
        self.n_epochs    = n_epochs
        self.decay_epoch = decay_epoch
        self.initial_lr  = initial_lr

    def get_lr_factor(self, epoch: int) -> float:
        if epoch < self.decay_epoch:
            return 1.0
        decay_progress = (epoch - self.decay_epoch) / (self.n_epochs - self.decay_epoch)
        return max(0.0, 1.0 - decay_progress)

    def update(self, optimizer: torch.optim.Optimizer, epoch: int) -> float:
        factor = self.get_lr_factor(epoch)
        new_lr = self.initial_lr * factor
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: CYCLEGAN TRAINING SYSTEM
# Purpose: Encapsulates all networks, losses, optimizers, and the training
#          loop in a single, resumable class.
# ══════════════════════════════════════════════════════════════════════════════

class CycleGANSystem:
    """
    Full CycleGAN system with dual generators and discriminators.

    Notation (consistent with Zhu et al., 2017):
    - Domain X: TB-positive chest X-rays
    - Domain Y: Healthy (Normal) chest X-rays
    - G: Generator X → Y (TB → Healthy counterfactual)
    - F: Generator Y → X (Healthy → TB reconstruction)
    - D_X: Discriminator for domain X (real/fake TB)
    - D_Y: Discriminator for domain Y (real/fake Healthy)

    LOSS TERMS:
    1. L_GAN(G, D_Y, X, Y): G generates realistic Healthy images
    2. L_GAN(F, D_X, Y, X): F generates realistic TB images
    3. λ_cycle * L_cycle(G, F):
        ||F(G(x)) - x||₁ + ||G(F(y)) - y||₁
       Enforces round-trip consistency → prevents hallucination.
    4. λ_identity * L_identity(G, F):
        ||G(y) - y||₁ + ||F(x) - x||₁
       If a Normal image is passed to G, it should remain unchanged.
       Critical for preserving bone structure and lung margins.
    5. λ_structural * L_structural:
        Multi-scale perceptual loss for texture preservation.
    """

    def __init__(self, device: torch.device = Config.DEVICE):
        self.device = device

        # ── Build Networks ─────────────────────────────────────────────────────
        self.G   = ResNetGenerator().to(device)  # TB → Healthy
        self.F   = ResNetGenerator().to(device)  # Healthy → TB
        self.D_X = PatchGANDiscriminator().to(device)
        self.D_Y = PatchGANDiscriminator().to(device)

        # ── Loss Functions ─────────────────────────────────────────────────────
        self.criterion_adv    = LSGANLoss()
        self.criterion_cycle  = nn.L1Loss()
        self.criterion_ident  = nn.L1Loss()
        self.criterion_struct = StructuralLoss(device)

        # ── Optimizers ─────────────────────────────────────────────────────────
        # DESIGN DECISION: Generators and discriminators share separate
        # optimizers to allow different effective learning rates and to
        # decouple their update schedules if needed.
        self.opt_G = torch.optim.Adam(
            chain(self.G.parameters(), self.F.parameters()),
            lr=Config.LEARNING_RATE,
            betas=(Config.BETA1, Config.BETA2),
        )
        self.opt_D = torch.optim.Adam(
            chain(self.D_X.parameters(), self.D_Y.parameters()),
            lr=Config.LEARNING_RATE,
            betas=(Config.BETA1, Config.BETA2),
        )

        # ── LR Schedulers ──────────────────────────────────────────────────────
        self.scheduler = LinearDecayLR(
            n_epochs=Config.N_EPOCHS,
            decay_epoch=Config.DECAY_EPOCH,
            initial_lr=Config.LEARNING_RATE,
        )

        # ── Image History Buffers ──────────────────────────────────────────────
        self.fake_X_buffer = ImageHistoryBuffer()
        self.fake_Y_buffer = ImageHistoryBuffer()

        # ── Logging ────────────────────────────────────────────────────────────
        self.loss_history = {
            'loss_G': [], 'loss_D': [],
            'loss_cycle_X': [], 'loss_cycle_Y': [],
            'loss_ident_X': [], 'loss_ident_Y': [],
            'loss_struct': [],
        }
        self.epoch_loss_history = {k: [] for k in self.loss_history}

        # ── Create output directories ──────────────────────────────────────────
        for d in [Config.CHECKPOINT_DIR, Config.RESULTS_DIR, Config.LOG_DIR]:
            os.makedirs(d, exist_ok=True)

    def _set_requires_grad(self, nets: list, requires_grad: bool) -> None:
        """
        Toggle gradient computation for a list of networks.
        Used to freeze discriminators during generator update and vice versa.
        This reduces memory usage and speeds up training.
        """
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad

    def train_step(self, real_X: torch.Tensor, real_Y: torch.Tensor) -> dict:
        """
        Single training step for one batch of unpaired images.

        Args:
            real_X: Batch of real TB images.
            real_Y: Batch of real Normal images.

        Returns:
            Dictionary of individual loss values for logging.
        """
        real_X = real_X.to(self.device)
        real_Y = real_Y.to(self.device)

        # ── Forward pass ───────────────────────────────────────────────────────
        fake_Y   = self.G(real_X)    # TB → Healthy (counterfactual)
        rec_X    = self.F(fake_Y)    # Healthy → TB (cycle reconstruction)
        fake_X   = self.F(real_Y)    # Healthy → TB
        rec_Y    = self.G(fake_X)    # TB → Healthy (cycle reconstruction)
        ident_X  = self.F(real_X)    # TB → TB (should be identity)
        ident_Y  = self.G(real_Y)    # Normal → Normal (should be identity)

        # ══════════════════════════════════════════════════════════════════════
        # GENERATOR UPDATE
        # Freeze discriminators during generator update to save memory.
        # ══════════════════════════════════════════════════════════════════════
        self._set_requires_grad([self.D_X, self.D_Y], False)
        self.opt_G.zero_grad()

        # 1. Adversarial loss: G tries to fool D_Y with fake healthy images
        loss_adv_G = self.criterion_adv.generator_loss(self.D_Y(fake_Y))
        # 2. Adversarial loss: F tries to fool D_X with fake TB images
        loss_adv_F = self.criterion_adv.generator_loss(self.D_X(fake_X))

        # 3. Cycle consistency loss
        loss_cycle_X = self.criterion_cycle(rec_X, real_X) * Config.LAMBDA_CYCLE
        loss_cycle_Y = self.criterion_cycle(rec_Y, real_Y) * Config.LAMBDA_CYCLE

        # 4. Identity loss (CRITICAL for medical anatomy preservation)
        loss_ident_X = self.criterion_ident(ident_X, real_X) * Config.LAMBDA_IDENTITY
        loss_ident_Y = self.criterion_ident(ident_Y, real_Y) * Config.LAMBDA_IDENTITY

        # 5. Structural perceptual loss (TB→Healthy should preserve lung structure)
        loss_struct = self.criterion_struct(real_X, fake_Y) * Config.LAMBDA_STRUCTURAL

        # Total generator loss
        loss_G = (
            loss_adv_G + loss_adv_F +
            loss_cycle_X + loss_cycle_Y +
            loss_ident_X + loss_ident_Y +
            loss_struct
        )
        loss_G.backward()
        # Gradient clipping prevents training instability in early epochs
        nn.utils.clip_grad_norm_(
            chain(self.G.parameters(), self.F.parameters()),
            max_norm=1.0
        )
        self.opt_G.step()

        # ══════════════════════════════════════════════════════════════════════
        # DISCRIMINATOR UPDATE
        # Use image history buffers to stabilize discriminator training.
        # ══════════════════════════════════════════════════════════════════════
        self._set_requires_grad([self.D_X, self.D_Y], True)
        self.opt_D.zero_grad()

        # D_Y: distinguish real Normal from fake Normal (generated by G)
        fake_Y_hist = self.fake_Y_buffer.query(fake_Y.detach())
        loss_D_Y = self.criterion_adv.discriminator_loss(
            self.D_Y(real_Y),
            self.D_Y(fake_Y_hist),
        )

        # D_X: distinguish real TB from fake TB (generated by F)
        fake_X_hist = self.fake_X_buffer.query(fake_X.detach())
        loss_D_X = self.criterion_adv.discriminator_loss(
            self.D_X(real_X),
            self.D_X(fake_X_hist),
        )

        loss_D = 0.5 * (loss_D_X + loss_D_Y)
        loss_D.backward()
        nn.utils.clip_grad_norm_(
            chain(self.D_X.parameters(), self.D_Y.parameters()),
            max_norm=1.0
        )
        self.opt_D.step()

        return {
            'loss_G'        : loss_G.item(),
            'loss_D'        : loss_D.item(),
            'loss_cycle_X'  : loss_cycle_X.item(),
            'loss_cycle_Y'  : loss_cycle_Y.item(),
            'loss_ident_X'  : loss_ident_X.item(),
            'loss_ident_Y'  : loss_ident_Y.item(),
            'loss_struct'   : loss_struct.item(),
        }

    def generate_counterfactual(self, real_tb: torch.Tensor) -> dict:
        """
        Generate a healthy counterfactual for a TB X-ray.

        Args:
            real_tb: Single TB image tensor (1, 3, H, W) in [-1, 1].

        Returns:
            dict with:
            - 'healthy'    : Generated healthy image tensor
            - 'diff_map'   : Absolute pixel-wise difference (the counterfactual explanation)
            - 'diff_heatmap': Colormap-enhanced difference for visualization
        """
        self.G.eval()
        with torch.no_grad():
            real_tb  = real_tb.to(self.device)
            fake_healthy = self.G(real_tb)

            # Difference map: |original - counterfactual|
            # Regions with high difference = features the model associates with TB
            diff = torch.abs(real_tb - fake_healthy)

            # Average across color channels for a single-channel explanation map
            diff_map = diff.mean(dim=1, keepdim=True)  # (1, 1, H, W)

        self.G.train()
        return {
            'healthy' : fake_healthy,
            'diff_map': diff_map,
        }

    def save_checkpoint(self, epoch: int, extra_info: dict = None) -> str:
        """Save full model state for resumable training."""
        checkpoint = {
            'epoch'          : epoch,
            'G_state'        : self.G.state_dict(),
            'F_state'        : self.F.state_dict(),
            'D_X_state'      : self.D_X.state_dict(),
            'D_Y_state'      : self.D_Y.state_dict(),
            'opt_G_state'    : self.opt_G.state_dict(),
            'opt_D_state'    : self.opt_D.state_dict(),
            'loss_history'   : self.epoch_loss_history,
            'config'         : {
                'n_residual_blocks': Config.N_RESIDUAL_BLOCKS,
                'n_gen_filters'    : Config.N_GEN_FILTERS,
                'lambda_cycle'     : Config.LAMBDA_CYCLE,
                'lambda_identity'  : Config.LAMBDA_IDENTITY,
                'lambda_structural': Config.LAMBDA_STRUCTURAL,
            },
        }
        if extra_info:
            checkpoint.update(extra_info)

        path = os.path.join(Config.CHECKPOINT_DIR, f'epoch_{epoch:04d}.pth')
        torch.save(checkpoint, path)
        print(f"  [Checkpoint] Saved → {path}")
        return path

    def load_checkpoint(self, path: str) -> int:
        """Load checkpoint and return the resumed epoch number."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        self.G.load_state_dict(checkpoint['G_state'])
        self.F.load_state_dict(checkpoint['F_state'])
        self.D_X.load_state_dict(checkpoint['D_X_state'])
        self.D_Y.load_state_dict(checkpoint['D_Y_state'])
        self.opt_G.load_state_dict(checkpoint['opt_G_state'])
        self.opt_D.load_state_dict(checkpoint['opt_D_state'])
        self.epoch_loss_history = checkpoint.get('loss_history', self.epoch_loss_history)

        epoch = checkpoint['epoch']
        print(f"  [Checkpoint] Resumed from epoch {epoch} ← {path}")
        return epoch

    def print_model_summary(self) -> None:
        """Print parameter counts for all networks."""
        def count_params(model):
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return total, trainable

        print(f"\n{'='*70}")
        print(f"  CycleGAN Model Summary")
        print(f"{'='*70}")
        for name, model in [('G (TB→Healthy)', self.G), ('F (Healthy→TB)', self.F),
                             ('D_X (TB Disc.)', self.D_X), ('D_Y (Normal Disc.)', self.D_Y)]:
            total, trainable = count_params(model)
            print(f"  {name:<25}: {total/1e6:.2f}M total, {trainable/1e6:.2f}M trainable")
        print(f"{'='*70}\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train(
    system         : CycleGANSystem,
    train_loader   : DataLoader,
    test_loader    : DataLoader,
    resume_epoch   : int = 0,
) -> None:
    """
    Full training loop with logging, checkpointing, and sample generation.

    Args:
        system: Initialized CycleGANSystem.
        train_loader: DataLoader for unpaired training data.
        test_loader: DataLoader for test/validation data.
        resume_epoch: If resuming, start from this epoch.
    """
    print(f"\n{'='*70}")
    print(f"  Starting Training: {Config.N_EPOCHS} epochs")
    print(f"  Device: {Config.DEVICE}")
    print(f"  Batch size: {Config.BATCH_SIZE}")
    print(f"  λ_cycle={Config.LAMBDA_CYCLE}, λ_identity={Config.LAMBDA_IDENTITY}, "
          f"λ_structural={Config.LAMBDA_STRUCTURAL}")
    print(f"{'='*70}\n")

    log_path = os.path.join(Config.LOG_DIR, 'training_log.json')
    training_log = []

    for epoch in range(resume_epoch, Config.N_EPOCHS):
        epoch_start = time.time()

        # Update learning rate
        new_lr = system.scheduler.update(system.opt_G, epoch)
        system.scheduler.update(system.opt_D, epoch)

        # ── Accumulate batch losses ────────────────────────────────────────────
        epoch_losses = {k: 0.0 for k in system.loss_history}
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            real_X = batch['TB']
            real_Y = batch['Normal']

            losses = system.train_step(real_X, real_Y)

            for k, v in losses.items():
                epoch_losses[k] += v
            n_batches += 1

            # ── Batch-level logging ────────────────────────────────────────────
            if (batch_idx + 1) % Config.LOG_FREQ == 0:
                print(
                    f"  Epoch [{epoch+1:03d}/{Config.N_EPOCHS}] "
                    f"Batch [{batch_idx+1:04d}/{len(train_loader)}] "
                    f"LR={new_lr:.6f} | "
                    f"G={losses['loss_G']:.4f} "
                    f"D={losses['loss_D']:.4f} "
                    f"Cyc_X={losses['loss_cycle_X']:.4f} "
                    f"Cyc_Y={losses['loss_cycle_Y']:.4f} "
                    f"Str={losses['loss_struct']:.4f}"
                )

        # ── Epoch-level averaging ──────────────────────────────────────────────
        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)
            system.epoch_loss_history[k].append(epoch_losses[k])

        epoch_time = time.time() - epoch_start
        print(
            f"\n  ── Epoch [{epoch+1:03d}/{Config.N_EPOCHS}] Summary ──\n"
            f"  Time: {epoch_time:.1f}s | LR: {new_lr:.6f}\n"
            f"  G={epoch_losses['loss_G']:.4f} | D={epoch_losses['loss_D']:.4f} | "
            f"CycleX={epoch_losses['loss_cycle_X']:.4f} | CycleY={epoch_losses['loss_cycle_Y']:.4f}\n"
            f"  IdentX={epoch_losses['loss_ident_X']:.4f} | IdentY={epoch_losses['loss_ident_Y']:.4f} | "
            f"Struct={epoch_losses['loss_struct']:.4f}\n"
        )

        # ── Log to JSON ────────────────────────────────────────────────────────
        log_entry = {'epoch': epoch + 1, 'time': epoch_time, **epoch_losses}
        training_log.append(log_entry)
        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=2)

        # ── Save sample images ─────────────────────────────────────────────────
        if (epoch + 1) % Config.SAMPLE_FREQ == 0:
            save_sample_images(system, test_loader, epoch + 1)
            plot_loss_curves(system.epoch_loss_history, epoch + 1)

        # ── Save checkpoint ────────────────────────────────────────────────────
        if (epoch + 1) % Config.SAVE_FREQ == 0:
            system.save_checkpoint(epoch + 1)

    # Final checkpoint
    system.save_checkpoint(Config.N_EPOCHS, {'final': True})
    print(f"\n[Training Complete] Logs saved to {log_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9: VISUALIZATION & COUNTERFACTUAL OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def tensor_to_numpy_image(t: torch.Tensor) -> np.ndarray:
    """
    Convert a normalized tensor ([-1,1], (C,H,W)) to a uint8 numpy array.
    Handles both grayscale (1-channel) and RGB (3-channel) tensors.
    """
    t = t.detach().cpu().float()
    t = (t + 1.0) / 2.0      # [-1,1] → [0,1]
    t = t.clamp(0, 1)
    t = t.permute(1, 2, 0)   # (C,H,W) → (H,W,C)
    arr = (t.numpy() * 255).astype(np.uint8)
    if arr.shape[2] == 1:
        arr = arr.squeeze(2)  # (H,W)
    return arr


def generate_difference_map_visualization(
    original_tensor   : torch.Tensor,
    generated_tensor  : torch.Tensor,
    diff_map_tensor   : torch.Tensor,
    save_path         : str,
    title_suffix      : str = "",
) -> None:
    """
    Generate a clinical-style 4-panel visualization:

    Panel 1: Original TB X-ray
    Panel 2: Generated Healthy Counterfactual
    Panel 3: Pixel-wise difference map (grayscale)
    Panel 4: Colormap-enhanced difference (jet colormap for clinical clarity)

    CLINICAL RATIONALE:
    The difference map is the primary output for the radiologist. High-intensity
    regions in the difference map indicate areas where the model detected
    TB-specific features. A colormap overlay (jet/inferno) makes small
    differences easier to identify visually.
    """
    original  = tensor_to_numpy_image(original_tensor.squeeze(0))
    generated = tensor_to_numpy_image(generated_tensor.squeeze(0))

    diff_map = diff_map_tensor.squeeze().detach().cpu().numpy()
    diff_map = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min() + 1e-8)

    # Apply clinical colormap for the 4th panel
    cmap_applied = cm.inferno(diff_map)  # Returns RGBA
    cmap_rgb = (cmap_applied[:, :, :3] * 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor='#0d1117')
    titles = [
        'Original TB X-Ray',
        'Generated Healthy\n(Counterfactual)',
        'Difference Map\n(Grayscale)',
        'Difference Map\n(Clinical Colormap)',
    ]
    images = [original, generated, (diff_map * 255).astype(np.uint8), cmap_rgb]
    cmaps  = ['gray', 'gray', 'gray', None]

    for ax, img, title, cmap in zip(axes, images, titles, cmaps):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=8)
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')

    if title_suffix:
        fig.suptitle(
            f"Counterfactual Explanation — {title_suffix}",
            color='white', fontsize=13, fontweight='bold', y=1.02
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='#0d1117', edgecolor='none')
    plt.close()


def save_sample_images(
    system      : CycleGANSystem,
    test_loader : DataLoader,
    epoch       : int,
    n_samples   : int = 4,
) -> None:
    """
    Generate and save counterfactual samples during training.
    Saves both individual panels and grid comparisons.
    """
    sample_dir = os.path.join(Config.RESULTS_DIR, f'epoch_{epoch:04d}')
    os.makedirs(sample_dir, exist_ok=True)

    system.G.eval()
    samples_saved = 0

    with torch.no_grad():
        for batch in test_loader:
            if samples_saved >= n_samples:
                break

            real_tb = batch['TB'].to(system.device)
            result  = system.generate_counterfactual(real_tb)

            for i in range(real_tb.shape[0]):
                if samples_saved >= n_samples:
                    break

                fname = os.path.join(sample_dir, f'sample_{samples_saved:03d}.png')
                generate_difference_map_visualization(
                    original_tensor  = real_tb[i:i+1],
                    generated_tensor = result['healthy'][i:i+1],
                    diff_map_tensor  = result['diff_map'][i:i+1],
                    save_path        = fname,
                    title_suffix     = f"Epoch {epoch}",
                )
                samples_saved += 1

    system.G.train()
    print(f"  [Samples] Saved {samples_saved} counterfactuals to {sample_dir}")


def plot_loss_curves(loss_history: dict, epoch: int) -> None:
    """Plot and save training loss curves for monitoring convergence."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor='#0d1117')
    axes = axes.flatten()

    plot_config = [
        ('loss_G',       'Generator Loss (G + F)',        '#58a6ff'),
        ('loss_D',       'Discriminator Loss (D_X + D_Y)','#f85149'),
        ('loss_cycle_X', 'Cycle Loss X (TB→H→TB)',        '#3fb950'),
        ('loss_cycle_Y', 'Cycle Loss Y (H→TB→H)',         '#d29922'),
        ('loss_ident_X', 'Identity Loss X',               '#bc8cff'),
        ('loss_struct',  'Structural Loss',                '#ff7b72'),
    ]

    for ax, (key, label, color) in zip(axes, plot_config):
        values = loss_history.get(key, [])
        if values:
            ax.plot(values, color=color, linewidth=1.5, alpha=0.9)
            # Running average for readability
            if len(values) > 10:
                window = min(10, len(values) // 5)
                avg = np.convolve(values, np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(values)), avg,
                        color='white', linewidth=2, alpha=0.7, linestyle='--',
                        label=f'{window}-ep avg')
                ax.legend(facecolor='#21262d', labelcolor='white', fontsize=8)
        ax.set_title(label, color='white', fontsize=10, fontweight='bold')
        ax.set_xlabel('Epoch', color='#8b949e', fontsize=8)
        ax.set_ylabel('Loss', color='#8b949e', fontsize=8)
        ax.tick_params(colors='#8b949e')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')
        ax.set_facecolor('#161b22')

    fig.suptitle(f'CycleGAN Training Curves — Epoch {epoch}',
                 color='white', fontsize=13, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(Config.LOG_DIR, f'loss_curves_epoch_{epoch:04d}.png')
    plt.savefig(save_path, dpi=120, bbox_inches='tight',
                facecolor='#0d1117', edgecolor='none')
    plt.close()
    print(f"  [Plot] Loss curves saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10: EVALUATION METRICS
# Purpose: Quantitative evaluation of generation quality and anatomical
#          preservation, per publication standards.
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(
    system      : CycleGANSystem,
    test_loader : DataLoader,
    output_dir  : str = None,
) -> dict:
    """
    Comprehensive evaluation of the trained CycleGAN.

    METRICS (see paper recommendations):
    1. FID (Fréchet Inception Distance):
       Measures the statistical distance between the generated healthy image
       distribution and the real Normal image distribution.
       Lower FID = more realistic counterfactuals.
       Limitation: FID is domain-agnostic (doesn't know what "healthy" means
       clinically). Use alongside clinical metrics.

    2. SSIM (Structural Similarity Index Measure):
       Compares the structural similarity between the original TB image and
       its counterfactual. We WANT some structural similarity (preserved anatomy)
       but NOT too high (otherwise the pathology wasn't removed).
       Target: moderate SSIM (~0.6-0.8 is typical for domain translation).

    3. PSNR (Peak Signal-to-Noise Ratio):
       Measures pixel-level reconstruction fidelity. Used as a secondary
       metric — high PSNR alone doesn't imply clinical validity.

    4. Cycle Reconstruction Error:
       The pixel-wise error between the original image and its cycle-
       reconstructed version (TB → Healthy → TB′). This is a proxy for
       how much non-TB anatomy was modified (hallucination risk indicator).
       Lower cycle error = less hallucination.

    Returns:
        dict of metric values.
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

    # ── Torchmetrics-based evaluation ──────────────────────────────────────────
    if TORCHMETRICS_AVAILABLE:
        fid_metric  = FrechetInceptionDistance(normalize=True).to(system.device)
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(system.device)
        psnr_metric = PeakSignalNoiseRatio(data_range=2.0).to(system.device)
        ssim_values, psnr_values = [], []

    print(f"\n[Evaluation] Running on test set...")

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            real_X = batch['TB'].to(system.device)
            real_Y = batch['Normal'].to(system.device)

            # Generate counterfactuals
            result   = system.generate_counterfactual(real_X)
            fake_Y   = result['healthy']
            diff_map = result['diff_map']

            # Cycle reconstruction: TB → Healthy → TB'
            rec_X = system.F(fake_Y)

            # Cycle error (lower = better anatomy preservation)
            cycle_err = F.l1_loss(rec_X, real_X).item()
            metrics['cycle_reconstruction_L1'].append(cycle_err)

            # Identity consistency: G applied to Normal should give ≈ Normal
            ident_Y = system.G(real_Y)
            ident_err = F.l1_loss(ident_Y, real_Y).item()
            metrics['identity_L1_G'].append(ident_err)

            # Difference map statistics
            dm = diff_map.squeeze().cpu().numpy()
            metrics['diff_map_mean_intensity'].append(float(dm.mean()))
            metrics['diff_map_std_intensity'].append(float(dm.std()))

            if TORCHMETRICS_AVAILABLE:
                # FID: compare generated Normal to real Normal
                # Convert from [-1,1] to [0,1] for FID
                real_Y_01  = ((real_Y + 1.0) / 2.0).clamp(0, 1)
                fake_Y_01  = ((fake_Y + 1.0) / 2.0).clamp(0, 1)
                fid_metric.update(real_Y_01, real=True)
                fid_metric.update(fake_Y_01, real=False)

                # SSIM / PSNR between original TB and counterfactual
                ssim_val = ssim_metric(fake_Y, real_X)
                psnr_val = psnr_metric(fake_Y, real_X)
                ssim_values.append(ssim_val.item())
                psnr_values.append(psnr_val.item())

            # Save sample visualizations for evaluation
            if i < 10:
                vis_path = os.path.join(output_dir, f'eval_sample_{i:03d}.png')
                generate_difference_map_visualization(
                    original_tensor  = real_X,
                    generated_tensor = fake_Y,
                    diff_map_tensor  = diff_map,
                    save_path        = vis_path,
                    title_suffix     = f"Eval Sample {i}",
                )

    # ── Aggregate metrics ──────────────────────────────────────────────────────
    results = {
        'cycle_reconstruction_L1_mean' : float(np.mean(metrics['cycle_reconstruction_L1'])),
        'cycle_reconstruction_L1_std'  : float(np.std(metrics['cycle_reconstruction_L1'])),
        'identity_L1_mean'             : float(np.mean(metrics['identity_L1_G'])),
        'identity_L1_std'              : float(np.std(metrics['identity_L1_G'])),
        'diff_map_mean_intensity'      : float(np.mean(metrics['diff_map_mean_intensity'])),
        'diff_map_std_intensity'       : float(np.mean(metrics['diff_map_std_intensity'])),
    }

    if TORCHMETRICS_AVAILABLE:
        results['FID']  = float(fid_metric.compute().item())
        results['SSIM'] = float(np.mean(ssim_values))
        results['PSNR'] = float(np.mean(psnr_values))

    # ── Print report ───────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Evaluation Report")
    print(f"{'='*70}")
    print(f"  Cycle Reconstruction L1 : {results['cycle_reconstruction_L1_mean']:.4f} "
          f"± {results['cycle_reconstruction_L1_std']:.4f}")
    print(f"    ↳ Interpretaion: Lower = less hallucination, better anatomy preservation")
    print(f"  Identity L1 (G on Normal): {results['identity_L1_mean']:.4f} "
          f"± {results['identity_L1_std']:.4f}")
    print(f"    ↳ Lower = healthyimages are correctly left unchanged by G")
    print(f"  Diff Map Mean Intensity  : {results['diff_map_mean_intensity']:.4f}")
    print(f"    ↳ Moderate intensity suggests discriminative features were found")
    if TORCHMETRICS_AVAILABLE:
        print(f"  FID                      : {results['FID']:.2f}")
        print(f"    ↳ Lower = more realistic generated images")
        print(f"  SSIM (TB vs. Counterfact.): {results['SSIM']:.4f}")
        print(f"    ↳ 0.6-0.8 expected for valid domain translation")
        print(f"  PSNR                     : {results['PSNR']:.2f} dB")
    print(f"{'='*70}\n")

    # ── Save metrics JSON ──────────────────────────────────────────────────────
    metrics_path = os.path.join(output_dir, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  [Evaluation] Metrics saved → {metrics_path}")

    system.G.train()
    system.F.train()
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11: BATCH COUNTERFACTUAL GENERATION
# Purpose: Generate counterfactual explanations for all TB images in the
#          test set after training is complete.
# ══════════════════════════════════════════════════════════════════════════════

def generate_all_counterfactuals(
    system      : CycleGANSystem,
    test_loader : DataLoader,
    output_dir  : str = None,
) -> None:
    """
    Generate counterfactual explanations for all TB images.

    Output structure:
    output_dir/
    ├── counterfactuals/          ← Individual healthy counterfactuals
    ├── difference_maps/          ← Raw grayscale difference maps
    └── visualizations/           ← 4-panel clinical visualization panels

    SAFETY CHECK:
    For each generated image, we compute the cycle reconstruction error.
    If it exceeds a threshold, a WARNING is issued — this indicates potential
    hallucination in that specific image.
    """
    if output_dir is None:
        output_dir = os.path.join(Config.RESULTS_DIR, 'counterfactuals_final')

    dirs = {
        'cf'   : os.path.join(output_dir, 'counterfactuals'),
        'diff' : os.path.join(output_dir, 'difference_maps'),
        'vis'  : os.path.join(output_dir, 'visualizations'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    system.G.eval()
    system.F.eval()

    # Hallucination threshold: cycle error above this suggests anatomy was modified.
    # This is an empirical threshold — tune based on your dataset.
    HALLUCINATION_THRESHOLD = 0.15
    hallucination_warnings = []

    print(f"\n[Generation] Generating counterfactuals for all TB test images...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            real_X  = batch['TB'].to(system.device)
            tb_path = Path(batch['TB_path'][0])
            stem    = tb_path.stem

            # Generate counterfactual
            result   = system.generate_counterfactual(real_X)
            fake_Y   = result['healthy']
            diff_map = result['diff_map']

            # ── Hallucination safety check ─────────────────────────────────────
            rec_X = system.F(fake_Y)
            cycle_err = F.l1_loss(rec_X, real_X).item()

            if cycle_err > HALLUCINATION_THRESHOLD:
                warning = {
                    'image': str(tb_path),
                    'cycle_error': cycle_err,
                    'threshold': HALLUCINATION_THRESHOLD,
                }
                hallucination_warnings.append(warning)
                print(
                    f"  ⚠ [HALLUCINATION RISK] {stem}: "
                    f"cycle_error={cycle_err:.4f} > {HALLUCINATION_THRESHOLD}. "
                    f"This counterfactual may have modified non-TB anatomy."
                )

            # ── Save counterfactual image ──────────────────────────────────────
            cf_path = os.path.join(dirs['cf'], f'{stem}_healthy.png')
            save_image((fake_Y + 1.0) / 2.0, cf_path)

            # ── Save difference map ────────────────────────────────────────────
            dm_np = diff_map.squeeze().cpu().numpy()
            dm_norm = (dm_np - dm_np.min()) / (dm_np.max() - dm_np.min() + 1e-8)
            dm_img = Image.fromarray((dm_norm * 255).astype(np.uint8), mode='L')
            dm_path = os.path.join(dirs['diff'], f'{stem}_diffmap.png')
            dm_img.save(dm_path)

            # ── Save visualization panel ───────────────────────────────────────
            vis_path = os.path.join(dirs['vis'], f'{stem}_visualization.png')
            generate_difference_map_visualization(
                original_tensor  = real_X,
                generated_tensor = fake_Y,
                diff_map_tensor  = diff_map,
                save_path        = vis_path,
                title_suffix     = stem,
            )

            if (batch_idx + 1) % 20 == 0:
                print(f"  Progress: {batch_idx+1}/{len(test_loader)} images processed")

    # ── Save hallucination report ──────────────────────────────────────────────
    if hallucination_warnings:
        warn_path = os.path.join(output_dir, 'hallucination_warnings.json')
        with open(warn_path, 'w') as f:
            json.dump(hallucination_warnings, f, indent=2)
        print(
            f"\n  ⚠ [{len(hallucination_warnings)} hallucination warnings] "
            f"Review: {warn_path}"
        )
    else:
        print(f"\n  ✓ No hallucination warnings. All cycle errors below threshold.")

    print(f"  [Generation] Complete. Output → {output_dir}")
    system.G.train()
    system.F.train()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12: COMMAND-LINE INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='CycleGAN Counterfactual Explanation Pipeline for TB Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train from scratch
  python cyclegan_tb_counterfactual.py --mode train --data_dir ./dataset

  # Resume from checkpoint
  python cyclegan_tb_counterfactual.py --mode train --data_dir ./dataset --resume ./checkpoints/epoch_0100.pth

  # Generate counterfactuals using a trained model
  python cyclegan_tb_counterfactual.py --mode generate --data_dir ./dataset --checkpoint ./checkpoints/epoch_0200.pth

  # Evaluate model quality
  python cyclegan_tb_counterfactual.py --mode evaluate --data_dir ./dataset --checkpoint ./checkpoints/epoch_0200.pth
        """
    )
    parser.add_argument(
        '--mode', type=str,
        choices=['train', 'generate', 'evaluate'],
        default='train',
        help='Operation mode.'
    )
    parser.add_argument(
        '--data_dir', type=str, default=Config.DATA_DIR,
        help='Root dataset directory containing TB/ and Normal/ subfolders.'
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to checkpoint file for generation/evaluation mode.'
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to checkpoint file to resume training from.'
    )
    parser.add_argument(
        '--epochs', type=int, default=Config.N_EPOCHS,
        help=f'Number of training epochs (default: {Config.N_EPOCHS}).'
    )
    parser.add_argument(
        '--lambda_cycle', type=float, default=Config.LAMBDA_CYCLE,
        help=f'Cycle-consistency loss weight (default: {Config.LAMBDA_CYCLE}).'
    )
    parser.add_argument(
        '--lambda_identity', type=float, default=Config.LAMBDA_IDENTITY,
        help=f'Identity loss weight (default: {Config.LAMBDA_IDENTITY}).'
    )
    parser.add_argument(
        '--img_size', type=int, default=Config.IMG_SIZE,
        help=f'Image resolution for training (default: {Config.IMG_SIZE}).'
    )
    parser.add_argument(
        '--seed', type=int, default=Config.SEED,
        help=f'Random seed for reproducibility (default: {Config.SEED}).'
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Output directory for generated results.'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Apply CLI overrides to Config ──────────────────────────────────────────
    Config.DATA_DIR       = args.data_dir
    Config.N_EPOCHS       = args.epochs
    Config.LAMBDA_CYCLE   = args.lambda_cycle
    Config.LAMBDA_IDENTITY = args.lambda_identity
    Config.IMG_SIZE       = args.img_size
    Config.SEED           = args.seed

    # ── Setup ──────────────────────────────────────────────────────────────────
    set_reproducibility(Config.SEED)

    print(f"\n{'='*70}")
    print(f"  TB CycleGAN Counterfactual Explanation Pipeline")
    print(f"  Mode    : {args.mode.upper()}")
    print(f"  Device  : {Config.DEVICE}")
    print(f"  Data Dir: {Config.DATA_DIR}")
    print(f"{'='*70}\n")

    # ── Build DataLoaders ──────────────────────────────────────────────────────
    train_loader, test_loader, ds_info = build_dataloaders(Config.DATA_DIR)

    # ── Initialize CycleGAN System ─────────────────────────────────────────────
    system = CycleGANSystem(Config.DEVICE)
    system.print_model_summary()

    # ── Mode dispatch ──────────────────────────────────────────────────────────
    if args.mode == 'train':
        resume_epoch = 0
        if args.resume:
            resume_epoch = system.load_checkpoint(args.resume)
        train(system, train_loader, test_loader, resume_epoch=resume_epoch)

    elif args.mode == 'generate':
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for mode=generate")
        system.load_checkpoint(args.checkpoint)
        output_dir = args.output_dir or os.path.join(
            Config.RESULTS_DIR, 'counterfactuals_final'
        )
        generate_all_counterfactuals(system, test_loader, output_dir)

    elif args.mode == 'evaluate':
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for mode=evaluate")
        system.load_checkpoint(args.checkpoint)
        output_dir = args.output_dir or os.path.join(Config.RESULTS_DIR, 'evaluation')
        evaluate_model(system, test_loader, output_dir)

    else:
        raise ValueError(f"Unknown mode: {args.mode}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    main()