"""
CycleGAN Inference Script for TB ↔ Normal Chest X-ray Translation
This script loads a trained CycleGAN generator and performs inference on single images.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path


# ============================================================================
# Generator Architecture (Standard CycleGAN ResNet Generator)
# ============================================================================

class ResidualBlock(nn.Module):
    """Residual block with two 3x3 conv layers"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels, affine=False),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels, affine=False)
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """CycleGAN Generator with ResNet blocks"""
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9):
        super(Generator, self).__init__()
        
        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, 7, bias=False),
            nn.InstanceNorm2d(ngf, affine=False),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, 3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ngf * mult * 2, affine=False),
                nn.ReLU(inplace=True)
            ]
        
        # Residual blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResidualBlock(ngf * mult)]
        
        # Upsampling
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), 3,
                                   stride=2, padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(int(ngf * mult / 2), affine=False),
                nn.ReLU(inplace=True)
            ]
        
        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, 7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)


# ============================================================================
# Checkpoint Loading with Multiple Format Support
# ============================================================================

def load_generator(checkpoint_path, device='cuda', generator_key=None):
    """
    Load generator from checkpoint with robust handling of different formats.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: 'cuda' or 'cpu'
        generator_key: Specific key to look for (e.g., 'G_AB', 'G_BA', 'G_state')
                       If None, will auto-detect
    
    Returns:
        Loaded generator model in eval mode
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize generator (6 residual blocks to match training)
    generator = Generator(input_nc=3, output_nc=3, ngf=64, n_blocks=6)
    
    # Try different checkpoint formats
    loaded = False
    
    # Format 1: Direct state_dict
    if isinstance(checkpoint, dict) and 'model' in str(type(checkpoint.get(list(checkpoint.keys())[0]))):
        try:
            generator.load_state_dict(checkpoint)
            loaded = True
            print("✓ Loaded from direct state_dict format")
        except:
            pass
    
    # Format 2: Nested with generator key
    if not loaded and isinstance(checkpoint, dict):
        # Common generator keys
        possible_keys = ['G_AB', 'G_BA', 'netG_A', 'netG_B', 'G_state', 'generator', 'gen_state_dict']
        
        if generator_key:
            possible_keys.insert(0, generator_key)
        
        for key in possible_keys:
            if key in checkpoint:
                try:
                    generator.load_state_dict(checkpoint[key])
                    loaded = True
                    print(f"✓ Loaded from checkpoint['{key}']")
                    break
                except Exception as e:
                    print(f"  Failed to load from '{key}': {e}")
                    continue
    
    # Format 3: Checkpoint has 'state_dict' wrapper
    if not loaded and isinstance(checkpoint, dict):
        for key in checkpoint.keys():
            if 'state_dict' in key.lower():
                try:
                    generator.load_state_dict(checkpoint[key])
                    loaded = True
                    print(f"✓ Loaded from checkpoint['{key}']")
                    break
                except:
                    continue
    
    if not loaded:
        print("\n⚠ Available keys in checkpoint:")
        if isinstance(checkpoint, dict):
            for key in checkpoint.keys():
                print(f"  - {key}: {type(checkpoint[key])}")
        raise ValueError(
            "Could not load generator from checkpoint. "
            "Please specify the correct generator_key parameter. "
            "Common keys: 'G_AB', 'G_BA', 'netG_A', 'netG_B', 'G_state'"
        )
    
    generator.to(device)
    generator.eval()
    return generator


# ============================================================================
# Image Preprocessing and Postprocessing
# ============================================================================

def preprocess_image(image_path, image_size=256):
    """
    Load and preprocess chest X-ray image.
    
    Args:
        image_path: Path to input image
        image_size: Size to resize to (default 256x256)
    
    Returns:
        Preprocessed tensor ready for model input
    """
    # Load image
    img = Image.open(image_path).convert('RGB')  # Convert to RGB
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    
    # Apply transforms and add batch dimension
    img_tensor = transform(img).unsqueeze(0)
    
    return img_tensor, img


def postprocess_image(tensor):
    """
    Convert model output tensor back to numpy image.
    
    Args:
        tensor: Output tensor from generator (range: [-1, 1])
    
    Returns:
        Numpy array (range: [0, 255])
    """
    # Remove batch dimension and denormalize
    img = tensor.squeeze(0).cpu().detach().numpy()  # (C, H, W)
    # Move channel dim to last for PIL: (H, W, C)
    if img.shape[0] == 3:  # RGB
        img = np.transpose(img, (1, 2, 0))
    img = (img + 1) / 2.0  # Denormalize from [-1, 1] to [0, 1]
    img = (img * 255).astype(np.uint8)
    
    return img


# ============================================================================
# Inference and Visualization
# ============================================================================

def run_inference(generator, image_path, output_dir, device='cuda', image_size=256):
    """
    Run inference on a single image and save results.
    
    Args:
        generator: Loaded generator model
        image_path: Path to input image
        output_dir: Directory to save outputs
        device: 'cuda' or 'cpu'
        image_size: Input image size
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Preprocess
    print(f"\nProcessing: {image_path}")
    input_tensor, original_pil = preprocess_image(image_path, image_size)
    input_tensor = input_tensor.to(device)
    
    # Generate
    with torch.no_grad():
        output_tensor = generator(input_tensor)
    
    # Postprocess
    input_img = postprocess_image(input_tensor)
    output_img = postprocess_image(output_tensor)
    
    # Compute difference map (convert to grayscale if RGB)
    if input_img.ndim == 3 and input_img.shape[2] == 3:
        # Convert RGB to grayscale for difference
        input_gray = 0.299 * input_img[:,:,0] + 0.587 * input_img[:,:,1] + 0.114 * input_img[:,:,2]
        output_gray = 0.299 * output_img[:,:,0] + 0.587 * output_img[:,:,1] + 0.114 * output_img[:,:,2]
        difference = np.abs(input_gray.astype(float) - output_gray.astype(float)).astype(np.uint8)
    else:
        difference = np.abs(input_img.astype(float) - output_img.astype(float)).astype(np.uint8)
    
    # Get base filename
    base_name = Path(image_path).stem
    
    # Save individual images
    Image.fromarray(input_img).save(os.path.join(output_dir, f"{base_name}_input.png"))
    Image.fromarray(output_img).save(os.path.join(output_dir, f"{base_name}_generated.png"))
    Image.fromarray(difference).save(os.path.join(output_dir, f"{base_name}_difference.png"))
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Original image
    axes[0, 0].imshow(input_img)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Generated image
    axes[0, 1].imshow(output_img)
    axes[0, 1].set_title('Generated Image', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Difference map
    im_diff = axes[1, 0].imshow(difference, cmap='hot')
    axes[1, 0].set_title('Absolute Difference Map', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im_diff, ax=axes[1, 0], fraction=0.046)
    
    # Overlay (difference highlighted on original)
    overlay = input_img.copy()
    overlay[:, :, 0] = np.clip(overlay[:, :, 0].astype(float) + difference, 0, 255).astype(np.uint8)
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Difference Overlay (Red)', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Compute and print statistics
    print(f"\n{'='*60}")
    print(f"Statistics for {base_name}:")
    print(f"{'='*60}")
    print(f"Mean Absolute Difference: {difference.mean():.2f}")
    print(f"Max Difference: {difference.max()}")
    print(f"Std Difference: {difference.std():.2f}")
    print(f"Percentage of pixels changed >10: {(difference > 10).sum() / difference.size * 100:.2f}%")
    print(f"Percentage of pixels changed >50: {(difference > 50).sum() / difference.size * 100:.2f}%")
    print(f"{'='*60}\n")
    
    print(f"✓ Saved outputs to: {output_dir}")
    print(f"  - {base_name}_input.png")
    print(f"  - {base_name}_generated.png")
    print(f"  - {base_name}_difference.png")
    print(f"  - {base_name}_comparison.png")
    
    return {
        'input': input_img,
        'output': output_img,
        'difference': difference,
        'stats': {
            'mean_diff': difference.mean(),
            'max_diff': difference.max(),
            'std_diff': difference.std(),
            'pct_changed_10': (difference > 10).sum() / difference.size * 100,
            'pct_changed_50': (difference > 50).sum() / difference.size * 100
        }
    }


# ============================================================================
# Batch Inference
# ============================================================================

def batch_inference(generator, input_dir, output_dir, device='cuda', image_size=256):
    """
    Run inference on all images in a directory.
    
    Args:
        generator: Loaded generator model
        input_dir: Directory containing input images
        output_dir: Directory to save outputs
        device: 'cuda' or 'cpu'
        image_size: Input image size
    """
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_dir).glob(f'*{ext}'))
        image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
    
    print(f"Found {len(image_files)} images in {input_dir}")
    
    results = []
    for img_path in image_files:
        result = run_inference(generator, str(img_path), output_dir, device, image_size)
        results.append(result)
    
    # Aggregate statistics
    print(f"\n{'='*60}")
    print("AGGREGATE STATISTICS:")
    print(f"{'='*60}")
    mean_diffs = [r['stats']['mean_diff'] for r in results]
    print(f"Average Mean Difference: {np.mean(mean_diffs):.2f} ± {np.std(mean_diffs):.2f}")
    print(f"Range: [{np.min(mean_diffs):.2f}, {np.max(mean_diffs):.2f}]")
    print(f"{'='*60}\n")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='CycleGAN Inference for TB ↔ Normal X-ray Translation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to generator checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input image or directory')
    parser.add_argument('--output', type=str, default='./inference_results',
                        help='Output directory for results')
    parser.add_argument('--generator_key', type=str, default=None,
                        help='Specific key in checkpoint (e.g., G_AB, G_BA)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device to run inference on')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size (default: 256)')
    
    args = parser.parse_args()
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠ CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    print(f"Using device: {args.device}")
    
    # Load generator
    generator = load_generator(args.checkpoint, args.device, args.generator_key)
    
    # Run inference
    input_path = Path(args.input)
    if input_path.is_file():
        # Single image
        run_inference(generator, str(input_path), args.output, args.device, args.image_size)
    elif input_path.is_dir():
        # Directory of images
        batch_inference(generator, str(input_path), args.output, args.device, args.image_size)
    else:
        raise ValueError(f"Input path does not exist: {args.input}")
    
    print("\n✅ Inference complete!")


if __name__ == '__main__':
    main()