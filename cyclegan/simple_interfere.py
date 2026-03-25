"""
Simple CycleGAN Inference Function - Notebook Friendly
Copy this into your Jupyter notebook for quick testing
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# Generator Architecture
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, ngf=64, n_blocks=9):
        super(Generator, self).__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling
        for i in range(2):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, 3, stride=2, padding=1),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(inplace=True)
            ]
        
        # Residual blocks
        for i in range(n_blocks):
            model += [ResidualBlock(ngf * 4)]
        
        # Upsampling
        for i in range(2):
            mult = 2 ** (2 - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), 3,
                                   stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(int(ngf * mult / 2)),
                nn.ReLU(inplace=True)
            ]
        
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, 7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)


def test_single_image(checkpoint_path, image_path, generator_key='G_AB', save_outputs=True):
    """
    Quick function to test a single image.
    
    Args:
        checkpoint_path: Path to your checkpoint file
        image_path: Path to input X-ray image
        generator_key: Key in checkpoint ('G_AB' for TB→Normal, 'G_BA' for Normal→TB)
        save_outputs: Whether to save output images
    
    Returns:
        Dictionary with input, output, and difference images
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load generator
    print("Loading generator...")
    generator = Generator(input_nc=1, output_nc=1, ngf=64, n_blocks=9)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to load state dict
    if generator_key in checkpoint:
        generator.load_state_dict(checkpoint[generator_key])
        print(f"✓ Loaded from checkpoint['{generator_key}']")
    else:
        print(f"Available keys: {list(checkpoint.keys())}")
        raise ValueError(f"Key '{generator_key}' not found in checkpoint")
    
    generator.to(device)
    generator.eval()
    
    # Load and preprocess image
    print(f"Loading image: {image_path}")
    img = Image.open(image_path).convert('L')
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    # Generate
    print("Generating output...")
    with torch.no_grad():
        output_tensor = generator(input_tensor)
    
    # Convert to numpy
    input_np = ((input_tensor.squeeze().cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
    output_np = ((output_tensor.squeeze().cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
    diff_np = np.abs(input_np.astype(float) - output_np.astype(float)).astype(np.uint8)
    
    # Visualize
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(input_np, cmap='gray')
    axes[0].set_title('Input', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(output_np, cmap='gray')
    axes[1].set_title('Generated', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(diff_np, cmap='hot')
    axes[2].set_title('Difference', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # Side-by-side comparison
    comparison = np.hstack([input_np, output_np])
    axes[3].imshow(comparison, cmap='gray')
    axes[3].set_title('Side-by-Side', fontsize=12, fontweight='bold')
    axes[3].axis('off')
    axes[3].axvline(x=256, color='red', linestyle='--', linewidth=2)
    
    plt.tight_layout()
    
    if save_outputs:
        plt.savefig('inference_result.png', dpi=150, bbox_inches='tight')
        Image.fromarray(input_np).save('input.png')
        Image.fromarray(output_np).save('generated.png')
        Image.fromarray(diff_np).save('difference.png')
        print("\n✓ Saved: inference_result.png, input.png, generated.png, difference.png")
    
    plt.show()
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"  Mean difference: {diff_np.mean():.2f}")
    print(f"  Max difference: {diff_np.max()}")
    print(f"  Pixels changed >10: {(diff_np > 10).sum() / diff_np.size * 100:.1f}%")
    print(f"  Pixels changed >50: {(diff_np > 50).sum() / diff_np.size * 100:.1f}%")
    
    return {
        'input': input_np,
        'output': output_np,
        'difference': diff_np
    }


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

"""
# Example 1: Quick test with automatic key detection
result = test_single_image(
    checkpoint_path='checkpoints/epoch_100.pth',
    image_path='test_images/tb_sample.png',
    generator_key='G_AB'  # TB → Normal
)

# Example 2: Test reverse direction
result = test_single_image(
    checkpoint_path='checkpoints/epoch_100.pth',
    image_path='test_images/normal_sample.png',
    generator_key='G_BA'  # Normal → TB
)

# Example 3: Access outputs programmatically
result = test_single_image('checkpoint.pth', 'image.png', save_outputs=False)
input_img = result['input']
output_img = result['output']
diff_img = result['difference']

# Calculate custom metrics
mean_intensity_change = np.mean(output_img) - np.mean(input_img)
print(f"Mean intensity change: {mean_intensity_change:.2f}")
"""