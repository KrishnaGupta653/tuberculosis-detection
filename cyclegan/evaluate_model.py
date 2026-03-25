"""
Model Evaluation Script for TB↔Normal CycleGAN
Evaluates accuracy using both quantitative and qualitative metrics.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from scipy import stats
import json

# ============================================================================
# 1. PERCEPTION-BASED METRICS (Most Important for Medical Imaging)
# ============================================================================

def compute_ssim(img1, img2):
    """Structural Similarity Index - measure of perceived image quality (0-1, 1=identical)"""
    if img1.shape != img2.shape:
        img2 = np.resize(img2, img1.shape)
    return ssim(img1, img2, data_range=255, channel_axis=2)

def compute_psnr(img1, img2):
    """Peak Signal-to-Noise Ratio - higher is better (>30 dB is good)"""
    if img1.shape != img2.shape:
        img2 = np.resize(img2, img1.shape)
    return psnr(img1, img2, data_range=255)

def compute_mae(img1, img2):
    """Mean Absolute Error - average pixel difference (0=identical)"""
    return np.mean(np.abs(img1.astype(float) - img2.astype(float)))

def compute_mse(img1, img2):
    """Mean Squared Error - penalizes large errors more (0=identical)"""
    return np.mean((img1.astype(float) - img2.astype(float)) ** 2)

# ============================================================================
# 2. MEDICAL IMAGING SPECIFIC METRICS
# ============================================================================

def compute_edge_preservation(img1, img2):
    """
    Measure how well edges (anatomical features) are preserved.
    Higher = better preservation of important details.
    """
    from scipy.ndimage import laplace
    
    edge1 = np.abs(laplace(img1.astype(float)))
    edge2 = np.abs(laplace(img2.astype(float)))
    
    # Correlation between edge maps
    if edge1.std() > 0 and edge2.std() > 0:
        correlation = np.corrcoef(edge1.flatten(), edge2.flatten())[0, 1]
        return correlation
    return 0.0

def compute_intensity_distribution(img1, img2):
    """
    Compare intensity histograms.
    Lower KL divergence = more similar to reference image.
    """
    hist1, _ = np.histogram(img1, bins=256, range=(0, 256))
    hist2, _ = np.histogram(img2, bins=256, range=(0, 256))
    
    # Normalize
    hist1 = hist1 / (hist1.sum() + 1e-7)
    hist2 = hist2 / (hist2.sum() + 1e-7)
    
    # KL divergence (0 = identical distributions)
    kl_div = np.sum(hist1 * np.log((hist1 + 1e-7) / (hist2 + 1e-7)))
    return kl_div

def compute_contrast_preservation(img1, img2):
    """
    Check if contrast is maintained (important for diagnostic features).
    Returns correlation of local contrast.
    """
    # Local contrast using local standard deviation
    from scipy.ndimage import uniform_filter
    
    mean1 = uniform_filter(img1.astype(float), size=5)
    mean2 = uniform_filter(img2.astype(float), size=5)
    
    contrast1 = np.sqrt(uniform_filter(img1.astype(float)**2, size=5) - mean1**2)
    contrast2 = np.sqrt(uniform_filter(img2.astype(float)**2, size=5) - mean2**2)
    
    if contrast1.std() > 0 and contrast2.std() > 0:
        correlation = np.corrcoef(contrast1.flatten(), contrast2.flatten())[0, 1]
        return correlation
    return 0.0

# ============================================================================
# 3. TRANSFORMATION QUALITY METRICS
# ============================================================================

def compute_transformation_magnitude(original, generated):
    """
    How much the transformation changed the image.
    High value = significant TB→Normal translation.
    Low value = minimal changes (potential underfitting).
    """
    diff = np.abs(original.astype(float) - generated.astype(float))
    percentage_changed_10 = (diff > 10).sum() / diff.size * 100
    percentage_changed_50 = (diff > 50).sum() / diff.size * 100
    
    return {
        'mean_difference': diff.mean(),
        'max_difference': diff.max(),
        'std_difference': diff.std(),
        'pixels_changed_>10': percentage_changed_10,
        'pixels_changed_>50': percentage_changed_50
    }

def compute_anatomical_plausibility(original, generated):
    """
    Check if transformation is anatomically plausible:
    - Shouldn't remove/add too much structure
    - Shouldn't create unrealistic artifacts
    """
    orig_mean = original.mean()
    gen_mean = generated.mean()
    
    # Check if means are too different (would indicate hallucination)
    mean_consistency = 1.0 - min(abs(orig_mean - gen_mean) / max(orig_mean, gen_mean, 1e-7), 1.0)
    
    # Check for extreme values (potential artifacts)
    extreme_pixels = np.sum(generated < 10) + np.sum(generated > 245)
    extreme_ratio = extreme_pixels / generated.size
    
    return {
        'mean_consistency': mean_consistency,  # 1.0 = good, should be > 0.8
        'extreme_pixels_ratio': extreme_ratio,  # should be < 0.05 (5%)
        'is_plausible': mean_consistency > 0.7 and extreme_ratio < 0.1
    }

# ============================================================================
# 4. COMPREHENSIVE EVALUATION
# ============================================================================

def evaluate_single_image(model, image_path, device='cuda'):
    """
    Run complete evaluation on a single TB image.
    
    Returns:
        Dict with all metrics and interpretations
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    # Convert to numpy
    def tensor_to_np(t):
        return np.transpose(
            ((t.squeeze(0).cpu().numpy() + 1) / 2.0 * 255).astype(np.uint8),
            (1, 2, 0)
        )
    
    original = tensor_to_np(input_tensor)
    generated = tensor_to_np(output_tensor)
    
    # Grayscale for metrics
    original_gray = 0.299 * original[:,:,0] + 0.587 * original[:,:,1] + 0.114 * original[:,:,2]
    generated_gray = 0.299 * generated[:,:,0] + 0.587 * generated[:,:,1] + 0.114 * generated[:,:,2]
    
    # Compute metrics
    results = {
        'image': Path(image_path).stem,
        'metrics': {
            'SSIM': compute_ssim(original_gray, generated_gray),
            'PSNR': compute_psnr(original_gray, generated_gray),
            'MAE': compute_mae(original_gray, generated_gray),
            'MSE': compute_mse(original_gray, generated_gray),
            'edge_preservation': compute_edge_preservation(original_gray, generated_gray),
            'intensity_distribution': compute_intensity_distribution(original_gray, generated_gray),
            'contrast_preservation': compute_contrast_preservation(original_gray, generated_gray),
        },
        'transformation': compute_transformation_magnitude(original_gray, generated_gray),
        'plausibility': compute_anatomical_plausibility(original_gray, generated_gray)
    }
    
    return results, original, generated

def print_evaluation_report(results):
    """Print human-readable evaluation report"""
    print("\n" + "="*70)
    print(f"EVALUATION REPORT: {results['image']}")
    print("="*70)
    
    # Perception Metrics
    print("\n📊 PERCEPTION METRICS (Lower MSE/MAE/distances = better):")
    print(f"  SSIM (Structural Similarity):    {results['metrics']['SSIM']:.4f}  (1.0=identical, >0.7=good)")
    print(f"  PSNR (Peak Signal-Noise Ratio):  {results['metrics']['PSNR']:.2f} dB  (>30=excellent, >20=good)")
    print(f"  MAE (Mean Absolute Error):       {results['metrics']['MAE']:.2f}  (pixels, 0-255)")
    print(f"  MSE (Mean Squared Error):        {results['metrics']['MSE']:.2f}  (0-65025)")
    
    # Medical Metrics
    print("\n🏥 MEDICAL IMAGING METRICS:")
    print(f"  Edge Preservation Correlation:   {results['metrics']['edge_preservation']:.4f}  (closer to 1.0 = better)")
    print(f"  Intensity Distribution (KL):     {results['metrics']['intensity_distribution']:.4f}  (lower = more realistic)")
    print(f"  Contrast Preservation:           {results['metrics']['contrast_preservation']:.4f}  (>0.5 = good)")
    
    # Transformation Quality
    print("\n🔄 TRANSFORMATION QUALITY:")
    trans = results['transformation']
    print(f"  Mean Pixel Difference:           {trans['mean_difference']:.2f}")
    print(f"  Max Pixel Difference:            {trans['max_difference']}")
    print(f"  Pixels Changed >10:              {trans['pixels_changed_>10']:.2f}%")
    print(f"  Pixels Changed >50:              {trans['pixels_changed_>50']:.2f}%")
    
    # Plausibility
    print("\n✅ ANATOMICAL PLAUSIBILITY:")
    plausi = results['plausibility']
    print(f"  Mean Intensity Consistency:      {plausi['mean_consistency']:.4f}  (>0.7 = good)")
    print(f"  Extreme Pixels Ratio:            {plausi['extreme_pixels_ratio']:.4f}  (<0.05 = good)")
    print(f"  Overall Plausibility:            {'✅ PASS' if plausi['is_plausible'] else '❌ FAIL'}")
    
    # Interpretation
    print("\n💡 INTERPRETATION:")
    print(f"  Image appears {'anatomically plausible' if plausi['is_plausible'] else 'SUSPICIOUS'}")
    print(f"  Transformation {'is significant' if trans['pixels_changed_>10'] > 50 else 'is subtle'}")
    print(f"  Output quality: {'EXCELLENT' if results['metrics']['SSIM'] > 0.85 else 'GOOD' if results['metrics']['SSIM'] > 0.75 else 'ACCEPTABLE' if results['metrics']['SSIM'] > 0.6 else 'POOR'}")
    print("="*70 + "\n")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate TB↔Normal CycleGAN')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint')
    parser.add_argument('--image', required=True, help='Path to TB image')
    parser.add_argument('--output', default='evaluation_results', help='Output directory')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    
    args = parser.parse_args()
    
    # Load model (copy from cyclegan_inference.py)
    from cyclegan_inference import Generator, load_generator
    
    print("Loading model...")
    generator = load_generator(args.checkpoint, device=args.device, generator_key='G_state')
    
    print("Evaluating...")
    results, original, generated = evaluate_single_image(generator, args.image, device=args.device)
    print_evaluation_report(results)
    
    # Save
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, f"{results['image']}_evaluation.json"), 'w') as f:
        # Convert numpy types to Python types for JSON
        results_json = {
            'image': results['image'],
            'metrics': {k: float(v) for k, v in results['metrics'].items()},
            'transformation': {k: float(v) for k, v in results['transformation'].items()},
            'plausibility': results['plausibility']
        }
        json.dump(results_json, f, indent=2)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original)
    axes[0].set_title(f"Original TB Image", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(generated)
    axes[1].set_title(f"Generated Normal Image", fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    diff = np.abs(original.astype(float) - generated.astype(float)).mean(axis=2)
    im = axes[2].imshow(diff, cmap='hot')
    axes[2].set_title(f"Difference Magnitude (SSIM: {results['metrics']['SSIM']:.3f})", fontsize=12, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, f"{results['image']}_evaluation.png"), dpi=150, bbox_inches='tight')
    print(f"✓ Results saved to {args.output}")
