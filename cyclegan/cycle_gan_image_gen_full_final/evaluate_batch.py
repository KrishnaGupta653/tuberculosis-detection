"""
Batch Evaluation Script - Test Multiple TB Images
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import json

def evaluate_batch(model, image_dir, output_dir, device='cuda'):
    """
    Evaluate model on all TB images in directory.
    Computes average metrics and creates comparison report.
    """
    
    # Load all TB images
    image_paths = list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.png"))
    if not image_paths:
        print(f"❌ No images found in {image_dir}")
        return
    
    print(f"Found {len(image_paths)} images to evaluate\n")
    
    results_all = []
    
    for idx, img_path in enumerate(image_paths, 1):
        print(f"[{idx}/{len(image_paths)}] Processing {img_path.name}...")
        
        try:
            # Load and preprocess
            img = Image.open(img_path).convert('RGB')
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
            original_gray = 0.299 * original[:,:,0] + 0.587 * original[:,:,1] + 0.114 * original[:,:,2]
            generated_gray = 0.299 * generated[:,:,0] + 0.587 * generated[:,:,1] + 0.114 * generated[:,:,2]
            
            # Quick metrics
            from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
            
            ssim_score = ssim(original_gray, generated_gray, data_range=255)
            psnr_score = psnr(original_gray, generated_gray, data_range=255)
            mae_score = np.mean(np.abs(original_gray.astype(float) - generated_gray.astype(float)))
            
            diff = np.abs(original_gray.astype(float) - generated_gray.astype(float))
            pixels_changed = (diff > 10).sum() / diff.size * 100
            
            mean_intensity_orig = original_gray.mean()
            mean_intensity_gen = generated_gray.mean()
            consistency = 1.0 - min(abs(mean_intensity_orig - mean_intensity_gen) / max(mean_intensity_orig, mean_intensity_gen, 1e-7), 1.0)
            
            results_all.append({
                'image': img_path.name,
                'ssim': ssim_score,
                'psnr': psnr_score,
                'mae': mae_score,
                'pixels_changed': pixels_changed,
                'consistency': consistency
            })
            
            print(f"  ✓ SSIM: {ssim_score:.3f}, PSNR: {psnr_score:.1f}, MAE: {mae_score:.1f}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
    
    # Summary Statistics
    if not results_all:
        print("❌ No images processed successfully")
        return
    
    print("\n" + "="*70)
    print("BATCH EVALUATION SUMMARY")
    print("="*70)
    
    ssim_vals = [r['ssim'] for r in results_all]
    psnr_vals = [r['psnr'] for r in results_all]
    mae_vals = [r['mae'] for r in results_all]
    pixels_changed_vals = [r['pixels_changed'] for r in results_all]
    consistency_vals = [r['consistency'] for r in results_all]
    
    print(f"\nTotal images evaluated: {len(results_all)}")
    print(f"\n📊 SSIM (Structural Similarity):")
    print(f"  Mean: {np.mean(ssim_vals):.4f}")
    print(f"  Median: {np.median(ssim_vals):.4f}")
    print(f"  Std: {np.std(ssim_vals):.4f}")
    print(f"  Range: [{np.min(ssim_vals):.4f}, {np.max(ssim_vals):.4f}]")
    
    print(f"\n📊 PSNR (Peak Signal-Noise Ratio) [dB]:")
    print(f"  Mean: {np.mean(psnr_vals):.2f}")
    print(f"  Median: {np.median(psnr_vals):.2f}")
    print(f"  Std: {np.std(psnr_vals):.2f}")
    print(f"  Range: [{np.min(psnr_vals):.2f}, {np.max(psnr_vals):.2f}]")
    
    print(f"\n📊 MAE (Mean Absolute Error):")
    print(f"  Mean: {np.mean(mae_vals):.2f}")
    print(f"  Median: {np.median(mae_vals):.2f}")
    print(f"  Std: {np.std(mae_vals):.2f}")
    
    print(f"\n🔄 Pixels Changed >10 (%):")
    print(f"  Mean: {np.mean(pixels_changed_vals):.2f}%")
    print(f"  Median: {np.median(pixels_changed_vals):.2f}%")
    print(f"  Range: [{np.min(pixels_changed_vals):.2f}%, {np.max(pixels_changed_vals):.2f}%]")
    
    print(f"\n✅ Mean Intensity Consistency:")
    print(f"  Mean: {np.mean(consistency_vals):.4f}")
    print(f"  Median: {np.median(consistency_vals):.4f}")
    
    # Quality assessment
    avg_ssim = np.mean(ssim_vals)
    quality = "EXCELLENT" if avg_ssim > 0.85 else "GOOD" if avg_ssim > 0.75 else "ACCEPTABLE" if avg_ssim > 0.65 else "POOR"
    
    print(f"\n🎯 OVERALL QUALITY ASSESSMENT: {quality}")
    print("="*70)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'batch_evaluation.json'), 'w') as f:
        json.dump({
            'images': results_all,
            'summary': {
                'total': len(results_all),
                'ssim_mean': float(np.mean(ssim_vals)),
                'psnr_mean': float(np.mean(psnr_vals)),
                'mae_mean': float(np.mean(mae_vals)),
                'quality': quality
            }
        }, f, indent=2)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].hist(ssim_vals, bins=10, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title(f'SSIM Distribution (Mean: {np.mean(ssim_vals):.3f})', fontweight='bold')
    axes[0, 0].set_xlabel('SSIM')
    axes[0, 0].axvline(np.mean(ssim_vals), color='red', linestyle='--', label='Mean')
    axes[0, 0].legend()
    
    axes[0, 1].hist(psnr_vals, bins=10, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 1].set_title(f'PSNR Distribution (Mean: {np.mean(psnr_vals):.2f} dB)', fontweight='bold')
    axes[0, 1].set_xlabel('PSNR (dB)')
    axes[0, 1].axvline(np.mean(psnr_vals), color='red', linestyle='--', label='Mean')
    axes[0, 1].legend()
    
    axes[1, 0].hist(mae_vals, bins=10, edgecolor='black', alpha=0.7, color='green')
    axes[1, 0].set_title(f'MAE Distribution (Mean: {np.mean(mae_vals):.2f})', fontweight='bold')
    axes[1, 0].set_xlabel('MAE')
    axes[1, 0].axvline(np.mean(mae_vals), color='red', linestyle='--', label='Mean')
    axes[1, 0].legend()
    
    axes[1, 1].hist(pixels_changed_vals, bins=10, edgecolor='black', alpha=0.7, color='purple')
    axes[1, 1].set_title(f'Pixels Changed Distribution (Mean: {np.mean(pixels_changed_vals):.2f}%)', fontweight='bold')
    axes[1, 1].set_xlabel('Pixels Changed >10 (%)')
    axes[1, 1].axvline(np.mean(pixels_changed_vals), color='red', linestyle='--', label='Mean')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'batch_evaluation_distributions.png'), dpi=150, bbox_inches='tight')
    print(f"\n✓ Results saved to {output_dir}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch Evaluate TB↔Normal CycleGAN')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint')
    parser.add_argument('--image-dir', default='dataset/TB', help='Directory with TB images')
    parser.add_argument('--output', default='batch_evaluation_results', help='Output directory')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    
    args = parser.parse_args()
    
    from cyclegan_inference import Generator, load_generator
    
    print("Loading model...")
    generator = load_generator(args.checkpoint, device=args.device, generator_key='G_state')
    
    print("Running batch evaluation...")
    evaluate_batch(generator, args.image_dir, args.output, device=args.device)
