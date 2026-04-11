#!/usr/bin/env python3
"""
Quick Model Testing Tool
========================
One-command script to verify model is working correctly and get quick accuracy metrics.

Usage:
  python quick_model_test.py                          # Uses default test image
  python quick_model_test.py --image <path>           # Test on custom image
  python quick_model_test.py --all-epochs             # Compare all epochs
  python quick_model_test.py --batch                  # Run on full TB dataset
"""

import argparse
import json
from pathlib import Path
import torch
from PIL import Image
import numpy as np
from collections import defaultdict
import sys

def load_model(checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Load model and checkpoint"""
    from cyclegan_inference import Generator
    
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None
    
    try:
        model = Generator(input_nc=3, output_nc=3, ngf=64, n_blocks=6).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['G_state'])
        model.eval()
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

def test_single_image(image_path, checkpoint_path, device='cuda'):
    """Test model on single image"""
    from cyclegan_inference import preprocess_image, postprocess_image
    from evaluate_model import (compute_ssim, compute_psnr, compute_mae, 
                                 compute_mse, compute_intensity_distribution,
                                 compute_transformation_magnitude)
    
    print(f"\n{'='*60}")
    print(f"Testing: {Path(image_path).name}")
    print(f"Checkpoint: {Path(checkpoint_path).name}")
    print(f"Device: {device}")
    print(f"{'='*60}")
    
    # Load model
    model = load_model(checkpoint_path, device)
    if model is None:
        return None
    
    # Load and preprocess image
    try:
        img, img_pil = preprocess_image(str(image_path))
        print(f"✓ Image loaded: {img_pil.size}")
    except Exception as e:
        print(f"❌ Failed to load image: {e}")
        return None
    
    # Generate
    img = img.to(device)
    with torch.no_grad():
        y = model(img)
    
    # Postprocess
    output_array = postprocess_image(y[0].cpu())
    output_img = Image.fromarray(output_array)
    
    print(f"✓ Generated image: {output_img.size}, {output_img.mode}")
    
    # Evaluate quality
    try:
        input_array = np.array(img_pil.convert('RGB'))
        transform_mag = compute_transformation_magnitude(input_array, output_array)
        results = {
            'ssim': float(compute_ssim(input_array, output_array)),
            'psnr': float(compute_psnr(input_array, output_array)),
            'mae': float(compute_mae(input_array, output_array)),
            'mse': float(compute_mse(input_array, output_array)),
            'consistency': float(1.0 / (1.0 + compute_intensity_distribution(input_array, output_array))),  # Convert KL div to score
            'pixels_changed': float(transform_mag['pixels_changed_>10']),
        }
        
        print(f"\n📊 QUALITY METRICS:")
        print(f"  SSIM:        {results['ssim']:.4f}  {'✓ Good' if results['ssim'] > 0.75 else '⚠ Low'}")
        print(f"  PSNR:        {results['psnr']:.2f} dB  {'✓ Good' if results['psnr'] > 15 else '⚠ Low'}")
        print(f"  MAE:         {results['mae']:.2f} pixels  {'✓ Controlled' if results['mae'] < 35 else '⚠ Aggressive'}")
        print(f"  Consistency: {results['consistency']:.4f}  {'✓ Excellent' if results['consistency'] > 0.8 else '⚠ Poor'}")
        print(f"  Pixels>10%:  {results['pixels_changed']:.1f}%  {'✓ Good' if 75 < results['pixels_changed'] < 95 else '⚠ Check'}")
        
        return results
        
    except Exception as e:
        print(f"⚠ Evaluation skipped: {e}")
        return None

def test_all_epochs(image_path, device='cuda'):
    """Compare all available epochs"""
    print(f"\n{'='*60}")
    print(f"COMPARING ALL EPOCHS ON: {Path(image_path).name}")
    print(f"{'='*60}")
    
    checkpoints = sorted(Path('checkpoints').glob('epoch_*.pth'))
    
    if not checkpoints:
        print("❌ No checkpoints found in ./checkpoints/")
        return
    
    results = {}
    for ckpt in checkpoints:
        ckpt_name = ckpt.stem
        print(f"\n▶ Testing {ckpt_name}...", end=' ')
        sys.stdout.flush()
        
        result = test_single_image(image_path, str(ckpt), device)
        if result:
            results[ckpt_name] = result
            print("✓")
        else:
            print("✗ Failed")
    
    # Ranking
    if results:
        print(f"\n{'='*60}")
        print("RANKING (by SSIM, then MAE):")
        print(f"{'='*60}")
        
        sorted_results = sorted(
            results.items(),
            key=lambda x: (-x[1]['ssim'], x[1]['mae'])
        )
        
        for rank, (epoch_name, metrics) in enumerate(sorted_results, 1):
            ssim = metrics['ssim']
            mae = metrics['mae']
            consistency = metrics['consistency']
            
            emoji = '🏆' if rank == 1 else '🥈' if rank == 2 else '🥉' if rank == 3 else '  '
            print(f"{emoji} {rank}. {epoch_name:12} | SSIM: {ssim:.4f} | MAE: {mae:.2f} | Consistency: {consistency:.4f}")
        
        # Save results
        with open('quick_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: quick_test_results.json")
        
        best_epoch = sorted_results[0][0]
        print(f"\n✅ BEST EPOCH: {best_epoch}")
        print(f"   Command: python cyclegan_inference.py --checkpoint checkpoints/{best_epoch}.pth --input <image>")

def run_batch_test(image_dir, checkpoint, device='cuda', limit=10):
    """Quick batch test on multiple images"""
    from cyclegan_inference import preprocess_image, postprocess_image
    
    print(f"\n{'='*60}")
    print(f"BATCH TEST: {image_dir} (limit: {limit} images)")
    print(f"Checkpoint: {checkpoint}")
    print(f"{'='*60}")
    
    model = load_model(checkpoint, device)
    if model is None:
        return
    
    image_dir = Path(image_dir)
    images = list(image_dir.glob('*.jpg'))[:limit]
    
    if not images:
        print(f"❌ No JPG images found in {image_dir}")
        return
    
    print(f"Found {len(images)} images\n")
    
    ssim_scores = []
    
    for idx, img_path in enumerate(images[:limit], 1):
        try:
            x = preprocess_image(str(img_path))
            x = x.unsqueeze(0).to(device)
            
            with torch.no_grad():
                y = model(x)
            
            output_array = postprocess_image(y[0].cpu())
            
            # Quick SSIM check
            from skimage.metrics import structural_similarity
            img_array = np.array(Image.open(img_path).convert('RGB').resize((256, 256)))
            ssim = structural_similarity(img_array, output_array, channel_axis=2)
            ssim_scores.append(ssim)
            
            print(f"[{idx:3d}/{len(images)}] {img_path.name:30s} SSIM: {ssim:.4f}")
            
        except Exception as e:
            print(f"[{idx:3d}/{len(images)}] {img_path.name:30s} ✗ Error: {e}")
    
    if ssim_scores:
        print(f"\n{'='*60}")
        print(f"BATCH STATISTICS:")
        print(f"  Mean SSIM:   {np.mean(ssim_scores):.4f}")
        print(f"  Std Dev:     {np.std(ssim_scores):.4f}")
        print(f"  Min SSIM:    {np.min(ssim_scores):.4f}")
        print(f"  Max SSIM:    {np.max(ssim_scores):.4f}")
        print(f"  Passing (>0.75): {sum(s > 0.75 for s in ssim_scores)}/{len(ssim_scores)}")
        print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(
        description="Quick TB Model Testing Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quick_model_test.py                    # Test on default image
  python quick_model_test.py --image data.jpg   # Test on custom image
  python quick_model_test.py --all-epochs       # Compare all epochs
  python quick_model_test.py --batch            # Batch test (10 images)
  python quick_model_test.py --batch --limit 50 # Batch test (50 images)
        """
    )
    
    parser.add_argument('--image', type=str, default='test_tb.jpg',
                        help='Image to test (default: test_tb.jpg)')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/epoch_0025.pth',
                        help='Checkpoint to use (default: epoch_0025.pth)')
    parser.add_argument('--all-epochs', action='store_true',
                        help='Compare all available epochs')
    parser.add_argument('--batch', action='store_true',
                        help='Run batch test on TB dataset')
    parser.add_argument('--batch-dir', type=str, default='dataset/TB',
                        help='Directory for batch test (default: dataset/TB)')
    parser.add_argument('--limit', type=int, default=10,
                        help='Limit for batch test (default: 10)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use (default: auto)')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"\n🔧 TB Model Test Tool")
    print(f"{'='*60}")
    print(f"Device: {device.upper()}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}")
    
    # Run tests
    if args.all_epochs:
        test_all_epochs(args.image, device)
    elif args.batch:
        run_batch_test(args.batch_dir, args.checkpoint, device, args.limit)
    else:
        test_single_image(args.image, args.checkpoint, device)

if __name__ == '__main__':
    main()
