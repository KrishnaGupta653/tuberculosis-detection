"""
Compare all available checkpoints to find best accuracy
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import json
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

def quick_evaluate(model, image_path, device='cuda'):
    """Quick evaluation of single image"""
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    def tensor_to_np(t):
        return np.transpose(
            ((t.squeeze(0).cpu().numpy() + 1) / 2.0 * 255).astype(np.uint8),
            (1, 2, 0)
        )
    
    original = tensor_to_np(input_tensor)
    generated = tensor_to_np(output_tensor)
    
    original_gray = 0.299 * original[:,:,0] + 0.587 * original[:,:,1] + 0.114 * original[:,:,2]
    generated_gray = 0.299 * generated[:,:,0] + 0.587 * generated[:,:,1] + 0.114 * generated[:,:,2]
    
    ssim_score = ssim(original_gray, generated_gray, data_range=255)
    psnr_score = psnr(original_gray, generated_gray, data_range=255)
    mae_score = np.mean(np.abs(original_gray.astype(float) - generated_gray.astype(float)))
    mse_score = np.mean((original_gray.astype(float) - generated_gray.astype(float)) ** 2)
    
    diff = np.abs(original_gray.astype(float) - generated_gray.astype(float))
    pixels_changed = (diff > 10).sum() / diff.size * 100
    
    mean_intensity_orig = original_gray.mean()
    mean_intensity_gen = generated_gray.mean()
    consistency = 1.0 - min(abs(mean_intensity_orig - mean_intensity_gen) / max(mean_intensity_orig, mean_intensity_gen, 1e-7), 1.0)
    
    return {
        'ssim': ssim_score,
        'psnr': psnr_score,
        'mae': mae_score,
        'mse': mse_score,
        'pixels_changed': pixels_changed,
        'consistency': consistency,
        'mean_orig': mean_intensity_orig,
        'mean_gen': mean_intensity_gen
    }

if __name__ == '__main__':
    from cyclegan_inference import Generator, load_generator
    
    device = 'cuda'
    test_image = 'test_tb.jpg'
    
    checkpoints = sorted(Path('checkpoints').glob('epoch_*.pth'))
    
    print("\n" + "="*90)
    print("EPOCH COMPARISON - MODEL ACCURACY ACROSS ALL CHECKPOINTS")
    print("="*90)
    print(f"\nTesting image: {test_image}")
    print(f"Total checkpoints: {len(checkpoints)}\n")
    
    results_all = {}
    
    for cp_path in checkpoints:
        epoch_num = cp_path.stem.split('_')[1]
        print(f"[{epoch_num}] Evaluating {cp_path.name}...", end=' ', flush=True)
        
        try:
            generator = load_generator(str(cp_path), device=device, generator_key='G_state')
            metrics = quick_evaluate(generator, test_image, device=device)
            results_all[epoch_num] = metrics
            print(f"✓ SSIM: {metrics['ssim']:.4f}")
        except Exception as e:
            print(f"✗ Error: {str(e)[:50]}")
            results_all[epoch_num] = None
    
    # Find best epoch for each metric
    valid_results = {k: v for k, v in results_all.items() if v is not None}
    
    if not valid_results:
        print("❌ No valid results")
        exit(1)
    
    print("\n" + "="*90)
    print("RESULTS COMPARISON TABLE")
    print("="*90)
    print(f"\n{'Epoch':<10} {'SSIM':<10} {'PSNR':<10} {'MAE':<10} {'MSE':<12} {'Pixels>10%':<12} {'Consistency':<12}")
    print("-" * 90)
    
    for epoch in sorted(valid_results.keys(), key=lambda x: int(x)):
        m = valid_results[epoch]
        print(f"{epoch:<10} {m['ssim']:<10.4f} {m['psnr']:<10.2f} {m['mae']:<10.2f} {m['mse']:<12.1f} {m['pixels_changed']:<12.1f} {m['consistency']:<12.4f}")
    
    # Find best epochs
    print("\n" + "="*90)
    print("BEST EPOCHS BY METRIC")
    print("="*90)
    
    best_ssim_epoch = max(valid_results.keys(), key=lambda x: valid_results[x]['ssim'])
    best_psnr_epoch = max(valid_results.keys(), key=lambda x: valid_results[x]['psnr'])
    best_mae_epoch = min(valid_results.keys(), key=lambda x: valid_results[x]['mae'])
    best_mse_epoch = min(valid_results.keys(), key=lambda x: valid_results[x]['mse'])
    
    print(f"\n✓ BEST SSIM (Perception Quality):  epoch_{best_ssim_epoch} = {valid_results[best_ssim_epoch]['ssim']:.4f}")
    print(f"✓ BEST PSNR (Signal Fidelity):    epoch_{best_psnr_epoch} = {valid_results[best_psnr_epoch]['psnr']:.2f} dB")
    print(f"✓ BEST MAE (Pixel Error):         epoch_{best_mae_epoch} = {valid_results[best_mae_epoch]['mae']:.2f}")
    print(f"✓ BEST MSE (Squared Error):       epoch_{best_mse_epoch} = {valid_results[best_mse_epoch]['mse']:.2f}")
    
    # Overall recommendation
    print("\n" + "="*90)
    print("RECOMMENDATION")
    print("="*90)
    
    # Score each epoch
    epoch_scores = {}
    for epoch in valid_results.keys():
        m = valid_results[epoch]
        # SSIM is most important for medical imaging
        ssim_score = (m['ssim'] / max(v['ssim'] for v in valid_results.values())) * 0.5
        mae_score = (1 - m['mae'] / max(v['mae'] for v in valid_results.values())) * 0.2
        consistency_score = m['consistency'] * 0.2
        pixels_score = min(m['pixels_changed'] / 100, 1.0) * 0.1
        
        epoch_scores[epoch] = ssim_score + mae_score + consistency_score + pixels_score
    
    best_overall = max(epoch_scores.keys(), key=lambda x: epoch_scores[x])
    
    print(f"\n🏆 BEST OVERALL EPOCH: epoch_{best_overall}")
    print(f"   Score: {epoch_scores[best_overall]:.4f}/1.0")
    print(f"\n   Metrics:")
    m = valid_results[best_overall]
    print(f"   ├─ SSIM: {m['ssim']:.4f} (Structural similarity)")
    print(f"   ├─ PSNR: {m['psnr']:.2f} dB (Signal fidelity)")
    print(f"   ├─ MAE: {m['mae']:.2f} (Pixel error)")
    print(f"   ├─ Pixels Changed: {m['pixels_changed']:.2f}%")
    print(f"   └─ Consistency: {m['consistency']:.4f}")
    
    print(f"\n📋 USAGE COMMAND:")
    print(f"   python cyclegan_inference.py --checkpoint checkpoints/epoch_{best_overall}.pth \\")
    print(f"       --input test_tb.jpg --output results --generator_key G_state")
    
    print(f"\n💡 ALTERNATIVE EPOCHS:")
    sorted_epochs = sorted(epoch_scores.items(), key=lambda x: x[1], reverse=True)
    for idx, (epoch, score) in enumerate(sorted_epochs[1:4], 2):
        print(f"   {idx}. epoch_{epoch} (score: {score:.4f})")
    
    print("\n" + "="*90)
    
    # Save detailed comparison
    with open('epoch_comparison.json', 'w') as f:
        json.dump({
            'all_results': {k: v for k, v in results_all.items() if v is not None},
            'best_epochs': {
                'ssim': best_ssim_epoch,
                'psnr': best_psnr_epoch,
                'mae': best_mae_epoch,
                'mse': best_mse_epoch,
                'overall': best_overall
            },
            'scores': {k: float(v) for k, v in epoch_scores.items()}
        }, f, indent=2)
    
    print("\n✓ Detailed results saved to: epoch_comparison.json")
