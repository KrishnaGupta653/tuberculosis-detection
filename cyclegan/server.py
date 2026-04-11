"""
================================================================================
🏥 CycleGAN TB IMAGE GENERATION SERVER
================================================================================
REST API server for TB↔Normal image generation with port forwarding support.

USAGE:
    python server.py                           # Start on localhost:5000
    python server.py --port 8080               # Custom port
    python server.py --checkpoint epoch_0100   # Set default epoch
    python server.py --host 0.0.0.0            # Allow external connections

QUICK EXAMPLES:
    URL Parameters (perfect for friends via port forwarding):
    http://localhost:8080/?image=TB1.jpg
    http://localhost:8080/?image=dataset/TB/TB.1.jpg&checkpoint=epoch_0100
    
    With output folder:
    http://localhost:8080/?image=TB1.jpg&output=my_results
    http://localhost:8080/?image=TB1.jpg&checkpoint=epoch_0075&output=test_run

PORT FORWARDING (Share with friends):
    1. VS Code Dev Tunnels (Recommended - FREE):
       code tunnel
       # Copy URL like: https://xyz-8080.inc1.devtunnels.ms/
       # Share with friend: https://xyz-8080.inc1.devtunnels.ms/?image=TB1.jpg
    
    2. ngrok (Alternative):
       ngrok http 8080
       # Copy forwarding URL and share with friend

WARRANTY:
    Works with file uploads (POST) AND URL parameters (GET)
    Output folders created locally on the server
    Friend sees results as HTML or JSON
================================================================================
"""

import os
import json
import argparse
import io
import base64
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

from cyclegan_inference import Generator, load_generator, preprocess_image, postprocess_image
from evaluate_model import (
    compute_ssim, compute_psnr, compute_mae, compute_mse,
    compute_edge_preservation, compute_intensity_distribution,
    compute_contrast_preservation, compute_transformation_magnitude,
    compute_anatomical_plausibility
)

# ============================================================================
# CHANGE THIS TO SELECT WHICH EPOCH TO USE
# ============================================================================
epoch = 'epoch_0025'  # <-- EDIT THIS to change default checkpoint (e.g., 'epoch_0149', 'epoch_0100', etc.)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    DEBUG = False
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}
    CHECKPOINT_DIR = 'checkpoints'
    IMAGE_SIZE = 256

class AppState:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DEFAULT_CHECKPOINT = None  # Will be set from CLI args
    OUTPUT_DIR = 'server_outputs'  # Default output directory
    MODEL_CACHE = {}

app = Flask(__name__)
CORS(app)
app.config.from_object(Config)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def get_available_checkpoints() -> list:
    """Get list of available checkpoint filenames"""
    checkpoint_dir = Path(Config.CHECKPOINT_DIR)
    if not checkpoint_dir.exists():
        return []
    return sorted([f.stem for f in checkpoint_dir.glob('epoch_*.pth')])

def load_model_cached(checkpoint_name: str) -> Optional[Generator]:
    """Load model from checkpoint with caching"""
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f'{checkpoint_name}.pth')
    
    if not os.path.exists(checkpoint_path):
        return None
    
    if checkpoint_name in AppState.MODEL_CACHE:
        return AppState.MODEL_CACHE[checkpoint_name]
    
    try:
        print(f"  Loading checkpoint: {checkpoint_path}")
        model = load_generator(checkpoint_path, device=AppState.DEVICE, generator_key='G_state')
        AppState.MODEL_CACHE[checkpoint_name] = model
        print(f"  ✓ Loaded successfully")
        return model
    except Exception as e:
        print(f"  ❌ Failed to load: {e}")
        return None

def image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string for JSON embedding"""
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def to_python_native(obj):
    """Convert numpy and torch types to native Python for JSON serialization"""
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: to_python_native(val) for key, val in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_python_native(item) for item in obj]
    return obj

def compute_all_metrics(original: np.ndarray, generated: np.ndarray) -> Dict:
    """Compute all evaluation metrics"""
    try:
        # Convert to grayscale if needed
        if original.ndim == 3:
            original_gray = 0.299 * original[:,:,0] + 0.587 * original[:,:,1] + 0.114 * original[:,:,2]
            generated_gray = 0.299 * generated[:,:,0] + 0.587 * generated[:,:,1] + 0.114 * generated[:,:,2]
        else:
            original_gray = original
            generated_gray = generated
        
        plausibility = compute_anatomical_plausibility(original_gray, generated_gray)
        
        metrics = {
            'ssim': float(compute_ssim(original_gray, generated_gray)),
            'psnr': float(compute_psnr(original_gray, generated_gray)),
            'mae': float(compute_mae(original_gray, generated_gray)),
            'mse': float(compute_mse(original_gray, generated_gray)),
            'edge_preservation': float(compute_edge_preservation(original_gray, generated_gray)),
            'intensity_distribution': float(compute_intensity_distribution(original_gray, generated_gray)),
            'contrast_preservation': float(compute_contrast_preservation(original_gray, generated_gray)),
            'transformation': to_python_native(compute_transformation_magnitude(original_gray, generated_gray)),
            'plausibility': {
                'mean_consistency': float(plausibility['mean_consistency']),
                'extreme_pixels_ratio': float(plausibility['extreme_pixels_ratio']),
                'is_plausible': bool(plausibility['is_plausible'])
            }
        }
        return metrics
    except Exception as e:
        print(f"  ⚠ Metrics error: {e}")
        return {}

def preprocess_from_stream(file_stream):
    """Load and preprocess image from file stream (for POST uploads)"""
    img_pil = Image.open(file_stream).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img_tensor = transform(img_pil).unsqueeze(0)
    return img_tensor, img_pil

def create_difference_map(input_np: np.ndarray, output_np: np.ndarray):
    """
    Create absolute difference map with hot colormap (EXACTLY like cyclegan_inference.py).
    
    Outputs a grayscale difference map that can be applied with 'hot' colormap.
    Returns the raw difference values for use with matplotlib's hot colormap.
    """
    try:
        # Ensure both are numpy arrays
        input_np = np.asarray(input_np, dtype=np.uint8)
        output_np = np.asarray(output_np, dtype=np.uint8)
        
        # Convert to float for calculations
        input_float = input_np.astype(float)
        output_float = output_np.astype(float)
        
        # Compute absolute difference
        diff = np.abs(input_float - output_float)
        
        # If 3D (RGB), convert to grayscale like cyclegan_inference.py does
        if len(diff.shape) == 3 and diff.shape[2] == 3:
            # Convert RGB to grayscale using standard weights
            diff_gray = 0.299 * diff[:,:,0] + 0.587 * diff[:,:,1] + 0.114 * diff[:,:,2]
        elif len(diff.shape) == 3:
            # Take mean if other 3D format
            diff_gray = np.mean(diff, axis=2)
        else:
            diff_gray = diff
        
        # Normalize to 0-255 for visualization
        if diff_gray.max() > 0:
            diff_normalized = (diff_gray / diff_gray.max() * 255).astype(np.uint8)
        else:
            diff_normalized = diff_gray.astype(np.uint8)
        
        return diff_normalized
    except Exception as e:
        print(f"  ⚠️ Difference map creation failed: {e}")
        return np.zeros((256, 256), dtype=np.uint8)

def create_comparison_grid(input_np: np.ndarray, output_np: np.ndarray) -> Image.Image:
    """
    Create 2x2 comparison grid EXACTLY like cyclegan_inference.py:
    
    Top-left:     Original image
    Top-right:    Generated image
    Bottom-left:  Absolute difference map (with hot colormap + colorbar)
    Bottom-right: Difference overlay (red tint)
    
    Returns a PIL Image of the complete comparison.
    """
    try:
        # Ensure inputs are uint8 numpy arrays
        input_np = np.asarray(input_np, dtype=np.uint8)
        output_np = np.asarray(output_np, dtype=np.uint8)
        
        # Ensure RGB (3D arrays)
        if len(input_np.shape) == 2:
            input_np = np.stack([input_np]*3, axis=2)
        if len(output_np.shape) == 2:
            output_np = np.stack([output_np]*3, axis=2)
        
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # Top-left: Original
        axes[0, 0].imshow(input_np)
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Top-right: Generated
        axes[0, 1].imshow(output_np)
        axes[0, 1].set_title('Generated Image', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Bottom-left: Difference map with hot colormap
        # Compute difference like cyclegan_inference.py does
        if input_np.ndim == 3 and input_np.shape[2] == 3:
            input_gray = 0.299 * input_np[:,:,0] + 0.587 * input_np[:,:,1] + 0.114 * input_np[:,:,2]
            output_gray = 0.299 * output_np[:,:,0] + 0.587 * output_np[:,:,1] + 0.114 * output_np[:,:,2]
            difference = np.abs(input_gray.astype(float) - output_gray.astype(float)).astype(np.uint8)
        else:
            difference = np.abs(input_np.astype(float) - output_np.astype(float)).astype(np.uint8)
        
        # Display difference with hot colormap and colorbar
        im_diff = axes[1, 0].imshow(difference, cmap='hot')
        axes[1, 0].set_title('Absolute Difference Map', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(im_diff, ax=axes[1, 0], fraction=0.046)
        
        # Bottom-right: Difference overlay (red tint)
        overlay = input_np.copy()
        overlay[:, :, 0] = np.clip(overlay[:, :, 0].astype(float) + difference, 0, 255).astype(np.uint8)
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Difference Overlay (Red)', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        # Tight layout and save to PIL Image
        plt.tight_layout()
        
        # Convert matplotlib figure to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        comparison_pil = Image.open(buf)
        comparison_pil = comparison_pil.convert('RGB')
        
        plt.close(fig)
        
        return comparison_pil
    except Exception as e:
        print(f"  ⚠️ Comparison grid creation failed: {e}")
        plt.close('all')
        return Image.new('RGB', (800, 800), color=(200, 200, 200))

def generate_html_result(input_np: np.ndarray, output_np: np.ndarray, metrics: Dict, 
                        image_path: str, checkpoint_name: str) -> str:
    """Generate HTML visualization of results"""
    input_b64 = image_to_base64(Image.fromarray(input_np))
    output_b64 = image_to_base64(Image.fromarray(output_np))
    
    ssim = metrics.get('ssim', 0)
    psnr = metrics.get('psnr', 0)
    quality = 'EXCELLENT' if ssim > 0.85 else 'GOOD' if ssim > 0.75 else 'FAIR'
    
    plausibility = metrics.get('plausibility', {})
    is_plausible = '✓ YES' if plausibility.get('is_plausible') else '✗ NO'
    
    html = f"""
    <html>
    <head>
        <title>CycleGAN - Generated Image</title>
        <style>
            * {{ margin: 0; padding: 0; }}
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; border-radius: 12px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); padding: 30px; }}
            h1 {{ color: #333; margin-bottom: 10px; text-align: center; }}
            .subtitle {{ text-align: center; color: #666; margin-bottom: 30px; font-size: 14px; }}
            .info {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 30px; padding: 15px; background: #f8f9fa; border-radius: 8px; }}
            .info-item {{ }}
            .info-label {{ font-weight: 600; color: #666; font-size: 12px; }}
            .info-value {{ color: #333; font-size: 14px; word-break: break-all; }}
            .images {{ display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-bottom: 30px; }}
            .image-box {{ text-align: center; }}
            .image-box h3 {{ font-size: 14px; color: #666; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 1px; }}
            .image-box img {{ max-width: 100%; border: 2px solid #667eea; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
            .metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-bottom: 30px; }}
            .metric-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 8px; text-align: center; }}
            .metric-name {{ font-size: 12px; opacity: 0.9; text-transform: uppercase; }}
            .metric-value {{ font-size: 20px; font-weight: bold; margin-top: 8px; }}
            .quality-badge {{ background: #10b981; color: white; padding: 4px 12px; border-radius: 20px; display: inline-block; margin-top: 8px; font-size: 12px; }}
            .footer {{ text-align: center; color: #999; font-size: 12px; margin-top: 20px; padding-top: 20px; border-top: 1px solid #eee; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>✅ Image Generation Complete</h1>
            <div class="subtitle">CycleGAN TB↔Normal Transformation</div>
            
            <div class="info">
                <div class="info-item">
                    <div class="info-label">📁 Image</div>
                    <div class="info-value">{image_path}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">🧠 Model</div>
                    <div class="info-value">{checkpoint_name}</div>
                </div>
            </div>
            
            <div class="images">
                <div class="image-box">
                    <h3>Input (TB X-ray)</h3>
                    <img src="data:image/png;base64,{input_b64}" />
                </div>
                <div class="image-box">
                    <h3>Generated (Normal)</h3>
                    <img src="data:image/png;base64,{output_b64}" />
                </div>
            </div>
            
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-name">SSIM</div>
                    <div class="metric-value">{ssim:.4f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-name">PSNR</div>
                    <div class="metric-value">{psnr:.2f} dB</div>
                </div>
                <div class="metric-card">
                    <div class="metric-name">Quality</div>
                    <div class="quality-badge">{quality}</div>
                </div>
            </div>
            
            <div class="info">
                <div class="info-item">
                    <div class="info-label">🔬 Anatomical Plausibility</div>
                    <div class="info-value">{is_plausible}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">⚠️ Extreme Pixels</div>
                    <div class="info-value">{plausibility.get('extreme_pixels_ratio', 0):.2%}</div>
                </div>
            </div>
            
            <div class="footer">
                Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Server running on {AppState.DEVICE.upper()}
            </div>
        </div>
    </body>
    </html>
    """
    return html

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/', methods=['GET'])
@app.route('/api/quick', methods=['GET'])
@app.route('/api/generate', methods=['GET', 'POST'])
def generate():
    """
    Main endpoint for image generation.
    
    Query parameters:
      - image: Path to input image (relative to server root)
      - checkpoint: Which epoch to use (default: latest)
      - output: Output folder name (default: auto-generated timestamp)
      - format: 'html' (default) or 'json'
    
    POST data (multipart):
      - image: Image file upload
      - checkpoint: Checkpoint name
      - output: Output folder name
    """
    try:
        img_tensor = None
        image_path_str = None
        output_format = request.args.get('format', 'html')
        
        # ---- GET Method: URL parameters ----
        if request.method == 'GET':
            image_path_str = request.args.get('image')
            
            # Show help if no image parameter
            if not image_path_str:
                help_html = f"""
                <html>
                <head>
                    <title>CycleGAN TB Generator</title>
                    <style>
                        body {{ font-family: Arial; margin: 20px; background: #f5f5f5; }}
                        .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                        h1 {{ color: #667eea; }}
                        .example {{ background: #f0f0f0; padding: 12px; margin: 10px 0; border-left: 4px solid #667eea; font-family: monospace; }}
                        .note {{ background: #fff3cd; padding: 12px; margin: 10px 0; border-left: 4px solid #ffc107; }}
                        .checkpoints {{ background: #f8f9fa; padding: 15px; border-radius: 4px; }}
                        ul {{ margin: 10px 0 10px 20px; }}
                        li {{ margin: 5px 0; }}
                        code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 3px; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>🏥 CycleGAN TB↔Normal Generator</h1>
                        
                        <h2>Quick Start</h2>
                        <div class="example">/?image=TB1.jpg</div>
                        
                        <h2>Examples</h2>
                        <div class="example">/?image=TB1.jpg&checkpoint={AppState.DEFAULT_CHECKPOINT}</div>
                        <div class="example">/?image=dataset/TB/TB.1.jpg&checkpoint=epoch_0100&output=my_results</div>
                        <div class="example">/?image=TB1.jpg&format=json</div>
                        
                        <h2>Parameters</h2>
                        <ul>
                            <li><code>image</code> (required): Path to TB X-ray image</li>
                            <li><code>checkpoint</code> (optional): Model epoch (default: {AppState.DEFAULT_CHECKPOINT})</li>
                            <li><code>output</code> (optional): Output folder name</li>
                            <li><code>format</code> (optional): 'html' or 'json' (default: html)</li>
                        </ul>
                        
                        <div class="note">
                            <strong>📁 For Friends:</strong><br>
                            If friend is in a folder with only TB.png, they can use:<br>
                            <code>/?image=TB.png&output=generated_results</code><br>
                            Output folder will be created in the server's current directory.
                        </div>
                        
                        <div class="checkpoints">
                            <strong>Available Checkpoints:</strong> {len(get_available_checkpoints())}
                            <ul>
                        """ + "".join(f"<li>{cp}</li>" for cp in get_available_checkpoints()[-10:]) + """
                            </ul>
                        </div>
                        
                        <div class="note">
                            <strong>💡 Port Forwarding:</strong><br>
                            Use VS Code Dev Tunnels: <code>code tunnel</code><br>
                            Then share: <code>https://your-tunnel-url.inc1.devtunnels.ms/?image=TB1.jpg</code>
                        </div>
                    </div>
                </body>
                </html>
                """
                return help_html, 200
            
            # Check image exists
            if not os.path.exists(image_path_str):
                return jsonify({'error': f'❌ Image not found: {image_path_str}'}), 404
            
            # Load image from file path
            try:
                img_tensor, _ = preprocess_image(image_path_str, Config.IMAGE_SIZE)
            except Exception as e:
                return jsonify({'error': f'❌ Failed to load image: {str(e)}'}), 400
        
        # ---- POST Method: File upload ----
        else:
            if 'image' not in request.files:
                return jsonify({'error': '❌ No image file in request'}), 400
            
            file = request.files['image']
            if not file or not allowed_file(file.filename):
                return jsonify({'error': '❌ Invalid file type'}), 400
            
            image_path_str = file.filename
            try:
                img_tensor, _ = preprocess_from_stream(file.stream)
            except Exception as e:
                return jsonify({'error': f'❌ Failed to process image: {str(e)}'}), 400
        
        # ---- Load checkpoint ----
        checkpoint_name = request.form.get('checkpoint') or request.args.get('checkpoint') or AppState.DEFAULT_CHECKPOINT
        if not checkpoint_name:
            checkpoint_name = get_available_checkpoints()[-1]
        
        model = load_model_cached(checkpoint_name)
        if not model:
            return jsonify({'error': f'❌ Failed to load checkpoint: {checkpoint_name}'}), 500
        
        # ---- Generate image ----
        print(f"  Generating with model {checkpoint_name}...")
        with torch.no_grad():
            output_tensor = model(img_tensor.to(AppState.DEVICE))
        
        # ---- Postprocess ----
        input_np = postprocess_image(img_tensor)
        output_np = postprocess_image(output_tensor.cpu())
        metrics = compute_all_metrics(input_np, output_np)
        
        # ---- Save to output folder ----
        output_folder_name = request.form.get('output') or request.args.get('output') or datetime.now().strftime('result_%Y%m%d_%H%M%S')
        output_dir = os.path.join(AppState.OUTPUT_DIR, output_folder_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save input and generated
        Image.fromarray(input_np).save(os.path.join(output_dir, 'input.png'))
        Image.fromarray(output_np).save(os.path.join(output_dir, 'generated.png'))
        
        # Generate difference map (grayscale)
        diff_map_gray = create_difference_map(input_np, output_np)
        diff_map_pil = Image.fromarray(diff_map_gray, mode='L')  # Save as grayscale
        diff_map_pil.save(os.path.join(output_dir, 'difference.png'))
        
        # Generate comparison grid (2x2 with hot colormap like cyclegan_inference.py)
        comparison_grid = create_comparison_grid(input_np, output_np)
        comparison_grid.save(os.path.join(output_dir, 'comparison.png'))
        
        # Save metrics
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"  ✓ Results saved to {output_dir}")
        print(f"    - input.png (original)")
        print(f"    - generated.png (model output)")
        print(f"    - difference.png (grayscale diff map)")
        print(f"    - comparison.png (2x2 grid with hot colormap)")
        print(f"    - metrics.json (quality scores)")
        
        # ---- Return result ----
        if output_format == 'json':
            # Load all 4 saved images from disk and encode to base64
            try:
                input_b64 = image_to_base64(Image.open(os.path.join(output_dir, 'input.png')))
                print(f"  ✓ Encoded input.png")
            except Exception as e:
                print(f"  ❌ Error encoding input: {e}")
                input_b64 = None
            
            try:
                output_b64 = image_to_base64(Image.open(os.path.join(output_dir, 'generated.png')))
                print(f"  ✓ Encoded generated.png")
            except Exception as e:
                print(f"  ❌ Error encoding generated: {e}")
                output_b64 = None
            
            try:
                # Difference map is grayscale, convert to RGB for display
                diff_img = Image.open(os.path.join(output_dir, 'difference.png')).convert('RGB')
                diff_b64 = image_to_base64(diff_img)
                print(f"  ✓ Encoded difference.png (grayscale → RGB)")
            except Exception as e:
                print(f"  ❌ Error encoding difference: {e}")
                diff_b64 = None
            
            try:
                comparison_b64 = image_to_base64(Image.open(os.path.join(output_dir, 'comparison.png')))
                print(f"  ✓ Encoded comparison.png (2x2 grid)")
            except Exception as e:
                print(f"  ❌ Error encoding comparison: {e}")
                comparison_b64 = None
            
            # Load metrics
            try:
                with open(os.path.join(output_dir, 'metrics.json'), 'r') as f:
                    metrics_data = json.load(f)
            except:
                metrics_data = metrics
            
            return jsonify({
                'status': 'success',
                'image': image_path_str,
                'checkpoint': checkpoint_name,
                'output_folder': output_dir,
                'input_image': input_b64,
                'generated_image': output_b64,
                'difference_image': diff_b64,
                'comparison_image': comparison_b64,
                'metrics': metrics_data
            }), 200
        else:
            html = generate_html_result(input_np, output_np, metrics, image_path_str, checkpoint_name)
            return html, 200
    
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return jsonify({'error': f'❌ Server error: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'device': AppState.DEVICE.upper(),
        'cuda_available': torch.cuda.is_available(),
        'default_checkpoint': AppState.DEFAULT_CHECKPOINT,
        'checkpoints_available': len(get_available_checkpoints()),
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/api/checkpoints', methods=['GET'])
def list_checkpoints():
    """List available checkpoints"""
    checkpoints = get_available_checkpoints()
    return jsonify({
        'count': len(checkpoints),
        'checkpoints': checkpoints,
        'default': AppState.DEFAULT_CHECKPOINT,
        'latest': checkpoints[-1] if checkpoints else None
    }), 200

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': '❌ File too large (max 50 MB)'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': '❌ Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': '❌ Internal server error'}), 500

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='CycleGAN TB↔Normal Image Generation Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python server.py
  python server.py --port 8080
  python server.py --checkpoint epoch_0100 --port 8080
  python server.py --port 8080 --output-dir ./my_results
  python server.py --checkpoint epoch_0149 --output-dir ./generated --host 0.0.0.0
        """
    )
    parser.add_argument('--host', type=str, default='localhost', help='Host to bind to (default: localhost)')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to (default: 5000)')
    parser.add_argument('--checkpoint', type=str, default=epoch, help='Default checkpoint/epoch to use (overrides module-level epoch variable)')
    parser.add_argument('--output-dir', type=str, default='server_outputs', help='Output directory for results (default: server_outputs)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default=AppState.DEVICE, help='Device to use')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Set configuration
    AppState.DEVICE = args.device
    AppState.OUTPUT_DIR = args.output_dir
    AppState.DEFAULT_CHECKPOINT = args.checkpoint or get_available_checkpoints()[-1]
    
    # Print banner
    print(f"\n{'='*80}")
    print(f"🏥 CycleGAN TB IMAGE GENERATION SERVER - PORT FORWARDING READY")
    print(f"{'='*80}")
    print(f"\n📍 Server Configuration:")
    print(f"   Host:        {args.host}")
    print(f"   Port:        {args.port}")
    print(f"   Device:      {AppState.DEVICE.upper()}")
    print(f"   Checkpoint:  {AppState.DEFAULT_CHECKPOINT}")
    print(f"   Output Dir:  {AppState.OUTPUT_DIR}")
    print(f"\n🌐 Access URLs:")
    if args.host == 'localhost':
        print(f"   Local:       http://{args.host}:{args.port}/")
        print(f"   Tunnel:      code tunnel (then share the URL)")
    else:
        print(f"   Remote:      http://0.0.0.0:{args.port}/")
    print(f"\n📝 Available Checkpoints: {len(get_available_checkpoints())}")
    for cp in get_available_checkpoints()[-5:]:
        print(f"   - {cp}")
    print(f"\n💡 Quick Examples:")
    print(f"   /?image=dataset/TB/TB.1.jpg")
    print(f"   /?image=TB.jpg&checkpoint=epoch_0100&format=json")
    print(f"\n{'='*80}\n")
    
    # Start server
    print(f"🚀 Starting server...\n")
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main()
    main()

    AppState.OUTPUT_DIR = args.output_dir
    AppState.DEFAULT_CHECKPOINT = args.checkpoint or get_available_checkpoints()[-1]
    
    # Print banner
    print(f"\n{'='*80}")
    print(f"🏥 CycleGAN TB IMAGE GENERATION SERVER - PORT FORWARDING READY")
    print(f"{'='*80}")
    print(f"\n📍 Server Configuration:")
    print(f"   Host:        {args.host}")
    print(f"   Port:        {args.port}")
    print(f"   Device:      {AppState.DEVICE.upper()}")
    if AppState.DEVICE == 'cuda':
        print(f"   GPU:         {torch.cuda.get_device_name(0)}")
    print(f"   Default Epoch: {AppState.DEFAULT_CHECKPOINT}")
    print(f"   Output Dir:  {os.path.abspath(AppState.OUTPUT_DIR)}")
    print(f"   Checkpoints: {len(get_available_checkpoints())} available")
    
    print(f"\n🔗 Quick Access URLs:")
    print(f"   Health:      http://{args.host}:{args.port}/api/health")
    print(f"   Generate:    http://{args.host}:{args.port}/?image=TB1.jpg")
    print(f"   With Epoch:  http://{args.host}:{args.port}/?image=TB1.jpg&checkpoint={AppState.DEFAULT_CHECKPOINT}")
    print(f"   Custom Out:  http://{args.host}:{args.port}/?image=TB1.jpg&output=my_results")
    
    print(f"\n📁 Port Forwarding (share with friends):")
    print(f"   1. Run: code tunnel")
    print(f"   2. Copy the public URL (https://xxx-{args.port}.inc1.devtunnels.ms/)")
    print(f"   3. Share: https://xxx-{args.port}.inc1.devtunnels.ms/?image=TB1.jpg")
    
    print(f"\n{'='*80}\n")
    
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main()
