import requests
import base64
import os
import json
from PIL import Image
from io import BytesIO

image_path = "C:\\Users\\kg060\\Desktop\\projects\\tuberculosis\\cyclegan\\TB1.jpg"

# Create folder to save results
filename_base = os.path.splitext(os.path.basename(image_path))[0]
output_folder = f"{filename_base}_results"
os.makedirs(output_folder, exist_ok=True)

# Send request to server
with open(image_path, 'rb') as f:
    response = requests.post(
        'https://6625-122-161-53-195.ngrok-free.app/api/generate?format=json',
        files={'image': f},
        data={'checkpoint': 'epoch_0025', 'output': 'results_001'}
    )

result = response.json()

# Debug: Check what keys are in response
print("Keys in response:", list(result.keys()))

# Save all 4 images
images_to_save = {
    'input': result.get('input_image'),
    'generated': result.get('generated_image'),
    'difference': result.get('difference_image'),
    'comparison': result.get('comparison_image'),
}

print("\nImage sizes (first 100 chars of base64):")
for img_type, base64_data in images_to_save.items():
    if base64_data:
        print(f"  {img_type}: {len(base64_data)} chars")
    else:
        print(f"  {img_type}: MISSING!")

print("\nSaving images:")
for img_type, base64_data in images_to_save.items():
    if base64_data:
        try:
            image_data = base64.b64decode(base64_data)
            img = Image.open(BytesIO(image_data))
            output_name = os.path.join(output_folder, f"{img_type}.png")
            img.save(output_name)
            print(f"✅ Saved {img_type}.png")
        except Exception as e:
            print(f"❌ Failed to save {img_type}.png: {e}")
    else:
        print(f"⚠️  {img_type} is missing from server response!")

# Save metrics
metrics_file = os.path.join(output_folder, 'metrics.json')
with open(metrics_file, 'w') as f:
    json.dump(result['metrics'], f, indent=2)
print(f"✅ Saved metrics.json")

print(f"\n📊 Key Metrics:")
print(f"   SSIM: {result['metrics']['ssim']:.4f}")
print(f"   PSNR: {result['metrics']['psnr']:.2f}")
print(f"   MAE: {result['metrics']['mae']:.2f}")
print(f"\n📁 All files saved in folder: {output_folder}")
