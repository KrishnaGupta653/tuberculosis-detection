#!/usr/bin/env python3
"""
test_server.py — Quick test script for TB Detection Server
"""

import requests
import argparse
import sys
from pathlib import Path


def test_health(base_url: str):
    """Test /health endpoint"""
    print(f"\n[TEST 1] Health Check")
    print(f"  URL: {base_url}/health")
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        if r.status_code == 200:
            print(f"  ✓ Server is UP")
            print(f"  Response: {r.json()}")
            return True
        else:
            print(f"  ✗ Unexpected status {r.status_code}")
            print(f"  Response: {r.text}")
            return False
    except requests.ConnectionError:
        print(f"  ✗ Cannot connect to {base_url}")
        print(f"  Make sure server is running: python server.py")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_models(base_url: str):
    """Test /models endpoint"""
    print(f"\n[TEST 2] List Models")
    print(f"  URL: {base_url}/models")
    try:
        r = requests.get(f"{base_url}/models", timeout=5)
        if r.status_code == 200:
            data = r.json()
            print(f"  ✓ Available models:")
            for model in data.get('available_models', []):
                print(f"    - {model}")
            return True
        else:
            print(f"  ✗ Status {r.status_code}: {r.text}")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_predict_image(base_url: str, image_path: str, model: str = "ensemble"):
    """Test /predict endpoint with file upload"""
    print(f"\n[TEST 3] Predict from Image File")
    print(f"  URL: {base_url}/predict")
    print(f"  Image: {image_path}")
    print(f"  Model: {model}")
    
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"  ✗ File not found: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {'model': model}
            r = requests.post(f"{base_url}/predict", files=files, data=data, timeout=30)
        
        if r.status_code == 200:
            result = r.json()
            print(f"  ✓ Prediction successful!")
            print(f"    Prediction: {result.get('prediction')}")
            print(f"    Confidence: {result.get('confidence')*100:.1f}%")
            print(f"    Risk Level: {result.get('risk_level')}")
            
            # Probability bars
            prob_normal = result.get('prob_normal', 0) * 100
            prob_tb = result.get('prob_tb', 0) * 100
            bar_width = 30
            
            normal_bar_len = int(prob_normal / 100 * bar_width)
            tb_bar_len = int(prob_tb / 100 * bar_width)
            
            normal_bar = "█" * normal_bar_len + "░" * (bar_width - normal_bar_len)
            tb_bar = "█" * tb_bar_len + "░" * (bar_width - tb_bar_len)
            
            print(f"    Normal: [{normal_bar}] {prob_normal:5.1f}%")
            print(f"    TB:     [{tb_bar}] {prob_tb:5.1f}%")
            return True
        else:
            print(f"  ✗ Status {r.status_code}")
            print(f"    Response: {r.json()}")
            return False
    except requests.Timeout:
        print(f"  ✗ Request timeout (>30s)")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_predict_url(base_url: str, image_url: str, model: str = "ensemble"):
    """Test /predict_url endpoint"""
    print(f"\n[TEST 4] Predict from URL")
    print(f"  URL: {base_url}/predict_url")
    print(f"  Image URL: {image_url}")
    print(f"  Model: {model}")
    
    try:
        payload = {
            'model': model,
            'image_url': image_url
        }
        r = requests.post(f"{base_url}/predict_url", json=payload, timeout=30)
        
        if r.status_code == 200:
            result = r.json()
            print(f"  ✓ Prediction successful!")
            print(f"    Prediction: {result.get('prediction')}")
            print(f"    Confidence: {result.get('confidence')*100:.1f}%")
            print(f"    Risk Level: {result.get('risk_level')}")
            
            # Probability bars
            prob_normal = result.get('prob_normal', 0) * 100
            prob_tb = result.get('prob_tb', 0) * 100
            bar_width = 30
            
            normal_bar_len = int(prob_normal / 100 * bar_width)
            tb_bar_len = int(prob_tb / 100 * bar_width)
            
            normal_bar = "█" * normal_bar_len + "░" * (bar_width - normal_bar_len)
            tb_bar = "█" * tb_bar_len + "░" * (bar_width - tb_bar_len)
            
            print(f"    Normal: [{normal_bar}] {prob_normal:5.1f}%")
            print(f"    TB:     [{tb_bar}] {prob_tb:5.1f}%")
            return True
        else:
            print(f"  ✗ Status {r.status_code}")
            print(f"    Response: {r.json()}")
            return False
    except requests.Timeout:
        print(f"  ✗ Request timeout (>30s)")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    ap = argparse.ArgumentParser(
        description='Test TB Detection Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_server.py --url http://localhost:5000
  python test_server.py --url https://YOUR-NGROK-URL
  python test_server.py --url http://localhost:5000 --image test.jpg
  python test_server.py --url http://localhost:5000 --image test.jpg --model svm
        """
    )
    ap.add_argument('--url', required=True,
                    help='Server URL (e.g., http://localhost:5000 or https://ngrok-url)')
    ap.add_argument('--image', default=None,
                    help='Path to test image (optional)')
    ap.add_argument('--model', default='ensemble',
                    choices=['svm', 'rf', 'gb', 'ensemble'],
                    help='Model to test (default: ensemble)')
    ap.add_argument('--image-url', default=None,
                    help='URL to test image (optional)')
    
    args = ap.parse_args()
    
    # Normalize URL
    base_url = args.url.rstrip('/')
    
    print("=" * 60)
    print("TB DETECTION SERVER — TEST SUITE")
    print("=" * 60)
    
    results = []
    
    # Test 1: Health
    results.append(("Health Check", test_health(base_url)))
    
    # Test 2: Models
    results.append(("List Models", test_models(base_url)))
    
    # Test 3: Predict from file
    if args.image:
        results.append((f"Predict Image ({args.model})", 
                       test_predict_image(base_url, args.image, args.model)))
    
    # Test 4: Predict from URL
    if args.image_url:
        results.append((f"Predict URL ({args.model})", 
                       test_predict_url(base_url, args.image_url, args.model)))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"\nPassed: {passed}/{total}")
    for name, result in results:
        status = "✓" if result else "✗"
        print(f"  {status} {name}")
    
    if passed == total:
        print(f"\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
