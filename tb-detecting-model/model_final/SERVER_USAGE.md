# TB Detection Server — Usage Guide

A Flask-based HTTP server for TB detection inference. Upload chest X-ray images and get predictions.

## Installation

Install Flask and requests (if not already installed):

```bash
pip install flask requests
```

## Running the Server

### Local (default)
```bash
python server.py
```
Runs on: `http://localhost:5000`

### Specify host & port
```bash
python server.py --host 0.0.0.0 --port 8000
```

### Debug mode (auto-reload)
```bash
python server.py --debug
```

### Verbose logging
```bash
python server.py --verbose
```

## Endpoints

### 1. **Health Check**
```bash
GET /health
```
Returns: `{"status": "ok", "service": "TB Detection Server"}`

### 2. **List Available Models**
```bash
GET /models
```
Returns available models and aliases.

### 3. **Predict from File Upload** ⭐
```bash
POST /predict
```
**Required Parameters (form-data):**
- `model`: `svm` | `rf` | `gb` | `ensemble` (required)
- `image`: Binary file upload (required)

**Supported formats:** `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`

**Response:**
```json
{
  "model": "ensemble",
  "filename": "test.jpg",
  "prediction": "TB",
  "confidence": 0.8542,
  "risk_level": "High",
  "prob_normal": 0.1458,
  "prob_tb": 0.8542
}
```

### 4. **Predict from URL**
```bash
POST /predict_url
```
**Required Parameters (JSON):**
- `model`: `svm` | `rf` | `gb` | `ensemble` (required)
- `image_url`: URL to image file (required)

**Response:** Same as `/predict`

---

## Usage Examples

### Via curl (local)

**Single image upload:**
```bash
curl -X POST -F "image=@test.jpg" -F "model=ensemble" \
     http://localhost:5000/predict
```

**Different models:**
```bash
curl -X POST -F "image=@test.jpg" -F "model=svm" \
     http://localhost:5000/predict

curl -X POST -F "image=@test.jpg" -F "model=rf" \
     http://localhost:5000/predict

curl -X POST -F "image=@test.jpg" -F "model=gb" \
     http://localhost:5000/predict
```

**From URL:**
```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"image_url":"https://example.com/xray.jpg", "model":"ensemble"}' \
     http://localhost:5000/predict_url
```

---

## Via ngrok (Public Access)

### 1. Start your server
```bash
python server.py --host 0.0.0.0 --port 5000
```

### 2. In another terminal, start ngrok
```bash
ngrok http 5000
```

You'll get a URL like: `https://1234-56-78-90-123.ngrok-free.app`

### 3. Make requests via ngrok URL

**Upload image:**
```bash
curl -X POST -F "image=@test.jpg" -F "model=ensemble" \
     https://1234-56-78-90-123.ngrok-free.app/predict
```

**From URL:**
```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"image_url":"https://example.com/xray.jpg", "model":"ensemble"}' \
     https://1234-56-78-90-123.ngrok-free.app/predict_url
```

---

## Python Client Example

```python
import requests
import json

# Upload image
url = 'http://localhost:5000/predict'
files = {'image': open('test.jpg', 'rb')}
data = {'model': 'ensemble'}
response = requests.post(url, files=files, data=data)
result = response.json()
print(result)
```

**From URL:**
```python
import requests
import json

url = 'http://localhost:5000/predict_url'
payload = {
    'model': 'ensemble',
    'image_url': 'https://example.com/xray.jpg'
}
response = requests.post(url, json=payload)
result = response.json()
print(result)
```

---

## Model Descriptions

| Model | Type | Characteristics |
|-------|------|-----------------|
| **SVM** | Support Vector Machine | Fast, linear classification, uses DenseNet features |
| **RF** | Random Forest | Robust ensemble, uses texture (GLCM) features |
| **GB** | Gradient Boosting | Sequential ensemble, hybrid features |
| **Ensemble** | Voting Ensemble | Best accuracy, combines all three models |

**Recommended:** Start with `ensemble` for highest accuracy.

---

## Response Fields

- **prediction**: `Normal` or `TB`
- **confidence**: 0–1 (probability of predicted class)
- **risk_level**: `Low` | `Medium` | `High` (for TB cases)
  - `Normal` → always `Low`
  - TB with confidence ≥ 0.90 → `High`
  - TB with confidence ≥ 0.70 → `Medium`
  - TB with confidence < 0.70 → `Low`
- **prob_normal**: Probability of Normal class (0–1)
- **prob_tb**: Probability of TB class (0–1)

---

## Error Handling

**Common Errors:**

| Error | Solution |
|-------|----------|
| `Missing "model" parameter` | Add `--data 'model=ensemble'` or `-F "model=ensemble"` |
| `Missing "image" file` | Include file upload with `-F "image=@filename"` |
| `Unsupported file format` | Use: `.jpg`, `.png`, `.bmp`, `.tiff` |
| `File too large (413)` | Max 50 MB. Compress image. |
| `Model files not found (500)` | Check `models/` folder exists with `.pkl` files |

---

## Performance Notes

- **First request:** ~2–3 seconds (models loaded in memory)
- **Subsequent requests:** ~0.5–1 second per image
- **Batch processing:** Submit multiple requests (server is threaded)
- **GPU:** If CUDA available, DenseNet features computed on GPU for speed

---

## Architecture

```
Request
  ↓
[Flask Route Handler]
  ↓
[Load Model (cached)]
  ↓
[Read & Decode Image]
  ↓
[Lung Segmentation]
  ↓
[Feature Extraction (DenseNet + GLCM)]
  ↓
[Feature Selection & Scaling]
  ↓
[Classification]
  ↓
[JSON Response]
```

---

## Troubleshooting

### Port already in use
```bash
python server.py --port 8001
```

### Models not loading
Ensure these files exist in `./models/`:
- `svm_model.pkl`
- `rf_model.pkl`
- `gb_model.pkl`
- `ensemble_model.pkl` + `ensemble_metadata.json`

### Segmentation returns None
- Image may be corrupted
- Try a different format (`.jpg` → `.png`)
- Check image dimensions (must be valid X-ray format)

### ngrok connection issues
- Ensure server is running on `0.0.0.0` not `127.0.0.1`
- Check firewall settings
- Restart ngrok: `ngrok http 5000`

---

## License & Credits

TB Detection System — Model Pipeline + Server
