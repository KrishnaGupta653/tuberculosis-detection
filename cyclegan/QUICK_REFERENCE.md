# 🚀 Quick 10-Line Guide

## For You (Local Testing)

```bash
# Default - outputs to server_outputs/
python server.py --port 8080

# Custom epoch - save to specific folder
python server.py --port 8080 --checkpoint epoch_0149 --output-dir ./my_results

# Test via browser
http://localhost:8080/?image=dataset/TB/TB.1.jpg&checkpoint=epoch_0100
```

## For Your Friend (Shared Laptop)

```bash
# Friend puts TB.png in their folder, runs this (saves output in their current directory)
python server.py --port 8080 --checkpoint epoch_0149 --output-dir ./generated

# Friend opens browser (replace XXX with actual tunnel URL from 'code tunnel')
https://xyz-8080.inc1.devtunnels.ms/?image=TB.png&output=result_001
```

## Port Forwarding Setup (1 minute)

```bash
# Terminal 1: Activate tunnel
code tunnel

# Terminal 2: Start server with custom epoch
python server.py --port 8080 --checkpoint epoch_0149 --output-dir ./results

# Share the tunnel URL with friend → they use it in browser with ?image=filename
```

## CLI Parameters Explained

| Parameter      | Purpose               | Example                              |
| -------------- | --------------------- | ------------------------------------ |
| `--port`       | Server port           | `--port 8080`                        |
| `--checkpoint` | Default epoch/model   | `--checkpoint epoch_0149`            |
| `--output-dir` | Where to save results | `--output-dir ./generated`           |
| `--host`       | For external access   | `--host 0.0.0.0` (for local network) |

## Query Parameters (URL)

| Parameter    | Purpose                | Example                        |
| ------------ | ---------------------- | ------------------------------ |
| `image`      | Input image path       | `?image=TB.png`                |
| `checkpoint` | Override default epoch | `?checkpoint=epoch_0100`       |
| `output`     | Result folder name     | `?output=my_run_001`           |
| `format`     | Response type          | `?format=json` (default: html) |

✅ **Done!** Your friend can now generate images with just TB.png in their folder.
