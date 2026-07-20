#!/bin/bash
set -e
echo "=== GeniVideo Worker Starting ==="
echo "Step 1: Downloading/verifying model assets..."
python /app/download_models.py
echo "Step 2: Launching RunPod handler..."
exec python -u /app/rp_handler.py
