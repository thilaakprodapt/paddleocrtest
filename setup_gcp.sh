#!/bin/bash
# GCP VM Setup Script for PaddleOCR CIOMS Extractor

# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Python and dependencies
sudo apt-get install -y python3 python3-pip python3-venv libgl1-mesa-glx libglib2.0-0

# Create virtual environment
python3 -m venv paddleocr_env
source paddleocr_env/bin/activate

# Install PaddleOCR
pip install --upgrade pip
pip install paddleocr paddlepaddle

echo ""
echo "=== Setup Complete ==="
echo "To run the extractor:"
echo "  source paddleocr_env/bin/activate"
echo "  python extractor.py <image_path>"
