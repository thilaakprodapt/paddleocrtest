# CIOMS Form Text Extractor

Extract key-value pairs from CIOMS (Council for International Organizations of Medical Sciences) forms using PaddleOCR 3.x.

## Features
- Extracts filled fields only (ignores empty fields)
- Handles scanned images, digital documents, and handwritten forms
- Groups extracted data by category (Patient Info, Drug Info, etc.)

## Requirements
- Python 3.8+
- PaddleOCR 3.x
- PaddlePaddle 3.x

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd rxlogic

# Create virtual environment
python3 -m venv paddleocr_env
source paddleocr_env/bin/activate  # Linux/Mac
# or: paddleocr_env\Scripts\activate  # Windows

# Install dependencies
pip install paddleocr paddlepaddle
```

## Usage

```bash
python extractor.py <image_path>
```

Example:
```bash
python extractor.py Filled_Cioms_form-1.jpg
```

## GCP VM Setup

For running on a GCP Linux VM (recommended for PaddleOCR 3.x):

```bash
chmod +x setup_gcp.sh
./setup_gcp.sh
source paddleocr_env/bin/activate
python extractor.py <image_path>
```

## Output

The extractor outputs filled fields grouped by category:
- Patient Information
- Adverse Event
- Drug Information
- Reporter Information
- Case Details
- Additional Information
