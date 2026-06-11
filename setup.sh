#!/bin/bash
# ──────────────────────────────────────────────
# Setup script for AI Resume Screening System
# ──────────────────────────────────────────────

set -e

echo "📦 Installing system dependencies..."

# Install Tesseract OCR (works for both root and non-root users)
if command -v sudo &> /dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y tesseract-ocr
else
    apt-get update -qq
    apt-get install -y tesseract-ocr
fi

echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

echo "📦 Downloading spaCy language model..."
python -m spacy download en_core_web_sm

echo ""
echo "✅ Setup complete!"
echo "   Run the app with:  streamlit run app.py"