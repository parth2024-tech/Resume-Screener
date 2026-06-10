#!/bin/bash
# Install Tesseract OCR
apt-get update
apt-get install -y tesseract-ocr

export TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata/
export PATH=$PATH:/usr/local/bin/