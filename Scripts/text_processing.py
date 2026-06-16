"""
text_processing.py — Single source of truth for NLP preprocessing and file extraction.

All text extraction (PDF, DOCX, Image) and preprocessing logic lives here.
Both app.py and CLI scripts import from this module.
"""

import shutil
import subprocess
import spacy
import pdfplumber
import pytesseract
from PIL import Image
from docx import Document


# ---------------------------------------------------------------------------
# Dynamic Tesseract path detection (cross-platform)
# ---------------------------------------------------------------------------
_tesseract_path = shutil.which("tesseract")
if _tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = _tesseract_path


# ---------------------------------------------------------------------------
# spaCy model — module-level singleton (loaded once, reused everywhere)
# ---------------------------------------------------------------------------
_nlp = None


def load_nlp_model():
    """Load spaCy model with parser and NER disabled for speed.

    Returns the cached model on subsequent calls.
    """
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except OSError:
            subprocess.run(
                ["python", "-m", "spacy", "download", "en_core_web_sm"],
                check=True,
            )
            _nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    return _nlp


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------
def preprocess_text(text):
    """Clean text via spaCy: lowercasing, lemmatization, stopword & punctuation removal."""
    nlp = load_nlp_model()
    # 1. Lowercase inputs to ensure case-insensitivity
    doc = nlp(text.lower())
    # 2. Tokenize, lemmatize, and filter stopwords/punctuation
    tokens = [
        token.lemma_ 
        for token in doc 
        if not token.is_stop and not token.is_punct and token.text.strip()
    ]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Text extraction — one function per format, each wrapped in try/except
# ---------------------------------------------------------------------------
def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file, falling back to OCR for scanned pages.

    Returns an empty string (never crashes) if extraction fails.
    """
    try:
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
                else:
                    # Scanned page — fall back to OCR
                    text += pytesseract.image_to_string(
                        page.to_image(resolution=300).original
                    ) + "\n"
        return text.strip()
    except Exception:
        return ""


def extract_text_from_image(image_file):
    """Extract text from an image (JPG/PNG/JPEG) via OCR.

    Returns an empty string if extraction fails.
    """
    try:
        return pytesseract.image_to_string(
            Image.open(image_file).convert("RGB")
        ).strip()
    except Exception:
        return ""


def extract_text_from_docx(docx_file):
    """Extract text from a Word document (.docx).

    Returns an empty string if extraction fails.
    """
    try:
        return "\n".join(
            para.text for para in Document(docx_file).paragraphs
        ).strip()
    except Exception:
        return ""
