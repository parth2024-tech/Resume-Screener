# 🚀 AI-Powered Resume Screening & Ranking System

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-frontend-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📌 Overview

**AI-Powered Resume Screening & Ranking System** is a smart, automated recruitment assistant that helps HR professionals and recruiters efficiently screen and rank resumes based on the relevance to a given job description using advanced **NLP** and **ML** techniques.

By automating resume filtering, this tool saves hours of manual review and ensures that only the most relevant candidates are shortlisted.

---

## ✨ Features

- 🔍 **Smart Job Description Input**
  - Paste directly or upload job description files
  - Supports: `.txt`, `.pdf`, `.docx`, `.jpg`, `.jpeg`, `.png`

- 📂 **Multi-Format Resume Upload**
  - Upload multiple resumes at once
  - Supports scanned and digital resumes in formats: `.pdf`, `.docx`, `.txt`, `.jpg`, `.png`, `.jpeg`

- 🧠 **AI-Based Resume Matching**
  - Uses **TF-IDF (with unigram and bigram representation)** and **Cosine Similarity** to evaluate matching scores.
  - Sublinear term-frequency scaling is applied to normalize high repetition keyword spamming.
  - Intelligent filtering using **spaCy** NLP preprocessing (lowercasing, lemmatization, stopword filtering, and punctuation removal).
  - Handles scanned documents seamlessly with an integrated Tesseract OCR fallback mechanism.

- 📊 **Candidate Ranking Dashboard**
  - Real-time ranking display with interactive Streamlit UI
  - Highlights resume scores in percentage

---

## 🛠 Tech Stack

| Layer       | Tools / Libraries |
|-------------|-------------------|
| 💻 Frontend | `Streamlit`       |
| 🧠 NLP      | `spaCy`, `pytesseract`, `pdfplumber`, `docx`, `PIL` |
| 📊 ML Logic | `scikit-learn` (TF-IDF, Cosine Similarity) |
| 📈 Data     | `pandas`, `NumPy` |
| 📁 File Handling | `pdfplumber`, `docx`, `pytesseract`, `PIL` |
| ⚙️ Utilities | `os`, `subprocess` |

---

## 📷 Screenshots

> 📸 *[Include app screenshots here if available]*  
> (For example: Upload JD, Upload Resumes, Ranked Resume Table)

---

## 🚀 Getting Started

### 🔧 Prerequisites

Ensure you have the following installed:
- Python 3.8+
- pip
- tesseract-ocr (for OCR support)

### 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-resume-ranker.git
cd ai-resume-ranker

# Install dependencies
pip install -r requirements.txt

sudo apt install tesseract-ocr

# Run the app
streamlit run app.py
```

### 🖥️ Running via CLI
You can also run the resume screening via the command line without the UI. Use the `--jd` and `--resume-dir` flags to customize the input paths:
```bash
python Scripts/resume_ranking.py --jd "data/job_description.txt" --resume-dir "data"
```

### 🧪 Running Unit Tests
To run the automated test suite for scoring and matching logic:
```bash
python -m unittest tests/test_nlp_logic.py
```

### 🧹 Cache Cleanup
To clean up compiled Python cache files (`.pyc`) and `__pycache__` directories:
```bash
python scripts/clean_pycache.py
```
