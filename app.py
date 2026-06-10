import streamlit as st
import os
import spacy
import pandas as pd
import subprocess
import pdfplumber
import pytesseract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from docx import Document

# Streamlit Cloud ke liye Tesseract ka path set kiya hai
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# Ensure spaCy model is installed
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")


# Load external CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Apply CSS
load_css("styles.css")


# Function to extract text from PDF (including OCR for scanned PDFs)
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            text += extracted_text + "\n" if extracted_text else pytesseract.image_to_string(
                page.to_image(resolution=300).original) + "\n"
    return text.strip()


# Function to extract text from images (JPG, PNG, JPEG) using OCR
def extract_text_from_image(image_file):
    return pytesseract.image_to_string(Image.open(image_file).convert("RGB")).strip()


# Function to extract text from Word documents (.docx)
def extract_text_from_docx(docx_file):
    return "\n".join([para.text for para in Document(docx_file).paragraphs]).strip()


# Text preprocessing using spaCy (lemmatization & stopword removal)
def preprocess_text(text):
    return " ".join([token.lemma_ for token in nlp(text.lower()) if not token.is_stop and not token.is_punct])


# Function to calculate similarity between job description and resumes
def calculate_similarity(job_desc, resumes):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([job_desc] + resumes)
    return cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])[0]


# Streamlit UI
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🚀 AI-Powered Resume Screening & Ranking System</h1>",
            unsafe_allow_html=True)

with st.container():
    st.markdown("### 📜 Job Description")
    jd_option = st.radio("How do you want to provide the Job Description?", ("Paste Job Description", "Upload a File"),
                         horizontal=True)

job_desc = ""

if jd_option == "Paste Job Description":
    job_desc = st.text_area("✍ Paste Job Description Here", height=150)

elif jd_option == "Upload a File":
    job_desc_file = st.file_uploader("📂 Upload a Job Description file (.txt, .pdf, .docx, .jpg, .png, .jpeg)",
                                     type=["txt", "pdf", "docx", "jpg", "jpeg", "png"])
    if job_desc_file:
        ext = job_desc_file.name.split(".")[-1].lower()
        if ext == "txt":
            job_desc = job_desc_file.read().decode("utf-8")
        elif ext == "pdf":
            job_desc = extract_text_from_pdf(job_desc_file)
        elif ext == "docx":
            job_desc = extract_text_from_docx(job_desc_file)
        elif ext in ["jpg", "jpeg", "png"]:
            job_desc = extract_text_from_image(job_desc_file)

if job_desc:
    job_desc = preprocess_text(job_desc)

st.markdown("---")

st.markdown("### 📂 Upload Resumes")
resume_files = st.file_uploader("📤 Upload Multiple Resumes (.txt, .pdf, .jpg, .png, .jpeg, .docx)",
                                type=["txt", "pdf", "jpg", "jpeg", "png", "docx"], accept_multiple_files=True)

if job_desc and resume_files:
    st.markdown("### 🔄 Processing Resumes...")
    progress_bar = st.progress(0)

    resumes = []
    resume_names = []

    for idx, resume_file in enumerate(resume_files):
        ext = resume_file.name.split(".")[-1].lower()
        if ext == "txt":
            resume_text = resume_file.read().decode("utf-8")
        elif ext == "pdf":
            resume_text = extract_text_from_pdf(resume_file)
        elif ext == "docx":
            resume_text = extract_text_from_docx(resume_file)
        elif ext in ["jpg", "jpeg", "png"]:
            resume_text = extract_text_from_image(resume_file)

        resumes.append(preprocess_text(resume_text))
        resume_names.append(resume_file.name)

        progress_bar.progress((idx + 1) / len(resume_files))

    scores = calculate_similarity(job_desc, resumes)
    scores_percentage = [round(score * 100, 2) for score in scores]

    ranked_resumes = sorted(zip(resume_names, scores_percentage), key=lambda x: x[1], reverse=True)

    df = pd.DataFrame(ranked_resumes, columns=["Resume", "Candidate Score (%)"])
    df = df.style.set_properties(**{"text-align": "left"}).bar(subset=["Candidate Score (%)"], color="#FF4B4B")

    st.markdown("### 📊 Ranked Resumes")
    st.dataframe(df)