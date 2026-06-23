"""
app.py — AI-Powered Resume Screening & Ranking System (Streamlit UI)

All extraction and NLP logic is imported from Scripts/ to avoid duplication.
Uses Streamlit caching to prevent redundant model loads and re-processing.
"""

import io
import os
import streamlit as st
import pandas as pd

from Scripts.text_processing import (
    load_nlp_model,
    preprocess_text,
    extract_text_from_pdf,
    extract_text_from_image,
    extract_text_from_docx,
)
from Scripts.resume_ranking import (
    calculate_similarity,
    normalize_scores,
    get_score_label,
)


# ──────────────────────────────────────────────
# Caching — model loaded ONCE across all reruns
# ──────────────────────────────────────────────
@st.cache_resource
def init_nlp():
    """Load and cache the spaCy NLP model (survives Streamlit reruns)."""
    return load_nlp_model()


@st.cache_data
def cached_extract_text(file_bytes, file_name):
    """Extract text from a file, cached by content hash so identical
    files are never re-processed even across reruns."""
    ext = file_name.split(".")[-1].lower()

    if ext == "txt":
        return file_bytes.decode("utf-8", errors="replace")
    elif ext == "pdf":
        return extract_text_from_pdf(io.BytesIO(file_bytes))
    elif ext == "docx":
        return extract_text_from_docx(io.BytesIO(file_bytes))
    elif ext in ("jpg", "jpeg", "png"):
        return extract_text_from_image(io.BytesIO(file_bytes))
    return ""


# Warm up the model on first load
init_nlp()


# ──────────────────────────────────────────────
# Load external CSS
# ──────────────────────────────────────────────
def load_css(file_name):
    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css("styles.css")


# ──────────────────────────────────────────────
# UI — Title
# ──────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align: center; color: #FF4B4B;'>"
    "🚀 AI-Powered Resume Screening & Ranking System"
    "</h1>",
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
# UI — Job Description input
# ──────────────────────────────────────────────
with st.container():
    st.markdown("### 📜 Job Description")
    jd_option = st.radio(
        "How do you want to provide the Job Description?",
        ("Paste Job Description", "Upload a File"),
        horizontal=True,
    )

job_desc_raw = ""

if jd_option == "Paste Job Description":
    job_desc_raw = st.text_area("✍ Paste Job Description Here", height=150)

elif jd_option == "Upload a File":
    job_desc_file = st.file_uploader(
        "📂 Upload a Job Description file (.txt, .pdf, .docx, .jpg, .png, .jpeg)",
        type=["txt", "pdf", "docx", "jpg", "jpeg", "png"],
        key="jd_uploader",
    )
    if job_desc_file:
        try:
            job_desc_raw = cached_extract_text(
                job_desc_file.getvalue(), job_desc_file.name
            )
        except Exception as e:
            st.warning(f"⚠️ Failed to extract job description: {e}")

st.markdown("---")


# ──────────────────────────────────────────────
# UI — Resume upload
# ──────────────────────────────────────────────
st.markdown("### 📂 Upload Resumes")
resume_files = st.file_uploader(
    "📤 Upload Multiple Resumes (.txt, .pdf, .jpg, .png, .jpeg, .docx)",
    type=["txt", "pdf", "jpg", "jpeg", "png", "docx"],
    accept_multiple_files=True,
    key="resume_uploader",
)


# ──────────────────────────────────────────────
# UI — Screen button (prevents auto-processing)
# ──────────────────────────────────────────────
if job_desc_raw and resume_files:
    screen_clicked = st.button(
        "🔍 Screen Resumes", type="primary", use_container_width=True
    )

    if screen_clicked:
        # Preprocess job description
        job_desc = preprocess_text(job_desc_raw)

        st.markdown("### 🔄 Processing Resumes...")
        progress_bar = st.progress(0)

        resumes = []
        resume_names = []
        extracted_texts = {}
        failed_files = []

        for idx, resume_file in enumerate(resume_files):
            try:
                # Limit size to 10MB to prevent potential memory issues
                if resume_file.size > 10 * 1024 * 1024:
                    failed_files.append(
                        (resume_file.name, "File size exceeds 10MB limit")
                    )
                    progress_bar.progress((idx + 1) / len(resume_files))
                    continue

                resume_text = cached_extract_text(
                    resume_file.getvalue(), resume_file.name
                )

                if not resume_text.strip():
                    failed_files.append(
                        (resume_file.name, "Empty or unreadable content")
                    )
                    progress_bar.progress((idx + 1) / len(resume_files))
                    continue

                extracted_texts[resume_file.name] = resume_text
                resumes.append(preprocess_text(resume_text))
                resume_names.append(resume_file.name)

            except Exception as e:
                failed_files.append((resume_file.name, str(e)))

            progress_bar.progress((idx + 1) / len(resume_files))

        # ── Show warnings for any files that failed ──
        if failed_files:
            for fname, reason in failed_files:
                st.warning(f"⚠️ Skipped **{fname}**: {reason}")

        # ── Build and display results ──
        if resumes:
            raw_scores = calculate_similarity(job_desc, resumes)
            normalized = normalize_scores(raw_scores)
            labels = [get_score_label(s) for s in normalized]
            raw_pct = [round(s * 100, 2) for s in raw_scores]

            ranked_data = sorted(
                zip(resume_names, normalized, raw_pct, labels),
                key=lambda x: x[1],
                reverse=True,
            )

            df = pd.DataFrame(
                ranked_data,
                columns=["Resume", "Score (%)", "Raw Similarity (%)", "Match Level"],
            )
            df.index = range(1, len(df) + 1)
            df.index.name = "Rank"

            st.markdown("### 📊 Ranked Resumes")
            st.dataframe(
                df.style.bar(
                    subset=["Score (%)"], color="#FF4B4B", vmin=0, vmax=100
                ),
                use_container_width=True,
            )

            # ── Expandable text preview ──
            st.markdown("### 🔎 Extracted Text Preview")
            for name, score, _, label in ranked_data:
                with st.expander(f"📄 {name} — {label}"):
                    st.text(extracted_texts.get(name, "Text not available"))
        else:
            st.error("❌ No resumes could be processed successfully.")

elif job_desc_raw and not resume_files:
    st.info("📤 Please upload resumes to begin screening.")
elif resume_files and not job_desc_raw:
    st.info("📜 Please provide a job description to begin screening.")