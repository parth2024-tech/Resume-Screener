"""
app.py — AI-Powered Resume Screening & Ranking System (Streamlit UI)

All extraction and NLP logic is imported from Scripts/ to avoid duplication.
Uses Streamlit caching to prevent redundant model loads and re-processing.
"""

import io
import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from Scripts.text_processing import (
    load_nlp_model,
    preprocess_text,
    extract_text_from_pdf,
    extract_text_from_image,
    extract_text_from_docx,
)
from Scripts.resume_ranking import (
    calculate_similarity,
    get_score_label,
)
from Scripts.skills_extractor import (
    extract_keywords,
    get_matched_keywords,
    get_missing_keywords,
    keyword_match_ratio,
    combined_score,
)


# ──────────────────────────────────────────────
# Page config — must be first Streamlit call
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ──────────────────────────────────────────────
# Caching — model loaded ONCE across all reruns
# ──────────────────────────────────────────────
@st.cache_resource
def init_nlp():
    """Load and cache the spaCy NLP model (survives Streamlit reruns)."""
    return load_nlp_model()


@st.cache_data
def cached_extract_text(file_bytes: bytes, file_name: str) -> str:
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
def load_css(file_name: str) -> None:
    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css("styles.css")


# ──────────────────────────────────────────────
# UI — Hero Header
# ──────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-icon">🚀</div>
    <h1>AI Resume Screening System</h1>
    <p class="hero-subtitle">Upload resumes and a job description — get instant AI-powered ranking with skills gap analysis</p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# UI — Job Description input
# ──────────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("### 📜 Step 1 — Job Description")
jd_option = st.radio(
    "How do you want to provide the Job Description?",
    ("✍️ Paste Text", "📂 Upload File"),
    horizontal=True,
)

job_desc_raw = ""

if jd_option == "✍️ Paste Text":
    job_desc_raw = st.text_area(
        "Paste the Job Description here",
        height=180,
        placeholder="e.g. We are looking for a Python developer with experience in machine learning, SQL, and REST APIs...",
    )
elif jd_option == "📂 Upload File":
    job_desc_file = st.file_uploader(
        "Upload a Job Description file (.txt, .pdf, .docx, .jpg, .png)",
        type=["txt", "pdf", "docx", "jpg", "jpeg", "png"],
        key="jd_uploader",
    )
    if job_desc_file:
        try:
            job_desc_raw = cached_extract_text(
                job_desc_file.getvalue(), job_desc_file.name
            )
            st.success(f"✅ Job description loaded from **{job_desc_file.name}**")
        except Exception as e:
            st.warning(f"⚠️ Failed to extract job description: {e}")
st.markdown('</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────
# UI — Resume upload
# ──────────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("### 📂 Step 2 — Upload Resumes")
resume_files = st.file_uploader(
    "Upload one or more resumes (.txt, .pdf, .docx, .jpg, .png)",
    type=["txt", "pdf", "jpg", "jpeg", "png", "docx"],
    accept_multiple_files=True,
    key="resume_uploader",
)
if resume_files:
    st.info(f"📎 **{len(resume_files)}** resume(s) loaded and ready.")
st.markdown('</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────
# UI — Screen button
# ──────────────────────────────────────────────
if job_desc_raw and resume_files:
    st.markdown('<div class="cta-row">', unsafe_allow_html=True)
    screen_clicked = st.button(
        "🔍 Screen & Rank Resumes", type="primary", use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if screen_clicked:

        # ── Preprocess JD ──
        job_desc_processed = preprocess_text(job_desc_raw)
        jd_keywords = extract_keywords(job_desc_raw)

        st.markdown("---")
        st.markdown("### 🔄 Processing Resumes...")
        progress_bar = st.progress(0)

        resumes_processed: list = []
        resume_names: list = []
        extracted_texts: dict = {}
        resume_keywords_map: dict = {}
        failed_files: list = []

        for idx, resume_file in enumerate(resume_files):
            try:
                if resume_file.size > 10 * 1024 * 1024:
                    failed_files.append((resume_file.name, "File size exceeds 10MB limit"))
                    progress_bar.progress((idx + 1) / len(resume_files))
                    continue

                resume_text = cached_extract_text(resume_file.getvalue(), resume_file.name)

                if not resume_text.strip():
                    failed_files.append((resume_file.name, "Empty or unreadable content"))
                    progress_bar.progress((idx + 1) / len(resume_files))
                    continue

                extracted_texts[resume_file.name] = resume_text
                resumes_processed.append(preprocess_text(resume_text))
                resume_names.append(resume_file.name)
                resume_keywords_map[resume_file.name] = extract_keywords(resume_text)

            except Exception as e:
                failed_files.append((resume_file.name, str(e)))

            progress_bar.progress((idx + 1) / len(resume_files))

        if failed_files:
            for fname, reason in failed_files:
                st.warning(f"⚠️ Skipped **{fname}**: {reason}")

        if resumes_processed:
            # ── Scoring: combined TF-IDF + keyword match ──
            tfidf_scores = calculate_similarity(job_desc_processed, resumes_processed)

            results = []
            for name, tfidf_raw in zip(resume_names, tfidf_scores):
                res_kws = resume_keywords_map[name]
                kw_ratio = keyword_match_ratio(jd_keywords, res_kws)
                score = combined_score(float(tfidf_raw), kw_ratio)
                matched = get_matched_keywords(jd_keywords, res_kws)
                missing = get_missing_keywords(jd_keywords, res_kws)
                label = get_score_label(score)
                results.append({
                    "name": name,
                    "score": score,
                    "tfidf_pct": round(float(tfidf_raw) * 100, 2),
                    "kw_ratio": round(kw_ratio * 100, 2),
                    "matched": matched,
                    "missing": missing,
                    "label": label,
                })

            results.sort(key=lambda x: x["score"], reverse=True)

            # ── Summary stats ──
            avg_score = round(sum(r["score"] for r in results) / len(results), 1)
            top = results[0]

            st.markdown("---")
            st.markdown("### 📊 Screening Results")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{len(results)}</div>
                    <div class="metric-label">Resumes Processed</div>
                </div>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{top['score']}%</div>
                    <div class="metric-label">Top Score</div>
                </div>""", unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{avg_score}%</div>
                    <div class="metric-label">Average Score</div>
                </div>""", unsafe_allow_html=True)

            # ── Bar chart ──
            st.markdown("#### 🏆 Candidate Score Comparison")
            names = [r["name"].replace(".pdf","").replace(".docx","").replace(".txt","") for r in results]
            scores = [r["score"] for r in results]
            colors = []
            for s in scores:
                if s >= 70:
                    colors.append("#22c55e")
                elif s >= 45:
                    colors.append("#f59e0b")
                else:
                    colors.append("#ef4444")

            fig = go.Figure(go.Bar(
                x=names,
                y=scores,
                marker_color=colors,
                text=[f"{s}%" for s in scores],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Score: %{y}%<extra></extra>",
            ))
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0", family="Inter"),
                yaxis=dict(range=[0, 110], gridcolor="rgba(255,255,255,0.05)", title="Match Score (%)"),
                xaxis=dict(gridcolor="rgba(0,0,0,0)"),
                margin=dict(t=30, b=20),
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── Ranked table + download ──
            st.markdown("#### 📋 Ranked Candidates Table")
            df = pd.DataFrame([{
                "Rank": i + 1,
                "Resume": r["name"],
                "Final Score (%)": r["score"],
                "TF-IDF Similarity (%)": r["tfidf_pct"],
                "Keyword Match (%)": r["kw_ratio"],
                "Match Level": r["label"],
            } for i, r in enumerate(results)])
            df = df.set_index("Rank")

            st.dataframe(
                df.style.bar(subset=["Final Score (%)"], color="#FF4B4B", vmin=0, vmax=100),
                use_container_width=True,
            )

            csv = df.to_csv().encode("utf-8")
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="ranked_resumes.csv",
                mime="text/csv",
                use_container_width=True,
            )

            # ── Per-candidate skill gap cards ──
            st.markdown("---")
            st.markdown("### 🔎 Candidate Deep-Dive")

            for rank, r in enumerate(results, start=1):
                matched_sorted = sorted(r["matched"])[:20]
                missing_sorted = sorted(r["missing"])[:15]

                matched_tags = " ".join(
                    f'<span class="tag tag-match">✅ {k}</span>'
                    for k in matched_sorted
                ) or "<em style='color:#94a3b8'>No keyword matches detected</em>"

                missing_tags = " ".join(
                    f'<span class="tag tag-miss">❌ {k}</span>'
                    for k in missing_sorted
                ) or "<em style='color:#94a3b8'>No missing keywords — great coverage!</em>"

                score_bar_color = "#22c55e" if r["score"] >= 70 else ("#f59e0b" if r["score"] >= 45 else "#ef4444")

                with st.expander(f"#{rank}  {r['name']}  —  {r['label']}  ({r['score']}%)"):
                    st.markdown(f"""
                    <div class="candidate-card">
                        <div class="score-row">
                            <div class="score-pill" style="background:{score_bar_color}22; border:1px solid {score_bar_color}; color:{score_bar_color};">
                                🎯 Final Score: <strong>{r['score']}%</strong>
                            </div>
                            <div class="score-pill" style="background:#6366f122; border:1px solid #6366f1; color:#a5b4fc;">
                                📐 TF-IDF: <strong>{r['tfidf_pct']}%</strong>
                            </div>
                            <div class="score-pill" style="background:#0ea5e922; border:1px solid #0ea5e9; color:#7dd3fc;">
                                🔑 Keyword Match: <strong>{r['kw_ratio']}%</strong>
                            </div>
                        </div>

                        <h4>✅ Matched Skills & Keywords</h4>
                        <div class="tags-container">{matched_tags}</div>

                        <h4 style="margin-top:16px">❌ Missing from Resume (JD requires)</h4>
                        <div class="tags-container">{missing_tags}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("**📄 Extracted Resume Text**")
                    st.text_area(
                        label="",
                        value=extracted_texts.get(r["name"], "Text not available"),
                        height=200,
                        disabled=True,
                        key=f"text_{rank}",
                    )
        else:
            st.error("❌ No resumes could be processed successfully.")

elif job_desc_raw and not resume_files:
    st.info("📤 Please upload at least one resume to begin screening.")
elif resume_files and not job_desc_raw:
    st.info("📜 Please provide a job description to begin screening.")