"""
app.py — AI-Powered Resume Screening & Ranking System (Streamlit UI)

All extraction and NLP logic is imported from Scripts/ to avoid duplication.
Uses Streamlit caching to prevent redundant model loads and re-processing.
"""

import io
import os
import re
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
    get_skills_by_category,
    combined_score,
    SKILL_CATEGORIES,
)
from Scripts.resume_parser import parse_resume, extract_experience_years


# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ──────────────────────────────────────────────
# Caching
# ──────────────────────────────────────────────
@st.cache_resource
def init_nlp():
    return load_nlp_model()


@st.cache_data
def cached_extract_text(file_bytes: bytes, file_name: str) -> str:
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


def _detect_required_years(jd_text: str) -> int:
    """Try to detect how many years of experience the JD requires."""
    m = re.search(r"(\d+)\+?\s*(?:to\s*\d+\+?)?\s*years?\s+(?:of\s+)?(?:work\s+)?experience", jd_text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return 0


init_nlp()


# ──────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────
def load_css(file_name: str) -> None:
    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("styles.css")


# ──────────────────────────────────────────────
# Hero header
# ──────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-icon">🚀</div>
    <h1>AI Resume Screening System</h1>
    <p class="hero-subtitle">
        Upload resumes &amp; a job description — get instant AI-powered ranking,
        structured candidate profiles, and skill-gap analysis
    </p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Step 1 — Job Description
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
        placeholder="e.g. We are looking for a Python developer with 3+ years of experience in machine learning, SQL, and REST APIs...",
    )
elif jd_option == "📂 Upload File":
    jd_file = st.file_uploader(
        "Upload a Job Description (.txt, .pdf, .docx, .jpg, .png)",
        type=["txt", "pdf", "docx", "jpg", "jpeg", "png"],
        key="jd_uploader",
    )
    if jd_file:
        try:
            job_desc_raw = cached_extract_text(jd_file.getvalue(), jd_file.name)
            st.success(f"✅ Job description loaded from **{jd_file.name}**")
        except Exception as e:
            st.warning(f"⚠️ Failed to extract JD: {e}")
st.markdown('</div>', unsafe_allow_html=True)

# Show JD keyword preview
if job_desc_raw.strip():
    jd_kw_preview = extract_keywords(job_desc_raw)
    jd_skills_preview = {k for cat in SKILL_CATEGORIES.values() for k in cat if k in jd_kw_preview}
    if jd_skills_preview:
        tags_html = " ".join(
            f'<span class="tag tag-jd">{k}</span>'
            for k in sorted(jd_skills_preview)[:30]
        )
        st.markdown(
            f'<div class="section-card" style="padding:1rem">'
            f'<p style="color:#94a3b8;font-size:0.8rem;margin-bottom:0.5rem;">📌 Skills detected in JD:</p>'
            f'<div class="tags-container">{tags_html}</div></div>',
            unsafe_allow_html=True
        )


# ──────────────────────────────────────────────
# Step 2 — Resume Upload
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
    st.info(f"📎 **{len(resume_files)}** resume(s) ready for screening.")
st.markdown('</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Screen button
# ──────────────────────────────────────────────
if job_desc_raw.strip() and resume_files:
    screen_clicked = st.button(
        "🔍 Screen & Rank Resumes", type="primary", use_container_width=True
    )

    if screen_clicked:
        job_desc_processed = preprocess_text(job_desc_raw)
        jd_keywords = extract_keywords(job_desc_raw)
        required_years = _detect_required_years(job_desc_raw)

        st.markdown("---")
        progress = st.progress(0, text="Extracting resume text…")

        results = []
        failed = []

        for idx, rf in enumerate(resume_files):
            progress.progress((idx + 1) / len(resume_files), text=f"Processing {rf.name}…")
            try:
                if rf.size > 10 * 1024 * 1024:
                    failed.append((rf.name, "File exceeds 10 MB limit"))
                    continue

                raw_text = cached_extract_text(rf.getvalue(), rf.name)
                if not raw_text.strip():
                    failed.append((rf.name, "Empty / unreadable content"))
                    continue

                # Structured parse
                parsed = parse_resume(raw_text)
                exp_years = parsed["experience_years"]

                # NLP scoring
                processed = preprocess_text(raw_text)
                resume_kw = extract_keywords(raw_text)
                kw_ratio = keyword_match_ratio(jd_keywords, resume_kw)
                tfidf_score = None  # filled below after batch vectorisation

                results.append({
                    "name": rf.name,
                    "raw_text": raw_text,
                    "processed": processed,
                    "parsed": parsed,
                    "exp_years": exp_years,
                    "resume_kw": resume_kw,
                    "kw_ratio": kw_ratio,
                })
            except Exception as e:
                failed.append((rf.name, str(e)))

        if failed:
            for fname, reason in failed:
                st.warning(f"⚠️ Skipped **{fname}**: {reason}")

        if results:
            # Batch TF-IDF
            tfidf_scores = calculate_similarity(
                job_desc_processed,
                [r["processed"] for r in results]
            )

            for r, tfidf_raw in zip(results, tfidf_scores):
                r["tfidf_pct"] = round(float(tfidf_raw) * 100, 2)
                r["score"] = combined_score(
                    float(tfidf_raw),
                    r["kw_ratio"],
                    r["exp_years"],
                    required_years,
                )
                r["matched"] = get_matched_keywords(jd_keywords, r["resume_kw"])
                r["missing"] = get_missing_keywords(jd_keywords, r["resume_kw"])
                r["matched_by_cat"] = get_skills_by_category(r["matched"])
                r["label"] = get_score_label(r["score"])

            results.sort(key=lambda x: x["score"], reverse=True)
            progress.empty()

            # ── Summary metrics ──
            avg = round(sum(r["score"] for r in results) / len(results), 1)
            top = results[0]

            st.markdown("### 📊 Screening Results")
            c1, c2, c3, c4 = st.columns(4)
            metrics = [
                (c1, str(len(results)), "Resumes Ranked"),
                (c2, f"{top['score']}%", "Top Match Score"),
                (c3, f"{avg}%", "Average Score"),
                (c4, str(required_years) + " yrs" if required_years else "N/A", "Experience Required"),
            ]
            for col, val, label in metrics:
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{val}</div>
                        <div class="metric-label">{label}</div>
                    </div>""", unsafe_allow_html=True)

            # ── Bar chart ──
            st.markdown("#### 🏆 Candidate Score Comparison")
            disp_names = [
                r["parsed"]["contact"]["name"] or r["name"].rsplit(".", 1)[0]
                for r in results
            ]
            scores = [r["score"] for r in results]
            bar_colors = [
                "#22c55e" if s >= 70 else "#f59e0b" if s >= 45 else "#ef4444"
                for s in scores
            ]
            fig = go.Figure(go.Bar(
                x=disp_names, y=scores,
                marker_color=bar_colors,
                text=[f"{s}%" for s in scores],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Score: %{y}%<extra></extra>",
            ))
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0", family="Inter"),
                yaxis=dict(range=[0, 115], gridcolor="rgba(255,255,255,0.05)", title="Match Score (%)"),
                xaxis=dict(gridcolor="rgba(0,0,0,0)"),
                margin=dict(t=30, b=10), height=340,
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── Ranked table ──
            st.markdown("#### 📋 Ranked Candidates")
            df = pd.DataFrame([{
                "Rank": i + 1,
                "Candidate": r["parsed"]["contact"]["name"] or r["name"],
                "File": r["name"],
                "Score (%)": r["score"],
                "TF-IDF (%)": r["tfidf_pct"],
                "Keyword Match (%)": round(r["kw_ratio"] * 100, 1),
                "Experience (yrs)": r["exp_years"],
                "Education": r["parsed"]["education"],
                "Match Level": r["label"],
            } for i, r in enumerate(results)]).set_index("Rank")

            st.dataframe(
                df.style.bar(subset=["Score (%)"], color="#FF4B4B", vmin=0, vmax=100),
                use_container_width=True,
            )
            csv = df.to_csv().encode("utf-8")
            st.download_button(
                "📥 Download Results as CSV", csv,
                "ranked_resumes.csv", "text/csv",
                use_container_width=True,
            )

            # ── Deep-dive candidate profiles ──
            st.markdown("---")
            st.markdown("### 🔎 Candidate Profiles")

            for rank, r in enumerate(results, 1):
                contact = r["parsed"]["contact"]
                edu = r["parsed"]["education"]
                exp = r["exp_years"]
                sections = r["parsed"]["sections"]
                label = r["label"]
                score = r["score"]

                score_color = "#22c55e" if score >= 70 else "#f59e0b" if score >= 45 else "#ef4444"
                display_name = contact["name"] or r["name"].rsplit(".", 1)[0]

                with st.expander(f"#{rank}  {display_name}  —  {label}  ({score}%)"):

                    # ── Profile header ──
                    col_a, col_b = st.columns([1, 2])
                    with col_a:
                        # Avatar initial
                        initials = "".join(w[0].upper() for w in display_name.split()[:2]) or "?"
                        st.markdown(f"""
                        <div class="avatar-circle" style="background:linear-gradient(135deg,{score_color}44,{score_color}22);border:2px solid {score_color};">
                            <span class="avatar-initials" style="color:{score_color};">{initials}</span>
                        </div>""", unsafe_allow_html=True)

                    with col_b:
                        st.markdown(f"<h3 style='margin:0;color:#f1f5f9'>{display_name}</h3>", unsafe_allow_html=True)
                        contact_items = []
                        if contact["email"]:
                            contact_items.append(f"📧 {contact['email']}")
                        if contact["phone"]:
                            contact_items.append(f"📱 {contact['phone']}")
                        if contact["linkedin"]:
                            contact_items.append(f"🔗 {contact['linkedin']}")
                        if contact["github"]:
                            contact_items.append(f"🐱 {contact['github']}")
                        if contact_items:
                            st.markdown(
                                " &nbsp;|&nbsp; ".join(
                                    f'<span style="color:#94a3b8;font-size:0.82rem">{i}</span>'
                                    for i in contact_items
                                ),
                                unsafe_allow_html=True
                            )

                    # ── Info badges ──
                    st.markdown("<div style='margin:0.75rem 0'>", unsafe_allow_html=True)
                    badges = [
                        ("🎓", edu, "#6366f1"),
                        ("💼", f"{exp} yr{'s' if exp != 1 else ''} experience" if exp else "Experience N/A", "#0ea5e9"),
                        ("📄", f"{len(sections)} sections detected", "#a855f7"),
                    ]
                    badges_html = " ".join(
                        f'<span class="badge" style="background:{c}22;border:1px solid {c}55;color:{c};">'
                        f'{icon} {txt}</span>'
                        for icon, txt, c in badges
                    )
                    st.markdown(f'<div class="score-row">{badges_html}</div>', unsafe_allow_html=True)

                    # ── Score breakdown ──
                    st.markdown(f"""
                    <div class="score-row" style="margin-top:0.5rem">
                        <div class="score-pill" style="background:{score_color}22;border:1px solid {score_color};color:{score_color};">
                            🎯 Final Score: <strong>{score}%</strong>
                        </div>
                        <div class="score-pill" style="background:#6366f122;border:1px solid #6366f1;color:#a5b4fc;">
                            📐 TF-IDF: <strong>{r['tfidf_pct']}%</strong>
                        </div>
                        <div class="score-pill" style="background:#0ea5e922;border:1px solid #0ea5e9;color:#7dd3fc;">
                            🔑 Keyword Match: <strong>{round(r['kw_ratio']*100, 1)}%</strong>
                        </div>
                    </div>""", unsafe_allow_html=True)

                    st.markdown("---")

                    # ── Matched skills by category ──
                    st.markdown("**✅ Matched Skills by Category**")
                    if r["matched_by_cat"]:
                        for cat, skills_list in r["matched_by_cat"].items():
                            tags = " ".join(
                                f'<span class="tag tag-match">{s}</span>'
                                for s in skills_list
                            )
                            st.markdown(
                                f'<p style="color:#94a3b8;font-size:0.75rem;margin:0.5rem 0 0.2rem">{cat}</p>'
                                f'<div class="tags-container">{tags}</div>',
                                unsafe_allow_html=True
                            )
                    else:
                        st.markdown('<em style="color:#64748b">No curated skills matched</em>', unsafe_allow_html=True)

                    # ── Missing skills ──
                    missing_sorted = sorted(r["missing"])[:20]
                    if missing_sorted:
                        st.markdown("**❌ Missing Skills (JD requires)**")
                        miss_tags = " ".join(
                            f'<span class="tag tag-miss">{k}</span>'
                            for k in missing_sorted
                        )
                        st.markdown(f'<div class="tags-container">{miss_tags}</div>', unsafe_allow_html=True)

                    # ── Sections detected ──
                    if sections:
                        st.markdown("**📑 Sections Detected in Resume**")
                        sec_tags = " ".join(
                            f'<span class="tag tag-section">{s}</span>'
                            for s in sections
                        )
                        st.markdown(f'<div class="tags-container">{sec_tags}</div>', unsafe_allow_html=True)

                    # ── Raw text preview ──
                    st.markdown("**📄 Extracted Resume Text**")
                    st.text_area(
                        label="",
                        value=r["raw_text"],
                        height=220,
                        disabled=True,
                        key=f"raw_{rank}",
                    )
        else:
            st.error("❌ No resumes could be processed successfully.")

elif job_desc_raw.strip() and not resume_files:
    st.info("📤 Please upload at least one resume to begin screening.")
elif resume_files and not job_desc_raw.strip():
    st.info("📜 Please provide a job description to begin screening.")