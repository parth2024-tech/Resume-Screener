"""
resume_ranking.py — Similarity matching and score normalization.

Provides improved TF-IDF (bigrams + sublinear TF), min-max score normalization,
and human-readable match labels.

Mathematical Background:
1. TF-IDF Representation:
   Term frequency is scaled sublinearly: TF_scaled = 1 + log(TF) if TF > 0 else 0.
   This prevents high term repetition from dominating the feature weights.
   Bigrams are included to capture key phrase patterns (e.g., "data science").
2. Cosine Similarity:
   Similarity(A, B) = (A . B) / (||A|| * ||B||)
   This computes the cosine of the angle between the document TF-IDF vectors.
"""

import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from Scripts.text_processing import preprocess_text, load_nlp_model
except ImportError:
    from text_processing import preprocess_text, load_nlp_model


# ---------------------------------------------------------------------------
# Similarity calculation
# ---------------------------------------------------------------------------
def calculate_similarity(job_desc, resumes):
    """Calculate cosine similarity between a job description and a list of resumes.

    Uses bigram TF-IDF with sublinear term-frequency scaling for better
    phrase-level matching compared to plain unigram TF-IDF.
    """
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    documents = [job_desc] + resumes
    tfidf_matrix = vectorizer.fit_transform(documents)
    similarity_scores = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])[0]
    return similarity_scores


# ---------------------------------------------------------------------------
# Score normalization
# ---------------------------------------------------------------------------
def normalize_scores(scores):
    """Convert raw cosine similarity scores directly to percentages (0-100).
    
    This replaces the batch-dependent min-max normalization to ensure that 
    a resume's score is absolute and does not change based on other resumes 
    uploaded in the same batch.
    """
    scores = np.array(scores, dtype=float)
    if len(scores) == 0:
        return scores
    
    # Cosine similarity ranges from 0 to 1. We multiply by 100 to get a percentage.
    normalized = np.clip(scores * 100, 0, 100)
    return np.round(normalized, 2)


def get_score_label(score):
    """Return a human-readable match label for a normalized score."""
    if score >= 80:
        return "🟢 Excellent Match"
    elif score >= 60:
        return "🔵 Good Match"
    elif score >= 40:
        return "🟡 Average Match"
    elif score >= 20:
        return "🟠 Weak Match"
    else:
        return "🔴 Poor Match"


# ---------------------------------------------------------------------------
# File helper (used by CLI scripts only)
# ---------------------------------------------------------------------------
def load_text_file(filepath):
    """Load a text file and return its content as a string."""
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as file:
            return file.read()
    return ""


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    job_desc = preprocess_text(load_text_file("data/job_description.txt"))

    resume_files = [
        f for f in os.listdir("data")
        if f.endswith(".txt") and "job_description" not in f
    ]
    resumes = [
        preprocess_text(load_text_file(f"data/{fname}"))
        for fname in resume_files
    ]

    raw_scores = calculate_similarity(job_desc, resumes)
    normalized = normalize_scores(raw_scores)
    labels = [get_score_label(s) for s in normalized]

    ranked = sorted(
        zip(resume_files, normalized, labels),
        key=lambda x: x[1],
        reverse=True,
    )

    df = pd.DataFrame(ranked, columns=["Resume", "Score (%)", "Match Level"])
    print(df.to_string(index=False))