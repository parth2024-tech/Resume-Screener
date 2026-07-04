"""
skills_extractor.py — Keyword and skills extraction for resume-JD matching.

Extracts meaningful keywords (nouns, proper nouns, technical terms) from text
using spaCy, then computes matched/missing skills between a job description
and a resume. Also provides a combined score blending TF-IDF and keyword match.
"""

from typing import Any, List, Set, Tuple
from Scripts.text_processing import load_nlp_model


# ---------------------------------------------------------------------------
# Keyword / skills extraction
# ---------------------------------------------------------------------------

# A curated set of common tech/domain skills used as a fallback boost.
# spaCy noun extraction already catches most skills; this list supplements
# short acronyms (e.g. "SQL", "ML") that spaCy may miss.
_SKILL_KEYWORDS: Set[str] = {
    # Programming languages
    "python", "java", "javascript", "typescript", "c++", "c#", "ruby", "go",
    "rust", "swift", "kotlin", "scala", "r", "matlab", "php", "sql", "nosql",
    # ML / AI
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "neural network", "tensorflow", "pytorch", "keras",
    "scikit-learn", "xgboost", "lightgbm", "bert", "gpt", "llm", "rag",
    "reinforcement learning", "data science", "ai", "ml",
    # Data
    "pandas", "numpy", "spark", "hadoop", "kafka", "airflow", "dbt",
    "tableau", "power bi", "excel", "sql", "postgresql", "mysql",
    "mongodb", "redis", "elasticsearch", "snowflake", "bigquery",
    # Web / Backend
    "react", "angular", "vue", "node", "django", "flask", "fastapi",
    "spring", "rest", "graphql", "api", "microservices", "docker",
    "kubernetes", "aws", "azure", "gcp", "ci/cd", "devops", "git",
    # Soft skills / domain
    "communication", "leadership", "agile", "scrum", "project management",
    "problem solving", "teamwork", "analytical", "research",
}


def extract_keywords(text: str) -> Set[str]:
    """Extract meaningful keywords from text using spaCy + skill dictionary.

    Combines:
      1. spaCy noun / proper-noun / verb tokens after lemmatisation
      2. A curated _SKILL_KEYWORDS lookup for short technical terms

    Returns a set of lowercase keyword strings.
    """
    nlp: Any = load_nlp_model()
    doc = nlp(text.lower())

    # Collect spaCy-identified content tokens
    spacy_tokens: Set[str] = {
        token.lemma_
        for token in doc
        if token.pos_ in ("NOUN", "PROPN", "VERB")
        and not token.is_stop
        and not token.is_punct
        and len(token.text.strip()) > 1
    }

    # Boost with curated skill keywords present in raw text
    text_lower = text.lower()
    skill_hits: Set[str] = {
        skill for skill in _SKILL_KEYWORDS if skill in text_lower
    }

    return spacy_tokens | skill_hits


def get_matched_keywords(
    jd_keywords: Set[str], resume_keywords: Set[str]
) -> Set[str]:
    """Return keywords that appear in both the JD and the resume."""
    return jd_keywords & resume_keywords


def get_missing_keywords(
    jd_keywords: Set[str], resume_keywords: Set[str]
) -> Set[str]:
    """Return JD keywords that are absent from the resume."""
    return jd_keywords - resume_keywords


def keyword_match_ratio(jd_keywords: Set[str], resume_keywords: Set[str]) -> float:
    """Fraction of JD keywords that are matched in the resume (0.0 – 1.0).

    Returns 0.0 if the JD has no extractable keywords.
    """
    if not jd_keywords:
        return 0.0
    matched = get_matched_keywords(jd_keywords, resume_keywords)
    return len(matched) / len(jd_keywords)


def combined_score(tfidf_score: float, kw_ratio: float) -> float:
    """Weighted combination of TF-IDF cosine similarity and keyword match ratio.

    Formula:
        score = 0.60 * tfidf_score + 0.40 * kw_ratio

    Both inputs should be in [0, 1]. The result is returned as a percentage
    value (0 – 100), rounded to 2 decimal places.
    """
    raw = 0.60 * tfidf_score + 0.40 * kw_ratio
    return round(min(max(raw, 0.0), 1.0) * 100, 2)
