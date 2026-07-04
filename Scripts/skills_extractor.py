"""
skills_extractor.py — Keyword and skills extraction for resume-JD matching.

Extracts meaningful keywords (nouns, proper nouns, technical terms) from text
using spaCy + a curated 300+ skills dictionary with aliases and categories.
Computes matched/missing skills between a job description and a resume,
and provides a multi-factor combined score.
"""

from typing import Any, Dict, List, Set, Tuple
from Scripts.text_processing import load_nlp_model


# ---------------------------------------------------------------------------
# Skill categories — used for grouped display in the UI
# ---------------------------------------------------------------------------
SKILL_CATEGORIES: Dict[str, Set[str]] = {
    "🐍 Programming Languages": {
        "python", "java", "javascript", "typescript", "c++", "c#", "c",
        "ruby", "go", "golang", "rust", "swift", "kotlin", "scala",
        "r", "matlab", "php", "perl", "bash", "shell", "powershell",
        "groovy", "dart", "elixir", "haskell", "lua", "fortran", "cobol",
        "assembly", "vba", "objective-c",
    },
    "🤖 AI / Machine Learning": {
        "machine learning", "deep learning", "nlp", "natural language processing",
        "computer vision", "neural network", "tensorflow", "pytorch", "keras",
        "scikit-learn", "xgboost", "lightgbm", "catboost", "bert", "gpt",
        "llm", "rag", "retrieval augmented generation", "transformers",
        "reinforcement learning", "supervised learning", "unsupervised learning",
        "random forest", "gradient boosting", "svm", "support vector machine",
        "regression", "classification", "clustering", "feature engineering",
        "model deployment", "mlops", "hugging face", "langchain", "openai",
        "stable diffusion", "generative ai", "transfer learning",
        "hyperparameter tuning", "cross validation",
    },
    "📊 Data & Analytics": {
        "data science", "data analysis", "data engineering", "data pipeline",
        "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",
        "tableau", "power bi", "looker", "metabase", "excel", "google sheets",
        "spark", "hadoop", "kafka", "airflow", "dbt", "flink",
        "etl", "data warehouse", "data lake", "snowflake", "bigquery",
        "redshift", "databricks", "dask", "statistics", "a/b testing",
        "hypothesis testing", "time series", "forecasting",
    },
    "🗄️ Databases": {
        "sql", "postgresql", "mysql", "sqlite", "oracle", "sql server",
        "nosql", "mongodb", "redis", "cassandra", "dynamodb", "couchdb",
        "elasticsearch", "neo4j", "firebase", "supabase", "planetscale",
        "cockroachdb", "influxdb",
    },
    "🌐 Web & Backend": {
        "react", "reactjs", "angular", "vue", "vuejs", "nextjs", "nuxtjs",
        "svelte", "html", "css", "sass", "tailwind", "bootstrap",
        "node", "nodejs", "express", "django", "flask", "fastapi",
        "spring", "spring boot", "rails", "laravel", "asp.net",
        "rest", "restful", "graphql", "grpc", "websocket", "api",
        "microservices", "serverless", "oauth", "jwt", "web scraping",
    },
    "☁️ Cloud & DevOps": {
        "aws", "azure", "gcp", "google cloud", "heroku", "vercel", "netlify",
        "docker", "kubernetes", "terraform", "ansible", "puppet", "chef",
        "jenkins", "github actions", "gitlab ci", "circleci", "travis ci",
        "ci/cd", "devops", "devsecops", "monitoring", "prometheus", "grafana",
        "nginx", "apache", "linux", "unix", "bash scripting",
    },
    "🔧 Tools & Practices": {
        "git", "github", "gitlab", "bitbucket", "jira", "confluence",
        "slack", "trello", "notion", "figma", "postman", "swagger",
        "agile", "scrum", "kanban", "tdd", "bdd", "unit testing",
        "integration testing", "code review", "pair programming",
        "design patterns", "solid principles", "oop", "functional programming",
        "data structures", "algorithms", "system design", "microservices",
    },
    "🧠 Soft Skills": {
        "communication", "leadership", "teamwork", "collaboration",
        "problem solving", "critical thinking", "time management",
        "project management", "stakeholder management", "mentoring",
        "analytical", "research", "presentation", "documentation",
        "adaptability", "creativity", "initiative",
    },
}

# Flat set of all curated skills for fast lookup
_ALL_SKILLS: Set[str] = {
    skill for category_skills in SKILL_CATEGORIES.values()
    for skill in category_skills
}

# Alias map — normalises common alternate forms to canonical name
_ALIASES: Dict[str, str] = {
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "golang": "go",
    "node.js": "nodejs",
    "react.js": "reactjs",
    "vue.js": "vuejs",
    "next.js": "nextjs",
    "ml": "machine learning",
    "ai": "machine learning",
    "dl": "deep learning",
    "cv": "computer vision",
    "nlp": "natural language processing",
    "sk-learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "tf": "tensorflow",
    "k8s": "kubernetes",
    "k3s": "kubernetes",
    "pg": "postgresql",
    "postgres": "postgresql",
    "mssql": "sql server",
    "gcp": "google cloud",
    "aws lambda": "serverless",
    "azure devops": "ci/cd",
    "hf": "hugging face",
}


def _normalise(text: str) -> str:
    """Apply alias normalisation to a lowercase skill string."""
    return _ALIASES.get(text.strip(), text.strip())


def extract_keywords(text: str) -> Set[str]:
    """Extract meaningful keywords from text using spaCy + skill dictionary.

    Strategy:
      1. Run spaCy to collect lemmatised NOUN / PROPN tokens
      2. Scan for all curated skills (multi-word phrases included)
      3. Normalise aliases
    """
    nlp: Any = load_nlp_model()
    doc = nlp(text.lower())

    # spaCy noun/verb tokens
    spacy_tokens: Set[str] = {
        _normalise(token.lemma_)
        for token in doc
        if token.pos_ in ("NOUN", "PROPN", "VERB")
        and not token.is_stop
        and not token.is_punct
        and len(token.text.strip()) > 1
    }

    text_lower = text.lower()

    # Multi-word + single-word skill lookups
    skill_hits: Set[str] = {
        _normalise(skill) for skill in _ALL_SKILLS if skill in text_lower
    }

    # Alias scanning directly on the text
    alias_hits: Set[str] = {
        canonical
        for alias, canonical in _ALIASES.items()
        if re.search(rf"\b{re.escape(alias)}\b", text_lower)
    }

    return spacy_tokens | skill_hits | alias_hits


def get_matched_keywords(jd_kw: Set[str], resume_kw: Set[str]) -> Set[str]:
    """Return keywords that appear in both the JD and the resume."""
    return jd_kw & resume_kw


def get_missing_keywords(jd_kw: Set[str], resume_kw: Set[str]) -> Set[str]:
    """Return JD keywords absent from the resume — capped at skill vocabulary."""
    # Only flag missing skills that are in our curated dictionary to avoid noise
    jd_skills_only = {k for k in jd_kw if k in _ALL_SKILLS}
    return jd_skills_only - resume_kw


def keyword_match_ratio(jd_kw: Set[str], resume_kw: Set[str]) -> float:
    """Fraction of JD *skill* keywords matched in the resume (0.0–1.0)."""
    jd_skills = {k for k in jd_kw if k in _ALL_SKILLS}
    if not jd_skills:
        return 0.0
    return len(jd_skills & resume_kw) / len(jd_skills)


def get_skills_by_category(matched: Set[str]) -> Dict[str, List[str]]:
    """Group a set of matched skill keywords into their named categories."""
    result: Dict[str, List[str]] = {}
    for cat, skills in SKILL_CATEGORIES.items():
        hits = sorted(s for s in skills if s in matched)
        if hits:
            result[cat] = hits
    return result


def combined_score(
    tfidf_score: float,
    kw_ratio: float,
    experience_years: int = 0,
    required_years: int = 0,
) -> float:
    """Multi-factor weighted score combining TF-IDF, keyword match, and experience.

    Weights:
        TF-IDF cosine similarity  → 50%
        Skill keyword match ratio → 40%
        Experience adequacy       → 10%

    All inputs normalised to [0, 1]. Output is a percentage (0–100).
    """
    # Experience adequacy (0–1): how well candidate meets required years
    if required_years > 0:
        exp_score = min(experience_years / required_years, 1.0)
    else:
        # No requirement stated — treat any experience positively
        exp_score = min(experience_years / 5, 1.0)

    raw = 0.50 * tfidf_score + 0.40 * kw_ratio + 0.10 * exp_score
    return round(min(max(raw, 0.0), 1.0) * 100, 2)


# Avoid circular import — import re only inside this module
import re
