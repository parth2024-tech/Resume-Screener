"""
resume_parser.py — Structured information extractor for resumes.

Extracts candidate contact details, education level, estimated years of
experience, and detected section headings from raw resume text using
regex patterns and keyword heuristics.

No external ML model is required — all logic is rule-based for speed
and determinism.
"""

import re
from typing import Any, Dict, List, Optional
from datetime import datetime


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
)

_PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[\s\-.]?)?"           # optional country code
    r"(?:\(?\d{2,4}\)?[\s\-.]?)"         # area code
    r"\d{3,4}[\s\-.]?\d{3,5}"            # main number
)

_LINKEDIN_RE = re.compile(
    r"linkedin\.com/in/[\w\-]+"
)

_GITHUB_RE = re.compile(
    r"github\.com/[\w\-]+"
)

# Degree keywords mapped to a numeric level (higher = more qualified)
_DEGREE_MAP: Dict[str, int] = {
    "phd": 5, "ph.d": 5, "doctorate": 5, "doctoral": 5,
    "master": 4, "m.tech": 4, "m.sc": 4, "msc": 4, "mba": 4, "m.e": 4,
    "m.s": 4, "ms ": 4, "me ": 4,
    "bachelor": 3, "b.tech": 3, "b.sc": 3, "bsc": 3, "b.e": 3,
    "b.s": 3, "be ": 3, "undergraduate": 3, "ug": 3,
    "diploma": 2, "associate": 2,
    "high school": 1, "secondary": 1, "12th": 1, "hsc": 1,
}

_DEGREE_LABELS: Dict[int, str] = {
    5: "PhD / Doctorate",
    4: "Master's Degree",
    3: "Bachelor's Degree",
    2: "Diploma / Associate",
    1: "High School",
    0: "Not Detected",
}

# Common resume section headings
_SECTION_KEYWORDS: List[str] = [
    "education", "experience", "work experience", "professional experience",
    "employment", "skills", "technical skills", "projects", "certifications",
    "achievements", "awards", "publications", "languages", "interests",
    "summary", "objective", "profile", "about", "training",
    "internship", "volunteer", "references",
]

# Patterns to detect year ranges for experience calculation
# Matches: "Jan 2019 – Mar 2022", "2018 - 2021", "2020 – Present"
_YEAR_RANGE_RE = re.compile(
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)?\s*"
    r"(20\d{2}|19\d{2})"               # start year
    r"\s*[-–—to]+\s*"                   # separator
    r"(20\d{2}|19\d{2}|present|current|now)",  # end year or present
    re.IGNORECASE,
)

_CURRENT_YEAR = datetime.now().year


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_contact_info(text: str) -> Dict[str, Optional[str]]:
    """Extract name, email, phone, LinkedIn and GitHub from resume text.

    Name heuristic: the first non-empty, non-URL line that contains only
    alphabetic characters and spaces (max 5 words) is treated as the name.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # Email
    email_match = _EMAIL_RE.search(text)
    email = email_match.group(0) if email_match else None

    # Phone
    phone_match = _PHONE_RE.search(text)
    phone = phone_match.group(0).strip() if phone_match else None

    # LinkedIn / GitHub
    li_match = _LINKEDIN_RE.search(text)
    gh_match = _GITHUB_RE.search(text)
    linkedin = li_match.group(0) if li_match else None
    github = gh_match.group(0) if gh_match else None

    # Name — first short alphabetic line, skip obvious non-names
    name: Optional[str] = None
    skip_tokens = {"resume", "cv", "curriculum", "vitae", "profile", "summary"}
    for line in lines[:8]:
        words = line.split()
        if (
            1 < len(words) <= 5
            and all(re.match(r"^[A-Za-z.\-']+$", w) for w in words)
            and not any(w.lower() in skip_tokens for w in words)
            and "@" not in line
        ):
            name = line.title()
            break

    return {
        "name": name,
        "email": email,
        "phone": phone,
        "linkedin": linkedin,
        "github": github,
    }


def extract_education(text: str) -> str:
    """Detect the highest education level mentioned in resume text."""
    text_lower = text.lower()
    best_level = 0
    for keyword, level in _DEGREE_MAP.items():
        if keyword in text_lower and level > best_level:
            best_level = level
    return _DEGREE_LABELS.get(best_level, "Not Detected")


def extract_experience_years(text: str) -> int:
    """Estimate total years of experience by scanning date ranges.

    Scans all year-range patterns (e.g. "2018 – 2021", "Jan 2020 – Present")
    and sums up the non-overlapping durations. Returns 0 if no ranges found.
    """
    text_lower = text.lower()
    total_months = 0

    for m in _YEAR_RANGE_RE.finditer(text_lower):
        start_year = int(m.group(2))
        end_str = m.group(3).lower()
        end_year = (
            _CURRENT_YEAR
            if end_str in ("present", "current", "now")
            else int(end_str)
        )
        if end_year >= start_year:
            total_months += (end_year - start_year) * 12

    # Also catch explicit "X years of experience" statements
    explicit_re = re.search(
        r"(\d+)\+?\s+years?\s+of\s+(?:work\s+)?experience", text_lower
    )
    if explicit_re:
        explicit_years = int(explicit_re.group(1))
        # Use whichever is larger
        return max(explicit_years, total_months // 12)

    return total_months // 12


def extract_sections(text: str) -> List[str]:
    """Return a list of section headings found in the resume."""
    text_lower = text.lower()
    found = []
    for section in _SECTION_KEYWORDS:
        # Match as a standalone line-level heading (not mid-sentence)
        pattern = rf"(?:^|\n)\s*{re.escape(section)}\s*(?:\n|:)"
        if re.search(pattern, text_lower):
            found.append(section.title())
    return found


def extract_skill_categories(matched_skills: set, skill_categories: dict) -> Dict[str, List[str]]:
    """Group matched skills into their categories for display."""
    result: Dict[str, List[str]] = {}
    for category, skills in skill_categories.items():
        hits = [s for s in skills if s in matched_skills]
        if hits:
            result[category] = sorted(hits)
    return result


def parse_resume(text: str) -> Dict[str, Any]:
    """Return a full structured parse of a resume.

    Keys:
        contact  — dict with name, email, phone, linkedin, github
        education — str, highest degree detected
        experience_years — int, estimated total years of experience
        sections — list of detected section headings
    """
    return {
        "contact": extract_contact_info(text),
        "education": extract_education(text),
        "experience_years": extract_experience_years(text),
        "sections": extract_sections(text),
    }
