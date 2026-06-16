"""
data_loader.py — Simple utility to load text files from the data/ directory.

Run from project root:  python -m Scripts.data_loader
"""

import os


def load_text_file(filepath):
    """Load a text file and return its content as a string.
    
    Includes exception handling for file read or decoding issues.
    """
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                return file.read()
        except Exception as e:
            print(f"❌ Error reading file at {filepath}: {e}")
    return ""


if __name__ == "__main__":
    data_dir = "data"

    # Load job description
    jd_path = os.path.join(data_dir, "job_description.txt")
    job_desc = load_text_file(jd_path)
    if job_desc:
        print("Job Description:\n", job_desc)
    else:
        print(f"⚠️  Job description not found at {jd_path}")

    # Auto-detect and load all resume files in data/
    resume_files = [
        f for f in os.listdir(data_dir)
        if f.endswith(".txt") and "job_description" not in f
    ]

    if not resume_files:
        print("⚠️  No resume .txt files found in data/")
    else:
        for fname in resume_files:
            resume = load_text_file(os.path.join(data_dir, fname))
            print(f"\n{'='*50}")
            print(f"Resume ({fname}):\n", resume)
