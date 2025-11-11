import re
from difflib import get_close_matches

# Simple keyword maps (can expand later with ontology or embeddings)
SUBJECTS = ["Math", "Science", "Literature", "Environment", "Sports", "Geography", "History"]
LANGUAGES = ["Arabic", "English"]

def infer_metadata_from_query(query: str):
    """Automatically infer filters from a natural-language query."""
    query_lower = query.lower()
    metadata = {}

    # Detect subject
    subject_match = get_close_matches(
        next((w for w in SUBJECTS if w.lower() in query_lower), None) or "", SUBJECTS, n=1, cutoff=0.6)
    if subject_match:
        metadata["subject"] = subject_match[0]

    # Detect language
    for lang in LANGUAGES:
        if lang.lower() in query_lower:
            metadata["language"] = lang
            break

    # Detect grade level (numbers like 'grade 10', '10th grade')
    grade_match = re.search(r'grade\s*(\d+)', query_lower)
    if grade_match:
        metadata["grade_level"] = int(grade_match.group(1))

    # Default source
    metadata["source"] = "books"

    return metadata
