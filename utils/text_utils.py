# utils/text_utils.py

"""
Text preprocessing utilities:
- clean_text: Normalize article content
- split_sentences: Break long paragraphs into sentences
- extract_dates: Extract date mentions from text
"""

import re
from dateutil.parser import parse
from typing import List, Tuple, Optional


def clean_text(text: str) -> str:
    """
    Cleans and normalizes raw text from articles.
    """
    text = re.sub(r"An ultra-low latency.*?platform", "", text)
    text = text.replace("-LRB-", "(").replace("-RRB-", ")")
    text = re.sub(r"\s+", " ", text)
    text = text.replace("`", "'").replace("\\", "")
    return text.strip()


# Sentence boundary detection (basic version, customizable later)
SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def split_sentences(text: str) -> List[str]:
    """
    Splits a block of text into individual sentences.
    """
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]


# Broad regex to match various human-written date strings
BROAD_DATE_RE = re.compile(
    r"\b("
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+"
    r"\d{1,2}(?:/\d{1,2})?"
    r"(?:,\s*\d{4})?"
    r")\b"
)


def extract_dates(sentence: str, fallback_year: Optional[int] = None) -> List[str]:
    """
    Extracts all date mentions from a sentence.
    Optionally fills in missing years using fallback_year.
    """
    matches = BROAD_DATE_RE.findall(sentence)
    dates = []

    for date_str in matches:
        try:
            # If year is not in the string and fallback is provided, add it
            if not re.search(r"\d{4}", date_str) and fallback_year:
                date_str += f", {fallback_year}"

            dt = parse(date_str, fuzzy=True)
            dates.append(dt.strftime("%Y-%m-%d"))
        except Exception:
            continue

    return dates
