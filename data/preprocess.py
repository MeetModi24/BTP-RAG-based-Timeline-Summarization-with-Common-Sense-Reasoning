# data/preprocess.py

"""
Document preprocessing and formatting utilities:
- group documents by date
- convert grouped results to model-ready input
- optional: reformat summaries or sentences
"""

import re
from collections import defaultdict


def list_to_grouped_dict(processed_results):
    """
    Convert processed result list (with Date/Content) to a dict:
    {
        "2021-01-01": ["content1", "content2"],
        ...
    }
    """
    grouped = defaultdict(list)
    for doc in processed_results:
        grouped[doc["Date"]].append(doc["Content"])
    return grouped


def trim_summary_sentences(summary_text, max_sentences=20):
    """
    Limit a long summary to the first N sentences.
    """
    sentences = re.split(r'\.\s+', summary_text.strip())
    trimmed = ". ".join(sentences[:max_sentences]).strip()
    if not trimmed.endswith("."):
        trimmed += "."
    return trimmed


def reformat_summaries_for_cleaning(summaries):
    """
    Takes LLaMA summaries {date: text} and returns a cleaned
    text block ready for `cleaned_summary.txt`.

    Format:
    Publication Date: 2021-01-01
    Content: actual summary...
    """
    cleaned_lines = []
    for date, summary_text in sorted(summaries.items()):
        if not summary_text:
            continue
        cleaned_lines.append(f"Publication Date: {date}\n")
        cleaned_lines.append(f"Content: {summary_text.strip()}\n\n")
    return "".join(cleaned_lines)
