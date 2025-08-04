# data/__init__.py
"""
Data module initializer for loading and preprocessing articles.

This module includes the following core functionalities:
- load_articles: Load preprocessed articles from a JSONL file
- clean_text: Remove unwanted tokens and normalize content
- process_retrieved_docs: Aggregate and clean retrieved documents by date
- write_processed_results_to_file: Save structured summaries to disk
"""

from .loader import (
    load_articles,
    clean_text,
    process_retrieved_docs,
    write_processed_results_to_file,
)
