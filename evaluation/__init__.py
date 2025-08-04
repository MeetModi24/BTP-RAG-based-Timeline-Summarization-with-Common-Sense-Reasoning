# evaluation/__init__.py

"""
Evaluation module initializer.

Includes:
- Baseline 1: Standard Tilse ROUGE evaluation
- Future extensions: Baseline 2, 3, 4, TIMO

Exports:
- evaluate_timeline (from tilse_eval.py)
"""

from .tilse_eval import (
    evaluate_timeline,
)
