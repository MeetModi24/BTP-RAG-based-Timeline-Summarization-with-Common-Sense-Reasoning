# evaluation/__init__.py

"""
Evaluation module initializer.

Includes:
- Baseline 1: Standard Tilse ROUGE evaluation
- Baseline 2: Regex-based date extraction and ROUGE
- Baseline 3: spaCy-based date mapping and ROUGE
- Baseline 4: TimeLLaMA graph ordering baseline
- Baseline 5: TIMO Syria-specific sorting baseline

Exports:
- evaluate_timeline          (Baseline 1)
- run_baseline2              (Baseline 2)
- run_baseline3              (Baseline 3)
- run_baseline4              (Baseline 4)
- run_timo_syria             (Baseline 5)
"""

from .baseline1_tilse import evaluate_timeline
from .baseline2_date_extraction import run_baseline2
from .baseline3_llama_summary import run_baseline3
from .baseline4_timellama import run_baseline4
from models.timo_model import run_timo_syria
