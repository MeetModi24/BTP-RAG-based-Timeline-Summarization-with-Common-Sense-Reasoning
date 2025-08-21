# models/timo_model.py

"""
TIMO Model Integration (Baseline 5)

This module:
- Loads the TIMO model (text-generation pipeline).
- Extracts Syria-related sentences from baseline_3_dated_sentences.txt.
- Builds a prompt asking TIMO to order them chronologically.
- Runs multiple trials and saves outputs.
"""

import random
import re
import torch
from transformers import pipeline

# -------------------------------
# Load TIMO model (7B HF version)
# -------------------------------
timo = pipeline("text-generation", model="Warrieryes/timo-7b-hf")

# Prompt template
TIMO_TEMPLATE = """Below is an instruction that describes a task. 
Write a response that appropriately completes the request.

### Instruction:
{query}

### Response:"""


# ----------------------------------------
# Utility: Parse sentences containing keyword
# ----------------------------------------
def parse_sentences_with_filter(txt_file, keyword="Syria"):
    """
    Parses baseline_3_dated_sentences.txt and returns a list of
    (date_str, sentence_text) that contain the keyword.
    """
    sentences = []
    pattern = re.compile(r"\d+\. \[(\d{4}-\d{2}-\d{2})\] (.*)")
    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            m = pattern.match(line.strip())
            if m:
                date_str, sent = m.groups()
                if keyword.lower() in sent.lower():
                    sentences.append((date_str, sent))
    return sentences


# ----------------------------------------
# Utility: Build TIMO input prompt
# ----------------------------------------
def build_prompt(sentences):
    """
    Builds a natural language prompt for TIMO given sentences.
    """
    prompt = "Sort the following Syria-related events in the correct chronological order:\n"
    for i, (date, sent) in enumerate(sentences):
        prompt += f"{i+1}. [{date}] {sent}\n"
    return prompt


# ----------------------------------------
# Run TIMO baseline for Syria-related events
# ----------------------------------------
def run_timo_syria(input_file, keyword="Syria", num_runs=5):
    """
    Runs TIMO to sort Syria-related sentences in multiple trials.
    Saves output to files timo_syria_run{N}.txt
    """
    all_syria_sentences = parse_sentences_with_filter(input_file, keyword)
    print(f"Total Syria-related sentences found: {len(all_syria_sentences)}")

    if len(all_syria_sentences) < 10:
        print("Not enough Syria-related sentences to perform multiple runs.")
        return

    for run_id in range(1, num_runs + 1):
        selected = random.sample(
            all_syria_sentences, 
            min(random.randint(10, 12), len(all_syria_sentences))
        )
        prompt = build_prompt(selected)
        input_text = TIMO_TEMPLATE.format(query=prompt)

        print(f"\n==================== TIMO Run {run_id} ====================")
        output = timo(input_text, max_new_tokens=512)[0]['generated_text']

        output_file = f"timo_syria_run{run_id}.txt"
        with open(output_file, 'w', encoding='utf-8') as out_f:
            out_f.write("Syria-Related Input Sentences:\n")
            for date, sent in selected:
                out_f.write(f"[{date}] {sent}\n")
            out_f.write("\nTIMO's Sorted Output:\n")
            out_f.write(output)

        print(f"TIMO output saved to '{output_file}'")
