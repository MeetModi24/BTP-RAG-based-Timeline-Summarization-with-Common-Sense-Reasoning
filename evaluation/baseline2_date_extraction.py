# evaluation/baseline2_date_extraction.py

import re
import json
from datetime import datetime
from dateutil.parser import parse
from collections import defaultdict

import tilse.data.timelines as timelines
import tilse.evaluation.rouge as rouge


# --- Helper regex and splitters ---
def get_publication_year(text):
    full_date_re = re.compile(
        r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"\s\d{1,2},\s(\d{4})"
    )
    m = full_date_re.search(text)
    if m:
        return int(m.group(1))
    return None


BROAD_DATE_RE = re.compile(
    r"\b("
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+"
    r"\d{1,2}(?:/\d{1,2})?"
    r"(?:,\s*\d{4})?"
    r")\b"
)

SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')


# --- Core pipeline functions ---
def extract_dates_and_summaries(infile, outfile):
    """Extract dated sentences from cleaned_summary.txt and save flat summary file."""
    entries = []

    with open(infile, 'r', encoding='utf-8') as f:
        text = f.read()

    pub_year = get_publication_year(text) or datetime.now().year
    lines = text.splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split(':', 1)
        if len(parts) < 2:
            continue
        sentence_text = parts[1].strip()

        # Split into sentences
        sentences = SENT_SPLIT.split(sentence_text)
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            # Find dates in sentence
            date_matches = list(BROAD_DATE_RE.finditer(sent))
            if not date_matches:
                continue

            for m in date_matches:
                date_str = m.group(1)

                if re.search(r"\d{4}", date_str):  # year present
                    try:
                        dt = parse(date_str)
                        entries.append((dt.date(), sent))
                    except Exception:
                        continue
                else:  # infer year
                    parts = re.match(r"(?P<month>[A-Za-z]+)\.?\s+(?P<day>\d{1,2}(?:/\d{1,2})?)", date_str)
                    if not parts:
                        continue
                    month, day = parts.group("month"), parts.group("day")

                    if "/" in day:  # handle ranges
                        for d in day.split("/"):
                            try:
                                dt = parse(f"{month} {d}, {pub_year}")
                                entries.append((dt.date(), sent))
                            except Exception:
                                continue
                    else:
                        try:
                            dt = parse(f"{month} {day}, {pub_year}")
                            entries.append((dt.date(), sent))
                        except Exception:
                            continue

    # Deduplicate by date
    deduped_entries = defaultdict(set)
    for date_obj, sentence in entries:
        deduped_entries[date_obj].add(sentence.lower().strip())

    flat_entries = [(d, s) for d in sorted(deduped_entries) for s in deduped_entries[d]]

    # Write results
    with open(outfile, "w", encoding="utf-8") as out:
        for date_obj, sentence in flat_entries:
            out.write(f"{date_obj.strftime('%Y-%m-%d')}: {sentence}\n")

    return outfile


def convert_to_jsonl(summary_file, jsonl_file):
    """Convert extracted summaries to JSONL format for Tilse evaluation."""
    current_timeline = {}

    with open(summary_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            date_str, summary_text = line.split(":", 1)
            date_obj = parse(date_str.strip()).date()
            current_timeline.setdefault(date_obj, []).append(summary_text.strip())

    formatted_timeline = [
        [date.strftime("%Y-%m-%dT%H:%M:%S"), summaries] for date, summaries in current_timeline.items()
    ]

    with open(jsonl_file, "w", encoding="utf-8") as out_file:
        json.dump(formatted_timeline, out_file)
        out_file.write("\n")

    return jsonl_file


def parse_summary_file(summary_file):
    with open(summary_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    summary_dict = {}
    for entry in data:
        for date_str, summaries in entry:
            date_obj = parse(date_str).date()
            summary_dict[date_obj] = summaries
    return timelines.Timeline(summary_dict)


def parse_groundtruth_file(groundtruth_file):
    groundtruth_timelines = []
    with open(groundtruth_file, "r", encoding="utf-8") as f:
        for line in f:
            timeline = json.loads(line)
            groundtruth_dict = {parse(entry[0]).date(): entry[1] for entry in timeline}
            groundtruth_timelines.append(timelines.Timeline(groundtruth_dict))
    return timelines.GroundTruth(groundtruth_timelines)


# --- Public function for main.py ---
def run_baseline2(cleaned_summary_file, groundtruth_file,
                  flat_out="baseline2_summary.txt",
                  jsonl_out="baseline2_timelines.jsonl"):
    """Run Baseline 2 pipeline and return ROUGE scores."""
    # Extract and convert
    extract_dates_and_summaries(cleaned_summary_file, flat_out)
    convert_to_jsonl(flat_out, jsonl_out)

    # Parse predictions + ground truth
    predicted_timeline = parse_summary_file(jsonl_out)
    groundtruth = parse_groundtruth_file(groundtruth_file)

    if not predicted_timeline.dates_to_summaries:
        print(" Baseline 2 ERROR: Predicted timeline is empty.")
        return None, None

    rouge_evaluator = rouge.TimelineRougeEvaluator(measures=["rouge_1"])
    concat_scores = rouge_evaluator.evaluate_concat(predicted_timeline, groundtruth)
    align_scores = rouge_evaluator.evaluate_align_date_content_costs(predicted_timeline, groundtruth)

    return concat_scores, align_scores
