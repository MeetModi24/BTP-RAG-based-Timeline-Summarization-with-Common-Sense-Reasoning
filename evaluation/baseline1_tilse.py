# evaluation/tilse_eval.py
# Baseline 1,2,3

import json
import datetime
from dateutil.parser import parse
from tilse.data import timelines
from tilse.evaluation import rouge


def parse_summary_file(summary_file):
    """
    Parses a cleaned summary file in format:
    Publication Date: ...
    Content: ...
    """
    summary_dict = {}
    with open(summary_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        date_line = lines[i].strip()
        content_line = lines[i+1].strip() if i+1 < len(lines) else ""

        if date_line.startswith("Publication Date:") and content_line.startswith("Content:"):
            date_str = date_line[len("Publication Date:"):].strip()
            content_str = content_line[len("Content:"):].strip()

            if "Unknown Date" in date_str or not content_str:
                i += 2
                continue

            try:
                date_obj = parse(date_str).date()
                summary_dict.setdefault(date_obj, []).append(content_str)
            except Exception as e:
                print(f"Error parsing date '{date_str}': {e}")

            i += 2
        else:
            i += 1  # skip malformed line

    return timelines.Timeline(summary_dict)


def parse_groundtruth_file(groundtruth_file):
    """
    Parses a JSONL file with format:
    [
      ["YYYY-MM-DDTHH:MM:SS", [summary1, summary2]],
      ...
    ]
    One such list per line.
    """
    groundtruth_timelines = []
    with open(groundtruth_file, "r", encoding="utf-8") as f:
        for line in f:
            timeline = json.loads(line)
            groundtruth_dict = {}
            for entry in timeline:
                try:
                    date_obj = parse(entry[0]).date()
                    groundtruth_dict[date_obj] = entry[1]
                except Exception as e:
                    print(f"Error parsing GT date: {entry[0]} → {e}")
            groundtruth_timelines.append(timelines.Timeline(groundtruth_dict))

    return timelines.GroundTruth(groundtruth_timelines)


def evaluate_timeline(predicted_file, groundtruth_file):
    """
    Evaluates predicted timeline summaries using Tilse.
    Returns both concat and align ROUGE-1 scores.
    """
    predicted = parse_summary_file(predicted_file)
    groundtruth = parse_groundtruth_file(groundtruth_file)

    if not predicted.dates_to_summaries:
        print("Predicted timeline is empty.")
        return None, None
    if not groundtruth.timelines:
        print("Ground truth timeline is empty.")
        return None, None

    evaluator = rouge.TimelineRougeEvaluator(measures=["rouge_1"])

    concat = evaluator.evaluate_concat(predicted, groundtruth)
    align = evaluator.evaluate_align_date_content_costs(predicted, groundtruth)

    return concat, align
