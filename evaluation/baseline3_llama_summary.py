# evaluation/baseline3_llama_summary.py

import re
import json
from datetime import datetime, timedelta
from dateutil.parser import parse
from collections import defaultdict

import spacy
import tilse.data.timelines as timelines
import tilse.evaluation.rouge as rouge

# Load spaCy English model (download with: python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm")

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def spacy_sent_tokenize(text):
    """Sentence splitter using spaCy."""
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


def extract_dates_with_full_mapping(input_file, output_file):
    """
    Extract sentences with mapped dates (heuristics: full dates, partial dates, weekdays).
    Writes output in a readable dated-sentence format.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        blocks = [b.strip() for b in f.read().split('\n\n') if b.strip()]

    outputs = []
    for block in blocks:
        lines = block.split('\n')
        pub_date_str = lines[0].replace("Publication Date:", "").strip()
        try:
            pub_date_obj = parse(pub_date_str)
        except Exception:
            continue

        last_date = None
        summary = ' '.join(lines[1:]).replace("Content:", "").strip()
        sentences = spacy_sent_tokenize(summary)

        mappings = []
        for sent in sentences:
            final_date = None

            # Regex full date
            full_date_regex = re.compile(
                r'\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)?\.?\s*'
                r'(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|'
                r'Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
                r'\s+\d{1,2}(?:\s*,)?\s+\d{4}\b',
                flags=re.IGNORECASE
            )

            # Regex partial date
            partial_date_regex = re.compile(
                r'\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|'
                r'Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}\b',
                flags=re.IGNORECASE
            )

            # Try full date
            match = full_date_regex.search(sent)
            if match:
                try:
                    dt = parse(match.group(), fuzzy=True)
                    final_date = dt.strftime("%Y-%m-%d")
                    last_date = final_date
                except Exception:
                    final_date = last_date or pub_date_obj.strftime("%Y-%m-%d")

            else:
                # Try partial
                match = partial_date_regex.search(sent)
                if match:
                    try:
                        dt = parse(f"{match.group()} {pub_date_obj.year}", fuzzy=True)
                        final_date = dt.strftime("%Y-%m-%d")
                        last_date = final_date
                    except Exception:
                        final_date = last_date or pub_date_obj.strftime("%Y-%m-%d")
                else:
                    # Try year
                    year_match = re.search(r'\b(\d{4})\b', sent)
                    if year_match:
                        final_date = f"{year_match.group(1)}-01-01"
                    else:
                        # Try weekday
                        weekday_match = next((wd for wd in WEEKDAYS if re.search(r'\b' + wd + r'\b', sent)), None)
                        if weekday_match:
                            idx = WEEKDAYS.index(weekday_match)
                            diff = (pub_date_obj.weekday() - idx) % 7
                            inferred = pub_date_obj - timedelta(days=diff)
                            final_date = inferred.strftime("%Y-%m-%d")
                            last_date = final_date
                        else:
                            final_date = last_date or pub_date_obj.strftime("%Y-%m-%d")

            mappings.append((final_date, sent.strip()))

        # Build block output
        header = f"Publication Date: {pub_date_obj.strftime('%a %b %d , %Y')}"
        body = "Dated Sentences:\n" + "\n".join(
            f"{i}. [{dt}] {s}" for i, (dt, s) in enumerate(mappings, 1)
        )
        outputs.append(f"{header}\n\n{body}")

    with open(output_file, 'w', encoding='utf-8') as out_f:
        out_f.write("\n\n".join(outputs))

    return output_file


def parse_baseline3_txt_file(txt_file):
    """Convert baseline3_dated_sentences.txt into a Tilse Timeline."""
    date_map = defaultdict(list)
    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            m = re.match(r"\d+\. \[(\d{4}-\d{2}-\d{2})\] (.*)", line)
            if m:
                date_str, sent = m.groups()
                try:
                    date_obj = parse(date_str).date()
                    date_map[date_obj].append(sent)
                except Exception:
                    continue
    return timelines.Timeline(date_map)


def parse_groundtruth_file(groundtruth_file):
    """Parse ground truth into Tilse GroundTruth object."""
    gt = []
    with open(groundtruth_file, 'r', encoding='utf-8') as f:
        for line in f:
            arr = json.loads(line)
            dmap = {parse(item[0]).date(): item[1] for item in arr}
            gt.append(timelines.Timeline(dmap))
    return timelines.GroundTruth(gt)


# --- Public function ---
def run_baseline3(processed_results_file, groundtruth_file,
                  dated_sentences_out="baseline3_dated_sentences.txt"):
    """
    Run Baseline 3 (date mapping using regex + heuristics).
    Returns (concat_scores, align_scores).
    """
    # Step 1: Extract date mappings
    extract_dates_with_full_mapping(processed_results_file, dated_sentences_out)

    # Step 2: Load predictions + ground truth
    predicted = parse_baseline3_txt_file(dated_sentences_out)
    groundtruth = parse_groundtruth_file(groundtruth_file)

    if not predicted.dates_to_summaries:
        print("Baseline 3 ERROR: Predicted timeline empty.")
        return None, None

    evaluator = rouge.TimelineRougeEvaluator(measures=["rouge_1"])
    concat = evaluator.evaluate_concat(predicted, groundtruth)
    align = evaluator.evaluate_align_date_content_costs(predicted, groundtruth)

    return concat, align
