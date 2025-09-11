"""
Processes a directory of plain text files into a single JSONL file with
detailed linguistic and temporal annotations. Handles dates in the document header.
"""

import os
import spacy
import json
import re
import argparse
import glob
from datetime import datetime
from dateutil.parser import parse, ParserError
from tqdm import tqdm

def process_document(doc_id, text_content, nlp_model):
    """
    Processes a raw text string to extract metadata and parse sentences.
    """
    try:
        # --- 1. Smarter Publication Time Extraction ---
        pub_time_str = None
        header_text = "\n".join(text_content.strip().split('\n')[:5])

        date_regex = r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b"
        date_match = re.search(date_regex, header_text)

        if not date_match:
            date_match = re.search(r"\[(.*?)\]", header_text)

        if date_match:
            try:
                date_text = date_match.group(0) if date_match.re.pattern == date_regex else date_match.group(1)
                default_date = datetime(1, 1, 1).replace(day=1)
                dt_obj = parse(date_text, default=default_date)
                pub_time_str = dt_obj.isoformat() + "+00:00"
            except (ValueError, TypeError, AttributeError):
                pub_time_str = None

        # --- Basic Metadata Extraction ---
        title_match = re.search(r"^\d+\.\s*(.*)", text_content, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else f"Document_{doc_id}"

        body_match = re.search(r"(DEAR SIR,|MY DEAR.*?)(.*?)(Yours sincerely,|Yours faithfully,)", text_content, re.DOTALL | re.IGNORECASE)
        if body_match:
            body_text = body_match.group(2).strip()
        else:
            lines = text_content.strip().split('\n')
            body_text = "\n".join(lines[5:])
        body_text = re.sub(r'\s+', ' ', body_text).strip()

        if not body_text:
            return None

        # --- Process the full body with SpaCy ---
        doc = nlp_model(body_text)
        processed_sentences = []

        for sent in doc.sents:
            sentence_level_time = None
            token_time_map = {}

            for ent in sent.ents:
                if ent.label_ == "DATE":
                    try:
                        parsed_time = parse(ent.text)
                        time_val = parsed_time.isoformat()
                        time_format = "YYYY-MM-DD"
                        if not sentence_level_time:
                            sentence_level_time = time_val
                        for token in ent:
                            token_time_map[token.i] = (time_val, time_format)
                    except (ParserError, ValueError):
                        continue

            sentence_data = {
                "raw": sent.text,
                "tokens": {
                    "keys": ["dep", "head", "lemma", "ner_iob", "ner_type", "pos", "raw", "time", "time_format"],
                    "data": []
                },
                "time": sentence_level_time,
                "pub_time": pub_time_str
            }

            token_list = []
            for token in sent:
                time_val, time_format = token_time_map.get(token.i, (None, None))
                token_list.append([
                    token.dep_, token.head.i - sent.start, token.lemma_,
                    token.ent_iob_, token.ent_type_, token.pos_, token.text,
                    time_val, time_format
                ])

            sentence_data["tokens"]["data"] = token_list
            processed_sentences.append(sentence_data)

        final_json = {
            "title": title,
            "text": text_content.strip(),
            "time": pub_time_str,
            "id": str(doc_id),
            "sentences": processed_sentences
        }
        return final_json

    except Exception as e:
        print(f"Error processing document {doc_id}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Process a directory of text files into a single JSONL file.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the directory containing .txt files.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output .jsonl file.")
    parser.add_argument("--model", type=str, default="en_core_web_sm", help="Name of the spaCy model to use.")
    args = parser.parse_args()

    print(f"Loading spaCy model '{args.model}'...")
    try:
        nlp = spacy.load(args.model)
    except OSError:
        print(f"Error: spaCy model '{args.model}' not found. Please run: python -m spacy download {args.model}")
        return
    print("Model loaded successfully.")

    # --- List of allowed filenames ---
    allowed_files = set([
        "volume43_book_102.txt",
        "volume43_book_117.txt",
        "volume92_book_299.txt",
        "volume43_book_70.txt",
        "volume43_book_78.txt",
        "volume43_book_118.txt",
        "volume43_book_125.txt",
        "volume43_book_164.txt",
        "volume43_book_151.txt",
        "volume43_book_83.txt",
        "volume43_book_154.txt",
        "volume92_book_291.txt",
        "volume92_book_293.txt",
        "volume43_book_111.txt",
        "volume43_book_65.txt",
        "volume92_book_289.txt",
        "volume43_book_92.txt",
        "volume43_book_109.txt",
        "volume43_book_58.txt",
        "volume43_book_75.txt",
        "volume43_book_177.txt",
        "volume43_book_71.txt",
        "volume43_book_116.txt",
        "volume92_book_298.txt",
        "volume43_book_82.txt",
        "volume43_book_123.txt",
        "volume43_book_140.txt",
        "volume43_book_64.txt",
        "volume43_book_155.txt",
        "volume43_book_63.txt",
        "volume92_book_292.txt",
        "volume43_book_86.txt",
        "volume97_book_153.txt",
        "volume43_book_87.txt",
        "volume43_book_90.txt",
        "volume43_book_93.txt",
        "volume43_book_128.txt",
        "volume43_book_122.txt",
        "volume43_book_62.txt",
        "volume43_book_110.txt",
        "volume97_book_152.txt",
        "volume43_book_160.txt",
        "volume43_book_61.txt",
        "volume43_book_178.txt",
        "volume43_book_91.txt",
        "volume43_book_98.txt",
        "volume43_book_66.txt",
        "volume43_book_72.txt",
        "volume43_book_170.txt",
        "volume43_book_84.txt",
        "volume43_book_60.txt",
        "volume43_book_121.txt",
        "volume92_book_296.txt",
        "volume43_book_183.txt",
        "volume43_book_89.txt",
        "volume43_book_181.txt",
        "volume43_book_88.txt",
        "volume43_book_100.txt",
        "volume43_book_150.txt",
        "volume92_book_300.txt",
        "volume92_book_290.txt",
        "volume92_book_295.txt",
        "volume95_book_100.txt",
        "volume43_book_74.txt"
    ])

    # --- Find all .txt files in the input directory ---
    search_path = os.path.join(args.input_dir, "*.txt")
    all_text_files = glob.glob(search_path)

    # --- Filter only allowed files ---
    text_files = [f for f in all_text_files if os.path.basename(f).strip().lower() in allowed_files]

    # --- Debug info ---
    print("Total files in folder:", len(all_text_files))
    print("Total allowed files:", len(allowed_files))
    print("Total matching files:", len(text_files))
    print("Example matches:", [os.path.basename(f) for f in text_files[:10]])

    # --- Process files and save to JSONL ---
    with open(args.output_file, 'w', encoding='utf-8') as outfile:
        for file_path in tqdm(text_files, desc="Processing files"):
            doc_id = os.path.splitext(os.path.basename(file_path))[0]
            try:
                with open(file_path, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                processed_data = process_document(doc_id, content, nlp)
                if processed_data:
                    outfile.write(json.dumps(processed_data) + '\n')
            except Exception as e:
                print(f"Failed to read or process file {file_path}: {e}")

    print(f"\n✅ Processing complete. Output saved to '{args.output_file}'.")

if __name__ == "__main__":
    main()
