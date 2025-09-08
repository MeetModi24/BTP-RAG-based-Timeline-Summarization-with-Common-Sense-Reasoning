# a.py (version 3)

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
        # --- 1. NEW: Smarter Publication Time Extraction ---
        pub_time_str = None
        # Isolate the first 5 lines to reliably find the header date
        header_text = "\n".join(text_content.strip().split('\n')[:5])
        
        # This regex looks for a full month name, day, and 4-digit year
        date_regex = r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b"
        date_match = re.search(date_regex, header_text)
        
        # Also check the old bracket format for backwards compatibility
        if not date_match:
            date_match = re.search(r"\[(.*?)\]", header_text)

        if date_match:
            try:
                # date_match.group(0) gets the full matched date string (e.g., "September 5, 1892")
                # date_match.group(1) is for the bracketed version
                date_text = date_match.group(0) if date_match.re.pattern == date_regex else date_match.group(1)
                default_date = datetime(1, 1, 1).replace(day=1)
                dt_obj = parse(date_text, default=default_date)
                pub_time_str = dt_obj.isoformat() + "+00:00"
            except (ValueError, TypeError, AttributeError):
                pub_time_str = None # Failed to parse

        # --- Basic Metadata Extraction ---
        title_match = re.search(r"^\d+\.\s*(.*)", text_content, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else f"Document_{doc_id}"
        
        # Isolate the main body of the letter
        body_match = re.search(r"(DEAR SIR,|MY DEAR.*?)(.*?)(Yours sincerely,|Yours faithfully,)", text_content, re.DOTALL | re.IGNORECASE)
        if body_match:
            body_text = body_match.group(2).strip()
        else:
            lines = text_content.strip().split('\n')
            body_text = "\n".join(lines[5:]) # Fallback: skip more lines
        body_text = re.sub(r'\s+', ' ', body_text).strip()

        if not body_text:
            return None

        # --- 2. Process the Full Body with SpaCy ---
        doc = nlp_model(body_text)
        processed_sentences = []

        # --- 3. Iterate Through Each Sentence to Find In-Sentence Times ---
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
            
            # --- 4. Build the JSON Structure with Correct Time Fields ---
            sentence_data = {
                "raw": sent.text,
                "tokens": {
                    "keys": ["dep", "head", "lemma", "ner_iob", "ner_type", "pos", "raw", "time", "time_format"],
                    "data": []
                },
                "time": sentence_level_time, # The time found IN this sentence
                "pub_time": pub_time_str      # The publication time of the whole document
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

        # --- 5. Assemble the Final Document JSON ---
        final_json = {
            "title": title,
            "text": text_content.strip(),
            "time": pub_time_str, # Main time for the document
            "id": str(doc_id),
            "sentences": processed_sentences
        }
        return final_json

    except Exception as e:
        print(f"Error processing document {doc_id}: {e}")
        return None

def main():
    """Main function to parse arguments and run the processing pipeline."""
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

    search_path = os.path.join(args.input_dir, '*.txt')
    text_files = glob.glob(search_path)
    if not text_files:
        print(f"Error: No .txt files found in '{args.input_dir}'.")
        return

    print(f"Found {len(text_files)} .txt files to process.")

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