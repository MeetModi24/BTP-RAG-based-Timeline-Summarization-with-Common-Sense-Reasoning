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

def _clean_footer(text: str) -> str:
    """
    Simplified footer removal: drop the last three lines of the body.
    This avoids brittle regex rules and matches the observation that
    trailing 3 lines are non-essential (sign-offs/citations).
    """
    if not text:
        return text
    lines = text.rstrip("\n").split("\n")
    if len(lines) <= 3:
        return ""
    return "\n".join(lines[:-3]).rstrip()

def process_document(doc_id, text_content, nlp_model):
    """
    Processes a raw text string to extract metadata and parse sentences.
    """
    try:
        # --- 1) Header parsing: id, title, publication date ---
        lines = text_content.strip().split('\n')
        header_lines = lines[:6]

        # Parse first line: "<id>. <TITLE>"
        parsed_id = None
        title = None
        if header_lines:
            first_line = header_lines[0].strip()
            m = re.match(r"^(\d+)\.\s*(.+)$", first_line)
            if m:
                parsed_id = m.group(1).strip()
                title = m.group(2).strip()

        if not parsed_id:
            parsed_id = str(doc_id)
        if not title:
            title = f"Document_{parsed_id}"

        # Robust date extraction from first 6 header lines
        pub_time_str = None
        default_date = datetime(1, 1, 1).replace(day=1)

        def try_parse_date(candidate: str):
            if not candidate:
                return None
            # Try exact bracket content like [December, 1888]
            bracket = re.search(r"\[(.*?)\]", candidate)
            if bracket:
                inner = bracket.group(1)
                try:
                    return parse(inner, default=default_date, fuzzy=True)
                except (ParserError, ValueError, TypeError):
                    pass
            # Try common explicit formats including weekday or plain
            try:
                return parse(candidate, default=default_date, fuzzy=True)
            except (ParserError, ValueError, TypeError):
                return None

        for hline in header_lines:
            dt_obj = try_parse_date(hline)
            if dt_obj:
                pub_time_str = dt_obj.isoformat() + "+00:00"
                break

        # --- 2) Remaining text after header becomes body ---
        body_text = "\n".join(lines[len(header_lines):]).strip()
        if not body_text:
            # Fallback: if header length misestimated, try removing only first line
            body_text = "\n".join(lines[1:]).strip()
        if not body_text:
            return None

        # --- 3) Remove trailing signatures/footers not useful for summarization ---
        body_text = _clean_footer(body_text)

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
            "text": body_text,
            "time": pub_time_str,
            "id": str(parsed_id),
            "sentences": processed_sentences
        }
        return final_json

    except Exception as e:
        print(f"Error processing document {doc_id}: {e}")
        return None

def _parse_header_id_and_date_from_file(file_path: str):
    """
    Read only the first lines to extract (id, title, iso_datetime) from header.
    Returns (id_str, title_str, iso_datetime_or_None).
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        lines = content.strip().split('\n')
        header_lines = lines[:6]

        parsed_id = None
        title = None
        if header_lines:
            first_line = header_lines[0].strip()
            m = re.match(r"^(\d+)\.\s*(.+)$", first_line)
            if m:
                parsed_id = m.group(1).strip()
                title = m.group(2).strip()
        if not parsed_id:
            parsed_id = os.path.splitext(os.path.basename(file_path))[0]
        if not title:
            title = f"Document_{parsed_id}"

        default_date = datetime(1, 1, 1).replace(day=1)

        def try_parse(candidate: str):
            if not candidate:
                return None
            bracket = re.search(r"\[(.*?)\]", candidate)
            if bracket:
                inner = bracket.group(1)
                try:
                    return parse(inner, default=default_date, fuzzy=True)
                except (ParserError, ValueError, TypeError):
                    pass
            try:
                return parse(candidate, default=default_date, fuzzy=True)
            except (ParserError, ValueError, TypeError):
                return None

        pub_time_str = None
        for hline in header_lines:
            dt = try_parse(hline)
            if dt:
                pub_time_str = dt.isoformat() + "+00:00"
                break

        return str(parsed_id), title, pub_time_str
    except Exception:
        return None, None, None

def main():
    parser = argparse.ArgumentParser(description="Process a directory of text files into a single JSONL file.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the directory containing .txt files.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output .jsonl file.")
    parser.add_argument("--model", type=str, default="en_core_web_sm", help="Name of the spaCy model to use.")
    parser.add_argument("--date_from", type=str, default=None, help="Inclusive start date (e.g., 1930-03-12).")
    parser.add_argument("--date_to", type=str, default=None, help="Inclusive end date (e.g., 1930-04-06).")
    parser.add_argument("--letters_xlsx", type=str, default=None, help="Optional path to filtered_letters_sent_by_mg.xlsx.")
    parser.add_argument("--xlsx_id_column", type=str, default="id", help="ID column name in the Excel file.")
    parser.add_argument("--xlsx_date_column", type=str, default="date", help="Date column name in the Excel file.")
    args = parser.parse_args()

    print(f"Loading spaCy model '{args.model}'...")
    try:
        nlp = spacy.load(args.model)
    except OSError:
        print(f"Error: spaCy model '{args.model}' not found. Please run: python -m spacy download {args.model}")
        return
    print("Model loaded successfully.")

    # --- Find all .txt files in the input directory ---
    search_path = os.path.join(args.input_dir, "*.txt")
    all_text_files = glob.glob(search_path)

    # --- Build date filter from CLI ---
    date_from = parse(args.date_from).date() if args.date_from else None
    date_to = parse(args.date_to).date() if args.date_to else None

    # --- Optionally load Excel id->date mapping ---
    id_to_date = {}
    if args.letters_xlsx:
        try:
            import pandas as pd
            df = pd.read_excel(args.letters_xlsx)
            id_col = args.xlsx_id_column
            date_col = args.xlsx_date_column
            if id_col in df.columns and date_col in df.columns:
                tmp = df[[id_col, date_col]].copy()
                tmp[id_col] = tmp[id_col].astype(str).str.extract(r"(\d+)")
                tmp[date_col] = pd.to_datetime(tmp[date_col], errors='coerce')
                for _, row in tmp.dropna(subset=[id_col, date_col]).iterrows():
                    id_to_date[str(row[id_col]).strip()] = row[date_col].date()
            else:
                print(f"Warning: Excel missing columns '{id_col}' or '{date_col}'. Skipping Excel mapping.")
        except Exception as e:
            print(f"Warning: Failed to read Excel '{args.letters_xlsx}': {e}. Proceeding without it.")

    # --- Filter files by date range (Excel preferred, else header) ---
    filtered_files = []
    for fpath in all_text_files:
        pid, _, iso_dt = _parse_header_id_and_date_from_file(fpath)
        file_date = None
        if pid and pid in id_to_date:
            file_date = id_to_date[pid]
        elif iso_dt:
            try:
                file_date = parse(iso_dt).date()
            except Exception:
                file_date = None

        if date_from or date_to:
            if not file_date:
                continue
            if date_from and file_date < date_from:
                continue
            if date_to and file_date > date_to:
                continue

        filtered_files.append(fpath)

    # --- Debug info ---
    print("Total files in folder:", len(all_text_files))
    print("Date filter:", args.date_from, "to", args.date_to)
    print("Using Excel mapping:", bool(id_to_date))
    print("Total matching files:", len(filtered_files))
    print("Example matches:", [os.path.basename(f) for f in filtered_files[:10]])

    # --- Process files and save to JSONL ---
    with open(args.output_file, 'w', encoding='utf-8') as outfile:
        for file_path in tqdm(filtered_files, desc="Processing files"):
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
