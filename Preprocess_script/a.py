"""
Simple processor that:
1) filters files by date range (prefers an Excel mapping file),
2) for matching files writes a JSONL where each record contains only: title, text, time

Assumptions / behavior:
- By default the Excel mapping is expected at `Preprocess_script/filtered_letters_sent_by_mg.xlsx`.
- Excel must contain a filename column (default: `file_name`) and a Date column (default: `Date`) in MM/DD/YYYY form.
- If Excel is provided the script selects filenames whose Date lies in the inclusive CLI range -- those filenames are then searched for (case-insensitive) inside the input directory.
- If Excel is not provided or no matches are found the script will fall back to parsing the header of each .txt file and use any date found in the first 6 lines.
- Output `time` format: `YYYY-MM-DDT00:00:00+00:00` (UTC midnight)

Usage example:
python a.py --input_dir cwmg_letters --output_file filtered.jsonl --date_from 1888-01-01 --date_to 1888-12-31

"""

import os
import glob
import json
import argparse
from datetime import datetime, date
from dateutil.parser import parse, ParserError

try:
    import pandas as pd
except Exception:
    pd = None

from tqdm import tqdm


def _try_parse_date_string(s: str):
    """Try to parse a date-like string. Returns a date object or None."""
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    # Remove surrounding brackets if present
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    try:
        dt = parse(s, default=datetime(1, 1, 1), fuzzy=True)
        return dt.date()
    except (ParserError, ValueError, TypeError):
        return None


def extract_title_and_header_date(full_text: str):
    """Given complete file content, return (title_str, date_obj_or_None, date_line_index).
    Title extraction: same as before, first line like "3. LETTER TO ..." -> title part after number.
    Date extraction: look through first 6 lines for any parseable date.
    Returns the date and its 0-based line index. Index is -1 if no date found.
    """
    lines = full_text.splitlines()
    header_lines = lines[:6]

    parsed_title = None
    if header_lines:
        first_line = header_lines[0].strip()
        m = None
        import re
        m = re.match(r"^(\d+)\.\s*(.+)$", first_line)
        if m:
            parsed_title = m.group(2).strip()
        else:
            # fallback: use entire first non-empty line
            for l in header_lines:
                if l.strip():
                    parsed_title = l.strip()
                    break

    parsed_date = None
    date_line_index = -1
    for i, l in enumerate(header_lines):
        d = _try_parse_date_string(l)
        if d:
            parsed_date = d
            date_line_index = i+1
            break

    if not parsed_title:
        parsed_title = "Untitled"

    return parsed_title, parsed_date, date_line_index


def load_excel_mapping(xlsx_path: str, id_col: str = "file_name", date_col: str = "Date"):
    """Load Excel mapping returning dict filename(lower)->date (datetime.date)
    If pandas is not available or file cannot be read, returns empty dict.
    """
    mapping = {}
    if not xlsx_path or not os.path.exists(xlsx_path):
        return mapping
    if pd is None:
        print("Warning: pandas not installed; cannot read Excel mapping. Skipping Excel.")
        return mapping
    try:
        df = pd.read_excel(xlsx_path)
        if id_col not in df.columns or date_col not in df.columns:
            print(f"Warning: Excel missing columns '{id_col}' or '{date_col}'. Skipping Excel mapping.")
            return mapping
        tmp = df[[id_col, date_col]].copy()
        # Normalize filename column to str and lower
        tmp[id_col] = tmp[id_col].astype(str).str.strip().str.lower()
        # Try parsing date column. The sheet often uses MM/DD/YYYY; try that first then fallback
        try:
            tmp[date_col] = pd.to_datetime(tmp[date_col], format="%m/%d/%Y", errors='coerce')
        except Exception:
            tmp[date_col] = pd.to_datetime(tmp[date_col], errors='coerce')
        for _, row in tmp.dropna(subset=[id_col, date_col]).iterrows():
            fname = str(row[id_col]).strip().lower()
            dt = row[date_col].date()
            mapping[fname] = dt
        return mapping
    except Exception as e:
        print(f"Warning: failed to read Excel '{xlsx_path}': {e}. Proceeding without it.")
        return {}


def find_files_for_filenames(base_dir: str, filenames: set):
    """For every filename in filenames try to find matching .txt file under base_dir.
    Matching is case-insensitive and allows filename with or without .txt. Returns dict filename->path
    """
    found = {}
    # Build list of existing files
    pattern = os.path.join(base_dir, "**", "*.txt")
    all_files = glob.glob(pattern, recursive=True)
    # map basenames lower -> path (if multiple, keep first but warn)
    basename_map = {}
    for p in all_files:
        bn = os.path.basename(p).lower()
        name_no_ext = os.path.splitext(bn)[0]
        if bn not in basename_map:
            basename_map[bn] = p
        if name_no_ext not in basename_map:
            basename_map[name_no_ext] = p

    for fname in filenames:
        # try exact match keys
        if fname in basename_map:
            found[fname] = basename_map[fname]
            continue
        # try adding .txt
        key = fname
        if not fname.endswith('.txt') and (fname + '.txt') in basename_map:
            found[fname] = basename_map[fname + '.txt']
            continue
        # try substring match: any key that contains fname
        matched = None
        for k, path in basename_map.items():
            if fname in k:
                matched = path
                break
        if matched:
            found[fname] = matched
    return found


def format_date_for_output(d: date):
    """Return string like YYYY-MM-DDT00:00:00+00:00 for a date object"""
    if d is None:
        return None
    dt = datetime(d.year, d.month, d.day, 0, 0, 0)
    return dt.isoformat() + "+00:00"


def main():
    parser = argparse.ArgumentParser(description="Filter cwmg .txt files by date range and emit simplified JSONL.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing .txt files (e.g., cwmg_letters)")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file path")
    parser.add_argument("--letters_xlsx", type=str, default="Preprocess_script/filtered_letters_sent_by_mg.xlsx",
                        help="Path to Excel mapping file (optional). Default: Preprocess_script/filtered_letters_sent_by_mg.xlsx")
    parser.add_argument("--xlsx_file_column", type=str, default="file_name", help="Column name in Excel for filename")
    parser.add_argument("--xlsx_date_column", type=str, default="Date", help="Column name in Excel for date (MM/DD/YYYY)")
    parser.add_argument("--date_from", type=str, default=None, help="Inclusive start date (YYYY-MM-DD)")
    parser.add_argument("--date_to", type=str, default=None, help="Inclusive end date (YYYY-MM-DD)")
    args = parser.parse_args()

    # parse CLI date range
    date_from = None
    date_to = None
    try:
        if args.date_from:
            date_from = parse(args.date_from).date()
        if args.date_to:
            date_to = parse(args.date_to).date()
    except Exception as e:
        print(f"Error parsing date_from/date_to: {e}")
        return

    # Load Excel mapping (filename -> date)
    filename_to_date = {}
    if args.letters_xlsx and os.path.exists(args.letters_xlsx):
        filename_to_date = load_excel_mapping(args.letters_xlsx, id_col=args.xlsx_file_column, date_col=args.xlsx_date_column)
    else:
        if args.letters_xlsx:
            print(f"Excel mapping not found at '{args.letters_xlsx}'. Will fall back to header parsing.")

    # If we have an excel mapping, select filenames within the date range
    selected_filenames = set()
    if filename_to_date:
        for fname, dt in filename_to_date.items():
            if date_from and dt < date_from:
                continue
            if date_to and dt > date_to:
                continue
            selected_filenames.add(fname)

    # Find matching file paths in input_dir
    files_to_process = []
    if selected_filenames:
        found = find_files_for_filenames(args.input_dir, selected_filenames)
        files_to_process = list(found.values())
        if not files_to_process:
            print("Warning: Excel yielded filenames in range but no matching .txt files were found under input_dir. Falling back to scanning all .txt files and header dates.")

    # If no files selected by excel or none found, fall back to scanning all .txt and parsing header dates
    if not files_to_process:
        pattern = os.path.join(args.input_dir, "*.txt")
        all_texts = glob.glob(pattern)
        for p in all_texts:
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                print(f"Failed to read {p}: {e}")
                continue
            _, header_date, _ = extract_title_and_header_date(content)
            if header_date is None:
                continue
            if date_from and header_date < date_from:
                continue
            if date_to and header_date > date_to:
                continue
            files_to_process.append(p)

    # Deduplicate and sort
    files_to_process = sorted(list(dict.fromkeys(files_to_process)))

    print(f"Total files found in input_dir: {len(glob.glob(os.path.join(args.input_dir, '*.txt')))}")
    print(f"Files selected for processing: {len(files_to_process)}")

    # Process and write simplified JSONL
    with open(args.output_file, 'w', encoding='utf-8') as out:
        for fpath in tqdm(files_to_process, desc="Processing files"):
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                print(f"Skipping {fpath} due to read error: {e}")
                continue

            title, header_date, date_line_index = extract_title_and_header_date(content)
            
            # Prefer date from excel mapping if present (lookup by basename)
            bn = os.path.basename(fpath).lower()
            name_no_ext = os.path.splitext(bn)[0]
            chosen_date = None
            if filename_to_date.get(bn):
                chosen_date = filename_to_date.get(bn)
            elif filename_to_date.get(name_no_ext):
                chosen_date = filename_to_date.get(name_no_ext)
            else:
                chosen_date = header_date

            time_field = format_date_for_output(chosen_date) if chosen_date else None

            # NEW: Determine the body of the text based on the date line
            lines = content.splitlines()
            start_line_index = 0
            if date_line_index != -1:
                # If date was found, body starts on the next line
                start_line_index = date_line_index + 1
            else:
                # Fallback: if no date in first 6 lines, assume header is 6 lines
                start_line_index = 6
            
            body_content = "\n".join(lines[start_line_index:])

            out_obj = {
                "title": title,
                "text": body_content, # Use the extracted body content
                "time": time_field
            }
            out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    print(f"\n✅ Done. Wrote {args.output_file} with {len(files_to_process)} records.")


if __name__ == '__main__':
    main()