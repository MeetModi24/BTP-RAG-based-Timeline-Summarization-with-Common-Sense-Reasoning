# data/loader.py

import json
import re
from dateutil import parser
from collections import defaultdict

# will load all the articles from the jsonl file
def load_articles(filename, limit=624):
    articles = []
    with open(filename, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            try:
                articles.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {e}")
    return articles

# will just remove the unwanted tokens and whitespaces and quotes
def clean_text(text):
    text = re.sub(r"An ultra-low latency.*?platform", "", text)
    text = text.replace("-LRB-", "(").replace("-RRB-", ")")
    text = re.sub(r"\s+", " ", text)
    text = text.replace("`", "'").replace("\\", "")
    return text.strip()

# will group the retrieved docs by date
# This function, process_retrieved_docs, takes a list of retrieved document dictionaries,
# groups their cleaned text content by date, and returns a list of dictionaries where each
# dictionary contains a "Date" and the aggregated "Content" for that date.
def process_retrieved_docs(retrieved_docs):
    grouped_docs = defaultdict(list)
    for doc in retrieved_docs:
        title = doc.get("title")
        if not title or title.strip().lower() == "none":
            title = None
        text = doc.get("text", "").strip()
        date = doc.get("time", "Unknown Date")
        if text:
            cleaned_text = clean_text(text)
            combined_text = f"{title.strip()}. {cleaned_text}" if title else cleaned_text
            grouped_docs[date].append(combined_text)

    processed_docs = []
    for date, contents in grouped_docs.items():
        all_content = " ".join(contents)
        all_content = re.sub(r"\s+", " ", all_content).strip()
        processed_docs.append({"Date": date, "Content": all_content})
    return processed_docs

# This function takes a list of processed document dictionaries and writes them to a text file.
# For each document, it attempts to parse and format the "Date" field, 
# then writes the publication date and content to the file.
def write_processed_results_to_file(processed_results, output_file="processed_results.txt"):
    with open(output_file, "w", encoding="utf-8") as f:
        for doc in processed_results:
            date_str = (doc.get("Date") or "").strip()
            content = doc.get("Content", "").strip()
            if not content:
                continue
            try:
                dt = parser.parse(date_str)
                formatted_date = dt.strftime('%a %b %d , %Y %I:%M %p EDT')
            except Exception:
                formatted_date = date_str
            f.write(f"Publication Date: {formatted_date}\n")
            f.write(f"Content: {content}\n\n")
