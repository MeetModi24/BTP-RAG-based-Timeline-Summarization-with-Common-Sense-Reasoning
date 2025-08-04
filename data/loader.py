# data/loader.py

import json
import re
from dateutil import parser
from collections import defaultdict

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

def clean_text(text):
    text = re.sub(r"An ultra-low latency.*?platform", "", text)
    text = text.replace("-LRB-", "(").replace("-RRB-", ")")
    text = re.sub(r"\s+", " ", text)
    text = text.replace("`", "'").replace("\\", "")
    return text.strip()

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

def write_processed_results_to_file(processed_results, output_file="processed_results.txt"):
    with open(output_file, "w", encoding="utf-8") as f:
        for doc in processed_results:
            date_str = doc.get("Date", "").strip()
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
