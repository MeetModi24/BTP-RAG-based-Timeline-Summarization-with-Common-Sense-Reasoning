# config.py
import os

HF_TOKEN = os.getenv("HF_TOKEN")
LLAMA_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
RETRIEVER_MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"

ARTICLE_FILE = "data/articles.preprocessed.jsonl"
GROUNDTRUTH_FILE = "data/timelines.jsonl"
PROCESSED_RESULTS_FILE = "processed_results.txt"
SUMMARY_FILE = "timeline_summary.txt"
CLEANED_SUMMARY_FILE = "cleaned_summary.txt"

TOP_K_RETRIEVAL = 10