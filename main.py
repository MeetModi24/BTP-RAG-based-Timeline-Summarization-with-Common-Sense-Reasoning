# main.py

"""
Entry point for the timeline summarization project.

This script:
1. Loads articles (passed at runtime)
2. Retrieves relevant documents using sentence-transformer + FAISS
3. Processes and groups documents by publication date
4. Loads LLaMA model and generates timeline summaries
5. Writes summary output to file
6. Runs all four baselines (Tilse, Regex, spaCy, TimeLLaMA)
7. Evaluates generated and baseline summaries using Tilse (ROUGE)

Usage:
    python main.py --articles data/ukraine_articles.jsonl --groundtruth data/ukraine_timelines.jsonl --query "Ukraine war"
"""

import argparse

from config import (
    ARTICLE_FILE as DEFAULT_ARTICLE_FILE,
    GROUNDTRUTH_FILE as DEFAULT_GROUNDTRUTH_FILE,
    PROCESSED_RESULTS_FILE,
    SUMMARY_FILE,
    CLEANED_SUMMARY_FILE,
)

from data import (
    load_articles,
    process_retrieved_docs,
    write_processed_results_to_file,
)
from data.preprocess import (
    list_to_grouped_dict,
    trim_summary_sentences,
    reformat_summaries_for_cleaning,
)

from models import (
    load_retriever,
    build_faiss_index,
    retrieve_documents,
    load_llama_model,
    generate_summary,
)

from utils import write_text

# Baseline evaluation modules
from evaluation import (
    evaluate_timeline,   # Baseline 1
    run_baseline2,       # Baseline 2
    run_baseline3,       # Baseline 3
    run_baseline4        # Baseline 4
)

# You can add this function to a utils file or directly in main.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_expanded_queries(topic_query, model, tokenizer, num_questions=5):
    """
    Uses an LLM to expand a simple topic query into a list of detailed questions.
    """
    prompt = f"""
    You are a research assistant. Your task is to generate {num_questions} diverse, detailed, and insightful questions about the topic: "{topic_query}".
    
    The questions should cover various aspects of the topic, such as:
    - The origins and initial outbreak.
    - The global response and key health organizations involved.
    - Key turning points, scientific discoveries, and public health measures.
    - The social and economic impact.
    - Prominent figures and their roles.

    Return ONLY a Python list of strings, like this:
    ["Question 1?", "Question 2?", "Question 3?"]
    """

    # This is a simplified generation call, adjust parameters as needed
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512, num_return_sequences=1)
    
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the list from the model's response
    try:
        # A simple way to find and evaluate the list in the response string
        list_str = response_text[response_text.find('['):response_text.rfind(']')+1]
        expanded_queries = eval(list_str)
        if isinstance(expanded_queries, list):
            return expanded_queries
    except:
        # Fallback if parsing fails
        print("Warning: Failed to parse LLM response for query expansion.")
        return [topic_query] # Return the original query as a fallback

    return [topic_query]

def parse_args():
    parser = argparse.ArgumentParser(description="Timeline Summarization Pipeline")
    parser.add_argument("--articles", type=str, default=DEFAULT_ARTICLE_FILE,
                        help="Path to JSONL file containing input articles (default: config.py setting)")
    parser.add_argument("--groundtruth", type=str, default=DEFAULT_GROUNDTRUTH_FILE,
                        help="Path to groundtruth timelines JSONL (default: config.py setting)")
    parser.add_argument("--query", type=str, required=True,
                        help="Query string for retrieval (e.g. 'Syrian uprising')")
    return parser.parse_args()


# def main():
#     args = parse_args()
#     ARTICLE_FILE = args.articles
#     GROUNDTRUTH_FILE = args.groundtruth
#     query = args.query

#     print(f"Using ARTICLE_FILE = {ARTICLE_FILE}")
#     print(f"Using GROUNDTRUTH_FILE = {GROUNDTRUTH_FILE}")
#     print(f"Query = '{query}'")

#     # Step 1: Load articles
#     print("Loading articles...")
#     articles = load_articles(ARTICLE_FILE)
#     print(f"Loaded {len(articles)} articles from {ARTICLE_FILE}")

#     # Step 2: Load retriever and create FAISS index
#     print("Initializing retriever and FAISS index...")
#     retriever = load_retriever()
#     index, _ = build_faiss_index(articles, retriever)

#     # Step 3: Retrieve documents for query
#     print(f"Retrieving top documents for query: '{query}'")
#     retrieved_docs = retrieve_documents(query, retriever, index, articles)

#     # Step 4: Process retrieved documents by date
#     print("Processing retrieved documents...")
#     grouped_results = process_retrieved_docs(retrieved_docs)
#     write_processed_results_to_file(grouped_results, output_file=PROCESSED_RESULTS_FILE)
#     print(f"Processed results saved to {PROCESSED_RESULTS_FILE}")

#     # Step 5: Convert grouped results to format for LLaMA
#     grouped_dict = list_to_grouped_dict(grouped_results)

#     # Step 6: Load LLaMA model and tokenizer
#     print("Loading LLaMA model...")
#     model, tokenizer = load_llama_model()

#     # Step 7: Generate summaries
#     print("Generating timeline summary...")
#     summaries = generate_summary(query, grouped_dict, model, tokenizer)

#     # Step 8: Save raw timeline summary (trimmed to 20 sentences)
#     print(f"Writing trimmed summaries to {SUMMARY_FILE}...")
#     with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
#         for date, text in sorted(summaries.items()):
#             trimmed = trim_summary_sentences(text, max_sentences=20)
#             f.write(f"{date}: {trimmed}\n")
#     print(f"Summary saved to {SUMMARY_FILE}")

#     # Step 9: Format and save cleaned summary for evaluation
#     print("Preparing cleaned summary for Tilse evaluation...")
#     cleaned_text = reformat_summaries_for_cleaning(summaries)
#     write_text(cleaned_text, CLEANED_SUMMARY_FILE)
#     print(f"Cleaned summary saved to {CLEANED_SUMMARY_FILE}")

#     # ================================
#     # Baseline 1: Tilse evaluation
#     # ================================
#     print("\nRunning ROUGE evaluation (Baseline 1: Tilse)...")
#     concat, align = evaluate_timeline(CLEANED_SUMMARY_FILE, GROUNDTRUTH_FILE)
#     if concat and align:
#         print("Baseline 1 ROUGE-1 (concat):", concat)
#         print("Baseline 1 ROUGE-1 (align):", align)
#     else:
#         print("Baseline 1 evaluation failed.")

#     # ================================
#     # Baseline 2: Regex-based
#     # ================================
#     print("\nRunning Baseline 2 (Regex-based)...")
#     b2_concat, b2_align = run_baseline2(CLEANED_SUMMARY_FILE, GROUNDTRUTH_FILE)
#     if b2_concat and b2_align:
#         print("Baseline 2 ROUGE-1 (concat):", b2_concat)
#         print("Baseline 2 ROUGE-1 (align):", b2_align)
#     else:
#         print("Baseline 2 evaluation failed.")

#     # ================================
#     # Baseline 3: spaCy-based
#     # ================================
#     print("\nRunning Baseline 3 (spaCy-based)...")
#     b3_concat, b3_align = run_baseline3(PROCESSED_RESULTS_FILE, GROUNDTRUTH_FILE,
#                                         dated_sentences_out="baseline3_dated_sentences.txt")
#     if b3_concat and b3_align:
#         print("Baseline 3 ROUGE-1 (concat):", b3_concat)
#         print("Baseline 3 ROUGE-1 (align):", b3_align)
#     else:
#         print("Baseline 3 evaluation failed.")

#     # ================================
#     # Baseline 4: TimeLLaMA-based ordering
#     # ================================
#     print("\nRunning Baseline 4 (TimeLLaMA ordering)...")
#     b4_concat, b4_align = run_baseline4("baseline3_dated_sentences.txt", GROUNDTRUTH_FILE)
#     if b4_concat and b4_align:
#         print("Baseline 4 ROUGE-1 (concat):", b4_concat)
#         print("Baseline 4 ROUGE-1 (align):", b4_align)
#     else:
#         print("Baseline 4 evaluation failed.")

#     print("\nAll baselines completed.")


def main():
    args = parse_args()
    ARTICLE_FILE = args.articles
    GROUNDTRUTH_FILE = args.groundtruth
    # Join the list of query words into a single string
    initial_query = " ".join(args.query)

    print(f"Using ARTICLE_FILE = {ARTICLE_FILE}")
    print(f"Using GROUNDTRUTH_FILE = {GROUNDTRUTH_FILE}")
    print(f"Initial Query = '{initial_query}'")

    # Step 1: Load articles
    print("Loading articles...")
    articles = load_articles(ARTICLE_FILE)
    print(f"Loaded {len(articles)} articles from {ARTICLE_FILE}")

    # =================================================================
    # NEW STEP 2: Expand the initial query using the LLM
    # =================================================================
    print("Loading LLaMA model for query expansion...")
    model, tokenizer = load_llama_model()

    print("Expanding query...")
    detailed_queries = generate_expanded_queries(initial_query, model, tokenizer)
    print("Generated the following detailed queries:")
    for q in detailed_queries:
        print(f"- {q}")

    # =================================================================
    # MODIFIED STEP 3: Retrieve documents for EACH detailed query
    # =================================================================
    print("Initializing retriever and FAISS index...")
    retriever = load_retriever()
    index, _ = build_faiss_index(articles, retriever)

    print(f"Retrieving top documents for {len(detailed_queries)} queries...")
    # Use a set to automatically handle and remove duplicate documents
    unique_retrieved_docs = set()

    for query in detailed_queries:
        # Retrieve documents for the current detailed question
        retrieved_docs_list = retrieve_documents(query, retriever, index, articles)

        # Add the retrieved documents to the set to ensure uniqueness
        # We convert each doc dict to a hashable tuple of items
        for doc in retrieved_docs_list:
            unique_retrieved_docs.add(tuple(sorted(doc.items())))

    # Convert the set of unique doc tuples back to a list of dictionaries
    final_retrieved_docs = [dict(doc_tuple) for doc_tuple in unique_retrieved_docs]
    print(f"Retrieved {len(final_retrieved_docs)} unique documents in total.")

    # Step 4: Process the aggregated retrieved documents
    print("Processing retrieved documents...")
    grouped_results = process_retrieved_docs(final_retrieved_docs)
    write_processed_results_to_file(grouped_results, output_file=PROCESSED_RESULTS_FILE)
    print(f"Processed results saved to {PROCESSED_RESULTS_FILE}")

    # Step 5: Convert grouped results to format for LLaMA
    grouped_dict = list_to_grouped_dict(grouped_results)

    # Step 6: The LLaMA model is already loaded, so we proceed to generate
    
    # Step 7: Generate summaries using the rich context
    print("Generating timeline summary...")
    # Note: We use the 'initial_query' here to guide the final summary's topic
    summaries = generate_summary(initial_query, grouped_dict, model, tokenizer)

    # Step 8: Save raw timeline summary
    print(f"Writing trimmed summaries to {SUMMARY_FILE}...")
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        for date, text in sorted(summaries.items()):
            trimmed = trim_summary_sentences(text, max_sentences=20)
            f.write(f"{date}: {trimmed}\n")
    print(f"Summary saved to {SUMMARY_FILE}")

    # Step 9: Format and save cleaned summary for evaluation
    print("Preparing cleaned summary for Tilse evaluation...")
    cleaned_text = reformat_summaries_for_cleaning(summaries)
    write_text(cleaned_text, CLEANED_SUMMARY_FILE)
    print(f"Cleaned summary saved to {CLEANED_SUMMARY_FILE}")

    # ================================
    # Baseline Evaluations (No changes needed below)
    # ================================
    print("\nRunning ROUGE evaluation (Baseline 1: Tilse)...")
    concat, align = evaluate_timeline(CLEANED_SUMMARY_FILE, GROUNDTRUTH_FILE)
    if concat and align:
        print("Baseline 1 ROUGE-1 (concat):", concat)
        print("Baseline 1 ROUGE-1 (align):", align)
    else:
        print("Baseline 1 evaluation failed.")

    print("\nRunning Baseline 2 (Regex-based)...")
    b2_concat, b2_align = run_baseline2(CLEANED_SUMMARY_FILE, GROUNDTRUTH_FILE)
    if b2_concat and b2_align:
        print("Baseline 2 ROUGE-1 (concat):", b2_concat)
        print("Baseline 2 ROUGE-1 (align):", b2_align)
    else:
        print("Baseline 2 evaluation failed.")

    print("\nRunning Baseline 3 (spaCy-based)...")
    b3_concat, b3_align = run_baseline3(PROCESSED_RESULTS_FILE, GROUNDTRUTH_FILE,
                                        dated_sentences_out="baseline3_dated_sentences.txt")
    if b3_concat and b3_align:
        print("Baseline 3 ROUGE-1 (concat):", b3_concat)
        print("Baseline 3 ROUGE-1 (align):", b3_align)
    else:
        print("Baseline 3 evaluation failed.")

    print("\nRunning Baseline 4 (TimeLLaMA ordering)...")
    b4_concat, b4_align = run_baseline4("baseline3_dated_sentences.txt", GROUNDTRUTH_FILE)
    if b4_concat and b4_align:
        print("Baseline 4 ROUGE-1 (concat):", b4_concat)
        print("Baseline 4 ROUGE-1 (align):", b4_align)
    else:
        print("Baseline 4 evaluation failed.")

    print("\nAll baselines completed.")

if __name__ == "__main__":
    main()
