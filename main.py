# main.py

"""
Entry point for the timeline summarization project.

This script:
1. Loads articles
2. Retrieves relevant documents using sentence-transformer + FAISS
3. Processes and groups documents by publication date
4. Loads LLaMA model and generates timeline summaries
5. Writes summary output to file
6. Runs all four baselines (Tilse, regex, spaCy, TimeLLaMA)
7. Evaluates generated and baseline summaries using Tilse (ROUGE)
"""

from config import (
    ARTICLE_FILE,
    PROCESSED_RESULTS_FILE,
    SUMMARY_FILE,
    CLEANED_SUMMARY_FILE,
    GROUNDTRUTH_FILE,
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
    evaluate_timeline,             # Baseline 1 (Tilse)
    run_baseline2,                 # Baseline 2 (Regex-based)
    run_baseline3,                 # Baseline 3 (spaCy-based)
    run_baseline4                  # Baseline 4 (TimeLLaMA ordering)
)


def main():
    # Step 1: Load articles
    print("Loading articles...")
    articles = load_articles(ARTICLE_FILE)
    print(f"Loaded {len(articles)} articles.")

    # Step 2: Load retriever and create FAISS index
    print("Initializing retriever and FAISS index...")
    retriever = load_retriever()
    index, _ = build_faiss_index(articles, retriever)

    # Step 3: Retrieve documents for a query
    query = "Syrian uprising"
    print(f"Retrieving top documents for query: '{query}'")
    retrieved_docs = retrieve_documents(query, retriever, index, articles)

    # Step 4: Process retrieved documents by date
    print("Processing retrieved documents...")
    grouped_results = process_retrieved_docs(retrieved_docs)
    write_processed_results_to_file(grouped_results, output_file=PROCESSED_RESULTS_FILE)
    print(f"Processed results saved to {PROCESSED_RESULTS_FILE}")

    # Step 5: Convert grouped results to format for LLaMA
    grouped_dict = list_to_grouped_dict(grouped_results)

    # Step 6: Load LLaMA model and tokenizer
    print("Loading LLaMA model...")
    model, tokenizer = load_llama_model()

    # Step 7: Generate summaries
    print("Generating timeline summary...")
    summaries = generate_summary(query, grouped_dict, model, tokenizer)

    # Step 8: Save raw timeline summary (trimmed to 20 sentences)
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
    # Baseline 1: Tilse evaluation
    # ================================
    print("\nRunning ROUGE evaluation (Baseline 1: Tilse)...")
    concat, align = evaluate_timeline(CLEANED_SUMMARY_FILE, GROUNDTRUTH_FILE)
    if concat and align:
        print("Baseline 1 ROUGE-1 (concat):", concat)
        print("Baseline 1 ROUGE-1 (align):", align)
    else:
        print("Baseline 1 evaluation failed.")

    # ================================
    # Baseline 2: Regex-based
    # ================================
    print("\nRunning Baseline 2 (Regex-based)...")
    b2_concat, b2_align = run_baseline2(CLEANED_SUMMARY_FILE, GROUNDTRUTH_FILE)
    if b2_concat and b2_align:
        print("Baseline 2 ROUGE-1 (concat):", b2_concat)
        print("Baseline 2 ROUGE-1 (align):", b2_align)
    else:
        print("Baseline 2 evaluation failed.")

    # ================================
    # Baseline 3: spaCy-based
    # ================================
    print("\nRunning Baseline 3 (spaCy-based)...")
    b3_concat, b3_align = run_baseline3(PROCESSED_RESULTS_FILE, GROUNDTRUTH_FILE)
    if b3_concat and b3_align:
        print("Baseline 3 ROUGE-1 (concat):", b3_concat)
        print("Baseline 3 ROUGE-1 (align):", b3_align)
    else:
        print("Baseline 3 evaluation failed.")

    # ================================
    # Baseline 4: TimeLLaMA-based ordering
    # ================================
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
