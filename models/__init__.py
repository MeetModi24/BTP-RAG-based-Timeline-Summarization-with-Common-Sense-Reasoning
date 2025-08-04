# models/__init__.py

"""
Model module initializer.

Includes:
- Retrieval model (SentenceTransformer + FAISS)
- LLaMA model (Meta LLaMA-3 for summarization)
- TIMO model (optional, for baseline prompting)

Available imports:
- load_retriever, build_faiss_index, retrieve_documents
- load_llama_model, generate_summary
"""

from .retrieval_model import (
    load_retriever,
    build_faiss_index,
    retrieve_documents,
)

from .llama_model import (
    load_llama_model,
    generate_summary,
)

# Optional: if timo_model.py is added
# from .timo_model import run_timo_syria
