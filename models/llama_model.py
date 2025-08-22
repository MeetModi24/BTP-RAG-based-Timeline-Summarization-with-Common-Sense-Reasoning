# models/llama_model.py

# --- Set env vars BEFORE importing torch/transformers ---
import os

# Silence tokenizer fork/parallelism warnings and avoid deadlocks
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Completely disable SDPA / FlashAttention backends
os.environ.setdefault("PYTORCH_USE_SDPA", "0")
os.environ.setdefault("PYTORCH_FLASH_ATTENTION", "0")

# Help with CUDA memory fragmentation on small GPUs
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Optional: reduce HF noise
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
)
from config import LLAMA_MODEL_NAME, HF_TOKEN


def load_llama_model(model_name: str = LLAMA_MODEL_NAME, hf_token: str = HF_TOKEN):
    """
    Loads a Llama 3 (or compatible) model in 4-bit quantization with attention forced to 'eager'
    so that FlashAttention/SDPA/xFormers are never used.
    """

    # Force eager attention (Transformers >= 4.38)
    config = AutoConfig.from_pretrained(model_name, use_auth_token=hf_token)
    # Valid values include "eager", "sdpa", "flash_attention_2" (if available).
    config._attn_implementation = "eager"

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    # Ensure pad token exists (some Llama tokenizers don't define it)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left padding is safer for decoder-only models at generation time
    tokenizer.padding_side = "left"

    # 4-bit quantization config (bnb)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # compute in fp16 for speed/memory balance
    )

    # Load the model with 4-bit quantization and device_map auto
    # offload_folder allows CPU/disk offload if GPU VRAM is tiny
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        quantization_config=quant_config,
        device_map="auto",
        offload_folder="offload",
        use_auth_token=hf_token,
    )

    # Align pad/eos ids for generation
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model.generation_config, "pad_token_id", None) is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    if getattr(model.config, "eos_token_id", None) is None:
        model.config.eos_token_id = tokenizer.eos_token_id

    # Best-effort: ensure any optional fast paths are disabled if present
    if hasattr(model, "enable_flash_attention"):
        model.enable_flash_attention(False)
    if hasattr(model, "enable_xformers_memory_efficient_attention"):
        model.enable_xformers_memory_efficient_attention(False)

    return model, tokenizer


def generate_summary(
    query,
    grouped_docs,
    model,
    tokenizer,
    max_summary_tokens: int = 128,
    # Keep context manageable for low VRAM; adjust if you have more memory
    max_context_tokens: int = 2048,
):
    """
    Generates per-date summaries. Uses deterministic (greedy) decoding to avoid
    temperature/top-p warnings. Keeps context modest for small GPUs.
    """
    summaries = {}

    # Safety margin for special tokens
    max_input_tokens = max(256, min(max_context_tokens, 8192 - max_summary_tokens - 50))

    for pub_date, contents in grouped_docs.items():
        full_content = " ".join(contents)

        prompt = (
            f"Query: {query}\n\n"
            f"Publication Date: {pub_date}\n\n"
            f"Content: {full_content}\n\n"
            "Your task is to generate a summary consisting of up to 20 sentences, "
            "strictly using only the information present in the content above for the given publication date and the query. "
            "Each sentence that mentions a date must begin with the date in YYYY-MM-DD format, followed by a colon and a space."
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
            padding=True,
        )

        # Send tensors to the same device as the model's first param
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Deterministic decoding; don't pass unsupported flags like temperature when do_sample=False
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_summary_tokens,
                do_sample=False,      # greedy
                top_p=1.0,            # no-op with do_sample=False
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Optionally trim echo of the prompt if model echoes it (common with some configs)
        # Try to split on the last occurrence of Content: to keep only the completion part
        split_key = "Content:"
        if split_key in text:
            # Keep the part after your prompt if the model echoed
            after = text.split(split_key, maxsplit=1)[-1]
            if "\n" in after:
                text = after.split("\n", maxsplit=1)[-1].strip()

        summaries[pub_date] = text

    return summaries
