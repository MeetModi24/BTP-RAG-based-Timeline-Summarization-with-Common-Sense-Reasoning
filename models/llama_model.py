# models/llama_model.py

# --- Set env vars BEFORE importing torch/transformers ---
import os

# Safer tokenizer behavior in forked environments
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Kill SDPA/FlashAttention everywhere
os.environ.setdefault("PYTORCH_USE_SDPA", "0")
os.environ.setdefault("PYTORCH_FLASH_ATTENTION", "0")

# More informative CUDA errors (sync kernels)
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

# Reduce CUDA fragmentation on small VRAM
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Optional: quieter transformers logs
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
)
from config import LLAMA_MODEL_NAME, HF_TOKEN


def _build_max_memory(default_gpu_gb: int = 3, cpu_gb: int = 120):
    """
    Constrain per-GPU memory so accelerate doesn't spill layers onto an almost-full device.
    You can override with env MAX_GPU_MEM_GB / MAX_CPU_MEM_GB.
    """
    gpu_gb = int(os.environ.get("MAX_GPU_MEM_GB", default_gpu_gb))
    cpu_limit = int(os.environ.get("MAX_CPU_MEM_GB", cpu_gb))
    max_mem = {"cpu": f"{cpu_limit}GiB"}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            max_mem[i] = f"{gpu_gb}GiB"
    return max_mem


def load_llama_model(model_name: str = LLAMA_MODEL_NAME, hf_token: str = HF_TOKEN):
    """
    Load Llama 3 (or compatible) in 4-bit with eager attention, constrained GPU memory,
    and CPU/disk offload for stability on small/contended GPUs.
    """

    # Force eager attention (disables SDPA/Flash/xformers)
    config = AutoConfig.from_pretrained(model_name, use_auth_token=hf_token)
    config._attn_implementation = "eager"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Prefer BF16 compute on A100 for bnb 4-bit stability
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    max_memory = _build_max_memory(default_gpu_gb=3, cpu_gb=120)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        quantization_config=quant_config,
        device_map="auto",          # accelerate chooses placement within max_memory limits
        max_memory=max_memory,      # <--- prevent spilling onto nearly-full GPUs
        offload_folder="offload",   # CPU/disk offload when VRAM is tight
        use_auth_token=hf_token,
    )
    model.eval()

    # Make sure pad/eos ids are set
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model.generation_config, "pad_token_id", None) is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    if getattr(model.config, "eos_token_id", None) is None:
        model.config.eos_token_id = tokenizer.eos_token_id

    # Best-effort disable any optional fast paths if present
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
    max_context_tokens: int = 1024,  # keep modest for low VRAM; increase if you have room
):
    """
    Deterministic generation (greedy) with conservative context length.
    """
    summaries = {}

    # Safety margin for special tokens
    max_input_tokens = max(256, min(max_context_tokens, 4096 - max_summary_tokens - 64))

    # Free any stale caches before a long loop (helps on shared GPUs)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    device = next(model.parameters()).device

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
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Greedy decoding; avoid unsupported sampling flags
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_summary_tokens,
                do_sample=False,
                top_p=1.0,
                use_cache=False,  # a bit more VRAM-friendly on small GPUs
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Trim prompt echo if present
        split_key = "Content:"
        if split_key in text:
            after = text.split(split_key, maxsplit=1)[-1]
            if "\n" in after:
                text = after.split("\n", maxsplit=1)[-1].strip()

        summaries[pub_date] = text

        # Optional: clear per-iteration cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return summaries
