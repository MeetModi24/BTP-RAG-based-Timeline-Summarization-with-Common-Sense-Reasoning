# models/llama_model.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from config import LLAMA_MODEL_NAME, HF_TOKEN
from transformers.integrations import sdpa_attention
import os


os.environ["TOKENIZERS_PARALLELISM"] = "false"   # silence tokenizer warning
os.environ["PYTORCH_USE_SDPA"] = "0"
os.environ["PYTORCH_FLASH_ATTENTION"] = "0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# ------------------ DISABLE FLASH ATTENTION ------------------
# This forces transformers to avoid FlashAttention on any GPU
sdpa_attention.USE_FLASH_ATTENTION = False
# -------------------------------------------------------------

def load_llama_model(model_name=LLAMA_MODEL_NAME, hf_token=HF_TOKEN):
    # Force eager attention (disables FlashAttention/SDPA completely)
    config = AutoConfig.from_pretrained(model_name, use_auth_token=hf_token)
    config._attn_implementation = "eager"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,                # <--- pass config with eager attention
        device_map="auto",
        torch_dtype=torch.float16,
        use_auth_token=hf_token
    )

    return model, tokenizer


def generate_summary(query, grouped_docs, model, tokenizer, max_summary_tokens=128):
    summaries = {}

    for pub_date, contents in grouped_docs.items():
        full_content = " ".join(contents)

        input_text = (
            f"Query: {query}\n\n"
            f"Publication Date: {pub_date}\n\n"
            f"Content: {full_content}\n\n"
            "Your task is to generate a summary consisting of up to 20 sentences, "
            "strictly using only the information present in the content above for the given publication date and the query. "
            "Each sentence that mentions a date must begin with the date in YYYY-MM-DD format, followed by a colon and a space."
        )

        max_input_tokens = 8192 - max_summary_tokens - 50
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens
        ).to(model.device)

        # Remove unsupported flags like 'temperature' if needed
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_summary_tokens,
            do_sample=False,
            # temperature=0.0,
            top_p=1.0
        )

        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        summaries[pub_date] = summary

    return summaries
