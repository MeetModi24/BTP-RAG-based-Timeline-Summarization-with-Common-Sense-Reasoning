# models/llama_model.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from config import LLAMA_MODEL_NAME, HF_TOKEN

def load_llama_model(model_name=LLAMA_MODEL_NAME, hf_token=HF_TOKEN):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        use_auth_token=hf_token
    )

    # ------------------ DISABLE FLASH ATTENTION ------------------
    if hasattr(model, "enable_flash_attention"):
        model.enable_flash_attention(False)

    # Optional: also disable xformers memory-efficient attention if present
    if hasattr(model, "enable_xformers_memory_efficient_attention"):
        model.enable_xformers_memory_efficient_attention(False)
    # -------------------------------------------------------------


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
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_input_tokens).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_summary_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0
        )

        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        summaries[pub_date] = summary

    return summaries
