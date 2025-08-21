# evaluation/baseline4_timellama.py

import os, re, json, random
from collections import deque, defaultdict
from datetime import datetime
from dateutil.parser import parse

import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
import tilse.data.timelines as timelines
import tilse.evaluation.rouge as rouge

# ------------------------
# Load TimeLLaMA once
# ------------------------
MODEL_NAME = "chrisyuan45/TimeLlama-7b-chat"
QUANTIZATION = True
tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME)
model = LlamaForCausalLM.from_pretrained(
    MODEL_NAME,
    return_dict=True,
    load_in_8bit=QUANTIZATION,
    device_map="auto",
    low_cpu_mem_usage=True
)

def timellama_generate(prompt, max_new_tokens=16, temperature=0.0):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=temperature,
        top_p=1.0
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# ------------------------
# Parsing + ordering helpers
# ------------------------
def parse_baseline3_sentences(txt_file):
    sentences = []
    pattern = re.compile(r"\d+\. \[(\d{4}-\d{2}-\d{2})\] (.*)")
    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            m = pattern.match(line)
            if m:
                date_str, sent = m.groups()
                sentences.append((date_str, sent))
    return sentences


def query_timellama_order(s1, s2):
    d1_str, sent1 = s1
    d2_str, sent2 = s2
    try:
        d1, d2 = datetime.strptime(d1_str, "%Y-%m-%d"), datetime.strptime(d2_str, "%Y-%m-%d")
        if d1 < d2: return -1
        if d1 > d2: return 1
    except ValueError:
        pass

    prompt = f"Which event happened first?\n1: [{d1_str}] {sent1}\n2: [{d2_str}] {sent2}\nAnswer with 1 or 2."
    response = timellama_generate(prompt, max_new_tokens=8, temperature=0.0)
    if re.search(r"\b1\b", response): return -1
    if re.search(r"\b2\b", response): return 1
    return 0


def build_order_graph(sentences, num_samples=None, cache_file="baseline4_cache.json"):
    n = len(sentences)
    total_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    cache = {}
    if os.path.exists(cache_file):
        cache = json.load(open(cache_file, 'r', encoding='utf-8'))

    graph, indegree = defaultdict(list), [0] * n

    for (i, j) in total_pairs:
        key = f"{i}_{j}"
        if key in cache:
            order = cache[key]
        else:
            order = query_timellama_order(sentences[i], sentences[j])
            cache[key] = order

        if order == -1:
            graph[i].append(j); indegree[j] += 1
        elif order == 1:
            graph[j].append(i); indegree[i] += 1

    json.dump(cache, open(cache_file, 'w', encoding='utf-8'))
    return graph, indegree


def topo_sort(graph, indegree, n):
    q, order = deque([i for i in range(n) if indegree[i] == 0]), []
    while q:
        u = q.popleft(); order.append(u)
        for v in graph[u]:
            indegree[v] -= 1
            if indegree[v] == 0: q.append(v)
    return order if len(order) == n else None


def timeline_from_sorted(sentences, order):
    dmap = defaultdict(list)
    for idx in order:
        try:
            dmap[datetime.strptime(sentences[idx][0], "%Y-%m-%d").date()].append(sentences[idx][1])
        except ValueError:
            continue
    return timelines.Timeline(dmap)


def parse_groundtruth_file(groundtruth_file):
    gt = []
    with open(groundtruth_file, 'r', encoding='utf-8') as f:
        for line in f:
            arr = json.loads(line)
            dmap = {parse(item[0]).date(): item[1] for item in arr}
            gt.append(timelines.Timeline(dmap))
    return timelines.GroundTruth(gt)


# ------------------------
# Main entry
# ------------------------
def run_baseline4(baseline3_txt, groundtruth_file, cache_file="baseline4_cache.json", k=12):
    sentences = parse_baseline3_sentences(baseline3_txt)
    if len(sentences) < k:
        k = len(sentences)

    subset = random.sample(sentences, k)
    graph, indegree = build_order_graph(subset, cache_file=cache_file)

    order = topo_sort(graph, indegree, len(subset))
    if order is None:
        order = sorted(range(len(subset)), key=lambda i: subset[i][0])

    predicted = timeline_from_sorted(subset, order)
    groundtruth = parse_groundtruth_file(groundtruth_file)

    evaluator = rouge.TimelineRougeEvaluator(measures=["rouge_1"])
    concat = evaluator.evaluate_concat(predicted, groundtruth)
    align = evaluator.evaluate_align_date_content_costs(predicted, groundtruth)

    return concat, align
