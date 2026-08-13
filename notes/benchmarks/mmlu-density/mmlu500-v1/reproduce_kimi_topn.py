#!/usr/bin/env python3
"""Score the frozen MMLU-500 bank through llama-server top-N logprobs."""

import argparse
import hashlib
import json
import math
import sys
import urllib.request
from pathlib import Path

ITEMS_SHA256 = "df6704c4d02550b9155e106bc9a9e1bfe1164a663d509e41a76736bb60d01ded"
CANDIDATES = (32, 33, 34, 35)

p = argparse.ArgumentParser()
p.add_argument("--url", default="http://127.0.0.1:8080")
p.add_argument("--top-n", type=int, default=100)
a = p.parse_args()

items = Path(__file__).with_name("items.jsonl")
assert hashlib.sha256(items.read_bytes()).hexdigest() == ITEMS_SHA256
rows = [json.loads(line) for line in items.open()]
correct = 0
ce = 0.0

def post(path, body):
    req = urllib.request.Request(
        a.url.rstrip("/") + path,
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as response:
        return json.load(response)


for n, row in enumerate(rows, 1):
    tokens = post("/tokenize", {"content": row["prompt"], "add_special": False})["tokens"]
    data = post("/completion", {"prompt": tokens, "n_predict": 1, "n_probs": a.top_n})
    top = data["completion_probabilities"][0]["top_logprobs"]
    scores = {entry["id"]: entry["logprob"] for entry in top}
    missing = set(CANDIDATES) - scores.keys()
    if missing:
        raise RuntimeError(f"row {n}: token IDs {sorted(missing)} are outside top {a.top_n}")
    logits = [scores[token] for token in CANDIDATES]
    gold = row["answer_index"]
    correct += max(range(4), key=logits.__getitem__) == gold
    peak = max(logits)
    ce -= math.log2(math.exp(logits[gold] - peak) / sum(math.exp(x - peak) for x in logits))
    print(f"{n}/500 correct={correct}", file=sys.stderr, flush=True)

print(json.dumps({"correct": correct, "total": 500, "gold_ce_bits": ce / 500}, indent=2))
