#!/usr/bin/env python3
import json
import urllib.request
from pathlib import Path

rows = [json.loads(x) for x in Path(__file__).with_name("items.jsonl").open()]
request = urllib.request.Request(
    "http://127.0.0.1:8080/completion",
    json.dumps({"prompt": [x["prompt"] for x in rows], "n_predict": 1, "n_probs": 100}).encode(),
    {"Content-Type": "application/json"},
)
answers = json.load(urllib.request.urlopen(request))
correct = 0
for row, answer in zip(rows, answers, strict=True):
    top = answer["completion_probabilities"][0]["top_logprobs"]
    scores = {x["id"]: x["logprob"] for x in top}
    correct += max(range(4), key=lambda x: scores[32 + x]) == row["answer_index"]
print(f"{correct}/{len(rows)}")
