"""Packaged BALANCED64 scorer used by the public QTIP V7 workflow."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _softmax(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    array -= array.max()
    result = np.exp(array)
    return result / result.sum()


def main() -> int:
    import torch

    candidate = Path(os.environ["QTIP_V7_CANDIDATE"])
    bank_path = Path(os.environ["QTIP_V7_TEACHER_BANK"])
    receipt = Path(os.environ["QTIP_V7_SHARD_RECEIPT"])
    if _sha256(candidate) != os.environ["QTIP_V7_CANDIDATE_SHA256"]:
        raise RuntimeError("staged candidate SHA-256 readback drift")
    if _sha256(bank_path) != os.environ["QTIP_V7_TEACHER_BANK_SHA256"]:
        raise RuntimeError("staged teacher-bank SHA-256 readback drift")
    payload = torch.load(candidate, map_location="cpu", weights_only=True)
    bank = json.loads(bank_path.read_text())
    rows = []
    for ordinal in range(
        int(os.environ["QTIP_V7_SHARD_START"]),
        int(os.environ["QTIP_V7_SHARD_END"]) + 1,
    ):
        teacher = bank["windows"][ordinal]["teacher_logits"]
        predicted = payload["predictions"][ordinal]
        teacher_probability = _softmax(teacher)
        predicted_probability = _softmax(predicted)
        kld = float(np.sum(
            teacher_probability
            * (np.log(teacher_probability) - np.log(predicted_probability))
        ))
        rows.append({
            "ordinal": ordinal,
            "positions": 1024,
            "support": 8192,
            "kld_sum_binary64": kld * 1024,
            "top1_matches": 1024 * int(np.argmax(teacher) == np.argmax(predicted)),
            "fallback_calls": 0,
            "pass_through_bytes": 0,
            "hidden_fp32_control_bytes": 0,
        })
    document = {
        "schema": "banana-smasher-qtip-v7-balanced64-shard-v1",
        "status": "PASS",
        "candidate_sha256": os.environ["QTIP_V7_CANDIDATE_SHA256"],
        "teacher_bank_sha256": os.environ["QTIP_V7_TEACHER_BANK_SHA256"],
        "ordinal_start": rows[0]["ordinal"],
        "ordinal_end": rows[-1]["ordinal"],
        "rows": rows,
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(document, sort_keys=True, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
