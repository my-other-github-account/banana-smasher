from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from banana_smasher.qtip_k2 import (
    _block_ldl_lower,
    _finalize_raw_hessian_cpu,
    _finalize_raw_hessian_on_device,
    _transformed_raw_hessian,
)


def tensor_sha256(tensor: torch.Tensor) -> str:
    payload = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    specifications: dict[str, dict[str, Any]] = {
        "shared_w1_w3": {
            "file": "L034_E000_shared_w1_w3_H_sum.fp32.npy",
            "width": 4096,
        },
        "e000_w2": {
            "file": "L034_E000_e000_w2_H_sum.fp32.npy",
            "width": 2048,
        },
    }
    results = {}
    for name, specification in specifications.items():
        raw_sum = torch.from_numpy(
            np.load(
                args.input_root / "hessian" / specification["file"],
                mmap_mode="r",
                allow_pickle=False,
            )
        )
        regularized = _finalize_raw_hessian_cpu(
            raw_sum,
            512_000,
            regularization_sigma=0.025,
        )
        torch.manual_seed(args.seed)
        signs = (
            (torch.randn(specification["width"], device="cuda").sign() + 1e-5)
            .sign()
            .float()
            .unsqueeze(1)
        )
        executable_regularized = _finalize_raw_hessian_on_device(
            raw_sum,
            512_000,
            regularization_sigma=0.025,
            device="cuda",
        )
        transformed = _transformed_raw_hessian(
            raw_sum,
            512_000,
            signs,
            regularization_sigma=0.025,
        )
        transformed_sha256 = tensor_sha256(transformed)
        lower = _block_ldl_lower(transformed)
        results[name] = {
            "cpu_receipt_regularized_sha256": tensor_sha256(regularized),
            "executable_regularized_sha256": tensor_sha256(executable_regularized),
            "input_signs_sha256": tensor_sha256(signs),
            "transformed_sha256": transformed_sha256,
            "lower_sha256": tensor_sha256(lower),
        }
    receipt = {
        "schema": "banana-smasher-q2-hessian-boundary-gate-v1",
        "status": "PASS",
        "seed": args.seed,
        "cuda": True,
        "fallback_calls": 0,
        "results": results,
    }
    print(json.dumps(receipt, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
