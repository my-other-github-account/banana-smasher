#!/usr/bin/env python3
from __future__ import annotations

import importlib.metadata
import json
import os

import torch

import flashinfer


def _synchronize(name: str) -> None:
    try:
        torch.cuda.synchronize()
    except Exception as exc:
        raise RuntimeError(f"FlashInfer {name} execution failed: {exc}") from exc


def _checked_sample(name: str, sample: torch.Tensor) -> list[int]:
    _synchronize(name)
    result = sample.tolist()
    if result != [1]:
        raise RuntimeError(f"unexpected {name} result: {result}")
    return result


def main() -> None:
    if os.environ.get("FLASHINFER_DISABLE_JIT") != "1":
        raise RuntimeError("FLASHINFER_DISABLE_JIT=1 is required for the AOT smoke")
    if os.environ.get("VLLM_HAS_FLASHINFER_CUBIN") != "1":
        raise RuntimeError("VLLM_HAS_FLASHINFER_CUBIN=1 is required for the AOT smoke")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability(device)
    if capability != (12, 1):
        raise RuntimeError(f"expected SM121, got compute capability {capability}")

    logits = torch.tensor([[-1000.0, 0.0]], dtype=torch.float32, device=device)
    probs = logits.softmax(dim=-1)
    top_k = torch.tensor([1], dtype=torch.int32, device=device)
    top_p = torch.tensor([1.0], dtype=torch.float32, device=device)
    sampling_results = {
        "top_p_sampling_from_probs": _checked_sample(
            "top_p_sampling_from_probs",
            flashinfer.sampling.top_p_sampling_from_probs(
                probs, top_p, deterministic=True
            ),
        ),
        "top_k_sampling_from_probs": _checked_sample(
            "top_k_sampling_from_probs",
            flashinfer.sampling.top_k_sampling_from_probs(
                probs, top_k, deterministic=True
            ),
        ),
        "top_k_top_p_sampling_from_logits": _checked_sample(
            "top_k_top_p_sampling_from_logits",
            flashinfer.sampling.top_k_top_p_sampling_from_logits(
                logits, top_k, top_p, deterministic=True
            ),
        ),
    }

    q = torch.zeros((1, 1, 512), dtype=torch.bfloat16, device=device)
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    head512 = flashinfer.single_prefill_with_kv_cache(q, k, v, backend="fa2")
    _synchronize("FA2 head512")
    if tuple(head512.shape) != (1, 1, 512) or not torch.isfinite(head512).all().item():
        raise RuntimeError(f"invalid FA2 head512 output: shape={tuple(head512.shape)}")

    num_tokens, num_heads, topk = 1, 8, 128
    page_block_size, bytes_per_token = 64, 584
    kv_cache = torch.zeros(
        (2, page_block_size, 1, bytes_per_token),
        dtype=torch.uint8,
        device=device,
    )
    sparse_query = torch.zeros(
        (num_tokens, 1, num_heads, 512),
        dtype=torch.bfloat16,
        device=device,
    )
    sparse_indices = torch.arange(topk, dtype=torch.int32, device=device).reshape(1, topk)
    sparse = flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4(
        query=sparse_query,
        swa_kv_cache=kv_cache,
        workspace_buffer=torch.empty(1, dtype=torch.int8, device=device),
        sparse_indices=sparse_indices,
        swa_topk_lens=torch.full(
            (num_tokens,), topk, dtype=torch.int32, device=device
        ),
        bmm1_scale=512**-0.5,
        kv_layout="NHD",
    )
    _synchronize("sparse_mla_sm120")
    expected_sparse_shape = (num_tokens, 1, num_heads, 512)
    if tuple(sparse.shape) != expected_sparse_shape or not torch.isfinite(sparse).all().item():
        raise RuntimeError(
            f"invalid sparse MLA output: shape={tuple(sparse.shape)} "
            f"expected={expected_sparse_shape}"
        )

    print(
        json.dumps(
            {
                "status": "PASS",
                "compute_capability": list(capability),
                "flashinfer_python": importlib.metadata.version("flashinfer-python"),
                "flashinfer_jit_cache": importlib.metadata.version(
                    "flashinfer-jit-cache"
                ),
                "executed": [
                    "top_p_sampling_from_probs",
                    "top_k_sampling_from_probs",
                    "top_k_top_p_sampling_from_logits",
                    "fa2_head512",
                    "sparse_mla_sm120",
                ],
                "sampling_results": sampling_results,
                "head512_shape": list(head512.shape),
                "sparse_mla_shape": list(sparse.shape),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
