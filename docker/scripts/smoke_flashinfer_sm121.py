#!/usr/bin/env python3
from __future__ import annotations

import importlib.metadata
import json
import os

import torch

import flashinfer
from flashinfer.autotuner import AutoTuner  # type: ignore[import-not-found]
from flashinfer.autotuner import autotune as flashinfer_autotune  # type: ignore[import-not-found]
from flashinfer.decode import (  # type: ignore[import-not-found]
    trtllm_batch_decode_sparse_mla_dsv4,
    trtllm_batch_decode_with_kv_cache_mla,
)


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


def _run_sparse_mla(device: torch.device, num_tokens: int) -> list[int]:
    num_heads = 64
    swa_topk, extra_topk = 128, 512
    page_block_size, extra_page_block_size, bytes_per_token = 64, 64, 584
    num_swa_blocks = 2
    num_extra_blocks = extra_topk // extra_page_block_size
    assert num_swa_blocks * page_block_size >= swa_topk
    assert num_extra_blocks * extra_page_block_size >= extra_topk
    swa_kv_cache = torch.zeros(
        (num_swa_blocks, page_block_size, 1, bytes_per_token),
        dtype=torch.uint8,
        device=device,
    )
    compressed_kv_cache = torch.zeros(
        (num_extra_blocks, extra_page_block_size, 1, bytes_per_token),
        dtype=torch.uint8,
        device=device,
    )
    sparse_query = torch.zeros(
        (num_tokens, num_heads, 512),
        dtype=torch.bfloat16,
        device=device,
    )
    sparse_indices = torch.arange(
        swa_topk, dtype=torch.int32, device=device
    ).repeat(num_tokens, 1)
    extra_sparse_indices = torch.arange(
        extra_topk, dtype=torch.int32, device=device
    ).repeat(num_tokens, 1, 1)
    swa_topk_lens = torch.full(
        (num_tokens,), swa_topk, dtype=torch.int32, device=device
    )
    extra_sparse_topk_lens = torch.full(
        (num_tokens,), extra_topk, dtype=torch.int32, device=device
    )
    sparse_out = torch.empty_like(sparse_query)
    sinks = torch.zeros((num_heads,), dtype=torch.float32, device=device)
    sparse = flashinfer.decode.trtllm_batch_decode_sparse_mla_dsv4(
        query=sparse_query,
        swa_kv_cache=swa_kv_cache,
        workspace_buffer=torch.zeros(
            128 * 1024 * 1024, dtype=torch.uint8, device=device
        ),
        sparse_indices=sparse_indices,
        compressed_kv_cache=compressed_kv_cache,
        out=sparse_out,
        swa_topk_lens=swa_topk_lens,
        extra_sparse_indices=extra_sparse_indices,
        extra_sparse_topk_lens=extra_sparse_topk_lens,
        bmm1_scale=512**-0.5,
        sinks=sinks,
        kv_layout="NHD",
    )
    _synchronize(f"sparse_mla_sm120_tokens_{num_tokens}")
    expected_sparse_shape = (num_tokens, num_heads, 512)
    if tuple(sparse.shape) != expected_sparse_shape or not torch.isfinite(sparse).all().item():
        raise RuntimeError(
            f"invalid sparse MLA output: shape={tuple(sparse.shape)} "
            f"expected={expected_sparse_shape}"
        )
    return list(sparse.shape)


def main() -> None:
    if os.environ.get("FLASHINFER_DISABLE_JIT") != "1":
        raise RuntimeError("FLASHINFER_DISABLE_JIT=1 is required for the AOT smoke")
    if os.environ.get("VLLM_HAS_FLASHINFER_CUBIN") != "1":
        raise RuntimeError("VLLM_HAS_FLASHINFER_CUBIN=1 is required for the AOT smoke")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if not callable(flashinfer_autotune):
        raise RuntimeError("FlashInfer autotune import gate is unavailable")
    if not callable(AutoTuner):
        raise RuntimeError("FlashInfer AutoTuner boot import is unavailable")
    if not callable(trtllm_batch_decode_sparse_mla_dsv4):
        raise RuntimeError("FlashInfer DSv4 sparse MLA import gate is unavailable")
    if not callable(trtllm_batch_decode_with_kv_cache_mla):
        raise RuntimeError("FlashInfer MLA decode import gate is unavailable")

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

    sparse_mla_shapes = {}
    for num_tokens in (1, 16):
        sparse_mla_shapes[str(num_tokens)] = _run_sparse_mla(device, num_tokens)

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
                    "sparse_mla_sm120_warmup16",
                ],
                "sampling_results": sampling_results,
                "head512_shape": list(head512.shape),
                "sparse_mla_shape": sparse_mla_shapes["1"],
                "sparse_mla_shapes": sparse_mla_shapes,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
