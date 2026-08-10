from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .q2_codec import MUL1_MULTIPLIER, tensor_core_permutation


_CODEBOOK_SCALE = 1.24371088


def _require_same_device(*tensors: Any) -> None:
    devices = {tensor.device for tensor in tensors}
    if len(devices) != 1:
        raise ValueError("K2 tensors must share one device")


def pack_k2(encoded: Any) -> Any:
    """Pack the low two transition bits from uint16 trellis states."""
    import torch

    if encoded.ndim < 1 or encoded.shape[-1] != 256 or encoded.dtype != torch.int16:
        raise ValueError("encoded trellis must be int16[..., 256]")
    codes = (encoded.to(torch.int32) & 3).reshape(*encoded.shape[:-1], 16, 2, 8)
    shifts = torch.arange(14, -1, -2, dtype=torch.int32, device=encoded.device)
    words = torch.sum(codes << shifts, dim=-1)
    words = words[..., [1, 0]].reshape(*encoded.shape[:-1], 32)
    return words.to(torch.int16)


def unpack_k2(packed: Any) -> Any:
    """Recover cyclic uint16 states from a packed two-bit transition stream."""
    import torch

    if packed.ndim < 1 or packed.shape[-1] != 32 or packed.dtype != torch.int16:
        raise ValueError("packed K2 trellis must be int16[..., 32]")
    words = (packed.to(torch.int32) & 0xFFFF).reshape(*packed.shape[:-1], 16, 2)
    words = words[..., [1, 0]]
    shifts = torch.arange(14, -1, -2, dtype=torch.int32, device=packed.device)
    codes = ((words.unsqueeze(-1) >> shifts) & 3).reshape(*packed.shape[:-1], 256)
    state = torch.zeros(packed.shape[:-1], dtype=torch.int64, device=packed.device)
    for position in range(248, 256):
        state = ((state << 2) | codes[..., position]) & 0xFFFF
    states = torch.empty_like(codes, dtype=torch.int16)
    for position in range(256):
        state = ((state << 2) | codes[..., position]) & 0xFFFF
        states[..., position] = state.to(torch.int16)
    return states


def decode_k2_states(states: Any, parent_lut: Any) -> Any:
    """Decode states procedurally through the exact FP16[1024] parent LUT."""
    import torch

    if states.dtype != torch.int16:
        raise ValueError("states must be int16")
    if parent_lut.dtype != torch.float16 or parent_lut.numel() != 1024:
        raise ValueError("parent_lut must be float16[1024]")
    _require_same_device(states, parent_lut)
    values = states.to(torch.int64) & 0xFFFF
    products = (values * MUL1_MULTIPLIER) & 0xFFFFFFFF
    parents = (
        (products & 0xFF)
        + ((products >> 8) & 0xFF)
        + ((products >> 16) & 0xFF)
        + ((products >> 24) & 0xFF)
    )
    return parent_lut[parents].to(torch.float32)


def decode_k2_matrix(packed: Any, parent_lut: Any) -> Any:
    """Decode packed tensor-core-order K2 tiles to an input-by-output matrix."""
    import torch

    if packed.ndim != 3:
        raise ValueError("packed K2 matrix must be int16[tiles_k, tiles_n, 32]")
    _require_same_device(packed, parent_lut)
    states = unpack_k2(packed)
    decoded = decode_k2_states(states, parent_lut)
    inverse = torch.argsort(
        torch.from_numpy(tensor_core_permutation()).to(device=packed.device, dtype=torch.int64)
    )
    decoded = decoded[..., inverse]
    tiles_k, tiles_n, _ = decoded.shape
    return decoded.reshape(tiles_k, tiles_n, 16, 16).permute(0, 2, 1, 3).reshape(
        tiles_k * 16, tiles_n * 16
    )


def normalized_hadamard_128(device: Any, dtype: Any) -> Any:
    import torch

    matrix = torch.ones((1, 1), dtype=dtype, device=device)
    while matrix.shape[0] < 128:
        matrix = torch.cat(
            (torch.cat((matrix, matrix), dim=1), torch.cat((matrix, -matrix), dim=1)),
            dim=0,
        )
    matrix *= 1.0 / math.sqrt(128)
    return matrix


def inverse_transform(weight: Any, su: Any, sv: Any) -> Any:
    """Apply the self-inverse H128 transforms and channel scales."""
    import torch

    if weight.ndim != 2 or weight.dtype != torch.float32:
        raise ValueError("weight must be a float32 matrix")
    matrix = weight
    hadamard = normalized_hadamard_128(matrix.device, matrix.dtype)
    matrix = (hadamard @ matrix.reshape(-1, 128, matrix.shape[1])).reshape_as(matrix)
    matrix *= su.reshape(-1, 1).to(device=matrix.device, dtype=matrix.dtype)
    matrix = (matrix.reshape(matrix.shape[0], -1, 128) @ hadamard).reshape_as(matrix)
    matrix *= sv.reshape(1, -1).to(device=matrix.device, dtype=matrix.dtype)
    return matrix


def _tensor_sha256(tensor: Any) -> str:
    import torch

    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _hadamard_last(tensor: Any, block: int = 128) -> Any:
    import torch

    if tensor.shape[-1] % block:
        raise ValueError("K2 Hadamard dimension must be divisible by 128")
    matrix = normalized_hadamard_128(tensor.device, torch.float32)
    shape = tensor.shape
    return (tensor.float().reshape(-1, block) @ matrix).reshape(shape)


def _block_rms(tensor: Any, *, dim: int, blocksize: int = 32) -> Any:
    import torch

    count = tensor.size(dim)
    square_sum = None
    for block in torch.split(tensor, blocksize, dim=dim):
        block_sum = block.square().sum(dim=dim, keepdim=True)
        square_sum = block_sum if square_sum is None else square_sum + block_sum
    return (square_sum / count).sqrt()


def _sample_scale_tiles(weight: Any, width: int = 3) -> Any:
    import torch

    tiles_k = weight.shape[0] // 16
    tiles_n = weight.shape[1] // 16
    tiled = weight.view(tiles_k, 16, tiles_n, 16)
    diagonal_length = max(tiles_k, tiles_n)
    indices = torch.arange(diagonal_length, device=weight.device).repeat_interleave(
        width
    )
    offsets = torch.arange(width, device=weight.device).repeat(diagonal_length)
    rows = indices % tiles_k
    columns = (indices + offsets) % tiles_n
    extreme_count = max(8, (diagonal_length * width) // 16)
    tile_ms = tiled.square().mean(dim=(1, 3)).flatten()
    extreme_count = min(extreme_count, (tile_ms.shape[0] + 1) // 2)
    high = torch.topk(tile_ms, extreme_count).indices
    low = torch.topk(tile_ms, extreme_count, largest=False).indices
    extreme = torch.cat((high, low))
    rows = torch.cat((rows, extreme // tiles_n))
    columns = torch.cat((columns, extreme % tiles_n))
    tiles = tiled[rows, :, columns, :].reshape(-1, 256)
    permutation = torch.from_numpy(tensor_core_permutation()).to(
        device=weight.device, dtype=torch.int64
    )
    return tiles[:, permutation].contiguous()


def _global_scale(
    samples: Any,
    parent_lut: Any,
    quantize_tiles_fn: Callable[[Any, Any], tuple[Any, ...]],
) -> float:
    import torch

    def evaluate(scales: list[float], tiles: Any) -> list[float]:
        count = tiles.shape[0]
        batch = torch.cat([tiles * scale for scale in scales])
        quantized, _ = quantize_tiles_fn(batch, parent_lut)
        if quantized.shape != batch.shape:
            raise ValueError("K2 scale-search quantizer changed tile geometry")
        errors = []
        for index, scale in enumerate(scales):
            restored = quantized[index * count : (index + 1) * count] / scale
            errors.append(float((restored - tiles).square().mean().item()))
        return errors

    coarse = [0.1 + 0.2 * index for index in range(10)]
    coarse_errors = evaluate(coarse, samples[::3])
    center = coarse[min(range(len(coarse)), key=coarse_errors.__getitem__)]
    step = 0.075
    fine = [center + step * (index - 2) for index in range(5)]
    fine_errors = evaluate(fine, samples)
    best = min(range(len(fine)), key=fine_errors.__getitem__)
    if 0 < best < len(fine) - 1:
        y0, y1, y2 = fine_errors[best - 1 : best + 2]
        denominator = y0 - 2.0 * y1 + y2
        offset = 0.5 * (y0 - y2) / denominator if denominator > 0 else 0.0
        offset = max(-0.5, min(0.5, offset))
    else:
        offset = 0.0
    return max(fine[best] + offset * step, 0.01)


def _source_transform(
    source_out_in: Any,
    parent_lut: Any,
    *,
    seed: int,
    quantize_tiles_fn: Callable[[Any, Any], tuple[Any, ...]],
    input_signs: Any | None = None,
    output_signs: Any | None = None,
) -> dict[str, Any]:
    import torch

    if not isinstance(source_out_in, torch.Tensor) or source_out_in.ndim != 2:
        raise ValueError("K2 source weight must be a two-dimensional Torch tensor")
    rows, columns = source_out_in.shape
    if rows % 128 or columns % 128:
        raise ValueError("K2 source weight dimensions must be divisible by 128")
    weight = source_out_in.transpose(0, 1).float().contiguous()
    size_k, size_n = weight.shape
    if input_signs is None or output_signs is None:
        if input_signs is not None or output_signs is not None:
            raise ValueError("K2 source transform signs must be supplied together")
        fork_devices = [] if weight.device.type == "cpu" else [weight.device]
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(seed)
            input_signs = _draw_signs(size_k, weight.device).unsqueeze(1)
            output_signs = _draw_signs(size_n, weight.device).unsqueeze(0)
    elif (
        input_signs.shape != (size_k, 1)
        or output_signs.shape != (1, size_n)
        or input_signs.dtype != torch.float32
        or output_signs.dtype != torch.float32
        or input_signs.device != weight.device
        or output_signs.device != weight.device
    ):
        raise ValueError("K2 source transform signs have incompatible geometry")

    output_scales = _block_rms(weight, dim=0)
    mean = float(output_scales.mean().item())
    if mean <= 1e-30:
        raise ValueError("K2 source weight has no nonzero output scale")
    output_scales /= mean
    zero_outputs = output_scales.abs() < 1e-30
    output_scales[zero_outputs] = 0.1
    output_transform = (output_signs * output_scales + 1e-10).float()
    weight /= output_transform
    output_transform[zero_outputs] = 0.0
    hadamard = normalized_hadamard_128(weight.device, torch.float32)
    for start in range(0, size_n, 128):
        stop = start + 128
        weight[:, start:stop].copy_(weight[:, start:stop] @ hadamard)

    input_scales = _block_rms(weight, dim=1)
    input_scales[input_scales.abs() < 1e-30] = 0.1
    input_transform = (
        input_signs * input_scales / (-_CODEBOOK_SCALE) + 1e-10
    ).float()
    weight /= input_transform
    for start in range(0, size_k, 128):
        stop = start + 128
        weight[start:stop, :].copy_(hadamard @ weight[start:stop, :])
    global_scale = _global_scale(
        _sample_scale_tiles(weight), parent_lut, quantize_tiles_fn
    )
    su = (input_transform / global_scale).flatten().contiguous()
    sv = output_transform.flatten().contiguous()
    return {
        "target_inner": (weight * global_scale).contiguous(),
        "su": su,
        "sv": sv,
        "suh": su.half().contiguous(),
        "svh": sv.half().contiguous(),
        "global_scale": global_scale,
        "input_signs": input_signs,
        "output_signs": output_signs,
    }


def _draw_signs(size: int, device: Any) -> Any:
    import torch

    return (torch.randn(size, device=device).sign() + 1e-5).sign().float()


def _transformed_hessian(
    calibration_samples: Any,
    input_signs: Any,
    *,
    sample_weights: Any | None,
    regularization_sigma: float,
) -> tuple[Any, int, float, int]:
    import torch

    batches = (
        [calibration_samples]
        if isinstance(calibration_samples, torch.Tensor)
        else list(calibration_samples)
    )
    if not batches:
        raise ValueError("K2 calibration requires ordered capture batches")
    if sample_weights is None:
        weight_batches = [None] * len(batches)
    elif isinstance(sample_weights, torch.Tensor):
        weight_batches = [sample_weights]
    else:
        weight_batches = list(sample_weights)
    if len(weight_batches) != len(batches):
        raise ValueError("K2 calibration weights must match ordered captures")

    hessian = torch.zeros(
        (input_signs.shape[0], input_signs.shape[0]),
        dtype=torch.float32,
        device=input_signs.device,
    )
    total_rows = 0
    total_mass = 0.0
    for batch, batch_weights in zip(batches, weight_batches):
        samples = batch.to(
            device=input_signs.device, dtype=torch.float32
        ).contiguous()
        if samples.ndim != 2 or samples.shape[1] != input_signs.shape[0]:
            raise ValueError("K2 calibration batch has incompatible geometry")
        if batch_weights is None:
            mass = float(samples.shape[0])
            hessian.addmm_(samples.T, samples)
        else:
            weights = batch_weights.to(
                device=input_signs.device, dtype=torch.float32
            ).contiguous()
            if weights.shape != (samples.shape[0],):
                raise ValueError("K2 calibration weights have incompatible geometry")
            mass = float(weights.double().sum().item())
            hessian.addmm_(samples.T, samples * weights[:, None])
        total_rows += int(samples.shape[0])
        total_mass += mass
    if not np.isfinite(total_mass) or total_mass <= 0:
        raise ValueError("K2 calibration weight mass must be positive")
    hessian /= total_mass
    diagonal_mean = float(hessian.diagonal().mean().item())
    if not np.isfinite(diagonal_mean) or diagonal_mean <= 0:
        raise ValueError("K2 calibration Hessian diagonal mean must be positive")
    hessian.diagonal().add_(regularization_sigma * diagonal_mean)
    hessian *= input_signs.T
    hessian = _hadamard_last(hessian)
    hessian *= input_signs
    hessian = _hadamard_last(hessian.T).T.contiguous()
    return hessian, total_rows, total_mass, len(batches)


def _finalize_raw_hessian_cpu(
    raw_hessian_sum: Any,
    raw_hessian_count: int,
    *,
    regularization_sigma: float,
) -> Any:
    import torch

    if raw_hessian_count <= 0:
        raise ValueError("K2 raw Hessian count must be positive")
    if raw_hessian_sum.device.type != "cpu":
        raise ValueError("K2 raw Hessian sum must be finalized on CPU")
    if raw_hessian_sum.ndim != 2 or raw_hessian_sum.shape[0] != raw_hessian_sum.shape[1]:
        raise ValueError("K2 raw Hessian sum must be square")
    hessian = raw_hessian_sum.detach().to(dtype=torch.float32).clone().contiguous()
    hessian /= raw_hessian_count
    diagonal_mean = float(torch.diag(hessian).mean().item())
    if not np.isfinite(diagonal_mean) or diagonal_mean <= 0:
        raise ValueError("K2 raw Hessian diagonal mean must be positive")
    hessian.diagonal().add_(regularization_sigma * diagonal_mean)
    return hessian


def _finalize_raw_hessian_on_device(
    raw_hessian_sum: Any,
    raw_hessian_count: int,
    *,
    regularization_sigma: float,
    device: Any,
) -> Any:
    """Replay executable K2 Hessian finalization on the assignment device."""
    import torch

    if raw_hessian_count <= 0:
        raise ValueError("K2 raw Hessian count must be positive")
    if raw_hessian_sum.ndim != 2 or raw_hessian_sum.shape[0] != raw_hessian_sum.shape[1]:
        raise ValueError("K2 raw Hessian sum must be square")
    hessian = (
        raw_hessian_sum.detach()
        .to(device=device, dtype=torch.float32)
        .clone()
        .contiguous()
    )
    hessian /= raw_hessian_count
    diagonal_mean = torch.diag(hessian).mean().item()
    if not np.isfinite(diagonal_mean) or diagonal_mean <= 0:
        raise ValueError("K2 raw Hessian diagonal mean must be positive")
    hessian.diagonal().add_(regularization_sigma * diagonal_mean)
    return hessian


def _transform_regularized_hessian(hessian: Any, input_signs: Any) -> Any:
    import torch

    block = 128
    hadamard = normalized_hadamard_128(hessian.device, torch.float32)
    hessian *= input_signs.T
    for start in range(0, hessian.shape[1], block):
        stop = start + block
        transformed = hessian[:, start:stop] @ hadamard
        hessian[:, start:stop].copy_(transformed)
    hessian *= input_signs
    for start in range(0, hessian.shape[0], block):
        stop = start + block
        transformed = hadamard @ hessian[start:stop, :]
        hessian[start:stop, :].copy_(transformed)
    return hessian


def _transformed_raw_hessian(
    raw_hessian_sum: Any,
    raw_hessian_count: int,
    input_signs: Any,
    *,
    regularization_sigma: float,
) -> Any:
    expected = (input_signs.shape[0], input_signs.shape[0])
    if raw_hessian_sum.shape != expected:
        raise ValueError("K2 raw Hessian sum has incompatible geometry")
    hessian = _finalize_raw_hessian_on_device(
        raw_hessian_sum,
        raw_hessian_count,
        regularization_sigma=regularization_sigma,
        device=input_signs.device,
    )
    return _transform_regularized_hessian(hessian, input_signs)


def _block_ldl_lower(hessian: Any, block_rows: int = 16) -> Any:
    """Build the normalized block-LDL feedback factor."""
    import torch

    size = int(hessian.shape[0])
    blocks = size // block_rows
    factor = torch.linalg.cholesky(hessian)
    hessian.copy_(factor)
    lower = hessian
    diagonal_blocks = torch.diagonal(
        lower.reshape(blocks, block_rows, blocks, block_rows), dim1=0, dim2=2
    ).permute(2, 0, 1)
    inverse_blocks = torch.linalg.inv(diagonal_blocks)
    lower = lower.view(size, blocks, block_rows)
    for index in range(blocks):
        lower[:, index, :] = lower[:, index, :] @ inverse_blocks[index]
    lower = lower.reshape(size, size).contiguous()
    lower_blocks = lower.view(
        blocks, block_rows, blocks, block_rows
    ).permute(0, 2, 1, 3)
    diagonal = torch.arange(blocks)
    lower_blocks[diagonal, diagonal] = torch.stack(
        [torch.eye(block_rows, dtype=lower.dtype, device=lower.device)] * blocks
    )
    lower.diagonal().zero_()
    return lower


def _block_trace(error: Any, hessian: Any, block_size: int = 1024) -> float:
    """Compute trace(error.T @ hessian @ error) in bounded column blocks."""
    import torch

    total = 0.0
    for start in range(0, hessian.shape[1], block_size):
        stop = min(start + block_size, hessian.shape[1])
        hessian_block = hessian[:, start:stop]
        error_block = error[start:stop, :]
        partial = torch.einsum(
            "ik,ij,jk->", error, hessian_block, error_block
        )
        total += partial.item()
    return total


@lru_cache(maxsize=1)
def _cuda_extension() -> Any:
    from torch.utils.cpp_extension import CUDA_HOME, load

    source = Path(__file__).with_name("qtip_k2_kernel.cu")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    extra_include_paths = []
    if CUDA_HOME:
        target_includes = sorted(Path(CUDA_HOME).glob("targets/*/include"))
        extra_include_paths = [
            str(path) for path in target_includes if (path / "cusparse.h").is_file()
        ]
    configured_includes = os.environ.get("BANANA_CUDA_INCLUDE_PATH", "")
    extra_include_paths.extend(
        path
        for path in configured_includes.split(os.pathsep)
        if path and (Path(path) / "cusparse.h").is_file()
    )
    return load(
        name=f"banana_qtip_k2_{digest}",
        sources=[str(source)],
        extra_cuda_cflags=["-O3"],
        extra_include_paths=extra_include_paths,
        with_cuda=True,
        verbose=False,
    )


def quantize_k2_tiles(
    tiles: Any,
    parent_lut: Any,
    *,
    chunk_tiles: int | None = None,
    return_edges: bool = False,
) -> tuple[Any, Any] | tuple[Any, Any, Any]:
    """Quantize CUDA float32[tiles,256] with strict-less FP16 K2 recurrence."""
    import torch

    if not tiles.is_cuda or tiles.dtype != torch.float32 or tiles.ndim != 2 or tiles.shape[1] != 256:
        raise ValueError("tiles must be CUDA float32[tiles, 256]")
    if not parent_lut.is_cuda or parent_lut.dtype != torch.float16 or parent_lut.numel() != 1024:
        raise ValueError("parent_lut must be CUDA float16[1024]")
    _require_same_device(tiles, parent_lut)
    if chunk_tiles is None:
        multiprocessors = torch.cuda.get_device_properties(tiles.device).multi_processor_count
        chunk_tiles = min(256, 2 * multiprocessors)
    if chunk_tiles < 1:
        raise ValueError("chunk_tiles must be positive")
    extension = _cuda_extension()
    quantized_parts = []
    index_parts = []
    edge_parts = []
    for start in range(0, tiles.shape[0], chunk_tiles):
        stop = min(start + chunk_tiles, tiles.shape[0])
        quantized, indices, edges = extension.quantize(
            tiles[start:stop].contiguous(), parent_lut.contiguous()
        )
        quantized_parts.append(quantized)
        index_parts.append(indices)
        if return_edges:
            edge_parts.append(edges)
    quantized = torch.cat(quantized_parts, dim=0)
    indices = torch.cat(index_parts, dim=0)
    if return_edges:
        return quantized, indices, torch.cat(edge_parts, dim=0)
    return quantized, indices


def buffered_ldlq(
    weight: Any,
    lower: Any,
    parent_lut: Any,
    *,
    buffer_rows: int = 128,
    quantize_tiles_fn: Callable[[Any, Any], tuple[Any, ...]] = quantize_k2_tiles,
    progress: Callable[[dict[str, int]], None] | None = None,
    resume_state: dict[str, Any] | None = None,
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[Any, Any]:
    """Run the reverse-buffer product-cache/target recurrence and K2 quantizer."""
    import torch

    if (
        weight.ndim != 2
        or lower.shape != (weight.shape[0], weight.shape[0])
        or weight.dtype != torch.float32
        or lower.dtype != torch.float32
        or not weight.is_cuda
        or not lower.is_cuda
    ):
        raise ValueError("weight/lower must be compatible CUDA float32 matrices")
    _require_same_device(weight, lower, parent_lut)
    size_k, size_n = weight.shape
    if size_k % buffer_rows or buffer_rows % 16 or size_n % 128:
        raise ValueError("LDLQ geometry must divide 128-row/16-row/128-column boundaries")
    permutation = torch.from_numpy(tensor_core_permutation()).to(
        device=weight.device, dtype=torch.int64
    )
    inverse = torch.argsort(permutation)
    total_buffers = size_k // buffer_rows
    encoded_shape = (size_k // 16, size_n // 16, 256)
    if resume_state is None:
        next_buffer = 0
        product_cache = torch.zeros_like(weight)
        quantized_weight = torch.zeros_like(weight)
        encoded = torch.zeros(encoded_shape, dtype=torch.int16, device=weight.device)
    else:
        next_buffer = int(resume_state["next_buffer"])
        product_cache = resume_state["product_cache"]
        quantized_weight = resume_state["quantized_weight"]
        encoded = resume_state["encoded"]
        if not 0 <= next_buffer <= total_buffers:
            raise ValueError("resume next_buffer is outside the LDLQ frontier")
        if (
            product_cache.shape != weight.shape
            or quantized_weight.shape != weight.shape
            or encoded.shape != encoded_shape
            or product_cache.dtype != torch.float32
            or quantized_weight.dtype != torch.float32
            or encoded.dtype != torch.int16
            or product_cache.device != weight.device
            or quantized_weight.device != weight.device
            or encoded.device != weight.device
        ):
            raise ValueError("resume tensors do not match the LDLQ geometry/device")

    buffer_highs = list(range(size_k, 0, -buffer_rows))
    for buffer_ordinal in range(next_buffer, total_buffers):
        high = buffer_highs[buffer_ordinal]
        low = high - buffer_rows
        buffer_weight = weight[low:high]
        buffer_quantized = quantized_weight[low:high]
        buffer_product = product_cache[low:high]
        buffer_lower = lower[low:high]
        for target_ordinal, target_high in enumerate(range(buffer_rows, 0, -16)):
            target_low = target_high - 16
            error = buffer_weight[target_high:] - buffer_quantized[target_high:]
            lower_slice = buffer_lower[target_high:, low + target_low : low + target_high]
            compensation = buffer_product[target_low:target_high]
            compensation.addmm_(lower_slice.T, error, alpha=1.0, beta=1.0)
            rows = buffer_weight[target_low:target_high] + compensation
            tiles = rows.reshape(16, size_n // 16, 16).permute(1, 0, 2).reshape(-1, 256)
            tiles = tiles[:, permutation]
            quantized_result = quantize_tiles_fn(tiles, parent_lut)
            quantized_tiles, indices = quantized_result[0], quantized_result[1]
            quantized_tiles = quantized_tiles[:, inverse]
            quantized_rows = (
                quantized_tiles.reshape(size_n // 16, 16, 16)
                .permute(1, 0, 2)
                .reshape(16, size_n)
            )
            buffer_quantized[target_low:target_high] = quantized_rows
            encoded[(low + target_low) // 16 : (low + target_high) // 16] = indices.unsqueeze(0)
            if progress is not None:
                progress(
                    {
                        "buffer": buffer_ordinal,
                        "buffers": total_buffers,
                        "target": target_ordinal,
                        "global_low": low + target_low,
                        "global_high": low + target_high,
                    }
                )
        buffer_error = buffer_weight - buffer_quantized
        product_cache.addmm_(buffer_lower.T, buffer_error, alpha=1.0, beta=1.0)
        if checkpoint is not None:
            checkpoint(
                {
                    "next_buffer": buffer_ordinal + 1,
                    "total_buffers": total_buffers,
                    "product_cache": product_cache,
                    "quantized_weight": quantized_weight,
                    "encoded": encoded,
                }
            )
    return quantized_weight, encoded


def assign_k2_source(
    source_out_in: Any,
    calibration_samples: Any | None,
    parent_lut: Any,
    *,
    calibration_sample_weights: Any | None = None,
    raw_hessian_sum: Any | None = None,
    raw_hessian_count: int | None = None,
    seed: int = 0,
    conversion_module_index: int = 0,
    regularization_sigma: float = 0.025,
    progress: Callable[[dict[str, int]], None] | None = None,
    resume_state: dict[str, Any] | None = None,
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
    quantize_tiles_fn: Callable[[Any, Any], tuple[Any, ...]] = quantize_k2_tiles,
) -> dict[str, Any]:
    """Generate one native K2/Q2 member from source weights and TRAIN captures."""
    import torch

    if not source_out_in.is_cuda or source_out_in.ndim != 2:
        raise ValueError("source_out_in must be a two-dimensional CUDA tensor")
    if (
        not parent_lut.is_cuda
        or parent_lut.dtype != torch.float16
        or parent_lut.shape != (1024,)
    ):
        raise ValueError("parent_lut must be CUDA float16[1024]")
    _require_same_device(source_out_in, parent_lut)
    if conversion_module_index < 0:
        raise ValueError("K2 conversion module index must be non-negative")
    effective_transform_seed = seed + conversion_module_index
    counters = {"cuda_calls": 0, "cuda_tiles": 0, "fallback_calls": 0}

    def counted_quantize(tiles: Any, lut: Any) -> tuple[Any, Any]:
        quantized, states = quantize_tiles_fn(tiles, lut)
        counters["cuda_calls"] += 1
        counters["cuda_tiles"] += int(tiles.shape[0])
        return quantized, states

    size_k = int(source_out_in.shape[1])
    size_n = int(source_out_in.shape[0])
    regularized_hessian_sha256 = None
    fork_devices = [] if source_out_in.device.type == "cpu" else [source_out_in.device]
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(effective_transform_seed)
        input_signs = _draw_signs(size_k, source_out_in.device).unsqueeze(1)
        if raw_hessian_sum is None:
            if raw_hessian_count is not None or calibration_samples is None:
                raise ValueError("K2 calibration requires captures or a raw Hessian sum")
            hessian, rows, mass, batches = _transformed_hessian(
                calibration_samples,
                input_signs,
                sample_weights=calibration_sample_weights,
                regularization_sigma=regularization_sigma,
            )
            raw_hessian_sum_sha256 = None
            calibration_mode = "ordered_captures"
        else:
            if calibration_samples is not None or calibration_sample_weights is not None:
                raise ValueError("K2 raw-Hessian mode forbids routed capture inputs")
            if raw_hessian_count is None:
                raise ValueError("K2 raw-Hessian mode requires the source row count")
            raw_hessian_sum_sha256 = _tensor_sha256(raw_hessian_sum)
            regularized_hessian = _finalize_raw_hessian_on_device(
                raw_hessian_sum,
                raw_hessian_count,
                regularization_sigma=regularization_sigma,
                device=source_out_in.device,
            )
            regularized_hessian_sha256 = _tensor_sha256(regularized_hessian)
            hessian = _transform_regularized_hessian(
                regularized_hessian, input_signs
            )
            rows = raw_hessian_count
            mass = float(raw_hessian_count)
            batches = 1
            calibration_mode = "raw_ordered_sum"
        hessian_sha256 = _tensor_sha256(hessian)
        proxy_hessian = hessian.cpu()
        lower = _block_ldl_lower(hessian)
        lower_sha256 = _tensor_sha256(lower)
        output_signs = _draw_signs(size_n, source_out_in.device).unsqueeze(0)
    transformed = _source_transform(
        source_out_in,
        parent_lut,
        seed=effective_transform_seed,
        quantize_tiles_fn=counted_quantize,
        input_signs=input_signs,
        output_signs=output_signs,
    )
    quantized_inner, states = buffered_ldlq(
        transformed["target_inner"],
        lower,
        parent_lut,
        quantize_tiles_fn=counted_quantize,
        progress=progress,
        resume_state=resume_state,
        checkpoint=checkpoint,
    )
    packed = pack_k2(states)
    physical_in_out = inverse_transform(
        quantized_inner.clone(), transformed["su"], transformed["sv"]
    )
    physical_out_in = physical_in_out.T.contiguous()
    physical_bfloat16 = physical_out_in.to(torch.bfloat16)
    source_float = source_out_in.float()
    source_bfloat16 = source_out_in.to(torch.bfloat16)
    physical_sse = float(
        (physical_out_in - source_float).double().square().sum().item()
    )
    physical_bfloat16_sse = float(
        (physical_bfloat16.float() - source_bfloat16.float())
        .double()
        .square()
        .sum()
        .item()
    )
    inner_sse = float(
        (quantized_inner - transformed["target_inner"])
        .double()
        .square()
        .sum()
        .item()
    )
    proxy_hessian_device = proxy_hessian.to(source_out_in.device)
    proxy_error = transformed["target_inner"] - quantized_inner
    proxy_numerator = _block_trace(proxy_error, proxy_hessian_device)
    proxy_denominator = _block_trace(
        transformed["target_inner"], proxy_hessian_device
    )
    objective_proxy_error = proxy_numerator / max(proxy_denominator, 1e-8)
    boundaries = {
        "source_sha256": _tensor_sha256(source_out_in),
        "parent_lut_sha256": _tensor_sha256(parent_lut),
        "input_signs_sha256": _tensor_sha256(transformed["input_signs"]),
        "output_signs_sha256": _tensor_sha256(transformed["output_signs"]),
        "target_inner_sha256": _tensor_sha256(transformed["target_inner"]),
        "hessian_sha256": hessian_sha256,
        "regularized_hessian_sha256": regularized_hessian_sha256,
        "raw_hessian_sum_sha256": raw_hessian_sum_sha256,
        "lower_sha256": lower_sha256,
        "states_sha256": _tensor_sha256(states),
        "packed_sha256": _tensor_sha256(packed),
        "su_sha256": _tensor_sha256(transformed["su"]),
        "sv_sha256": _tensor_sha256(transformed["sv"]),
        "suh_sha256": _tensor_sha256(transformed["suh"]),
        "svh_sha256": _tensor_sha256(transformed["svh"]),
        "physical_fp32_sha256": _tensor_sha256(physical_out_in),
        "physical_bfloat16_sha256": _tensor_sha256(physical_bfloat16),
    }
    return {
        "states": states,
        "packed_codes": packed,
        "su": transformed["su"],
        "sv": transformed["sv"],
        "suh": transformed["suh"],
        "svh": transformed["svh"],
        "global_scale": transformed["global_scale"],
        "decoded_inner": quantized_inner,
        "physical_fp32": physical_out_in,
        "physical_bfloat16": physical_bfloat16,
        "inner_sse": inner_sse,
        "objective_proxy_error": objective_proxy_error,
        "physical_sse": physical_sse,
        "physical_bfloat16_sse": physical_bfloat16_sse,
        "calibration_rows": rows,
        "calibration_mass": mass,
        "calibration_batches": batches,
        "calibration_mode": calibration_mode,
        "seed": seed,
        "conversion_module_index": conversion_module_index,
        "effective_transform_seed": effective_transform_seed,
        "source_only": True,
        "artifact_seed_inputs": 0,
        "comparator_inputs": 0,
        "external_state_map": False,
        "assignment_calls": 1,
        "solver_counters": counters,
        "boundaries": boundaries,
    }
