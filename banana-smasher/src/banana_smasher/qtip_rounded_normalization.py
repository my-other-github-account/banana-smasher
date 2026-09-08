"""Explicit correctly-rounded FP32 CUDA normalization, no CPU fallback."""
import triton
import triton.language as tl


@triton.jit
def _divide_rn(X, Y, count: tl.constexpr, divisor: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < count, other=0)
    y = tl.div_rn(x, tl.full((), divisor, tl.float32))
    tl.store(Y + i, y, i < count)


def rounded_divide_cuda(x, divisor):
    import torch
    if not x.is_cuda or x.dtype != torch.float32 or not x.is_contiguous():
        raise ValueError('rounded normalization requires contiguous CUDA float32')
    y = torch.empty_like(x)
    if x.numel():
        _divide_rn[(triton.cdiv(x.numel(), 256),)](x, y, x.numel(), divisor, BLOCK=256)
    return y
