"""Bounded-buffer FWHT for nondifferentiable float32 tensors.

Same ordered add/sub butterflies and final division as the reference; two
output buffers replace per-level add/sub/cat allocations. Not a new default.
"""
import math
import torch

def fwht_pingpong(x):
    n = x.shape[-1]
    if n <= 0 or n & (n - 1):
        raise ValueError('power-of-two last dimension required')
    if x.requires_grad or x.dtype != torch.float32:
        raise ValueError('nongrad float32 required')
    y = x.contiguous()
    if n == 1:
        return y / math.sqrt(n)
    buffers = [torch.empty_like(y), torch.empty_like(y)]
    width = 1
    step = 0
    while width < n:
        z = y.reshape(*y.shape[:-1], n // (2 * width), 2, width)
        out = buffers[step % 2]
        target = out.reshape(*y.shape[:-1], n // (2 * width), 2, width)
        torch.add(z[..., 0, :], z[..., 1, :], out=target[..., 0, :])
        torch.sub(z[..., 0, :], z[..., 1, :], out=target[..., 1, :])
        y = out
        step += 1
        width *= 2
    return y / math.sqrt(n)
