"""Experimental fused row-wise float32 FWHT; explicit CUDA-only admission."""
import math
import torch
_KERNEL = None

def _kernel():
    global _KERNEL
    if _KERNEL is None:
        import triton
        import triton.language as tl
        @triton.jit
        def transform(X, Y, N:tl.constexpr, LOG:tl.constexpr, DIV:tl.constexpr):
            row=tl.program_id(0)
            i=tl.arange(0,N)
            y=tl.load(X+row*N+i)
            for k in tl.static_range(LOG):
                width=1<<k
                other=tl.gather(y,i^width,0)
                y=tl.where((i&width)==0,y+other,other-y)
            # PyTorch CUDA division by a scalar uses reciprocal multiplication.
            y=y*tl.div_rn(1.0,DIV)
            tl.store(Y+row*N+i,y)
        _KERNEL=transform
    return _KERNEL

def fwht_fused(x):
    n=x.shape[-1]
    if not x.is_cuda or x.requires_grad or x.dtype!=torch.float32 or n<1 or n>4096 or n&(n-1):
        raise ValueError('CUDA nongrad float32 power-of-two width1..4096 required')
    y=x.contiguous()
    out=torch.empty_like(y)
    if y.numel():
        _kernel()[(y.numel()//n,)](y,out,n,n.bit_length()-1,math.sqrt(n),num_warps=4,enable_fp_fusion=False)
    return out
