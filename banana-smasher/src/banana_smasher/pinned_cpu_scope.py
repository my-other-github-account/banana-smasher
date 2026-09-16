"""Experimental synchronous pinned transfers for a sole serial executor.

Opt-in only. Process-global interception is incompatible with concurrent users.
No change to mathematical kernels, dtype, shape or completion semantics.
"""
from contextlib import contextmanager

@contextmanager
def pinned_cpu_scope(torch, enabled=False, minimum_bytes=1048576, *, event_fence=False):
    if type(event_fence) is not bool:
        raise ValueError('event_fence must be bool')
    if type(enabled) is not bool:
        raise ValueError('enabled must be bool')
    if type(minimum_bytes) is not int or minimum_bytes < 1:
        raise ValueError('minimum_bytes must be positive int')
    events = []
    original = torch.Tensor.cpu
    def transfer(t, *args, **kwargs):
        if (enabled and not args and not kwargs and t.is_cuda
                and t.is_contiguous() and not t.requires_grad
                and t.numel() * t.element_size() >= minimum_bytes):
            out = torch.empty_like(t, device='cpu', pin_memory=True)
            out.copy_(t, non_blocking=True)
            stream = torch.cuda.current_stream(t.device)
            if event_fence:
                event = torch.cuda.Event()
                event.record(stream)
                event.synchronize()
            else:
                stream.synchronize()
            events.append({'bytes': t.numel() * t.element_size(),
                           'dtype': str(t.dtype), 'pinned': out.is_pinned()})
            return out
        return original(t, *args, **kwargs)
    if enabled:
        torch.Tensor.cpu = transfer
    try:
        yield events
    finally:
        if enabled:
            torch.Tensor.cpu = original
