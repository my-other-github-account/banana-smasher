"""Install admitted calibrated producer wire into one native materialized layer.

This boundary does not fit, encode, score, or claim whole-model acceptance. The
caller must basis-gate the source and admit the complete model member roster.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def checked_install_calibrated_layer(layer: Any, *, installer: Any = None, **kwargs) -> dict[str, Any]:
    """Measure installed tensors and prove every native-rest tensor unchanged.

    Hash in bounded chunks, so verifying a materialized layer does not clone its
    full expert matrices onto CPU. No encoding or source-weight replay occurs.
    """
    import torch

    expert_names = {'mlp.experts.gate_up_proj', 'mlp.experts.down_proj'}

    def digest(tensor):
        h = hashlib.sha256()
        value = tensor.detach()
        if not value.is_contiguous():
            raise ValueError('canary requires contiguous native tensors')
        flat = value.reshape(-1)
        for offset in range(0, flat.numel(), 1048576):
            chunk = flat[offset:offset + 1048576].cpu().contiguous()
            h.update(chunk.view(torch.uint8).numpy().tobytes())
        return {'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': h.hexdigest()}

    before = {name: digest(tensor) for name, tensor in layer.state_dict().items()
              if name not in expert_names}
    result = (installer or install_calibrated_layer)(layer, **kwargs)
    after = {name: digest(tensor) for name, tensor in layer.state_dict().items()
             if name not in expert_names}
    if before != after:
        raise ValueError('native rest changed during calibrated installation')
    installed = {name: digest(tensor) for name, tensor in layer.state_dict().items()
                 if name in expert_names}
    if set(installed) != expert_names:
        raise ValueError('missing installed expert tensors')
    return dict(result, native_rest_unchanged=True, native_rest_tensor_count=len(before),
                native_rest_tensor_sha256=before, installed_tensor_sha256=installed)


def install_calibrated_layer(
    layer: Any, *, layer_id: int, members: list[dict[str, Any]],
    expected_experts: int, tier: int, decoder: Any,
) -> dict[str, Any]:
    """Copy exact decoded FP16->working-dtype values, leaving native rest intact.

    ``decoder`` is the pinned reference-free packed-wire decoder. It receives
    the payload and destination device. Only one decoded expert is held at once.
    """
    import torch

    if type(tier) is not int or tier not in (1, 2, 3, 4):
        raise ValueError('uniform tier must be integer K1..4')
    targets = {'fused13': layer.mlp.experts.gate_up_proj,
               'down': layer.mlp.experts.down_proj}
    by_cell = {row['cell']: row for row in members}
    expected = {f'L{layer_id:03d}/E{e:03d}_{p}'
                for e in range(expected_experts) for p in targets}
    if len(by_cell) != len(members) or set(by_cell) != expected:
        raise ValueError('exact layer expert/projection inventory required')
    if any(t.shape[0] != expected_experts for t in targets.values()):
        raise ValueError('native expert inventory mismatch')
    installed = []
    with torch.no_grad():
        for expert in range(expected_experts):
            for projection, target in targets.items():
                cell = f'L{layer_id:03d}/E{expert:03d}_{projection}'
                binding = by_cell[cell]['bindings']['artifact']
                path = Path(binding['path'])
                if (path.stat().st_size != binding['bytes'] or
                        hashlib.sha256(path.read_bytes()).hexdigest() != binding['sha256']):
                    raise ValueError(f'artifact bytes mismatch: {cell}')
                payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
                if payload['geometry']['K'] != tier:
                    raise ValueError(f'artifact tier mismatch: {cell}')
                if tuple(payload['shape']) != tuple(target[expert].shape):
                    raise ValueError(f'artifact shape mismatch: {cell}')
                decoded = decoder(payload, target.device).half().to(target.dtype)
                if tuple(decoded.shape) != tuple(target[expert].shape) or not torch.isfinite(decoded).all():
                    raise ValueError(f'invalid decoded weight: {cell}')
                target[expert].copy_(decoded)
                if not torch.equal(target[expert], decoded):
                    raise ValueError(f'installed weight mismatch: {cell}')
                installed.append(cell)
                del decoded, payload
    return {'layer': layer_id, 'tier': tier, 'installed_count': len(installed),
            'installed_cells': installed, 'native_rest_modified': False,
            'fit_invocations': 0, 'whole_model_acceptance': False}
