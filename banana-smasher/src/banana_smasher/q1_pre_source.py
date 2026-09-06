"""Read-only frozen K1 packed source for the existing Balanced64 forward.

No fitting, table generation, native substitution or Q4 admission weakening.
Scientific admission is separate from packed-wire decoding.
"""
from __future__ import annotations

import json
from pathlib import Path

from .q4_pre_balanced64 import BASIS, require, sha


class CleanK1Source:
    """Hash-bound full clean source implementing ``forward.source.fill_layer``.

    The inventory is an externally sealed clean-admission ledger, not a raw
    production count. Every cell is checked again on consumption. A partial
    staging prefix is deliberately not a source for a uniform score.
    """
    def __init__(self, inventory, inventory_sha256, output, *, source_root=None):
        require(sha(inventory) == inventory_sha256, "K1 inventory bytes")
        ledger = json.loads(Path(inventory).read_text())
        require(ledger["basis"] == BASIS, "K1 source basis")
        rows = ledger["accepted_clean"]
        self.rows = {tuple(row["cell"]): row for row in rows}
        required = {(l, e, p) for l in range(43) for e in range(256)
                    for p in ("fused13", "down")}
        require(len(rows) == len(self.rows) and set(self.rows) == required,
                "K1 full routed-expert coverage")
        self.output = Path(output)
        self.source_root = None if source_root is None else Path(source_root).resolve()
        self.decoded_units = 0

    def resolve_path(self, original):
        """Locate unchanged source bytes in an explicitly selected staging tree.

        No basename search or native-path fallback: all subsequent hash gates
        still authenticate the original ledger/config bindings.
        """
        path = Path(original)
        if self.source_root is None:
            return path
        require(path.is_absolute(), "K1 absolute source path")
        require('..' not in path.parts, "K1 source path traversal")
        staged = self.source_root.joinpath(*path.parts[1:]).resolve()
        require(staged.is_relative_to(self.source_root), "K1 source path escape")
        return staged

    def decode(self, layer, expert, projection):
        import torch
        row = self.rows[(layer, expert, projection)]
        path = self.resolve_path(row["artifact"])
        require(path.stat().st_size == row["artifact_bytes"] and
                sha(path) == row["artifact_sha256"], "K1 artifact bytes")
        receipt_path = self.resolve_path(row["receipt"])
        require(sha(receipt_path) == row["sha256"], "K1 solve receipt bytes")
        receipt = json.loads(receipt_path.read_text())
        require(receipt["status"] == "PASS" and receipt["fresh_no_warm_start"] is True,
                "K1 clean solve status")
        require((receipt["layer"], receipt["expert"], receipt["projection"]) ==
                (layer, expert, projection), "K1 solve identity")
        require(receipt["artifact_sha256"] == row["artifact_sha256"] and
                receipt["basis_gate"]["index_sha256"] == BASIS, "K1 solve binding")
        config = row["config"]
        config_path = self.resolve_path(config["path"])
        require(sha(config_path) == config["sha256"] == receipt["config_sha256"],
                "K1 config bytes")
        cfg = json.loads(config_path.read_text())
        population = cfg["fit_population_manifest"]
        # This pin authenticates original IDs, whole-window text exclusion,
        # frozen64 identity and unchanged evaluation; never admit full32 here.
        require(population["sha256"] ==
                "b8e512c966acd58db2e3bbdda477a2520ed35dc5086dd8309878c420174fbcbd"
                == sha(self.resolve_path(population["path"])), "K1 clean population bytes")
        require(sha(self.resolve_path(cfg["hessian_layer_manifest"])) ==
                cfg["hessian_layer_manifest_sha256"] ==
                receipt["hessian_layer_manifest"]["sha256"], "K1 capture binding")
        require(receipt["build"]["packed_decode"]["fp16_bit_exact"] is True and
                receipt["build"]["packed_decode"]["runtime_check_performed"] is True and
                receipt["build"]["canonical_pack"]["canonical_pack_roundtrip_exact"] is True,
                "K1 packed conformance")
        unit = torch.load(path, map_location="cpu", weights_only=True)
        require(tuple(unit["shape"]) ==
                (4096, 4096 if projection == "fused13" else 2048), "K1 projection shape")
        value = decode_unit(unit)
        self.decoded_units += 1
        return value

    def fill_layer(self, layer, gate_up, down):
        from .q4_pre_forward import atomic_json
        for expert in range(256):
            gate_up[expert] = self.decode(layer, expert, "fused13")
            down[expert] = self.decode(layer, expert, "down")
        atomic_json(str(self.output / "DECODE_PROGRESS.json"),
                    dict(layer=layer, decoded_units=self.decoded_units, basis=BASIS))

GEOMETRY = dict(L=16, K=1, V=2, tlut_bits=9, decode_mode="quantlut_sym", td_x=16, td_y=16)


def decode_unit(unit, *, device="cuda", kernel_decode=None):
    """Decode existing K1 bytes using the canonical QTIP wire and inverse RHT."""
    import torch
    from .fwht import bounded_fwht
    if kernel_decode is None:
        from .qtip_kernel_decompress import decode_compressed
        kernel_decode = decode_compressed
    require(unit["geometry"] == GEOMETRY, "frozen K1 geometry")
    m, k = unit["shape"]
    require(m > 0 and k > 0 and m % 32 == k % 32 == 0, "K1 tile geometry")
    index = torch.arange(1 << 16, device=device)
    quadratic = (index + 1) * index
    expanded = unit["tlut"].float().to(device)[(quadratic >> 6) & 511]
    expanded[:, 0] *= 1 - ((quadratic >> 15) & 1) * 2
    raw = kernel_decode(16, 9, 1, 1, m, k, unit["trellis"].to(device).reshape(-1), expanded)
    value = raw * unit["Wscale"].to(device)
    value = bounded_fwht(value.T).T * unit["SV"].float().to(device)[:, None]
    value = bounded_fwht(value) * unit["SU"].float().to(device)
    require(bool(torch.isfinite(value).all()), "nonfinite K1 decode")
    result = value.to(torch.bfloat16)
    require(bool(torch.isfinite(result).all()), "nonfinite K1 BF16 boundary")
    return result
