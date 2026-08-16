#!/usr/bin/env python3
"""Two-rank overlapped grouped official-K2 V7 pipeline trainer."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

from fast_k2_grouped import grouped_k2_stats
from fast_v7_expert_base import FullyResidentGroupedV7Experts

SCHEMA = "banana-smasher-green-u0-2node-runner-v1"
CHECKPOINT_FORMAT = "banana-smasher-qtip2-v7-joint-checkpoint-v1"
MODEL_INDEX_SHA256 = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
ADMISSION_SHA256 = "76d0674eb0cd37fc9022bac5e048c2b77c721826182222ae0a0609e29607a2c5"
DIAGNOSTIC_PARENT_CHECKPOINT_SHA256 = "2ecbe60917d03e705304b1f3b3adaab66b718bda75fdd2854119f8b1e4c16f61"
PUBLISHED_PRE_ROUTE_CENSUS_SHA256 = "2dcc28497deb834164be26e267fdf4c30cc951342c73f47ce78b207354275fc9"
PUBLISHED_PRE_TERMINAL_SHA256 = "1392d7aa34387d9d555257781919e452d0c35c4a99ff660a92250a960bce1d88"
PUBLISHED_PRE_METRICS_SHA256 = "6088ebe3545cdef387f4586532964cb4b02606f3aef7e38cc12bcff01fc6f591"
PUBLISHED_PRE_SCORER_INPUTS_SHA256 = "a3814092c1a2dab253b348a444e5a9c5bdc426c0b85a05965c404e5bae954091"
PUBLISHED_PRE_PHYSICAL_SCORING_SHA256 = "d1c4a4d9c9b88c9acae375283dcd53778228fcfb740a220eba35a4f5ebf698eb"
EXACT_PRE_BINDING_SHA256 = "c8597091144761efd21dcb6070cc495544e816821c1d66f1ff892a9a7c5fb5fb"
DENSE_L034_ROSTER_SHA256 = "13aaa61931aa362a355854aad7bfdb78db328833dfcb83f2444435d058ad2140"
PARENT_SHARED_LUT_SHA256 = "1fcb3546038bc65ab7847ef4473a2d1a8c66631315655c1b3d9f989325572a3c"
DENSE_L034_PHYSICAL_BYTES = 12_884_901_888
COMPACT_LAYERS = tuple(range(43))
L034_ROSTER_SHA256 = "cea2d8aa9cf8ba8dde0d4b699acc24295a03d0ab0dddae1950e20f4b0e8e269e"
LAYERS = 43
NORMS = 235
OUTPUTS = 43
GAIN_CLAMP = 0.25
WINDOWS_PER_STEP = 16
PIPELINE_MICROBATCH = 4
BASE_LRS = {"luts": 1.0e-2, "norms": 1.0e-4, "outputs": 1.0e-2}
WARMUP_STEPS = 16
COSINE_STEPS = 64
COSINE_MIN_RATIO = 0.1
COMM_TIMEOUT_SECONDS = 14_400
DORMANT_NORMS = {
    f"model.layers.{layer}.self_attn.compressor.indexer.kv_norm"
    for layer in range(2, 43, 2)
}
PROJECTIONS = ("w1", "w2", "w3")
PROJECTION_SHAPES = {
    "w1": (2048, 4096),
    "w2": (4096, 2048),
    "w3": (2048, 4096),
}
PACKED_BYTES = 2_097_152


def sha256_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
        fsync_dir(path.parent)
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass


def install_bytes_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.link(name, path)
        os.chmod(path, 0o444)
        with path.open("rb") as f:
            os.fsync(f.fileno())
        fsync_dir(path.parent)
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass


def install_json_once(path: Path, value: object) -> None:
    install_bytes_once(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def immutable_torch_save(torch: Any, path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as f:
            os.fsync(f.fileno())
        os.link(temporary, path)
        os.chmod(path, 0o444)
        with path.open("rb") as f:
            os.fsync(f.fileno())
        fsync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(path)


def layer_from_name(name: str) -> int | None:
    m = re.match(r"model\.layers\.(\d+)\.", name)
    return None if m is None else int(m.group(1))


def owned_name(name: str, first: int, last: int, rank: int) -> bool:
    layer = layer_from_name(name)
    if layer is not None:
        return first <= layer <= last
    return rank == 1 and name == "model.norm"


def current_multiplier(update: int) -> float:
    warm = min((update + 1) / WARMUP_STEPS, 1.0)
    progress = max(0, update - (WARMUP_STEPS - 1))
    span = max(1, COSINE_STEPS - WARMUP_STEPS)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, span) / span))
    return warm * (COSINE_MIN_RATIO + (1.0 - COSINE_MIN_RATIO) * cosine)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def require_file(path: Path, digest: str, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"{label} missing: {path}")
    observed = sha256_file(path)
    if observed != digest:
        raise RuntimeError(f"{label} sha drift: {observed} != {digest}: {path}")


REMOTE_FILESYSTEMS = {"nfs", "nfs4", "cifs", "smb3", "sshfs"}


def require_local_path(path: Path, label: str) -> dict[str, str]:
    """Refuse any compute input whose longest mount is network-backed."""
    resolved = path.resolve(strict=True)
    candidates: list[tuple[int, str, str]] = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        fields = line.split()
        separator = fields.index("-")
        mountpoint = fields[4].replace("\\040", " ")
        fstype = fields[separator + 1]
        source = fields[separator + 2]
        try:
            resolved.relative_to(mountpoint)
        except ValueError:
            continue
        candidates.append((len(mountpoint), fstype, source))
    if not candidates:
        raise RuntimeError(f"{label} has no mount identity: {resolved}")
    _length, fstype, source = max(candidates)
    if fstype in REMOTE_FILESYSTEMS or fstype.startswith("fuse."):
        raise RuntimeError(
            f"{label} must be copied local before compute: {resolved} is {fstype} {source}"
        )
    return {"path": str(resolved), "fstype": fstype, "source": source}


class PlaneSource:
    def __init__(
        self,
        *,
        torch: Any,
        np: Any,
        row: Mapping[str, Any],
        parent_root: Path,
        l034_roster: Path,
        device: Any,
    ) -> None:
        self.torch = torch
        self.layer = int(row["layer"])
        self.name = str(row["name"])
        manifest = Path(str(row["source_manifest"]["path"])).resolve()
        lut_path = Path(str(row["wire"]["source_path"])).resolve()
        require_file(manifest, str(row["source_manifest"]["sha256"]), f"L{self.layer:03d} manifest")
        require_file(lut_path, str(row["wire"]["sha256"]), f"L{self.layer:03d} LUT")
        initial = torch.from_numpy(np.fromfile(lut_path, dtype="<f2").astype("float32", copy=True)).to(device)
        if tuple(initial.shape) != (1024,) or str(initial.dtype) != "torch.float32":
            raise RuntimeError(f"L{self.layer:03d} official LUT is not fp32[1024]")
        self.master = torch.nn.Parameter(initial)
        self._uses = 0
        self.disk_read_calls = 2
        self.disk_read_bytes = manifest.stat().st_size + lut_path.stat().st_size
        self.member_paths: dict[tuple[int, str], Path] = {}
        if self.layer == 34:
            roster_doc = json.loads(Path(l034_roster).read_text())
            if roster_doc.get("schema") != "banana-smasher-qtip2-v7-l034-selected-wire-roster-v1" or int(roster_doc.get("layer", -1)) != 34 or int(roster_doc.get("member_count", -1)) != 768:
                raise RuntimeError("L034 selected-wire roster identity drift")
            root = Path(l034_roster).resolve().parent
            for row in roster_doc["members"]:
                expert, projection = int(row["expert"]), str(row["projection"])
                path = (root / str(row["path"])).resolve()
                if not path.is_file() or path.is_symlink() or path.stat().st_size != int(row["bytes"]):
                    raise RuntimeError(f"L034 selected-wire member drift: {path}")
                self.member_paths[(expert, projection)] = path
        else:
            root = parent_root.resolve() / f"L{self.layer:03d}"
            for expert in range(256):
                for projection in PROJECTIONS:
                    candidates = [
                        root / f"E{expert:03d}_{projection}.q2v7wire",
                        root / f"E{expert:03d}_{projection}.k2wire",
                    ]
                    present = [p.resolve() for p in candidates if p.is_file()]
                    if len(present) != 1:
                        raise RuntimeError(f"L{self.layer:03d} member ambiguity E{expert:03d}/{projection}")
                    self.member_paths[(expert, projection)] = present[0]
        expected = {(e, p) for e in range(256) for p in PROJECTIONS}
        if set(self.member_paths) != expected:
            raise RuntimeError(f"L{self.layer:03d} member coverage drift")
        for path in self.member_paths.values():
            if not path.is_file() or path.is_symlink() or path.stat().st_size != 2_109_444:
                raise RuntimeError(f"L{self.layer:03d} member geometry drift: {path}")

    def reset_usage(self) -> None:
        self._uses = 0

    def member_path(self, expert: int, projection: str) -> Path:
        return self.member_paths[(int(expert), str(projection))]

    @property
    def uses(self) -> int:
        return self._uses

    def wire_lut(self):
        self._uses += 1
        lut = self.master.to(dtype=self.torch.float16).to(dtype=self.torch.float32).reshape(-1).contiguous()
        if tuple(lut.shape) != (1024,):
            raise RuntimeError("official qtip_k2 LUT geometry drift")
        return lut


class ResidentOfficialExperts:
    def __init__(self, *, torch: Any, np: Any, official_k2: Any, source: PlaneSource, device: Any) -> None:
        from torch import nn
        import torch.nn.functional as F
        from torch.utils.checkpoint import checkpoint as checkpoint_fn

        class Module(nn.Module):
            pass

        self.module = Module()
        self.module.L = source.layer
        self.module.act = F.silu
        self.module.__dict__["plane_source"] = source
        self.module.__dict__["official_k2"] = official_k2
        self.module.__dict__["payloads"] = {}
        self.module.__dict__["resident_bytes"] = 0
        self.module.__dict__["load_seconds"] = 0.0
        started = time.time()
        for expert in range(256):
            for projection in PROJECTIONS:
                path = source.member_paths[(expert, projection)]
                m, k = PROJECTION_SHAPES[projection]
                payload = path.read_bytes()
                expected = PACKED_BYTES + (k + m) * 2 + 4
                if len(payload) != expected:
                    raise RuntimeError(f"packed member geometry drift: {path}")
                packed = torch.from_numpy(
                    np.frombuffer(payload[:PACKED_BYTES], dtype="<i2").copy().reshape(k // 16, m // 16, 32)
                ).to(device=device)
                su = torch.from_numpy(
                    np.frombuffer(payload[PACKED_BYTES:PACKED_BYTES + k * 2], dtype="<f2").copy()
                ).to(device=device, dtype=torch.float32)
                sv = torch.from_numpy(
                    np.frombuffer(payload[PACKED_BYTES + k * 2:PACKED_BYTES + (k + m) * 2], dtype="<f2").copy()
                ).to(device=device, dtype=torch.float32)
                self.module.payloads[(expert, projection)] = (packed, su, sv)
                self.module.resident_bytes += sum(t.numel() * t.element_size() for t in (packed, su, sv))
                source.disk_read_calls += 1
                source.disk_read_bytes += len(payload)
        self.module.load_seconds = time.time() - started

        def project(mod, hidden, expert: int, projection: str):
            packed, su, sv = mod.payloads[(expert, projection)]
            lut = mod.plane_source.wire_lut()

            def project_fn(current_hidden, current_lut):
                decoded = mod.official_k2.decode_k2_matrix(packed, current_lut)
                weight = mod.official_k2.inverse_transform(decoded, su, sv).T.contiguous().to(torch.bfloat16)
                return torch.matmul(current_hidden.to(torch.bfloat16), weight.transpose(0, 1)).float()

            # The outer layer checkpoint alone can retain every expert's decoded
            # dense matrix during a layer backward. Recompute each projection
            # independently so the full shard stays resident in packed form
            # without an O(experts * dense-weight) transient graph peak.
            if torch.is_grad_enabled():
                return checkpoint_fn(project_fn, hidden, lut, use_reentrant=False)
            return project_fn(hidden, lut)

        def forward(mod, hidden_states, top_k_index, top_k_weights):
            final = torch.zeros_like(hidden_states)
            with torch.no_grad():
                mask = F.one_hot(top_k_index, num_classes=256).permute(2, 1, 0)
                hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero().flatten().tolist()
            for expert in hit:
                top_k_pos, token_idx = torch.where(mask[expert])
                hidden = hidden_states[token_idx]
                gate = project(mod, hidden.unsqueeze(0), expert, "w1").squeeze(0)
                up = project(mod, hidden.unsqueeze(0), expert, "w3").squeeze(0)
                activated = mod.act(gate) * up
                current = project(mod, activated.unsqueeze(0), expert, "w2").squeeze(0)
                current = current * top_k_weights[token_idx, top_k_pos, None]
                final.index_add_(0, token_idx, current.to(final.dtype))
            return final

        self.module.forward = forward.__get__(self.module, type(self.module))


class ResidentDenseL034:
    """Frozen, checkpoint-bound dense QTIP reconstruction for published PRE L034."""

    def __init__(self, *, torch: Any, dense_state: Mapping[str, Any], device: Any) -> None:
        from torch import nn
        import torch.nn.functional as F
        from torch.utils.checkpoint import checkpoint as checkpoint_fn

        if dense_state.get("roster_sha256") != DENSE_L034_ROSTER_SHA256:
            raise RuntimeError("dense L034 roster identity refused")
        if dense_state.get("parent_lut_sha256") != PARENT_SHARED_LUT_SHA256:
            raise RuntimeError("dense L034 parent LUT identity refused")
        gate_up_cpu = dense_state.get("gate_up")
        down_cpu = dense_state.get("down")
        if not isinstance(gate_up_cpu, torch.Tensor) or not isinstance(down_cpu, torch.Tensor):
            raise RuntimeError("dense L034 tensors missing from exact PRE checkpoint")
        if gate_up_cpu.dtype != torch.bfloat16 or tuple(gate_up_cpu.shape) != (256, 4096, 4096):
            raise RuntimeError(f"dense L034 gate_up contract refused: {getattr(gate_up_cpu, 'dtype', None)} {getattr(gate_up_cpu, 'shape', None)}")
        if down_cpu.dtype != torch.bfloat16 or tuple(down_cpu.shape) != (256, 4096, 2048):
            raise RuntimeError(f"dense L034 down contract refused: {getattr(down_cpu, 'dtype', None)} {getattr(down_cpu, 'shape', None)}")
        physical_bytes = gate_up_cpu.numel() * gate_up_cpu.element_size() + down_cpu.numel() * down_cpu.element_size()
        if physical_bytes != DENSE_L034_PHYSICAL_BYTES:
            raise RuntimeError(f"dense L034 physical-byte accounting refused: {physical_bytes}")

        class Module(nn.Module):
            def __init__(self):
                super().__init__()
                # Frozen Candidate5 L034 is resident in host RAM, not tmpfs and
                # not reloaded from disk.  Only the selected expert slices cross
                # to CUDA for each microbatch; this avoids the failed rank1's
                # 12.9-GB extra unified-memory residency while preserving bytes.
                self.register_buffer("gate_up", gate_up_cpu.contiguous(), persistent=False)
                self.register_buffer("down", down_cpu.contiguous(), persistent=False)
                self.resident_bytes = physical_bytes
                self.residency = "cpu-resident-selected-expert-cuda-relay"
                self.disk_read_calls = 0
                self.act = F.silu

            def forward(self, hidden_states, top_k_index, top_k_weights):
                final = torch.zeros_like(hidden_states)
                with torch.no_grad():
                    mask = F.one_hot(top_k_index, num_classes=256).permute(2, 1, 0)
                    hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero().flatten().tolist()
                for expert in hit:
                    top_k_pos, token_idx = torch.where(mask[expert])
                    hidden = hidden_states[token_idx]

                    def dense_expert(current_hidden):
                        gate_up_weight = self.gate_up[expert].to(current_hidden.device, non_blocking=False)
                        down_weight = self.down[expert].to(current_hidden.device, non_blocking=False)
                        gate_up = F.linear(current_hidden.to(torch.bfloat16), gate_up_weight).float()
                        gate, up = gate_up.chunk(2, dim=-1)
                        activated = self.act(gate) * up
                        result = F.linear(activated.to(torch.bfloat16), down_weight).float()
                        del gate_up_weight, down_weight
                        return result

                    if torch.is_grad_enabled():
                        current = checkpoint_fn(dense_expert, hidden, use_reentrant=False)
                    else:
                        current = dense_expert(hidden)
                    current = current * top_k_weights[token_idx, top_k_pos, None]
                    final.index_add_(0, token_idx, current.to(final.dtype))
                return final

        self.module = Module()


class ShardStudent:
    def __init__(
        self,
        *,
        torch: Any,
        np: Any,
        base: Any,
        official_k2: Any,
        model_root: Path,
        admission: Mapping[str, Any],
        parent_root: Path,
        l034_roster: Path,
        input_state: Mapping[str, Any],
        rank: int,
        first: int,
        last: int,
        status_cb: Any,
        defer_dense_l034: bool = False,
    ) -> None:
        from safetensors import safe_open
        from torch import nn
        from transformers import AutoConfig, AutoModelForCausalLM
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RotaryEmbedding

        self.rank = rank
        self.first = first
        self.last = last
        self.device = torch.device("cuda")
        index = model_root / "model.safetensors.index.json"
        require_file(index, MODEL_INDEX_SHA256, "model index")
        self.config = AutoConfig.from_pretrained(model_root)
        self.wm = load_json(index)["weight_map"]
        with torch.device("meta"):
            self.model = AutoModelForCausalLM.from_config(self.config, attn_implementation="eager")
        self.model.eval()
        handles: dict[str, Any] = {}

        def get_tensor(name: str):
            shard = self.wm[name]
            if shard not in handles:
                while len(handles) >= 3:
                    handles.pop(next(iter(handles)))
                handles[shard] = safe_open(str(model_root / shard), framework="pt")
            return handles[shard].get_tensor(name)

        self.get_tensor = get_tensor
        m = self.model
        if rank == 0:
            m.model.embed_tokens.weight = nn.Parameter(
                get_tensor("embed.weight").to(self.device).to(torch.bfloat16), requires_grad=False
            )
        else:
            m.lm_head.weight = nn.Parameter(
                get_tensor("head.weight").to(self.device).to(torch.bfloat16), requires_grad=False
            )
            m.model.norm.weight = nn.Parameter(
                get_tensor("norm.weight").to(self.device).to(torch.bfloat16), requires_grad=False
            )
            m.model.hc_head.hc_fn = nn.Parameter(get_tensor("hc_head_fn").to(self.device).float(), requires_grad=False)
            m.model.hc_head.hc_base = nn.Parameter(get_tensor("hc_head_base").to(self.device).float(), requires_grad=False)
            m.model.hc_head.hc_scale = nn.Parameter(get_tensor("hc_head_scale").to(self.device).float(), requires_grad=False)
        m.model.rotary_emb = DeepseekV4RotaryEmbedding(self.config).to(self.device)
        rows = {int(r["layer"]): r for r in admission["trainable_roster"]["luts"]}
        self.sources: dict[int, PlaneSource] = {}
        self.experts: dict[int, Any] = {}
        load_started = time.time()
        for layer in range(first, last + 1):
            layer_started = time.time()
            source = PlaneSource(
                torch=torch,
                np=np,
                row=rows[layer],
                parent_root=parent_root,
                l034_roster=l034_roster,
                device=self.device,
            )
            resident = FullyResidentGroupedV7Experts(
                layer=layer, pilot=True, plane_source=source
            )
            m.model.layers[layer].mlp.experts = resident
            self.sources[layer] = source
            self.experts[layer] = resident
            sd = base.T.build_nonexpert_sd(layer, self.wm, get_tensor)
            base.v3.materialize_layer(m, layer, sd, self.config)
            del sd
            status_cb(
                phase="loading",
                loaded_layer=layer,
                loaded_layers=layer - first + 1,
                shard_layers=last - first + 1,
                layer_seconds=time.time() - layer_started,
                resident_payload_bytes=sum(x.resident_bytes for x in self.experts.values()),
                payload_disk_reads=sum(x.disk_read_calls for x in self.sources.values()),
            )
        self.load_seconds = time.time() - load_started
        for layer in range(first, last + 1):
            for parameter in m.model.layers[layer].parameters():
                parameter.requires_grad_(False)
        if rank == 0:
            m.model.embed_tokens.weight.requires_grad_(False)
        else:
            for module in (m.lm_head, m.model.hc_head):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        for source in self.sources.values():
            source.master.requires_grad_(True)


    @property
    def routed_payload_bytes(self) -> int:
        """Resident packed K2 payloads/scales only; never whole-model residency."""
        return sum(x.resident_bytes for x in self.experts.values())

    @property
    def resident_payload_bytes(self) -> int:
        """Compatibility alias for routed payload bytes, not total model bytes."""
        return self.routed_payload_bytes

    @property
    def routed_lut_parameter_bytes(self) -> int:
        return sum(
            source.master.numel() * source.master.element_size()
            for source in self.sources.values()
        )

    @property
    def native_parameter_bytes(self) -> int:
        """CUDA-resident registered non-routed parameters, deduped by storage."""
        total = 0
        storages: set[tuple[int, int]] = set()
        for parameter in self.model.parameters():
            if parameter.device.type != "cuda" or parameter.is_meta:
                continue
            storage = parameter.untyped_storage()
            key = (int(storage.data_ptr()), int(storage.nbytes()))
            if key in storages:
                continue
            storages.add(key)
            total += int(storage.nbytes())
        return total

    @property
    def resident_parameter_bytes(self) -> int:
        """Explicit resident tensors: native parameters + routed wire/LUT tensors."""
        return (
            self.native_parameter_bytes
            + self.routed_payload_bytes
            + self.routed_lut_parameter_bytes
        )

    @property
    def payload_disk_reads(self) -> int:
        return sum(x.disk_read_calls for x in self.sources.values())


def expose_local_dense(torch: Any, student: ShardStudent, admission: Mapping[str, Any]):
    from torch.nn.utils import parametrize

    class WireBf16(torch.nn.Module):
        def forward(self, master):
            return master.to(torch.bfloat16)

    def output_hook(module, _inputs, output):
        gain = torch.exp(module._banana_smasher_output_log_gain.clamp(-GAIN_CLAMP, GAIN_CLAMP)).to(output.dtype)
        return output * gain

    norms = []
    for row in admission["trainable_roster"]["rmsnorms"]:
        name = str(row["name"])
        if not owned_name(name, student.first, student.last, student.rank):
            continue
        module = student.model.get_submodule(name)
        wire = module._parameters.get("weight")
        if wire is None or wire.ndim != 1 or wire.device.type != "cuda":
            raise RuntimeError(f"local RMSNorm seam drift: {name}")
        before = wire.detach().clone()
        parametrize.register_parametrization(module, "weight", WireBf16(), unsafe=True)
        master = module.parametrizations.weight.original
        master.data = master.data.float()
        master.requires_grad_(True)
        if module.weight.dtype != torch.bfloat16 or not torch.equal(module.weight.detach(), before):
            raise RuntimeError(f"RMSNorm wire identity changed: {name}")
        norms.append((name, master))
    outputs = []
    for row in admission["trainable_roster"]["output_gains"]:
        name = str(row["name"])
        if not owned_name(name, student.first, student.last, student.rank):
            continue
        module_name = name.rsplit(".output_log_gain", 1)[0]
        module = student.model.get_submodule(module_name)
        wire = module._parameters.get("weight")
        if wire is None or wire.device.type != "cuda":
            raise RuntimeError(f"local output gain seam drift: {name}")
        parameter = torch.nn.Parameter(torch.zeros((), dtype=torch.float32, device=wire.device))
        module.register_parameter("_banana_smasher_output_log_gain", parameter)
        module.register_forward_hook(output_hook)
        outputs.append((name, parameter))
    luts = [
        (student.sources[layer].name, student.sources[layer].master)
        for layer in range(student.first, student.last + 1)
        if layer in student.sources
    ]
    return luts, norms, outputs


def load_local_state(rows: Iterable[tuple[str, Any]], saved: Mapping[str, Any], device: Any) -> None:
    for name, parameter in rows:
        if name not in saved:
            raise RuntimeError(f"checkpoint missing local state: {name}")
        parameter.data.copy_(saved[name].to(device))


def coverage(torch: Any, rows: list[tuple[str, Any]], *, gradient: bool) -> dict[str, Any]:
    nonzero, missing, nonfinite = [], [], []
    for name, parameter in rows:
        value = parameter.grad if gradient else parameter
        if value is None or not bool(torch.count_nonzero(value.detach()).item()):
            missing.append(name)
        else:
            nonzero.append(name)
        if value is not None and not bool(torch.isfinite(value.detach()).all().item()):
            nonfinite.append(name)
    return {"nonzero_names": nonzero, "missing_or_zero_names": missing, "nonfinite_names": nonfinite, "total": len(rows)}


def update_coverage(torch: Any, rows: list[tuple[str, Any]], before: list[Any]) -> dict[str, Any]:
    nonzero, missing, nonfinite = [], [], []
    for (name, parameter), old in zip(rows, before):
        delta = parameter.detach() - old
        if bool(torch.count_nonzero(delta).item()):
            nonzero.append(name)
        else:
            missing.append(name)
        if not bool(torch.isfinite(delta).all().item()):
            nonfinite.append(name)
    return {"nonzero_names": nonzero, "missing_or_zero_names": missing, "nonfinite_names": nonfinite, "total": len(rows)}


def merge_coverage(parts: list[Mapping[str, Any]], total: int) -> dict[str, Any]:
    nonzero = [name for part in parts for name in part["nonzero_names"]]
    missing = [name for part in parts for name in part["missing_or_zero_names"]]
    nonfinite = [name for part in parts for name in part["nonfinite_names"]]
    if sum(int(part["total"]) for part in parts) != total:
        raise RuntimeError(f"distributed coverage total drift: {parts}")
    return {
        "nonzero": len(nonzero),
        "total": total,
        "ratio": f"{len(nonzero)}/{total}",
        "nonzero_names": nonzero,
        "missing_or_zero_names": missing,
        "nonfinite_names": nonfinite,
    }


def validate_global_coverage(value: Mapping[str, Mapping[str, Any]]) -> None:
    expected = {"luts": (43, set()), "norms": (235, DORMANT_NORMS), "outputs": (43, set())}
    for surface, (total, missing_expected) in expected.items():
        row = value[surface]
        if (
            int(row["total"]) != total
            or int(row["nonzero"]) != total - len(missing_expected)
            or set(row["missing_or_zero_names"]) != missing_expected
            or row["nonfinite_names"]
        ):
            raise RuntimeError(f"authentic distributed coverage gate failed for {surface}: {row}")


def gather_object(dist: Any, value: object) -> list[Any]:
    out = [None, None]
    dist.all_gather_object(out, value)
    return out


def cpu_tree(torch: Any, value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(torch, item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(torch, item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(torch, item) for item in value)
    return value


def merge_optimizer_state(
    state_rows: list[Mapping[str, Any]], ordered_state: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    surfaces = ("luts", "norms", "outputs")
    ordered_names = {surface: list(ordered_state[surface]) for surface in surfaces}
    global_ids: dict[str, int] = {}
    next_id = 0
    for surface in surfaces:
        for name in ordered_names[surface]:
            global_ids[name] = next_id
            next_id += 1
    merged_state: dict[int, Any] = {}
    templates: dict[str, dict[str, Any]] = {}
    for row in state_rows:
        local = row["optimizer"]
        local_names = row["param_names"]
        if len(local["param_groups"]) != len(surfaces):
            raise RuntimeError("local optimizer parameter-group count drift")
        for surface, group in zip(surfaces, local["param_groups"]):
            names = list(local_names[surface])
            ids = list(group["params"])
            if len(names) != len(ids):
                raise RuntimeError(f"local optimizer name/id drift: {surface}")
            template = {key: value for key, value in group.items() if key != "params"}
            previous = templates.setdefault(surface, template)
            if previous != template:
                raise RuntimeError(f"optimizer group setting drift across ranks: {surface}")
            for name, local_id in zip(names, ids):
                global_id = global_ids[name]
                if global_id in merged_state or local_id not in local["state"]:
                    raise RuntimeError(f"optimizer state overlap/missing: {name}")
                merged_state[global_id] = local["state"][local_id]
    if set(merged_state) != set(range(next_id)):
        raise RuntimeError("global optimizer state coverage drift")
    param_groups = []
    for surface in surfaces:
        group = dict(templates[surface])
        group["params"] = [global_ids[name] for name in ordered_names[surface]]
        param_groups.append(group)
    return {"state": merged_state, "param_groups": param_groups}


def start_identity() -> dict[str, Any]:
    stat = Path(f"/proc/{os.getpid()}/stat").read_text().split()
    return {
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "pgid": os.getpgid(0),
        "startticks": int(stat[21]),
        "argv": list(sys.argv),
        "host": os.uname().nodename,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-smoke", action="store_true")
    parser.add_argument("--rank", type=int, choices=(0, 1))
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--parent-root", type=Path)
    parser.add_argument("--l034-roster", type=Path)
    parser.add_argument("--input-checkpoint", type=Path)
    parser.add_argument("--expected-input-checkpoint-sha256")
    parser.add_argument("--serialized-pre-receipt", type=Path)
    parser.add_argument("--fresh-u0", action="store_true", help="construct U0 from sealed wire parents; never load a continuation/PRE checkpoint")
    parser.add_argument("--expected-claim-owner")
    parser.add_argument("--task-id", default="t_5727296e")
    parser.add_argument("--exact-pre-gate", type=Path,
                        help="rank1-only fsynced gate JSON; preload compact shard while waiting")
    parser.add_argument("--exact-pre-binding", type=Path)
    parser.add_argument("--published-pre-route-census", type=Path)
    parser.add_argument("--published-pre-terminal", type=Path)
    parser.add_argument("--published-pre-metrics", type=Path)
    parser.add_argument("--published-pre-scorer-inputs", type=Path)
    parser.add_argument("--published-pre-physical-scoring", type=Path)
    parser.add_argument("--master-addr", default="192.168.200.4")
    parser.add_argument("--master-port", type=int, default=29598)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument(
        "--resident-benchmark-steps",
        type=int,
        default=0,
        help="consecutive fully resident train steps timed before restoring PRE state",
    )
    parser.add_argument(
        "--split-layer",
        type=int,
        default=21,
        help="first layer owned by rank 1; permits capacity-matched 2-node splits",
    )
    args = parser.parse_args(argv)
    if args.contract_smoke:
        source = Path(__file__).read_text()
        banned = "Qtip2" + "PhysicalLayer"
        if banned in source:
            raise RuntimeError("banned quadratic student symbol present")
        print(json.dumps({
            "schema": SCHEMA,
            "status": "PASS_CONTRACT_SMOKE",
            "split": {"rank0": [0, 21], "rank1": [22, 42]},
            "compact_layers": list(COMPACT_LAYERS),
            "compact_lut_count": len(COMPACT_LAYERS),
            "all43_uniform_grouped_k2": True,
            "windows_per_step": WINDOWS_PER_STEP,
            "pipeline_microbatch": PIPELINE_MICROBATCH,
            "official_lut_shape": [1024],
            "base_lrs": BASE_LRS,
            "warmup_steps": WARMUP_STEPS,
            "model_index_sha256": MODEL_INDEX_SHA256,
        }, sort_keys=True))
        return 0
    required = (
        args.rank,
        args.run_root,
        args.asset_root,
        args.model_root,
        args.parent_root,
        args.l034_roster,
        args.expected_claim_owner,
    )
    if any(value is None for value in required):
        parser.error("compute mode requires rank, run root, asset root, model root, parent root, and expected claim owner")
    if not args.fresh_u0 and any(value is None for value in (args.input_checkpoint, args.expected_input_checkpoint_sha256, args.serialized_pre_receipt)):
        parser.error("non-fresh mode requires checkpoint, checkpoint SHA, and serialized PRE receipt")
    if args.exact_pre_gate is not None:
        parser.error("superseded PRE gate mode is forbidden; pass the exact serialized PRE directly")

    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.utils.checkpoint import checkpoint

    rank = int(args.rank)
    split_layer = int(args.split_layer)
    if not 1 <= split_layer <= 42:
        raise RuntimeError(f"split-layer must be in [1,42], got {split_layer}")
    first, last = (
        (0, split_layer - 1) if rank == 0 else (split_layer, 42)
    )
    run_root = args.run_root.resolve()
    for directory in (run_root / "logs", run_root / "run", run_root / "receipts", run_root / "checkpoints"):
        directory.mkdir(parents=True, exist_ok=True)
    log_path = run_root / "logs" / f"rank{rank}.jsonl"
    status_path = run_root / "run" / f"RANK{rank}_STATUS.json"
    identity_os = start_identity()

    def emit(event: str, **fields: Any) -> None:
        row = {"schema": SCHEMA, "event": event, "rank": rank, "unix": time.time(), **fields}
        with log_path.open("a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        print(json.dumps(row, sort_keys=True), flush=True)

    def status_cb(**fields: Any) -> None:
        atomic_json(status_path, {"schema": SCHEMA, "rank": rank, "identity": identity_os, "updated_unix": time.time(), **fields})

    status_cb(phase="preflight", layers=[first, last])
    claim_path = Path("/home/dnola/HOST_CLAIM.json")
    claim = load_json(claim_path)
    expected_owner = str(args.expected_claim_owner)
    if claim.get("state") != "CLAIMED" or claim.get("owner") != expected_owner:
        raise RuntimeError(f"host claim drift: {claim}")
    claim_sha = sha256_file(claim_path)
    shards = load_json(run_root / "SHARDS.json")
    if (
        shards.get("schema") != "banana-smasher-2node-layer-shards-v1"
        or shards.get("task_id") != str(args.task_id)
        or shards.get("intended_basis", {}).get("model_index_sha256") != MODEL_INDEX_SHA256
        or shards["ranks"][str(rank)]["layers"] != [first, last]
    ):
        raise RuntimeError(f"SHARDS gate drift: {shards}")
    require_file(args.model_root / "model.safetensors.index.json", MODEL_INDEX_SHA256, "source model index")
    admission_path = args.asset_root / "code" / "JOINT_REPAIR_ADMISSION.json"
    require_file(admission_path, ADMISSION_SHA256, "joint admission")
    if args.fresh_u0:
        expected_parent_sha = None
        serialized_pre_receipt = {"status": "FRESH_U0_FROM_CODEBOOK", "forbidden_inputs": ["SERIALIZED_PRE.pt", "UPDATE_023.pt", "UPDATE_012.pt"]}
    else:
        expected_parent_sha = str(args.expected_input_checkpoint_sha256).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_parent_sha):
            raise RuntimeError("expected serialized PRE SHA is not lowercase sha256")
        require_file(args.input_checkpoint, expected_parent_sha, "exact serialized PRE checkpoint")
        serialized_pre_receipt = load_json(args.serialized_pre_receipt)
        if (
            serialized_pre_receipt.get("schema")
            != "banana-smasher.qtip2-v7.serialized-pre-auth-release.v1"
            or serialized_pre_receipt.get("status") != "SEALED_PASS_RELEASED"
            or serialized_pre_receipt.get("checkpoint_sha256") != expected_parent_sha
            or serialized_pre_receipt.get("basis_sha256") != MODEL_INDEX_SHA256
            or serialized_pre_receipt.get("legal_official_k2_layer_coverage") != 43
            or serialized_pre_receipt.get("frozen_member_count") != 33_024
        ):
            raise RuntimeError(f"serialized PRE receipt identity drift: {serialized_pre_receipt}")
    admission = load_json(admission_path)
    if admission.get("framework") != "banana-smasher" or len(admission["trainable_roster"]["luts"]) != 43:
        raise RuntimeError("admission identity drift")
    ordered_wins = list(map(int, admission["train_objective"]["full_model_train_bank"]["ordered_train_windows"]))
    if ordered_wins != list(range(20, 84)):
        raise RuntimeError("TRAIN bank drift")

    locality = {
        "model_root": require_local_path(args.model_root, "model root"),
        "parent_root": require_local_path(args.parent_root, "routed parent root"),
        "asset_root": require_local_path(args.asset_root, "asset root"),
        "corpus": require_local_path(Path(os.environ["BR_CORPUS"]), "training corpus"),
        "teachers": require_local_path(Path(os.environ["BR_TEACH"]), "teacher bank"),
    }
    model_weight_map = load_json(
        args.model_root / "model.safetensors.index.json"
    )["weight_map"]
    required_model_keys = {
        "embed.weight" if rank == 0 else "head.weight",
    }
    if rank == 1:
        required_model_keys.update(
            {"norm.weight", "hc_head_fn", "hc_head_base", "hc_head_scale"}
        )
    for name in model_weight_map:
        if (
            any(name.startswith(f"layers.{layer}.") for layer in range(first, last + 1))
            and ".ffn.experts." not in name
        ):
            required_model_keys.add(name)
    required_shards = sorted({model_weight_map[name] for name in required_model_keys})
    locality["native_model_shards"] = [
        require_local_path(args.model_root / shard, f"native model shard {shard}")
        for shard in required_shards
    ]
    emit(
        "locality_preflight_pass",
        required_native_keys=len(required_model_keys),
        required_native_shards=len(required_shards),
        locality=locality,
    )

    torch.cuda.set_device(0)

    runner_root = (args.asset_root / "runner").resolve()
    sys.path.insert(0, str(runner_root))
    vendor_root = (args.asset_root / "vendor").resolve()
    sys.path.insert(0, str(vendor_root / "src_lp4"))
    sys.path.insert(0, str(vendor_root / "src"))
    sys.path.insert(0, str(vendor_root / "site"))
    import importlib.util

    base_path = runner_root / "base_binrepair_e2e.py"
    spec = importlib.util.spec_from_file_location("banana_smasher_2node_base", base_path)
    if spec is None or spec.loader is None:
        raise ImportError(base_path)
    base = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = base
    spec.loader.exec_module(base)
    from banana_smasher import qtip_k2 as official_k2

    base.T.CKPT = str(args.model_root)
    base.T.DEV = "cuda"
    base.T.PACK = os.environ.get("V7_LP4_PACK", base.T.PACK)
    base.T.MAN = os.environ.get("V7_LP4_MANIFEST", base.T.MAN)
    base.T.SEL = os.environ.get("V7_LP4_SELECTION", getattr(base.T, "SEL", ""))
    base.T.TEACH = os.environ.get("BR_TEACH", base.T.TEACH)
    base.T.CORPUS = os.environ.get("BR_CORPUS", base.T.CORPUS)
    random.seed(1701)
    torch.manual_seed(1701)
    torch.cuda.manual_seed_all(1701)
    input_checkpoint = {}
    if not args.fresh_u0:
        input_checkpoint = torch.load(args.input_checkpoint, map_location="cpu", weights_only=False)
        if input_checkpoint.get("format") != CHECKPOINT_FORMAT:
            raise RuntimeError("input checkpoint format drift")
    load_started = time.time()
    student = ShardStudent(
        torch=torch,
        np=np,
        base=base,
        official_k2=official_k2,
        model_root=args.model_root,
        admission=admission,
        parent_root=args.parent_root,
        l034_roster=args.l034_roster,
        input_state=input_checkpoint,
        rank=rank,
        first=first,
        last=last,
        status_cb=status_cb,
        defer_dense_l034=False,
    )

    luts, norms, outputs = expose_local_dense(torch, student, admission)
    if not args.fresh_u0:
        load_local_state(luts, input_checkpoint["state"]["luts"], student.device)
        load_local_state(norms, input_checkpoint["state"]["norms"], student.device)
        load_local_state(outputs, input_checkpoint["state"]["outputs"], student.device)
    optimizer = torch.optim.Adam(
        [
            {"params": [p for _n, p in luts], "lr": BASE_LRS["luts"], "group_name": "luts"},
            {"params": [p for _n, p in norms], "lr": BASE_LRS["norms"], "group_name": "norms"},
            {"params": [p for _n, p in outputs], "lr": BASE_LRS["outputs"], "group_name": "outputs"},
        ],
        foreach=False,
    )
    corpus = base.T.load_corpus()
    ids_cache = {
        win: base.T.window_ids(corpus, win)[0].unsqueeze(0).to(student.device)
        for win in ordered_wins[:WINDOWS_PER_STEP]
    }
    real_lengths = {win: base.T.window_ids(corpus, win)[1] for win in ordered_wins[:WINDOWS_PER_STEP]}
    teacher_cache = {}
    if rank == 1:
        for win in ordered_wins[:WINDOWS_PER_STEP]:
            teacher_cache[win] = base.T.teacher_rows(win)

    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)
    os.environ["GLOO_SOCKET_IFNAME"] = "enp1s0f1np1"
    os.environ["NCCL_SOCKET_IFNAME"] = "enp1s0f1np1"
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    communication_timeout = timedelta(seconds=COMM_TIMEOUT_SECONDS)
    dist.init_process_group(
        "gloo", rank=rank, world_size=2, timeout=communication_timeout
    )
    gpu_group = dist.new_group(
        ranks=[0, 1], backend="nccl", timeout=communication_timeout
    )
    emit("distributed_ready", claim_sha256=claim_sha, claim_owner=expected_owner, shards_sha256=sha256_file(run_root / "SHARDS.json"))
    torch.cuda.synchronize()
    local_load = {
        "rank": rank,
        "host": os.uname().nodename,
        "layers": [first, last],
        "seconds": time.time() - load_started,
        "routed_payload_bytes": student.routed_payload_bytes,
        "routed_lut_parameter_bytes": student.routed_lut_parameter_bytes,
        "native_parameter_bytes": student.native_parameter_bytes,
        "resident_parameter_bytes": student.resident_parameter_bytes,
        "payload_disk_reads_init": student.payload_disk_reads,
        "cuda_allocated_bytes": torch.cuda.memory_allocated(),
        "cuda_reserved_bytes": torch.cuda.memory_reserved(),
        "trainables": {"luts": len(luts), "norms": len(norms), "outputs": len(outputs)},
        "claim_sha256": claim_sha,
        "shards_sha256": sha256_file(run_root / "SHARDS.json"),
        "process": identity_os,
    }
    load_rows = gather_object(dist, local_load)
    emit("resident_ready", **local_load)
    status_cb(phase="resident_ready", **local_load)
    emit("TRAINING_ENTERED", layers=[first, last], all43_legal_mutable_surface={"luts": 43, "norms": 235, "outputs": 43}, fresh_u0=args.fresh_u0)
    status_cb(phase="TRAINING_ENTERED", layers=[first, last], fresh_u0=args.fresh_u0)

    from transformers.cache_utils import DynamicCache
    from transformers.masking_utils import create_sliding_window_causal_mask

    hidden_shape = (
        PIPELINE_MICROBATCH,
        base.T.T_TRAIN,
        int(student.config.hc_mult),
        int(student.config.hidden_size),
    )

    def positional(ids: Any, template: Any):
        pos = torch.arange(ids.shape[1], device=student.device).unsqueeze(0)
        pe = {
            "main": student.model.model.rotary_emb(template, position_ids=pos, layer_type="main"),
            "compress": student.model.model.rotary_emb(template, position_ids=pos, layer_type="compress"),
        }
        mask = create_sliding_window_causal_mask(
            config=student.config,
            inputs_embeds=template,
            attention_mask=None,
            past_key_values=DynamicCache(config=student.config),
            position_ids=pos,
        )
        return pos, pe, mask

    def run_layers(hidden: Any, ids: Any, train: bool):
        template = hidden[:, :, 0, :] if hidden.ndim == 4 else hidden
        pos, pe, mask = positional(ids, template)
        for layer_index in range(first, last + 1):
            layer = student.model.model.layers[layer_index]

            def layer_fn(h, layer=layer):
                return layer(
                    h,
                    position_embeddings=pe,
                    position_ids=pos,
                    attention_mask=mask,
                    input_ids=ids,
                    past_key_values=DynamicCache(config=student.config),
                )

            if train:
                hidden = checkpoint(layer_fn, hidden, use_reentrant=False)
            else:
                with torch.no_grad():
                    hidden = layer_fn(hidden)
        return hidden

    def group_ids(group: list[int]):
        if len(group) != PIPELINE_MICROBATCH:
            raise RuntimeError(f"pipeline tail geometry refused: {group}")
        return torch.cat([ids_cache[win] for win in group], dim=0)

    def loss_group(hidden: Any, group: list[int]):
        final = student.model.model.norm(student.model.model.hc_head(hidden))
        losses = []
        for row, win in enumerate(group):
            idx, lp_n, p_n = teacher_cache[win]
            length = real_lengths[win]
            logits = student.model.lm_head(final[row, :length].to(torch.bfloat16))
            q = logits.gather(1, idx[:length]).float()
            qn = q - q.logsumexp(-1, keepdim=True)
            losses.append((p_n[:length] * (lp_n[:length] - qn)).sum(-1).mean())
        return torch.stack(losses).mean()

    def p2p_isend(tensor: Any, peer: int):
        return dist.batch_isend_irecv(
            [dist.P2POp(dist.isend, tensor, peer, group=gpu_group)]
        )[0]

    def p2p_irecv(tensor: Any, peer: int):
        return dist.batch_isend_irecv(
            [dist.P2POp(dist.irecv, tensor, peer, group=gpu_group)]
        )[0]

    def pipeline_pass(
        wins: list[int], train: bool, *, emit_progress: bool = True
    ) -> tuple[float | None, dict[str, float]]:
        pass_started = time.time()
        comm_seconds = 0.0
        objective_sum = 0.0
        forward_events: list[tuple[Any, Any]] = []
        backward_events: list[tuple[Any, Any]] = []
        groups = [wins[i : i + PIPELINE_MICROBATCH] for i in range(0, len(wins), PIPELINE_MICROBATCH)]
        if any(len(group) != PIPELINE_MICROBATCH for group in groups):
            raise RuntimeError("WINDOWS_PER_STEP must divide PIPELINE_MICROBATCH")
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            pending: list[tuple[Any, Any, Any]] = []
            for group_index, group in enumerate(groups):
                ids = group_ids(group)
                if rank == 0:
                    forward_start = torch.cuda.Event(enable_timing=True)
                    forward_end = torch.cuda.Event(enable_timing=True)
                    forward_start.record()
                    embeds = student.model.model.embed_tokens(ids)
                    hidden = embeds.unsqueeze(2).expand(-1, -1, student.config.hc_mult, -1).contiguous()
                    hidden = run_layers(hidden, ids, train)
                    forward_end.record()
                    forward_events.append((forward_start, forward_end))
                    if tuple(hidden.shape) != hidden_shape or hidden.dtype != torch.bfloat16:
                        raise RuntimeError(f"pipeline activation geometry drift: {tuple(hidden.shape)} {hidden.dtype}")
                    wire = hidden.detach().contiguous()
                    send_work = p2p_isend(wire, 1)
                    pending.append((hidden, wire, send_work))
                    if train and len(pending) >= 2:
                        old_hidden, old_wire, old_send = pending.pop(0)
                        old_send.wait()
                        grad = torch.empty_like(old_hidden)
                        started = time.time()
                        p2p_irecv(grad, 1).wait()
                        comm_seconds += time.time() - started
                        backward_start = torch.cuda.Event(enable_timing=True)
                        backward_end = torch.cuda.Event(enable_timing=True)
                        backward_start.record()
                        old_hidden.backward(grad)
                        backward_end.record()
                        backward_events.append((backward_start, backward_end))
                        del old_hidden, old_wire, grad
                    elif not train:
                        send_work.wait()
                        pending.pop()
                    del embeds
                else:
                    activation = torch.empty(hidden_shape, dtype=torch.bfloat16, device=student.device)
                    started = time.time()
                    p2p_irecv(activation, 0).wait()
                    comm_seconds += time.time() - started
                    if train:
                        activation.requires_grad_(True)
                    forward_start = torch.cuda.Event(enable_timing=True)
                    forward_end = torch.cuda.Event(enable_timing=True)
                    forward_start.record()
                    hidden = run_layers(activation, ids, train)
                    loss = loss_group(hidden, group)
                    forward_end.record()
                    forward_events.append((forward_start, forward_end))
                    objective_sum += float(loss.detach()) * len(group)
                    if train:
                        backward_start = torch.cuda.Event(enable_timing=True)
                        backward_end = torch.cuda.Event(enable_timing=True)
                        backward_start.record()
                        (loss * (len(group) / len(wins))).backward()
                        backward_end.record()
                        backward_events.append((backward_start, backward_end))
                        if activation.grad is None:
                            raise RuntimeError("pipeline boundary gradient missing")
                        grad_wire = activation.grad.contiguous()
                        pending.append((grad_wire, p2p_isend(grad_wire, 0), None))
                        while len(pending) > 2:
                            _wire, work, _unused = pending.pop(0)
                            work.wait()
                    del hidden, activation, loss
                progress = {
                    "pass_kind": "train" if train else "eval",
                    "window_group": list(map(int, group)),
                    "windows_completed": (group_index + 1) * PIPELINE_MICROBATCH,
                    "windows_total": len(wins),
                    "pipeline_inflight": len(pending),
                    "pass_elapsed_seconds": time.time() - pass_started,
                    "payload_disk_reads": student.payload_disk_reads,
                    "cuda_allocated_bytes": torch.cuda.memory_allocated(),
                    "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                }
                if emit_progress:
                    emit("pass_window", **progress)
                    status_cb(phase="pass_window", **progress)
            if train and rank == 0:
                while pending:
                    old_hidden, old_wire, old_send = pending.pop(0)
                    old_send.wait()
                    grad = torch.empty_like(old_hidden)
                    started = time.time()
                    p2p_irecv(grad, 1).wait()
                    comm_seconds += time.time() - started
                    backward_start = torch.cuda.Event(enable_timing=True)
                    backward_end = torch.cuda.Event(enable_timing=True)
                    backward_start.record()
                    old_hidden.backward(grad)
                    backward_end.record()
                    backward_events.append((backward_start, backward_end))
                    del old_hidden, old_wire, grad
            elif train and rank == 1:
                while pending:
                    _wire, work, _unused = pending.pop(0)
                    work.wait()
        torch.cuda.synchronize()
        forward_gpu_seconds = sum(
            start.elapsed_time(end) for start, end in forward_events
        ) / 1000.0
        backward_gpu_seconds = sum(
            start.elapsed_time(end) for start, end in backward_events
        ) / 1000.0
        input_tokens = len(wins) * int(base.T.T_TRAIN)
        hc_token_vectors = input_tokens * int(student.config.hc_mult)
        values = gather_object(dist, objective_sum if rank == 1 else None)
        return (
            None if values[1] is None else values[1] / len(wins),
            {
                "wall_seconds": time.time() - pass_started,
                "comm_seconds": comm_seconds,
                "forward_gpu_seconds": forward_gpu_seconds,
                "backward_gpu_seconds": backward_gpu_seconds,
                "input_tokens": input_tokens,
                "hc_token_vectors": hc_token_vectors,
                "forward_input_tokens_per_gpu_second": (
                    input_tokens / forward_gpu_seconds if forward_gpu_seconds > 0 else None
                ),
                "forward_hc_vectors_per_gpu_second": (
                    hc_token_vectors / forward_gpu_seconds if forward_gpu_seconds > 0 else None
                ),
                "pipeline_microbatch": PIPELINE_MICROBATCH,
                "pipeline_groups": len(groups),
            },
        )

    if args.updates != 1:
        raise RuntimeError("acceptance launcher currently requires exactly one update")
    if args.resident_benchmark_steps < 0:
        raise RuntimeError("resident-benchmark-steps must be nonnegative")
    wins = ordered_wins[:WINDOWS_PER_STEP]
    benchmark_step_rows: list[list[dict[str, Any]]] = []
    if args.resident_benchmark_steps:
        benchmark_params = [
            p for collection in (luts, norms, outputs) for _name, p in collection
        ]
        benchmark_snapshots = [p.detach().clone() for p in benchmark_params]
        for benchmark_step in range(args.resident_benchmark_steps):
            for source in student.sources.values():
                source.reset_usage()
            optimizer.zero_grad(set_to_none=True)
            benchmark_multiplier = current_multiplier(benchmark_step)
            for group in optimizer.param_groups:
                group["lr"] = BASE_LRS[group["group_name"]] * benchmark_multiplier
            reads_before = student.payload_disk_reads
            dist.barrier()
            torch.cuda.synchronize()
            benchmark_started = time.monotonic()
            benchmark_objective, benchmark_timing = pipeline_pass(
                wins, True, emit_progress=True
            )
            optimizer_started = time.monotonic()
            optimizer.step()
            torch.cuda.synchronize()
            optimizer_seconds = time.monotonic() - optimizer_started
            local_benchmark = {
                "rank": rank,
                "benchmark_step": benchmark_step,
                "objective": benchmark_objective,
                "pipeline": benchmark_timing,
                "optimizer_seconds": optimizer_seconds,
                "resident_step_wall_seconds": time.monotonic() - benchmark_started,
                "step_layer_payload_disk_reads": student.payload_disk_reads - reads_before,
            }
            gathered_benchmark = gather_object(dist, local_benchmark)
            benchmark_step_rows.append(gathered_benchmark)
            if rank == 0:
                emit(
                    "resident_benchmark_step",
                    benchmark_step=benchmark_step,
                    rows=gathered_benchmark,
                )
        with torch.no_grad():
            for parameter, snapshot in zip(benchmark_params, benchmark_snapshots):
                parameter.copy_(snapshot)
        optimizer.state.clear()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
    update = 0
    for source in student.sources.values():
        source.reset_usage()
    payload_reads_before = student.payload_disk_reads
    optimizer.zero_grad(set_to_none=True)
    snapshots = {
        "luts": [p.detach().clone() for _n, p in luts],
        "norms": [p.detach().clone() for _n, p in norms],
        "outputs": [p.detach().clone() for _n, p in outputs],
    }
    multiplier = current_multiplier(update)
    for group in optimizer.param_groups:
        group["lr"] = BASE_LRS[group["group_name"]] * multiplier
    dist.barrier()
    torch.cuda.synchronize()
    step_started = time.monotonic()
    objective_before, before_timing = pipeline_pass(
        wins, True, emit_progress=False
    )
    optimizer_started = time.monotonic()
    optimizer.step()
    torch.cuda.synchronize()
    optimizer_seconds = time.monotonic() - optimizer_started
    full_step_fwd_bwd_optimizer_seconds = time.monotonic() - step_started
    local_grad = {
        "luts": coverage(torch, luts, gradient=True),
        "norms": coverage(torch, norms, gradient=True),
        "outputs": coverage(torch, outputs, gradient=True),
    }
    gathered_grad = gather_object(dist, local_grad)
    global_grad = {
        "luts": merge_coverage([x["luts"] for x in gathered_grad], 43),
        "norms": merge_coverage([x["norms"] for x in gathered_grad], 235),
        "outputs": merge_coverage([x["outputs"] for x in gathered_grad], 43),
    }
    gate_error = None
    try:
        validate_global_coverage(global_grad)
    except Exception as exc:
        gate_error = f"{type(exc).__name__}: {exc}"
    gate_errors = gather_object(dist, gate_error)
    if any(gate_errors):
        raise RuntimeError(f"gradient gate failed: {gate_errors}")

    local_update = {
        "luts": update_coverage(torch, luts, snapshots["luts"]),
        "norms": update_coverage(torch, norms, snapshots["norms"]),
        "outputs": update_coverage(torch, outputs, snapshots["outputs"]),
    }
    gathered_update = gather_object(dist, local_update)
    global_update = {
        "luts": merge_coverage([x["luts"] for x in gathered_update], 43),
        "norms": merge_coverage([x["norms"] for x in gathered_update], 235),
        "outputs": merge_coverage([x["outputs"] for x in gathered_update], 43),
    }
    gate_error = None
    try:
        validate_global_coverage(global_update)
    except Exception as exc:
        gate_error = f"{type(exc).__name__}: {exc}"
    gate_errors = gather_object(dist, gate_error)
    if any(gate_errors):
        raise RuntimeError(f"update gate failed: {gate_errors}")
    objective_after = None
    after_timing = {
        "wall_seconds": 0.0,
        "comm_seconds": 0.0,
        "pipeline_microbatch": PIPELINE_MICROBATCH,
        "pipeline_groups": 0,
        "deferred_to_async_balanced64": True,
    }
    payload_reads_after = student.payload_disk_reads
    local_step = {
        "rank": rank,
        "before_pass": before_timing,
        "optimizer_seconds": optimizer_seconds,
        "full_step_fwd_bwd_optimizer_seconds": full_step_fwd_bwd_optimizer_seconds,
        "after_pass": after_timing,
        "step_wall_seconds": full_step_fwd_bwd_optimizer_seconds,
        "payload_disk_reads_before": payload_reads_before,
        "payload_disk_reads_after": payload_reads_after,
        "step_layer_payload_disk_reads": payload_reads_after - payload_reads_before,
        "plane_source_uses_after": {str(k): v.uses for k, v in student.sources.items()},
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "mechanism_counters": grouped_k2_stats(),
        "routed_payload_bytes": student.routed_payload_bytes,
        "routed_lut_parameter_bytes": student.routed_lut_parameter_bytes,
        "native_parameter_bytes": student.native_parameter_bytes,
        "resident_parameter_bytes": student.resident_parameter_bytes,
    }
    step_rows = gather_object(dist, local_step)
    gate_pass = (
        objective_before is not None
        and math.isfinite(float(objective_before))
        and all(int(row["step_layer_payload_disk_reads"]) == 0 for row in step_rows)
        and all(int(row["mechanism_counters"]["fallback_calls"]) == 0 for row in step_rows)
        and all(int(row["mechanism_counters"]["reconstruction_calls"]) == 0 for row in step_rows)
        and all(int(row["mechanism_counters"]["cpu_relay_bytes"]) == 0 for row in step_rows)
    )
    gate_values = gather_object(dist, gate_pass)
    if not all(gate_values):
        raise RuntimeError(
            f"objective/residency gate failed: before={objective_before} rows={step_rows}"
        )

    local_state = {
        "rank": rank,
        "luts": {name: p.detach().cpu().clone() for name, p in luts},
        "norms": {name: p.detach().cpu().clone() for name, p in norms},
        "outputs": {name: p.detach().cpu().clone() for name, p in outputs},
        "param_names": {
            "luts": [name for name, _p in luts],
            "norms": [name for name, _p in norms],
            "outputs": [name for name, _p in outputs],
        },
        "optimizer": cpu_tree(torch, optimizer.state_dict()),
    }
    state_rows = gather_object(dist, local_state)
    if rank == 0:
        merged = {"luts": {}, "norms": {}, "outputs": {}}
        for row in state_rows:
            for surface in merged:
                overlap = set(merged[surface]) & set(row[surface])
                if overlap:
                    raise RuntimeError(f"distributed state overlap: {surface} {sorted(overlap)[:3]}")
                merged[surface].update(row[surface])
        trainable_state = {
            surface: {
                name: merged[surface][name]
                for name in input_checkpoint["state"][surface]
                if name in merged[surface]
            }
            for surface in ("luts", "norms", "outputs")
        }
        if {k: len(v) for k, v in trainable_state.items()} != {"luts": 43, "norms": 235, "outputs": 43}:
            raise RuntimeError("merged trainable checkpoint surface count drift")
        checkpoint_state = trainable_state
        merged_optimizer = merge_optimizer_state(state_rows, trainable_state)
        identity = {
            "schema": SCHEMA,
            "framework": "banana-smasher",
            "student": "official-qtip-k2-fp32-master-fp16-wire-lut1024-mul1",
            "model_index_sha256": MODEL_INDEX_SHA256,
            "admission_sha256": ADMISSION_SHA256,
            "input_checkpoint_sha256": expected_parent_sha,
            "published_pre": {
                "route_census_sha256": PUBLISHED_PRE_ROUTE_CENSUS_SHA256,
                "terminal_sha256": PUBLISHED_PRE_TERMINAL_SHA256,
                "binary64_metrics_sha256": PUBLISHED_PRE_METRICS_SHA256,
                "scorer_inputs_sha256": PUBLISHED_PRE_SCORER_INPUTS_SHA256,
                "physical_scoring_sha256": PUBLISHED_PRE_PHYSICAL_SCORING_SHA256,
                "expected_kld": 0.22939197531977115,
                "expected_top1": 56533,
                "positions": 65536,
            },
            "split": {
                "rank0": [0, split_layer - 1],
                "rank1": [split_layer, 42],
            },
            "transport": {
                "interface": "enp1s0f1np1",
                "master_addr": args.master_addr,
                "hosts": [row["host"] for row in load_rows],
            },
            "windows_per_step": WINDOWS_PER_STEP,
            "pipeline_microbatch": PIPELINE_MICROBATCH,
            "ordered_train_windows": ordered_wins,
            "base_learning_rates": BASE_LRS,
            "warmup_steps": WARMUP_STEPS,
            "cosine_steps": COSINE_STEPS,
            "cosine_min_ratio": COSINE_MIN_RATIO,
            "trainables": {"luts": 43, "norms": 235, "outputs": 43},
            "all43_uniform_grouped_k2": True,
            "resident_loads": load_rows,
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "shards_sha256": sha256_file(run_root / "SHARDS.json"),
        }
        identity_sha = canonical_sha256(identity)
        payload = {
            "format": CHECKPOINT_FORMAT,
            "mechanism": "joint-v7-luts-plus-rmsnorms-plus-attention-output-gains",
            "next_update": 1,
            "identity": identity,
            "identity_sha256": identity_sha,
            "state": checkpoint_state,
            "optimizer": merged_optimizer,
            "scheduler": {
                "base_lrs": [BASE_LRS["luts"], BASE_LRS["norms"], BASE_LRS["outputs"]],
                "last_epoch": 0,
                "verbose": False,
                "_step_count": 1,
                "_get_lr_called_within_step": False,
                "_last_lr": [
                    BASE_LRS["luts"] * multiplier,
                    BASE_LRS["norms"] * multiplier,
                    BASE_LRS["outputs"] * multiplier,
                ],
                "lr_lambdas": [None, None, None],
            },
            "objective": {
                "update": 0,
                "before": float(objective_before),
                "after": None,
                "evaluation": "deferred-to-asynchronous-balanced64",
            },
            "invariants": {
                "codes_frozen": True,
                "assignments_frozen": True,
                "scales_frozen": True,
                "packed_geometry_frozen": True,
                "optimizer_steps_this_update": 1,
                "ordered_whole_model_objective": True,
                "resident_layer_payloads": True,
                "step_layer_payload_disk_reads": 0,
            },
            "saved_unix": time.time(),
        }
        checkpoint_path = run_root / "checkpoints" / "UPDATE_001.pt"
        checkpoint_sha = immutable_torch_save(torch, checkpoint_path, payload)
        receipt = {
            "schema": "banana-smasher-qtip2-v7-joint-update-receipt-v1",
            "status": "PASS",
            "gate_pass": True,
            "update": 0,
            "next_update": 1,
            "train_windows": wins,
            "batch": WINDOWS_PER_STEP,
            "pipeline_microbatch": PIPELINE_MICROBATCH,
            "objective": {
                "before": float(objective_before),
                "after": None,
                "delta": None,
                "evaluation": "deferred-to-asynchronous-balanced64",
            },
            "gradient_coverage": global_grad,
            "update_coverage": global_update,
            "optimizer": "Adam",
            "base_learning_rates": BASE_LRS,
            "warmup_multiplier": multiplier,
            "applied_learning_rates": {k: v * multiplier for k, v in BASE_LRS.items()},
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "identity_sha256": identity_sha,
            "resident_loads": load_rows,
            "step_timings": step_rows,
            "resident_benchmark_steps": benchmark_step_rows,
            "windows_per_step": WINDOWS_PER_STEP,
            "seconds_per_step": max(
                float(row["full_step_fwd_bwd_optimizer_seconds"]) for row in step_rows
            ),
            "step_layer_payload_disk_reads": 0,
            "claim_sha256_by_rank": [row["claim_sha256"] for row in load_rows],
            "shards_sha256": sha256_file(run_root / "SHARDS.json"),
        }
        receipt_path = run_root / "receipts" / "UPDATE_000.json"
        install_json_once(receipt_path, receipt)
        sidecar = {
            "schema": "banana-smasher-qtip2-v7-joint-checkpoint-sidecar-v1",
            "completed_update": 0,
            "next_update": 1,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "identity_sha256": identity_sha,
            "objective": receipt["objective"],
            "gradient_coverage": global_grad,
            "update_coverage": global_update,
            "gate_pass": True,
        }
        install_json_once(run_root / "checkpoints" / "UPDATE_001.json", sidecar)
        status_cb(
            phase="sealed",
            gate_pass=True,
            checkpoint=str(checkpoint_path),
            checkpoint_sha256=checkpoint_sha,
            receipt=str(receipt_path),
            windows_per_step=WINDOWS_PER_STEP,
            seconds_per_step=receipt["seconds_per_step"],
            objective=receipt["objective"],
        )
        emit("sealed_update", checkpoint=str(checkpoint_path), checkpoint_sha256=checkpoint_sha, receipt=str(receipt_path), objective=receipt["objective"], seconds_per_step=receipt["seconds_per_step"])
    else:
        status_cb(phase="rank1_complete", gate_pass=True, step=local_step)
        emit("rank1_complete", step=local_step)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
