#!/usr/bin/env python3
"""Source-native + frozen QTIP2-V7 partial-anchor PlaneSource for run 1489.

This module only replaces routed-expert materialization in the sealed DS4
scalar-logit builder. Unselected layers are decoded from the 0731 source
checkpoint's native MXFP4 tensors. Frozen layers read immutable compact wire
payloads (except preserved L034, whose admitted complete assignment-physical
provider is used exactly as sealed).
"""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any

import numpy as np

TASK = "t_58f64456"
RUN = 1996
BASIS = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
FREEZE_SHA = "2dcc28497deb834164be26e267fdf4c30cc951342c73f47ce78b207354275fc9"
LUT_SHA = "1fcb3546038bc65ab7847ef4473a2d1a8c66631315655c1b3d9f989325572a3c"
WIRE_BYTES = 2_109_444
MODEL = Path("/home/dnola/models/hf/DeepSeek-V4-Flash-0731")
CLAIM = Path("/home/dnola/HOST_CLAIM.json")
MISSION = Path("/home/dnola/missions/QTIP2_V7_U4_KLD_t_5c4c9419_s2")
ROOT = MISSION
FREEZE = MISSION / "inputs/FINAL_43_ROUTE_CENSUS_RUN1698.json"
STAGE_ROOT = Path("/dev/shm/PRE_SCORE_t_9e5a36e1_track_b_s2/runtime_layer")
CACHE_ROOT = Path("/dev/shm/PRE_SCORE_t_9e5a36e1_track_b_s2/cache")
PRESTAGED_L024 = ROOT / "support/L024"
PROGRESS = Path("/home/dnola/missions/PRE_SCORE_t_9e5a36e1/track-b-s2/receipts/RUNTIME_PROGRESS.json")
BUNDLE = Path("/home/dnola/missions/QTIP2_V7_t_a9b5af65_s2/incoming/attempt9_production_v1/python")
COMPLETE34_ROOT = Path("/dev/shm/QTIP2_V7_U4_KLD_t_5c4c9419_s2/l034")
COMPLETE34_TERMINAL = COMPLETE34_ROOT / "L034_SELECTED_WIRE_STAGE_TERMINAL.json"
COMPLETE34_TERMINAL_SHA = "3f2325acd4075bb4c90d65bfcb18f1d35ade5c0806c4312d9ef45f963a1a28f5"
COMPLETE34_BINDING = COMPLETE34_ROOT / "L034_SELECTED_WIRE_BINDING.json"
COMPLETE34_BINDING_SHA = "418c1cd803413fb0cfad3ae93eae6ac93095de00e106114a64c6b5f7983286a5"
CHECKPOINT = Path(os.environ["BANANA_SMASHER_CHECKPOINT"])
CHECKPOINT_SHA = os.environ["BANANA_SMASHER_CHECKPOINT_SHA256"]
CANDIDATE_IDENTITY = os.environ["BANANA_SMASHER_CANDIDATE_IDENTITY"]
BUILDER: Any = None


def sha256(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def atomic_json(path: Path, value: object) -> None:
    payload = canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        dfd = os.open(path.parent, os.O_RDONLY)
        os.fsync(dfd)
        os.close(dfd)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_authority() -> dict:
    claim = json.loads(CLAIM.read_text())
    if (
        claim.get("task_id") != TASK
        or claim.get("intended_basis") != BASIS
        or claim.get("state", claim.get("status")) != "CLAIMED"
        or float(claim.get("lease_until_unix", claim.get("lease_expires_unix", claim.get("expiry_unix", 0)))) <= time.time()
        or int(claim.get("workload_pid", -1)) != os.getpid()
        or int(claim.get("workload_startticks", -1)) != int(Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19])
    ):
        raise RuntimeError("stage7 host claim authority refused")
    return claim


def candidate_lut(layer: int):
    import torch
    if sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise RuntimeError("serialized checkpoint SHA refused")
    document = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if document.get("format") != "banana-smasher-qtip2-v7-joint-checkpoint-v1" or document.get("identity_sha256") != CANDIDATE_IDENTITY or int(document.get("next_update", -1)) != 0:
        raise RuntimeError("serialized checkpoint identity refused")
    key = f"layers.{layer}.qtip2_v7.layer_lut"
    rows = document["state"]["luts"]
    if set(rows) != {f"layers.{i}.qtip2_v7.layer_lut" for i in range(43)}:
        raise RuntimeError("serialized checkpoint LUT coverage refused")
    value = rows[key]
    if tuple(value.shape) != (1024,) or value.dtype != torch.float32:
        raise RuntimeError(f"serialized checkpoint LUT geometry refused L{layer:03d}")
    return value.to(BUILDER.DEV).to(torch.float16)


# Each route names only immutable transfer sources already admitted by FREEZE.
# layout: nested => E000/w1.{ext}; flat => E000_w1.{ext}.
ROUTES: dict[int, dict[str, Any]] = {}
for layer in range(0, 6):
    ROUTES[layer] = {
        "kind": "nas", "relay": "192.168.200.7",
        "source": f"/home/dnola/t_a9995518_s7_nas/banana-smasher/QTIP2_V7_t_54aa4a00_s1/L000-L005/wire/L{layer:03d}",
        "layout": "nested", "ext": "q2v7wire",
    }
ROUTES[6] = {
    "kind": "ssh", "host": "192.168.200.6",
    "source": "/home/dnola/missions/QTIP2_V7_t_a9b5af65_transfer_only/product_archive/L006/compact",
    "layout": "flat", "ext": "q2v7wire",
}
for layer in range(7, 12):
    ROUTES[layer] = {
        "kind": "nas", "relay": "192.168.200.7",
        "source": f"/volume1/dnola/banana-smasher/qtip2-v7/t_a9b5af65/L006-L011/layers/L{layer:03d}/compact",
        "layout": "flat", "ext": "q2v7wire",
    }
for layer in range(12, 14):
    ROUTES[layer] = {
        "kind": "nas", "relay": "192.168.200.7",
        "source": f"/volume1/dnola/banana-smasher/qtip2-v7/t_645797e8/L012-L017/layers/L{layer:03d}/compact",
        "layout": "flat", "ext": "q2v7wire",
    }
for layer in range(18, 24):
    ROUTES[layer] = {
        "kind": "nas", "relay": "192.168.200.7",
        "source": f"/home/dnola/t_a9995518_s7_nas/banana-smasher/QTIP2_V7_FANIN_t_bb990c93_s8/slot4_t_aa8d4e46/product_artifacts/L{layer:03d}",
        "layout": "nested", "ext": "k2wire",
    }
ROUTES[24] = {
    "kind": "ssh", "host": "192.168.200.4",
    "source": "/home/dnola/missions/QTIP2_V7_L024_L028_t_5367791c_s4/sealed_product/product_artifacts/L024",
    "layout": "nested", "ext": "k2wire",
}
for layer in range(29, 32):
    ROUTES[layer] = {
        "kind": "nas", "relay": "192.168.200.7",
        "source": f"/home/dnola/t_a9995518_s7_nas/banana-smasher/QTIP2_V7_ALL43_t_cf18512f_s6/L{layer:03d}/compact",
        "layout": "flat", "ext": "q2v7wire",
    }
ROUTES[39] = {
    "kind": "nas", "relay": "192.168.200.7",
    "source": "/home/dnola/t_a9995518_s7_nas/banana-smasher/QTIP2_V7_ALL43_t_bb990c93_s8_fanin_store/L039/compact",
    "verify_terminal": str(MISSION / "inputs/L039_PRODUCT_LAYER_TERMINAL.json"),
    "verify_terminal_sha256": "1ee173b3a5bdb7966684d0027054a68eb35433ee5d26e184198f71675bc2c56a",
    "layout": "flat", "ext": "q2v7wire",
}
ROUTES[40] = {
    "kind": "nas", "relay": "192.168.200.7",
    "source": "/home/dnola/banana-smasher/t_bb990c93_qtip2_v7_all43/L040/compact",
    "layout": "flat", "ext": "q2v7wire",
}
ROUTES[41] = {
    "kind": "ssh", "host": "192.168.200.7",
    "source": "/home/dnola/missions/QTIP2_V7_L024_L028_t_2d1f3fa5_s5w/incoming/fanin_s8/L041/compact",
    "layout": "flat", "ext": "q2v7wire",
}
ROUTES[42] = {
    "kind": "split",
    "parts": [
        {"host": "192.168.200.1", "source": "/home/dnola/missions/QTIP2_V7_t_54aa4a00_s1/incoming/fanin_s8/L042_w1/compact", "projections": ["w1"]},
        {"host": "192.168.200.6", "source": "/home/dnola/missions/QTIP2_V7_ALL43_t_bb990c93_s8_fanin_store/L042_w2w3/compact", "projections": ["w2", "w3"]},
    ],
    "layout": "flat", "ext": "q2v7wire",
}
# The authenticated census is the source of truth for prestaged rows omitted by
# the original hand-written route table (notably L014-L017 and L025-L038).
# Preserve the narrow manual overrides above; fill only genuinely missing rows.
for _row in json.loads(FREEZE.read_text())["rows"]:
    _layer = int(_row["layer"])
    if _layer == 34 or _layer in ROUTES:
        continue
    _route = dict(_row["route"])
    if "source" not in _route and "root" in _route:
        _route["source"] = _route["root"]
    ROUTES[_layer] = _route


class PlaneSource:
    def __init__(self, _contract_path: str):
        if BUILDER is None:
            raise RuntimeError("PlaneSource builder binding missing")
        require_authority()
        if sha256(MODEL / "model.safetensors.index.json") != BASIS:
            raise RuntimeError("source model basis gate refused")
        if sha256(FREEZE) != FREEZE_SHA:
            raise RuntimeError("freeze manifest identity refused")
        self.freeze = json.loads(FREEZE.read_text())
        self.selected = set(self.freeze["frozen_layers"])
        expected = set(range(43))
        if self.selected != expected:
            raise RuntimeError("frozen layer set drift")
        self.rows = {int(row["layer"]): row for row in self.freeze["rows"]}
        if set(self.rows) != expected:
            raise RuntimeError("route census layer rows drift")
        self.routes = {layer: dict(ROUTES[layer]) for layer in expected if layer != 34}
        self.weight_map = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]
        self.handles: dict[str, Any] = {}
        self.active_stage: Path | None = None
        self.active_route: dict[str, Any] | None = None
        self.active_lut = None
        self.complete34 = None
        self.counters = {
            "compact_layers_touched": [], "source_native_layers_touched": [],
            "compact_member_payloads_read": 0, "compact_payload_bytes_read": 0,
            "compact_shared_lut_bytes_read": 0, "source_native_payload_bytes_read": 0,
            "pass_through_bytes": 0, "hidden_fp32_control_bytes": 0, "fallback_calls": 0,
            "local_staged_layers": [], "local_staged_count": 0,
            "nas_bulk_tar_layers": [], "nas_bulk_tar_bytes": 0,
        }
        sys.path.insert(0, str(BUNDLE))
        from packed4_bs import qtip_k2
        self.q2 = qtip_k2
        self._write_progress(status="INITIALIZED", active_layer=None)

    def _write_progress(self, *, status: str, active_layer: int | None) -> None:
        atomic_json(PROGRESS, {
            "schema": "banana-smasher.qtip2_v7.partial_anchor_balanced64_runtime_progress.v1",
            "status": status, "task_id": TASK, "board_run_id": RUN,
            "basis_sha256": BASIS, "freeze_manifest_sha256": FREEZE_SHA,
            "active_layer": active_layer, "runtime_counters": self.counters,
            "updated_unix": time.time(),
        })

    def _cleanup_previous(self) -> None:
        self.active_lut = None
        self.complete34 = None
        gc.collect()
        import torch
        torch.cuda.empty_cache()
        if self.active_stage is not None and self.active_stage == STAGE_ROOT and STAGE_ROOT.exists():
            try:
                shutil.rmtree(STAGE_ROOT)
            except PermissionError:
                # Producer ownership preserved by authenticated tar can make a
                # fully-consumed stage non-removable. Detach it atomically so
                # the next layer can proceed; it remains read-only evidence.
                retired = STAGE_ROOT.with_name(f"{STAGE_ROOT.name}.retired.{int(time.time_ns())}")
                os.replace(STAGE_ROOT, retired)
        self.active_stage = None
        self.active_route = None

    def _source_get(self, name: str):
        from safetensors import safe_open
        shard = self.weight_map[name]
        if shard not in self.handles:
            if len(self.handles) >= 3:
                self.handles.pop(next(iter(self.handles)))
            self.handles[shard] = safe_open(MODEL / shard, framework="pt")
        value = self.handles[shard].get_tensor(name)
        self.counters["source_native_payload_bytes_read"] += value.numel() * value.element_size()
        return value

    def _source_expert(self, layer: int, expert: int, which: str):
        import torch
        pre = f"layers.{layer}.ffn.experts.{expert}."
        names = ("w1", "w3") if which == "13" else ("w2",)
        values = []
        for name in names:
            wb = self._source_get(pre + name + ".weight").view(torch.uint8).to(BUILDER.DEV)
            sb = self._source_get(pre + name + ".scale").view(torch.uint8).to(BUILDER.DEV)
            values.append(BUILDER.deq_fp4_block32(wb, sb, "e2m1"))
            del wb, sb
        return torch.cat(values, dim=0) if which == "13" else values[0]

    @staticmethod
    def _run_checked(argv: list[str], *, timeout: int) -> None:
        p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        if p.returncode:
            raise RuntimeError(f"transfer failed rc={p.returncode} argv={argv[:5]} stderr={p.stderr.decode(errors='replace')[-2000:]}")

    def _stage_ssh_dir(self, host: str, source: str, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        # The receiving rsync runs as dnola for the authorized SSH identity.
        # Bind only this fresh task-owned staging leaf to that exact UID/GID so
        # rsync can create its atomic temporary files; the root-owned parent
        # remains unchanged and cleanup stays under this workload.
        account = __import__("pwd").getpwnam("dnola")
        os.chown(destination, account.pw_uid, account.pw_gid)
        os.chmod(destination, 0o700)
        self._run_checked([
            "sudo", "-u", "dnola", "rsync", "-a", "--partial", "--timeout=180",
            "-e", "ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4",
            f"{host}:{source.rstrip('/')}/", str(destination) + "/",
        ], timeout=1800)

    def _stage_nas(self, route: dict[str, Any], destination: Path) -> None:
        wire = destination / "wire"
        wire.mkdir(parents=True, exist_ok=True)
        physical_source = route["source"]
        if not (physical_source.startswith("/home/dnola/") or physical_source.startswith("/volume1/dnola/")):
            raise RuntimeError(f"NAS physical-path mapping refused: {physical_source}")
        nas_command = shlex.join(["tar", "-C", physical_source, "-cf", "-", "."])
        gateway_command = shlex.join([
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            "192.168.88.100", nas_command,
        ])
        started = time.monotonic()
        producer = subprocess.Popen([
            "sudo", "-u", "dnola", "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            route["relay"], gateway_command,
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert producer.stdout is not None
        consumer = subprocess.run(["tar", "-C", str(wire), "-xf", "-"], stdin=producer.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
        producer.stdout.close()
        rc = producer.wait(timeout=60)
        error = producer.stderr.read().decode(errors="replace") if producer.stderr else ""
        if rc or consumer.returncode:
            raise RuntimeError(f"NAS bulk-tar failed producer={rc} consumer={consumer.returncode} stderr={error[-2000:]} tar={consumer.stderr.decode(errors='replace')[-1000:]}")
        files = [p for p in wire.rglob(f"*.{route['ext']}") if p.is_file()]
        if len(files) != 768:
            raise RuntimeError(f"NAS bulk-tar inventory refused files={len(files)} source={physical_source}")
        if route.get("verify_terminal"):
            terminal_path = Path(route["verify_terminal"])
            if sha256(terminal_path) != route["verify_terminal_sha256"]:
                raise RuntimeError(f"NAS source terminal identity refused: {terminal_path}")
            terminal = json.loads(terminal_path.read_text())
            if terminal.get("basis") != BASIS or terminal.get("status") != "PASS" or int(terminal.get("complete_members", -1)) != 768:
                raise RuntimeError(f"NAS source terminal contract refused: {terminal_path}")
            expected = {
                row["member"].split("/", 1)[1].replace("/", "_") + ".q2v7wire": row["artifact_sha256"]
                for row in terminal["members"]
            }
            if set(expected) != {p.name for p in files}:
                raise RuntimeError(f"NAS source file-set refused: {physical_source}")
            bad = [p.name for p in files if p.stat().st_size != WIRE_BYTES or sha256(p) != expected[p.name]]
            if bad:
                raise RuntimeError(f"NAS source member readback refused: {bad[:8]}")
        self.counters["nas_bulk_tar_bytes"] += sum(p.stat().st_size for p in files)
        self.counters["last_nas_bulk_tar"] = {
            "physical_source": physical_source,
            "files": len(files),
            "bytes": sum(p.stat().st_size for p in files),
            "seconds": time.monotonic() - started,
            "gateway": route["relay"],
            "nas": "192.168.88.100",
        }

    def _stage_nas_tranches(self, route: dict[str, Any], destination: Path) -> None:
        raw_root = destination / "wire"
        raw_root.mkdir(parents=True, exist_ok=True)
        physical_source = route["source"]
        nas_command = shlex.join(["tar", "-C", physical_source, "-cf", "-", "."])
        gateway_command = shlex.join([
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            "192.168.88.100", nas_command,
        ])
        producer = subprocess.Popen([
            "sudo", "-u", "dnola", "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            route["relay"], gateway_command,
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert producer.stdout is not None
        consumer = subprocess.run(["tar", "-C", str(raw_root), "-xf", "-"], stdin=producer.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
        producer.stdout.close()
        rc = producer.wait(timeout=60)
        error = producer.stderr.read().decode(errors="replace") if producer.stderr else ""
        if rc or consumer.returncode:
            raise RuntimeError(f"NAS tranche tar failed producer={rc} consumer={consumer.returncode} stderr={error[-2000:]} tar={consumer.stderr.decode(errors='replace')[-1000:]}")
        candidates = [p for p in raw_root.rglob(f"*.{route['ext']}") if p.is_file() and p.stat().st_size == WIRE_BYTES]
        if len(candidates) != 768:
            raise RuntimeError(f"NAS tranche compact inventory refused files={len(candidates)} source={physical_source}")

    @staticmethod
    def _flatten_staged_wire(route: dict[str, Any]) -> None:
        ext = route["ext"]
        source_root = STAGE_ROOT / "wire"
        candidates = [p for p in source_root.rglob(f"*.{ext}") if p.is_file()]
        normalized: dict[str, Path] = {}
        for path in candidates:
            if path.name.startswith("E") and "_" in path.stem:
                member = path.stem
            elif path.parent.name.startswith("E") and path.stem in {"w1", "w2", "w3"}:
                member = f"{path.parent.name}_{path.stem}"
            else:
                continue
            if len(member) != 7 or member[0] != "E" or member[4] != "_" or member[5:] not in {"w1", "w2", "w3"}:
                continue
            if path.stat().st_size != WIRE_BYTES:
                raise RuntimeError(f"compact size drift {path}")
            prior = normalized.get(member)
            if prior is not None and sha256(prior) != sha256(path):
                raise RuntimeError(f"staged conflicting duplicate {member}")
            normalized[member] = path
        expected = {f"E{expert:03d}_{projection}" for expert in range(256) for projection in ("w1", "w2", "w3")}
        if set(normalized) != expected:
            raise RuntimeError(f"staged inventory refused files={len(normalized)} gaps={len(expected-set(normalized))}")
        flat = STAGE_ROOT / "wire_flat"
        flat.mkdir(parents=True, exist_ok=True)
        for member, source in normalized.items():
            target = flat / f"{member}.{ext}"
            try:
                os.link(source, target)
            except OSError:
                shutil.copyfile(source, target)
        # Some authenticated tar sources preserve producer ownership/modes that
        # make the extracted tree non-removable by this task.  Keep that tree;
        # consume the complete task-owned flat copy instead of failing cleanup.
        try:
            shutil.rmtree(source_root)
        except PermissionError:
            source_root = STAGE_ROOT / "wire_original"
            os.replace(STAGE_ROOT / "wire", source_root)
        os.replace(flat, STAGE_ROOT / "wire")
        route["layout"] = "flat"

    @staticmethod
    def _normalize_route(layer: int, raw: dict[str, Any]) -> dict[str, Any]:
        route = dict(raw)
        kind = route.get("kind")
        if kind in {"nas_sftp", "nas_shell", "nas_shell_stream"}:
            source = route.get("source", route.get("root"))
            if source.startswith("/home/t_"):
                source = "/home/dnola/" + source.removeprefix("/home/")
            elif source.startswith("/home/banana-smasher/"):
                source = "/home/dnola/" + source.removeprefix("/home/")
            route.update(kind="nas", relay="192.168.200.7", source=source)
            route.setdefault("layout", "flat")
            route.setdefault("ext", "q2v7wire")
        elif kind == "nas_sftp_tranches":
            source = route.get("source", route.get("root"))
            if source.startswith("/home/t_"):
                source = "/home/dnola/" + source.removeprefix("/home/")
            route.update(kind="nas_tranches", relay="192.168.200.7", source=source, layout="flat", ext="q2v7wire")
        elif kind in {"ssh", "ssh_transfer_manifest_full_tree"}:
            route.update(kind="ssh", source=route.get("source", route.get("root")))
            route.setdefault("layout", "nested" if route.get("ext") == "k2wire" else "flat")
            route.setdefault("ext", "k2wire" if route["layout"] == "nested" and 24 <= layer <= 28 else "q2v7wire")
        elif kind == "local":
            route.update(kind="ssh", host="192.168.200.7", source=route.get("source", route.get("root")), layout=route.get("layout", "flat"), ext=route.get("ext", "q2v7wire"))
        elif kind == "split":
            parts = []
            for part0 in route["parts"]:
                part = dict(part0)
                part["source"] = part.get("source", part.get("root"))
                parts.append(part)
            route["parts"] = parts
            route.setdefault("layout", "flat")
            route.setdefault("ext", "q2v7wire")
        if route.get("kind") not in {"nas", "nas_tranches", "ssh", "prestage", "split"}:
            raise RuntimeError(f"unsupported final route L{layer:03d}: {route}")
        return route

    def _stage_compact(self, layer: int) -> tuple[Path, dict[str, Any]]:
        route = self._normalize_route(layer, self.routes[layer])
        cached = CACHE_ROOT / f"L{layer:03d}"
        cache_marker = cached / "CACHE_ADMISSION.json"
        if cached.is_dir() and cache_marker.is_file():
            marker = json.loads(cache_marker.read_text())
            if marker.get("status") != "PASS" or int(marker.get("layer", -1)) != layer or marker.get("basis_sha256") != BASIS or int(marker.get("files", -1)) != 768:
                raise RuntimeError(f"verified cache marker refused L{layer:03d}")
            return cached, route
        if route["kind"] == "prestage":
            stage = Path(route["source"])
            if not stage.is_dir():
                raise FileNotFoundError(stage)
            return stage, route
        if STAGE_ROOT.exists():
            shutil.rmtree(STAGE_ROOT)
        STAGE_ROOT.mkdir(parents=True)
        if route["kind"] == "ssh":
            self._stage_ssh_dir(route["host"], route["source"], STAGE_ROOT / "wire")
        elif route["kind"] == "nas":
            self._stage_nas(route, STAGE_ROOT)
        elif route["kind"] == "nas_tranches":
            self._stage_nas_tranches(route, STAGE_ROOT)
        elif route["kind"] == "split":
            for part in route["parts"]:
                label = "_".join(part.get("projections", [])) or part["host"].replace(".", "_")
                temp = STAGE_ROOT / ("part_" + label)
                self._stage_ssh_dir(part["host"], part["source"], temp)
                if part.get("projections"):
                    selected = [temp / f"E{expert:03d}_{projection}.{route['ext']}" for projection in part["projections"] for expert in range(256)]
                else:
                    selected = [p for p in temp.rglob(f"*.{route['ext']}") if p.is_file()]
                for src in selected:
                    dst = STAGE_ROOT / "wire" / src.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if dst.exists() and sha256(dst) != sha256(src):
                        raise RuntimeError(f"split route collision {src.name}")
                    if not dst.exists():
                        os.replace(src, dst)
                shutil.rmtree(temp)
        else:
            raise RuntimeError(route)
        self._flatten_staged_wire(route)
        if layer not in self.counters["local_staged_layers"]:
            self.counters["local_staged_layers"].append(layer)
            self.counters["local_staged_count"] = len(self.counters["local_staged_layers"])
        if route["kind"] in {"nas", "nas_tranches"} and layer not in self.counters["nas_bulk_tar_layers"]:
            self.counters["nas_bulk_tar_layers"].append(layer)
        return STAGE_ROOT, route

    def _wire_path(self, stage: Path, route: dict[str, Any], expert: int, projection: str) -> Path:
        base = stage if route["kind"] == "prestage" else stage / "wire"
        if route["layout"] == "nested":
            return base / f"E{expert:03d}" / f"{projection}.{route['ext']}"
        return base / f"E{expert:03d}_{projection}.{route['ext']}"

    def _decode(self, path: Path, projection: str):
        import torch
        if not path.is_file() or path.stat().st_size != WIRE_BYTES:
            raise RuntimeError(f"compact wire size/path refused {path}")
        raw = path.read_bytes()
        if projection == "w2":
            packed_shape, suh_count, svh_count = (128, 256, 32), 2048, 4096
            expected_shape = (4096, 2048)
        elif projection in ("w1", "w3"):
            packed_shape, suh_count, svh_count = (256, 128, 32), 4096, 2048
            expected_shape = (2048, 4096)
        else:
            raise RuntimeError(f"unknown compact projection {projection}")
        packed_end = 2_097_152
        suh_end = packed_end + 2 * suh_count
        svh_end = suh_end + 2 * svh_count
        if svh_end != 2_109_440:
            raise RuntimeError(f"compact layout accounting refused {projection} {svh_end}")
        packed = torch.from_numpy(np.frombuffer(raw[:packed_end], dtype="<i2").copy().reshape(packed_shape)).to(BUILDER.DEV)
        suh = torch.from_numpy(np.frombuffer(raw[packed_end:suh_end], dtype="<f2").copy()).to(BUILDER.DEV)
        svh = torch.from_numpy(np.frombuffer(raw[suh_end:svh_end], dtype="<f2").copy()).to(BUILDER.DEV)
        decoded = self.q2.decode_k2_matrix(packed, self.active_lut)
        physical = self.q2.inverse_transform(decoded, suh.float(), svh.float()).T.contiguous().to(torch.bfloat16)
        if tuple(physical.shape) != expected_shape:
            raise RuntimeError(f"compact physical shape refused {projection} {tuple(physical.shape)} expected={expected_shape}")
        self.counters["compact_member_payloads_read"] += 1
        self.counters["compact_payload_bytes_read"] += len(raw)
        n = self.counters["compact_member_payloads_read"]
        if n % 32 == 0:
            self._write_progress(status="RUNNING", active_layer=self.counters["compact_layers_touched"][-1])
        del raw, packed, suh, svh, decoded
        return physical

    def _load_complete34(self):
        import torch
        if sha256(COMPLETE34_TERMINAL) != COMPLETE34_TERMINAL_SHA or sha256(COMPLETE34_BINDING) != COMPLETE34_BINDING_SHA:
            raise RuntimeError("L034 selected-wire terminal/binding identity refused")
        terminal = json.loads(COMPLETE34_TERMINAL.read_text())
        binding = json.loads(COMPLETE34_BINDING.read_text())
        if (terminal.get("status") != "PASS" or terminal.get("member_count") != 768
                or terminal.get("member_bytes_total") != 1_620_052_992
                or terminal.get("shared_lut_sha256") != LUT_SHA
                or binding.get("member_count") != 768
                or binding.get("shared_lut_sha256") != LUT_SHA):
            raise RuntimeError("L034 selected-wire contract refused")
        rows = {(int(row["expert"]), str(row["projection"])): row for row in binding["members"]}
        if set(rows) != {(e, p) for e in range(256) for p in ("w1", "w2", "w3")}:
            raise RuntimeError("L034 selected-wire coverage refused")
        self.active_lut = candidate_lut(34)
        self.counters["compact_shared_lut_bytes_read"] += 2048
        self.counters["compact_layers_touched"].append(34)
        self.counters["local_staged_layers"].append(34)
        self.counters["local_staged_count"] = len(self.counters["local_staged_layers"])
        self._write_progress(status="RUNNING_PRESERVED_L034", active_layer=34)
        def read(expert: int, projection: str):
            row = rows[(expert, projection)]
            path = COMPLETE34_ROOT / row["path"]
            if path.stat().st_size != WIRE_BYTES or sha256(path) != row["sha256"]:
                raise RuntimeError(f"L034 selected-wire member identity refused E{expert:03d}/{projection}")
            return self._decode(path, projection)
        return read

    def _predecode_layer(self, read):
        """Materialize one admitted layer with four independent CUDA streams."""
        import torch
        from concurrent.futures import ThreadPoolExecutor

        workers = 4
        streams = [torch.cuda.Stream(device=BUILDER.DEV) for _ in range(workers)]

        def decode(item):
            ordinal, expert, projection = item
            with torch.cuda.stream(streams[ordinal % workers]):
                return (expert, projection), read(expert, projection)

        items = [
            (ordinal, expert, projection)
            for ordinal, (expert, projection) in enumerate(
                (e, p) for e in range(256) for p in ("w1", "w2", "w3")
            )
        ]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            decoded = dict(pool.map(decode, items))
        torch.cuda.synchronize(BUILDER.DEV)
        return lambda expert, which: torch.cat(
            [decoded[(expert, "w1")], decoded[(expert, "w3")]], dim=0
        ) if which == "13" else decoded[(expert, "w2")]

    def layer(self, layer: int):
        import torch
        require_authority()
        self._cleanup_previous()
        if layer not in self.selected:
            if layer not in self.counters["source_native_layers_touched"]:
                self.counters["source_native_layers_touched"].append(layer)
            self._write_progress(status="RUNNING_SOURCE_NATIVE", active_layer=layer)
            return (lambda expert, which: self._source_expert(layer, expert, which)), (256, 4096, 4096, 4096, 2048)
        if layer == 34:
            read = self._load_complete34()
            return self._predecode_layer(read), (256, 4096, 4096, 4096, 2048)
        stage, route = self._stage_compact(layer)
        if layer not in self.counters["local_staged_layers"]:
            self.counters["local_staged_layers"].append(layer)
            self.counters["local_staged_count"] = len(self.counters["local_staged_layers"])
        self.active_stage, self.active_route = stage, route
        paths = [self._wire_path(stage, route, expert, projection) for expert in range(256) for projection in ("w1", "w2", "w3")]
        bad = [str(path) for path in paths if not path.is_file() or path.stat().st_size != WIRE_BYTES]
        if bad:
            raise RuntimeError(f"compact layer admission refused L{layer}: {bad[:8]}")
        import torch
        self.active_lut = candidate_lut(layer)
        self.counters["compact_shared_lut_bytes_read"] += 2048
        if layer not in self.counters["compact_layers_touched"]:
            self.counters["compact_layers_touched"].append(layer)
        self._write_progress(status="RUNNING_COMPACT", active_layer=layer)
        def read(expert: int, projection: str):
            return self._decode(self._wire_path(stage, route, expert, projection), projection)
        return self._predecode_layer(read), (256, 4096, 4096, 4096, 2048)
