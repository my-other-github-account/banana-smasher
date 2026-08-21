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
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np

TASK = "t_03c6894c"
RUN = 1698
CLAIM_RUN = int(os.environ.get("W328_CLAIM_RUN_ID", "4725"))
BASIS = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
FREEZE_SHA = "2dcc28497deb834164be26e267fdf4c30cc951342c73f47ce78b207354275fc9"
LUT_SHA = "1fcb3546038bc65ab7847ef4473a2d1a8c66631315655c1b3d9f989325572a3c"
WIRE_BYTES = 2_109_444
ROOT = Path("/home/dnola/missions/W328_SEALED_RECON_t_03c6894c_s5w")
FREEZE = ROOT / "receipts/FINAL_43_ROUTE_CENSUS_RUN1698.json"
MODEL = Path("/home/dnola/models/hf/DeepSeek-V4-Flash-0731")
CLAIM = Path("/home/dnola/HOST_CLAIM.json")
STAGE_ROOT = Path("/dev/shm/W328_SEALED_RECON_t_03c6894c/runtime_layer")
CACHE_ROOT = Path("/dev/shm/W328_SEALED_RECON_t_03c6894c/cache")
PRESTAGED_L024 = ROOT / "l024"
PROGRESS = ROOT / "receipts/W328_RECON_RUNTIME_PROGRESS.json"
BUNDLE = ROOT / "bundles/attempt9_production_v1/python"
COMPLETE34_ROOT = ROOT / "l034"
COMPLETE34_PROVIDER = ROOT / "code/complete_provider_recovery4_local30.py"
COMPLETE34_PROVIDER_SHA = "37addd4d86479194d15eb727a17a7920aa2bcc063f1645e74a5bcf7b24c60780"
COMPLETE34_ROSTER = COMPLETE34_ROOT / "COMPLETE_768_ROSTER.json"
COMPLETE34_ROSTER_SHA = "13aaa61931aa362a355854aad7bfdb78db328833dfcb83f2444435d058ad2140"
EXACT34_PROVIDER = ROOT / "code/exact_k2_provider.py"
EXACT34_PROVIDER_SHA = "04b70b06450bf94320543e0f34b806f8ac705382fad04dc6b4e6cc401fd9bb7c"
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
        or int(claim.get("owner_run_id", -1)) != CLAIM_RUN
        or claim.get("intended_basis") != BASIS
        or claim.get("status") != "CLAIMED"
        or float(claim.get("lease_expires_unix", claim.get("expiry_unix", 0))) <= time.time()
    ):
        raise RuntimeError("stage7 host claim authority refused")
    return claim


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
    "kind": "prestage", "source": str(PRESTAGED_L024 / "product_artifacts/L024"),
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
    "verify_terminal": "/home/dnola/missions/QTIP2_V7_ALL43_t_bb990c93_s8/product/L039/receipts/PRODUCT_LAYER_TERMINAL.json",
    "verify_terminal_sha256": "1ee173b3a5bdb7966684d0027054a68eb35433ee5d26e184198f71675bc2c56a",
    "layout": "flat", "ext": "q2v7wire",
}
ROUTES[40] = {
    "kind": "nas", "relay": "192.168.200.7",
    "source": "/home/dnola/banana-smasher/t_bb990c93_qtip2_v7_all43/L040/compact",
    "layout": "flat", "ext": "q2v7wire",
}
ROUTES[41] = {
    "kind": "ssh", "host": "192.168.200.6",
    "source": "/home/dnola/missions/QTIP2_V7_L024_L028_t_2d1f3fa5_s5w/incoming/fanin_s8/L041/compact",
    "layout": "flat", "ext": "q2v7wire",
}
ROUTES[42] = {
    "kind": "split",
    "parts": [
        {"host": "192.168.200.2", "source": "/home/dnola/missions/QTIP2_V7_t_54aa4a00_s1/incoming/fanin_s8/L042_w1/compact", "projections": ["w1"]},
        {"host": "192.168.200.6", "source": "/home/dnola/missions/QTIP2_V7_ALL43_t_bb990c93_s8_fanin_store/L042_w2w3/compact", "projections": ["w2", "w3"]},
    ],
    "layout": "flat", "ext": "q2v7wire",
}


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
        self.routes = {layer: dict(self.rows[layer]["route"]) for layer in expected if layer != 34}
        self.routes[24].update({
            "kind": "prestage",
            "source": str(PRESTAGED_L024 / "product_artifacts/L024"),
            "layout": "nested",
            "ext": "k2wire",
        })
        self.routes[39]["verify_terminal"] = str(ROOT / "receipts/L039_PRODUCT_LAYER_TERMINAL.json")
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
            shutil.rmtree(STAGE_ROOT)
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
        errors: list[str] = []
        for attempt in range(3):
            p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
            if p.returncode == 0:
                return
            errors.append(f"attempt={attempt + 1} rc={p.returncode} stderr={p.stderr.decode(errors='replace')[-2000:]}")
            if attempt < 2:
                time.sleep(10)
        raise RuntimeError(f"transfer failed argv={argv[:5]} {' | '.join(errors)}")

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
            os.link(source, flat / f"{member}.{ext}")
        shutil.rmtree(source_root)
        os.replace(flat, source_root)
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
        lut_path = PRESTAGED_L024 / "artifacts/config/L024_PARENT_LUT.fp16.bin"
        if not lut_path.is_file() or sha256(lut_path) != LUT_SHA:
            raise RuntimeError("L034 compact-recovery LUT identity refused")
        self.active_lut = torch.from_numpy(np.frombuffer(lut_path.read_bytes(), dtype="<f2").copy()).to(BUILDER.DEV)
        self.counters["compact_shared_lut_bytes_read"] += lut_path.stat().st_size
        if sha256(COMPLETE34_PROVIDER) != COMPLETE34_PROVIDER_SHA or sha256(COMPLETE34_ROSTER) != COMPLETE34_ROSTER_SHA or sha256(EXACT34_PROVIDER) != EXACT34_PROVIDER_SHA:
            raise RuntimeError("L034 provider input identity drift")
        module = load_module("partial_anchor_complete34", COMPLETE34_PROVIDER)
        result = module.load_l034(
            torch=torch, device=BUILDER.DEV, source_state={}, basis_sha256=BASIS,
            config={
                "basis_sha256": BASIS,
                "candidate_roster": {"path": str(COMPLETE34_ROSTER), "sha256": COMPLETE34_ROSTER_SHA},
                "exact_provider": {"path": str(EXACT34_PROVIDER), "sha256": EXACT34_PROVIDER_SHA},
                "exact_config": {},
                "q2_module": self.q2,
                "active_lut": self.active_lut,
            },
        )
        return result

    def layer(self, layer: int):
        require_authority()
        self._cleanup_previous()
        if layer not in self.selected:
            if layer not in self.counters["source_native_layers_touched"]:
                self.counters["source_native_layers_touched"].append(layer)
            self._write_progress(status="RUNNING_SOURCE_NATIVE", active_layer=layer)
            return (lambda expert, which: self._source_expert(layer, expert, which)), (256, 4096, 4096, 4096, 2048)
        if layer == 34:
            self.complete34 = self._load_complete34()
            gate_up, down = self.complete34["gate_up"], self.complete34["down"]
            if layer not in self.counters["compact_layers_touched"]:
                self.counters["compact_layers_touched"].append(layer)
            if layer not in self.counters["local_staged_layers"]:
                self.counters["local_staged_layers"].append(layer)
                self.counters["local_staged_count"] = len(self.counters["local_staged_layers"])
            self._write_progress(status="RUNNING_PRESERVED_L034", active_layer=layer)
            return (lambda expert, which: gate_up[expert] if which == "13" else down[expert]), (256, 4096, 4096, 4096, 2048)
        stage, route = self._stage_compact(layer)
        if layer not in self.counters["local_staged_layers"]:
            self.counters["local_staged_layers"].append(layer)
            self.counters["local_staged_count"] = len(self.counters["local_staged_layers"])
        self.active_stage, self.active_route = stage, route
        paths = [self._wire_path(stage, route, expert, projection) for expert in range(256) for projection in ("w1", "w2", "w3")]
        bad = [str(path) for path in paths if not path.is_file() or path.stat().st_size != WIRE_BYTES]
        if bad:
            raise RuntimeError(f"compact layer admission refused L{layer}: {bad[:8]}")
        lut_path = PRESTAGED_L024 / "artifacts/config/L024_PARENT_LUT.fp16.bin"
        if not lut_path.is_file() or sha256(lut_path) != LUT_SHA:
            raise RuntimeError("shared compact LUT identity refused")
        import torch
        self.active_lut = torch.from_numpy(np.frombuffer(lut_path.read_bytes(), dtype="<f2").copy()).to(BUILDER.DEV)
        self.counters["compact_shared_lut_bytes_read"] += lut_path.stat().st_size
        if layer not in self.counters["compact_layers_touched"]:
            self.counters["compact_layers_touched"].append(layer)
        self._write_progress(status="RUNNING_COMPACT", active_layer=layer)
        return (lambda expert, which: torch.cat([
                    self._decode(self._wire_path(stage, route, expert, "w1"), "w1"),
                    self._decode(self._wire_path(stage, route, expert, "w3"), "w3"),
                ], dim=0) if which == "13" else self._decode(self._wire_path(stage, route, expert, "w2"), "w2")), (256, 4096, 4096, 4096, 2048)
