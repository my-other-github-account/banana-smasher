#!/usr/bin/env python3
"""Exact all-layer QTIP2-V7 joint TRAIN repair runner.

The runtime adapter named by --runtime-module must export
build_joint_v7_runtime(...).  It receives 43 local compact-wire PlaneSource
objects.  Its whole-model objective must call source.wire_lut() for every layer
in every forward; this runner enforces that use and the resulting gradients.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
import random
import sys
import time
import traceback
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

SCHEMA = "banana-smasher-qtip2-v7-joint-runner-v1"
CHECKPOINT_FORMAT = "banana-smasher-qtip2-v7-joint-checkpoint-v1"
UPDATES = 64
BATCH = 4
LAYERS = 43
NORMS = 235
OUTPUTS = 43
NORM_NUMEL = 446080
GAIN_CLAMP = 0.25
LUT_LR = 1.0e-2
NORM_LR = 1.0e-4
OUTPUT_LR = 1.0e-2
COSINE_MIN_RATIO = 0.1


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def install_bytes_once(path: Path, data: bytes) -> None:
    """Durably install immutable bytes without overwriting a winner."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        os.chmod(path, 0o444)
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
        fsync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def install_json_once(path: Path, value: object) -> None:
    install_bytes_once(
        path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    )


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def require_sha(path: Path, expected: str, label: str) -> str:
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(f"{label} SHA drift: {observed} != {expected}: {path}")
    return observed


class CompleteV7MemberResolver:
    """Resolve one member from the complete staged 42-layer V7 parent."""

    def __init__(
        self,
        *,
        full_parent_root: str | Path,
        l034_roster: str | Path | None = None,
    ) -> None:
        self.full_parent_root = Path(full_parent_root).expanduser().resolve()
        self.l034_roster = (
            None if l034_roster is None else Path(l034_roster).expanduser().resolve()
        )

    def resolve(self, *, layer: int, expert: int, projection: str) -> Path:
        if projection not in {"w1", "w2", "w3"}:
            raise ValueError(f"unsupported QTIP V7 projection {projection!r}")
        if int(layer) == 34:
            if self.l034_roster is None:
                raise RuntimeError("L034 resolution requires the selected-wire roster")
            roster = load_json(self.l034_roster)
            rows = roster.get("members")
            if not isinstance(rows, list):
                raise RuntimeError("L034 selected-wire roster coverage drift")
            identities = [
                (int(row["expert"]), str(row["projection"]))
                for row in rows
            ]
            expected = [
                (selected_expert, selected_projection)
                for selected_expert in range(256)
                for selected_projection in ("w1", "w2", "w3")
            ]
            if (
                roster.get("schema")
                != "banana-smasher-qtip2-v7-l034-selected-wire-roster-v1"
                or int(roster.get("layer", -1)) != 34
                or identities != expected
                or len(set(identities)) != 768
            ):
                raise RuntimeError("L034 selected-wire roster coverage drift")
            row = rows[expected.index((int(expert), projection))]
            relative = Path(str(row["path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError("L034 selected-wire path escapes roster root")
            path = (self.l034_roster.parent / relative).resolve()
            if (
                self.l034_roster.parent not in path.parents
                or not path.is_file()
                or path.is_symlink()
                or path.stat().st_size != int(row.get("bytes", -1))
                or sha256_file(path) != str(row.get("sha256"))
            ):
                raise RuntimeError(f"L034 selected-wire member drift: {path}")
            return path
        layer_root = self.full_parent_root / f"L{int(layer):03d}"
        candidates = [
            layer_root / f"E{int(expert):03d}_{projection}.q2v7wire",
            layer_root / f"E{int(expert):03d}_{projection}.k2wire",
        ]
        present = [path.resolve() for path in candidates if path.is_file()]
        if (
            len(present) != 1
            or present[0].is_symlink()
            or present[0].stat().st_size <= 0
        ):
            raise RuntimeError(
                f"L{int(layer):03d} complete-parent member drift: "
                f"E{int(expert):03d}/{projection}"
            )
        return present[0]


def _atomic_torch_save(torch: Any, path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def run_joint_optimizer_steps(
    *,
    parameter_groups: list[Mapping[str, Any]],
    objective: Callable[[], Any],
    optimizer_steps: int,
    checkpoint: str | Path,
    resume: bool = True,
) -> dict[str, Any]:
    """Public joint Adam loop with identity-bound parameter and optimizer resume."""
    import torch

    if isinstance(optimizer_steps, bool) or not isinstance(optimizer_steps, int) or optimizer_steps <= 0:
        raise ValueError("optimizer_steps must be a positive integer")
    names = [str(group.get("name", "")) for group in parameter_groups]
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("joint parameter groups require unique non-empty names")
    normalized = []
    named_parameters: dict[str, list[Any]] = {}
    seen: set[int] = set()
    for group, name in zip(parameter_groups, names, strict=True):
        parameters = list(group.get("params", ()))
        learning_rate = group.get("learning_rate")
        if (
            not parameters
            or not isinstance(learning_rate, (int, float))
            or not math.isfinite(float(learning_rate))
            or float(learning_rate) <= 0
        ):
            raise ValueError(f"invalid joint parameter group {name!r}")
        for parameter in parameters:
            if not isinstance(parameter, torch.nn.Parameter) or id(parameter) in seen:
                raise ValueError(f"joint parameter group {name!r} has invalid/shared parameters")
            seen.add(id(parameter))
        named_parameters[name] = parameters
        normalized.append(
            {"params": parameters, "lr": float(learning_rate), "group_name": name}
        )
    optimizer = torch.optim.Adam(normalized, foreach=False)
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    start_step = 0
    if checkpoint_path.is_file():
        if not resume:
            raise FileExistsError(checkpoint_path)
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved.get("format") != "banana-smasher-joint-optimizer-checkpoint-v1" or saved.get("group_names") != names:
            raise RuntimeError("joint optimizer checkpoint identity drift")
        for name, parameters in named_parameters.items():
            values = saved["parameters"][name]
            if len(values) != len(parameters):
                raise RuntimeError(f"joint optimizer checkpoint cardinality drift: {name}")
            for parameter, value in zip(parameters, values, strict=True):
                parameter.data.copy_(value.to(parameter.device))
        optimizer.load_state_dict(saved["optimizer"])
        start_step = int(saved["next_step"])
    if not 0 <= start_step <= optimizer_steps:
        raise RuntimeError("joint optimizer checkpoint next_step exceeds requested target")

    gradient_coverage = {name: False for name in names}
    movement_coverage = {name: False for name in names}
    objective_before = objective_after = float("nan")
    for step in range(start_step, optimizer_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        if not isinstance(loss, torch.Tensor) or loss.numel() != 1 or not torch.isfinite(loss):
            raise RuntimeError("joint objective must return one finite scalar tensor")
        objective_before = float(loss.detach())
        loss.backward()
        snapshots = {
            name: [parameter.detach().clone() for parameter in parameters]
            for name, parameters in named_parameters.items()
        }
        gradient_coverage = {
            name: all(
                parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all())
                and bool(torch.count_nonzero(parameter.grad).item())
                for parameter in parameters
            )
            for name, parameters in named_parameters.items()
        }
        if not all(gradient_coverage.values()):
            raise RuntimeError(f"joint optimizer gradient coverage failed: {gradient_coverage}")
        optimizer.step()
        movement_coverage = {
            name: all(
                bool(torch.count_nonzero(parameter.detach() - before).item())
                for parameter, before in zip(parameters, snapshots[name], strict=True)
            )
            for name, parameters in named_parameters.items()
        }
        if not all(movement_coverage.values()):
            raise RuntimeError(f"joint optimizer movement coverage failed: {movement_coverage}")
        with torch.no_grad():
            after = objective()
        if not isinstance(after, torch.Tensor) or after.numel() != 1 or not torch.isfinite(after):
            raise RuntimeError("joint post-step objective must be finite")
        objective_after = float(after.detach())
        _atomic_torch_save(
            torch,
            checkpoint_path,
            {
                "format": "banana-smasher-joint-optimizer-checkpoint-v1",
                "group_names": names,
                "next_step": step + 1,
                "parameters": {
                    name: [parameter.detach().cpu().clone() for parameter in parameters]
                    for name, parameters in named_parameters.items()
                },
                "optimizer": optimizer.state_dict(),
                "objective": {"before": objective_before, "after": objective_after},
            },
        )
    return {
        "schema": "banana-smasher-joint-optimizer-result-v1",
        "status": "PASS_UPDATE",
        "optimizer_steps_completed": optimizer_steps,
        "resumed_from_step": start_step,
        "gradient_coverage": gradient_coverage,
        "movement_coverage": movement_coverage,
        "objective": {"before": objective_before, "after": objective_after},
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def cosine_multiplier(step: int) -> float:
    clamped = min(max(int(step), 0), UPDATES)
    cosine = 0.5 * (1.0 + math.cos(math.pi * clamped / UPDATES))
    return COSINE_MIN_RATIO + (1.0 - COSINE_MIN_RATIO) * cosine


def _bound_manifest_file(root: Path, member_root: Path, row: Mapping[str, Any]) -> Path:
    relative = row.get("path")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise RuntimeError(f"unsafe compact-wire relative path: {relative!r}")
    if ".." in Path(relative).parts:
        raise RuntimeError(f"compact-wire path escapes root: {relative!r}")
    base = member_root if row.get("packed_code_bytes") is not None else root
    path = (base / relative).resolve()
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"compact-wire file unavailable/non-regular: {path}")
    declared = row.get("bytes")
    if not isinstance(declared, int) or declared <= 0 or path.stat().st_size != declared:
        raise RuntimeError(f"compact-wire byte drift: {path}")
    digest = row.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError(f"compact-wire row has no SHA-256: {path}")
    return path


class PlaneSource:
    """One fixed local compact-wire layer plus one shared FP32 LUT master.

    The packed trellis, transforms, scales, geometry, assignments, and member
    ordering are immutable descriptors.  Only wire_lut() is differentiable.
    Runtime adapters must use this method rather than copying the LUT.
    """

    def __init__(
        self,
        *,
        torch: Any,
        layer: int,
        roster: Mapping[str, Any],
        device: Any,
        full_wire_hash: bool,
    ) -> None:
        self.torch = torch
        self.layer = int(layer)
        if self.layer != int(roster["layer"]):
            raise RuntimeError("PlaneSource layer roster drift")
        self.name = str(roster["name"])
        manifest_path = Path(str(roster["source_manifest"]["path"])).resolve()
        require_sha(
            manifest_path,
            str(roster["source_manifest"]["sha256"]),
            f"L{self.layer:03d} manifest",
        )
        document = load_json(manifest_path)
        if document.get("schema") != "banana-smasher-qtip-v7-artifact-v1":
            raise RuntimeError(f"L{self.layer:03d} compact-wire schema drift")
        root = manifest_path.parent
        members = document.get("members")
        luts = document.get("layer_luts")
        if not isinstance(members, list) or not isinstance(luts, list):
            raise RuntimeError(f"L{self.layer:03d} malformed compact-wire roster")
        local_members = [row for row in members if int(row.get("layer", -1)) == self.layer]
        if not local_members:
            raise RuntimeError(f"L{self.layer:03d} compact wire has no local members")

        # The diagnostic parent manifests authenticate the per-layer LUT slots but
        # contain only E000. Bind routed members to the separately sealed local
        # complete-parent transport. L034 uses its independently sealed selected-
        # wire roster; all other layers use the exact 42-layer staging terminal.
        expected_identities = [
            (self.layer, expert, projection)
            for expert in range(256)
            for projection in ("w1", "w2", "w3")
        ]
        if self.layer == 34:
            roster_path = Path(os.environ["JOINT_V7_L034_ROSTER"]).resolve()
            roster_sha = os.environ["JOINT_V7_L034_ROSTER_SHA256"]
            require_sha(roster_path, roster_sha, "L034 selected-wire roster")
            selected = load_json(roster_path)
            if (
                selected.get("schema")
                != "banana-smasher-qtip2-v7-l034-selected-wire-roster-v1"
                or int(selected.get("layer", -1)) != 34
                or int(selected.get("member_count", -1)) != 768
                or int(selected.get("selected_payload_bytes", -1)) != 1_620_052_992
            ):
                raise RuntimeError("L034 selected-wire roster contract drift")
            selected_rows = selected.get("members")
            if not isinstance(selected_rows, list) or len(selected_rows) != 768:
                raise RuntimeError("L034 selected-wire roster coverage drift")
            selected_root = roster_path.parent
            local_members = []
            member_paths = []
            for row in selected_rows:
                relative = Path(str(row["path"]))
                if relative.is_absolute() or ".." in relative.parts:
                    raise RuntimeError("L034 selected-wire path escapes roster root")
                path = (selected_root / relative).resolve()
                if (
                    selected_root not in path.parents
                    or not path.is_file()
                    or path.is_symlink()
                    or path.stat().st_size != 2_109_444
                    or int(row.get("bytes", -1)) != 2_109_444
                ):
                    raise RuntimeError(f"L034 selected-wire member drift: {path}")
                local_members.append({
                    "layer": 34,
                    "expert": int(row["expert"]),
                    "projection": str(row["projection"]),
                    "bytes": int(row["bytes"]),
                    "sha256": str(row["sha256"]),
                })
                member_paths.append(path)
        else:
            staged_root = Path(os.environ["JOINT_V7_FULL_PARENT_ROOT"]).resolve()
            resolver = CompleteV7MemberResolver(full_parent_root=staged_root)
            local_members = []
            member_paths = []
            for _layer, expert, projection in expected_identities:
                path = resolver.resolve(
                    layer=self.layer, expert=expert, projection=projection
                )
                if path.stat().st_size != 2_109_444:
                    raise RuntimeError(
                        f"L{self.layer:03d} complete-parent member drift: E{expert:03d}/{projection}"
                    )
                local_members.append({
                    "layer": self.layer,
                    "expert": expert,
                    "projection": projection,
                    "bytes": 2_109_444,
                })
                member_paths.append(path)

        identities = [
            (int(row["layer"]), int(row["expert"]), str(row["projection"]))
            for row in local_members
        ]
        if identities != expected_identities or len(set(identities)) != 768:
            raise RuntimeError(f"L{self.layer:03d} complete compact-wire order/coverage drift")
        if full_wire_hash:
            for row, path in zip(local_members, member_paths, strict=True):
                digest = row.get("sha256")
                if not isinstance(digest, str):
                    raise RuntimeError(
                        f"L{self.layer:03d} full-wire hash requested without per-member SHA"
                    )
                require_sha(path, digest, f"L{self.layer:03d} member")
        lut_rows = [row for row in luts if int(row.get("layer", -1)) == self.layer]
        if len(lut_rows) != 1:
            raise RuntimeError(f"L{self.layer:03d} requires exactly one LUT slot")
        lut_row = lut_rows[0]
        lut_path = _bound_manifest_file(root, root, lut_row)
        wire = roster["wire"]
        if (
            lut_row.get("dtype") != "float16"
            or lut_row.get("shape") != [1024]
            or lut_path.stat().st_size != 2048
            or str(lut_path) != str(Path(str(wire["source_path"])).resolve())
        ):
            raise RuntimeError(f"L{self.layer:03d} LUT geometry/path drift")
        require_sha(lut_path, str(wire["sha256"]), f"L{self.layer:03d} LUT")
        import numpy as np

        initial = torch.from_numpy(
            np.fromfile(lut_path, dtype="<f2").astype("float32", copy=True)
        ).to(device)
        if tuple(initial.shape) != (1024,):
            raise RuntimeError(f"L{self.layer:03d} LUT shape drift")
        self.master = torch.nn.Parameter(initial)
        self.manifest_path = manifest_path
        self.document = document
        self.members = tuple(dict(row) for row in local_members)
        self.member_paths = tuple(member_paths)
        self._uses = 0
        self._stat_identity = self._capture_stats(lut_path)

    def _capture_stats(self, lut_path: Path | None = None) -> dict[str, Any]:
        paths = list(self.member_paths) if hasattr(self, "member_paths") else []
        if lut_path is not None:
            paths.append(lut_path)
        return {
            "manifest": {
                "path": str(self.manifest_path) if hasattr(self, "manifest_path") else "",
                "sha256": sha256_file(self.manifest_path) if hasattr(self, "manifest_path") else "",
            },
            "files": [
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "device": path.stat().st_dev,
                    "inode": path.stat().st_ino,
                    "mtime_ns": path.stat().st_mtime_ns,
                }
                for path in paths
            ],
        }

    def reset_usage(self) -> None:
        self._uses = 0

    @property
    def uses(self) -> int:
        return self._uses

    def wire_lut(self):
        self._uses += 1
        # Preserve the exact FP16 LUT slot in forward while retaining an FP32 master.
        return self.master.to(self.torch.float16).to(self.torch.float32).reshape(512, 2)

    def member_path(self, expert: int, projection: str) -> Path:
        matches = [
            path
            for row, path in zip(self.members, self.member_paths, strict=True)
            if int(row["expert"]) == int(expert) and str(row["projection"]) == projection
        ]
        if len(matches) != 1:
            raise KeyError((self.layer, expert, projection))
        return matches[0]

    def assert_frozen(self) -> None:
        current = self._capture_stats()
        # _capture_stats without lut_path covers the immutable packed members;
        # manifest SHA is rechecked, and the source LUT is identity-bound at launch.
        expected = {
            "manifest": self._stat_identity["manifest"],
            "files": self._stat_identity["files"][: len(self.member_paths)],
        }
        if current != expected:
            raise RuntimeError(f"L{self.layer:03d} frozen compact-wire stat drift")


def runtime_field(runtime: Any, name: str, default: Any = None) -> Any:
    if isinstance(runtime, Mapping):
        return runtime.get(name, default)
    return getattr(runtime, name, default)


def bind_objective(runtime: Any) -> Callable[[list[int], bool], Any]:
    direct = runtime_field(runtime, "objective")
    if callable(direct):
        return lambda wins, requires_grad: direct(wins, requires_grad=requires_grad)
    batch_loss = runtime_field(runtime, "batch_loss")
    if callable(batch_loss):
        return lambda wins, requires_grad: batch_loss(wins, requires_grad=requires_grad)
    base = runtime_field(runtime, "B")
    student = runtime_field(runtime, "student")
    corpus = runtime_field(runtime, "corpus")
    acache = runtime_field(runtime, "activation_cache")
    if base is not None and student is not None and corpus is not None and acache is not None:
        return lambda wins, requires_grad: base.batch_loss(
            student, corpus, acache, wins, requires_grad
        )
    raise RuntimeError(
        "runtime adapter must expose objective(wins, requires_grad=...), "
        "batch_loss(...), or B/student/corpus/activation_cache"
    )


def expose_dense_parameters(torch: Any, student: Any, admission: Mapping[str, Any]):
    from torch.nn.utils import parametrize

    class WireBf16(torch.nn.Module):
        def forward(self, master):
            return master.to(torch.bfloat16)

    def output_hook(module, _inputs, output):
        gain = torch.exp(
            module._banana_smasher_output_log_gain.clamp(-GAIN_CLAMP, GAIN_CLAMP)
        ).to(output.dtype)
        return output * gain

    norms = []
    outputs = []
    modules = list(student.model.named_modules())
    for name, module in modules:
        leaf = name.rsplit(".", 1)[-1].lower()
        wire = module._parameters.get("weight")
        if "norm" not in leaf or wire is None or wire.ndim != 1:
            continue
        before = wire.detach().clone()
        parametrize.register_parametrization(module, "weight", WireBf16(), unsafe=True)
        master = module.parametrizations.weight.original
        master.data = master.data.float()
        master.requires_grad_(True)
        if module.weight.dtype != torch.bfloat16 or not torch.equal(module.weight.detach(), before):
            raise RuntimeError(f"RMSNorm wire identity changed at {name}")
        norms.append((name, module, master))
    for name, module in modules:
        if name.endswith(".self_attn.o_b_proj"):
            wire = module._parameters.get("weight")
            if wire is None or hasattr(module, "_banana_smasher_output_log_gain"):
                raise RuntimeError(f"output-gain seam drift at {name}")
            parameter = torch.nn.Parameter(
                torch.zeros((), dtype=torch.float32, device=wire.device)
            )
            module.register_parameter("_banana_smasher_output_log_gain", parameter)
            module.register_forward_hook(output_hook)
            outputs.append((name + ".output_log_gain", module, parameter))
    expected_norms = admission["trainable_roster"]["rmsnorms"]
    expected_outputs = admission["trainable_roster"]["output_gains"]
    expected_norm_names = [str(row["name"]) for row in expected_norms]
    expected_output_names = [str(row["name"]) for row in expected_outputs]
    norm_by_name = {name: (name, module, parameter) for name, module, parameter in norms}
    output_by_name = {name: (name, module, parameter) for name, module, parameter in outputs}
    if len(norm_by_name) != len(norms) or set(norm_by_name) != set(expected_norm_names):
        raise RuntimeError("exact 235 RMSNorm roster identity drift")
    if len(output_by_name) != len(outputs) or set(output_by_name) != set(expected_output_names):
        raise RuntimeError("exact 43 output-gain roster identity drift")
    # Historical checkpoint/admission order is lexicographic, whereas the live
    # module traversal is architecture order. Canonicalize by the admitted names.
    norms = [norm_by_name[name] for name in expected_norm_names]
    outputs = [output_by_name[name] for name in expected_output_names]
    norm_names = [name for name, _module, _parameter in norms]
    output_names = [name for name, _module, _parameter in outputs]
    if (
        len(norms) != NORMS
        or sum(parameter.numel() for _n, _m, parameter in norms) != NORM_NUMEL
        or norm_names != expected_norm_names
    ):
        raise RuntimeError("exact 235 RMSNorm roster/order/numel drift")
    if len(outputs) != OUTPUTS or output_names != expected_output_names:
        raise RuntimeError("exact 43 output-gain roster/order drift")
    return norms, outputs


def tensor_state(rows: Iterable[tuple[str, Any, Any]]) -> dict[str, Any]:
    return {
        name: parameter.detach().cpu().clone()
        for name, _module, parameter in rows
    }


def load_tensor_state(rows: Iterable[tuple[str, Any, Any]], saved: Mapping[str, Any], device: Any) -> None:
    live = {name: parameter for name, _module, parameter in rows}
    if list(live) != list(saved):
        raise RuntimeError("checkpoint dense state key/order drift")
    for name, parameter in live.items():
        parameter.data.copy_(saved[name].to(device))


def coverage(torch: Any, named: list[tuple[str, Any]], *, gradient: bool) -> dict[str, Any]:
    nonzero = []
    missing = []
    nonfinite = []
    for name, parameter in named:
        tensor = parameter.grad if gradient else parameter
        if tensor is None:
            missing.append(name)
        elif not bool(torch.isfinite(tensor.detach()).all()):
            nonfinite.append(name)
        elif bool(torch.count_nonzero(tensor.detach()).item()):
            nonzero.append(name)
        else:
            missing.append(name)
    return {
        "nonzero": len(nonzero),
        "total": len(named),
        "ratio": f"{len(nonzero)}/{len(named)}",
        "nonzero_names": nonzero,
        "missing_or_zero_names": missing,
        "nonfinite_names": nonfinite,
    }


def update_coverage(torch: Any, named: list[tuple[str, Any]], before: list[Any]) -> dict[str, Any]:
    if len(named) != len(before):
        raise RuntimeError("update snapshot cardinality drift")
    rows = []
    missing = []
    nonfinite = []
    for (name, parameter), old in zip(named, before, strict=True):
        delta = parameter.detach() - old
        if not bool(torch.isfinite(delta).all()):
            nonfinite.append(name)
        elif bool(torch.count_nonzero(delta).item()):
            rows.append(name)
        else:
            missing.append(name)
    return {
        "nonzero": len(rows),
        "total": len(named),
        "ratio": f"{len(rows)}/{len(named)}",
        "nonzero_names": rows,
        "missing_or_zero_names": missing,
        "nonfinite_names": nonfinite,
    }


_DORMANT_INDEXER_NORMS = {
    f"model.layers.{layer}.self_attn.compressor.indexer.kv_norm"
    for layer in range(2, 43, 2)
}


def require_authentic_coverage(
    observed: Mapping[str, Any], *, surface: str, phase: str, update: int
) -> None:
    """Require every differentiable member of the admitted repair surface.

    The 21 lightning-indexer RMSNorms influence a discrete top-k index only, so
    autograd authentically leaves them without gradients or Adam movement.  The
    recovered update-12 checkpoint confirms the same 214 realized norm states
    and 21 exact dormant names.  Keep all 235 masters checkpointed while gating
    the exact differentiable subset rather than fabricating surrogate gradients.
    """
    totals = {"luts": LAYERS, "norms": NORMS, "outputs": OUTPUTS}
    if surface not in totals:
        raise ValueError(f"unknown joint repair surface {surface!r}")
    expected_missing = _DORMANT_INDEXER_NORMS if surface == "norms" else set()
    missing = set(map(str, observed.get("missing_or_zero_names", ())))
    nonfinite = set(map(str, observed.get("nonfinite_names", ())))
    expected_nonzero = totals[surface] - len(expected_missing)
    if (
        int(observed.get("total", -1)) != totals[surface]
        or int(observed.get("nonzero", -1)) != expected_nonzero
        or missing != expected_missing
        or nonfinite
    ):
        raise RuntimeError(
            f"authentic {phase} coverage gate failed at update {update} "
            f"for {surface}: {observed}"
        )


def objective_value(torch: Any, objective: Callable, sources: Mapping[int, PlaneSource], wins: list[int], requires_grad: bool):
    for source in sources.values():
        source.reset_usage()
    context = torch.enable_grad() if requires_grad else torch.no_grad()
    with context:
        loss = objective(wins, requires_grad)
    if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
        raise RuntimeError("joint objective must return one scalar tensor")
    used = sorted(layer for layer, source in sources.items() if source.uses > 0)
    if used != list(range(LAYERS)):
        raise RuntimeError(f"joint objective did not consume all43 PlaneSources: {used}")
    value = float(loss.detach())
    if not math.isfinite(value):
        raise RuntimeError(f"joint objective is non-finite: {value}")
    return loss, value, {str(layer): sources[layer].uses for layer in range(LAYERS)}


def immutable_torch_checkpoint(torch: Any, path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, path)
        os.chmod(path, 0o444)
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
        fsync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(path)


def update_latest(latest: Path, checkpoint: Path) -> None:
    temporary = latest.with_name(f".{latest.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(checkpoint, temporary)
        os.replace(temporary, latest)
        fsync_dir(latest.parent)
    finally:
        temporary.unlink(missing_ok=True)


def validate_contract(admission: Mapping[str, Any]) -> None:
    schedule = admission["optimizer_schedule_checkpoint"]
    groups = schedule["parameter_groups"]
    expected_groups = [
        ("all43_luts", LUT_LR),
        ("all_rmsnorms", NORM_LR),
        ("all43_output_log_gains", OUTPUT_LR),
    ]
    observed_groups = [(row["name"], float(row["learning_rate"])) for row in groups]
    frozen = admission["frozen_policy"]
    if (
        admission.get("schema") != "banana-smasher-qtip2-v7-joint-repair-admission-v1"
        or admission.get("status")
        != "SEALED_ADMITTED_EXACT_PRIOR_JOINT_PATH_WITH_FULL_MODEL_BANK_AND_EXACT_OVERLAY_CONSUMER"
        or int(schedule["updates"]) != UPDATES
        or int(schedule["batch"]) != BATCH
        or schedule["optimizer"] != "Adam"
        or schedule["scheduler"] != "cosine"
        or float(schedule["cosine_min_ratio"]) != COSINE_MIN_RATIO
        or observed_groups != expected_groups
        or len(admission["trainable_roster"]["luts"]) != LAYERS
        or len(admission["trainable_roster"]["rmsnorms"]) != NORMS
        or len(admission["trainable_roster"]["output_gains"]) != OUTPUTS
        or not all(bool(value) for value in frozen.values())
        or not admission["train_objective"]["evaluation_feedback_forbidden"]
    ):
        raise RuntimeError("sealed joint-repair admission contract drift")


def gate_receipt_valid(path: Path, update: int) -> bool:
    row = load_json(path)
    expected = {
        "schema": "banana-smasher-qtip2-v7-joint-update-receipt-v1",
        "update": update,
        "gate_pass": True,
    }
    if any(row.get(key) != value for key, value in expected.items()):
        return False
    for surface, total in (("luts", LAYERS), ("norms", NORMS), ("outputs", OUTPUTS)):
        try:
            require_authentic_coverage(
                row["gradient_coverage"][surface],
                surface=surface,
                phase="gradient",
                update=update,
            )
            require_authentic_coverage(
                row["update_coverage"][surface],
                surface=surface,
                phase="update",
                update=update,
            )
        except RuntimeError:
            return False
    return float(row["objective"]["after"]) < float(row["objective"]["before"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-smoke", action="store_true")
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--historical-roster", type=Path)
    parser.add_argument("--runtime-module", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--full-wire-hash", action="store_true")
    parser.add_argument("--diagnostic-every", type=int, default=8)
    parser.add_argument("--activation-checkpoint", action="store_true")
    parser.add_argument("--frozen-cache-bytes", type=int, default=2 << 30)
    parser.add_argument("--fwht-backend", choices=("quack",), default="quack")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--resume-checkpoint-sha256")
    args = parser.parse_args(argv)
    if args.contract_smoke:
        print(json.dumps({
            "schema": SCHEMA,
            "status": "PASS_IMPORT_CONTRACT_SMOKE",
            "updates": UPDATES,
            "batch": BATCH,
            "optimizer_steps_per_update": 1,
            "coverage": {"luts": LAYERS, "norms": NORMS, "outputs": OUTPUTS},
        }, sort_keys=True))
        return 0
    required = {
        "admission": args.admission,
        "inventory": args.inventory,
        "historical_roster": args.historical_roster,
        "runtime_module": args.runtime_module,
        "run_root": args.run_root,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error(f"compute mode requires: {', '.join('--' + name.replace('_', '-') for name in missing)}")
    if args.diagnostic_every <= 0:
        parser.error("--diagnostic-every must be positive")
    if args.frozen_cache_bytes <= 0:
        parser.error("--frozen-cache-bytes must be positive")
    if (args.resume_checkpoint is None) != (args.resume_checkpoint_sha256 is None):
        parser.error(
            "--resume-checkpoint and --resume-checkpoint-sha256 are required together"
        )

    # One public fast path: never silently import the historical expert that
    # parses and decodes every compact member on every objective traversal.
    os.environ["JOINT_V7_EXPERT_BASE"] = (
        "banana_smasher.joint_v7_expert_base:JointV7ExpertBase"
    )
    os.environ["JOINT_V7_FROZEN_CACHE_BYTES"] = str(args.frozen_cache_bytes)
    os.environ["JOINT_V7_FWHT_BACKEND"] = args.fwht_backend
    os.environ["BANANA_SMASHER_REPAIR_ACTIVATION_CHECKPOINT"] = (
        "1" if args.activation_checkpoint else "0"
    )

    admission_path = args.admission.resolve()
    inventory_path = args.inventory.resolve()
    roster_path = args.historical_roster.resolve()
    runtime_path = args.runtime_module.resolve()
    root = args.run_root.resolve()
    admission = load_json(admission_path)
    validate_contract(admission)
    source_hashes = admission["prior_working_path_proof"]["local_source_hashes"]
    require_sha(inventory_path, str(source_hashes[inventory_path.name]), "V7 inventory")
    require_sha(roster_path, str(source_hashes[roster_path.name]), "historical roster")
    inventory = load_json(inventory_path)
    historical = load_json(roster_path)
    if int(historical["config"]["batch"]) != BATCH or historical["config"]["optimizer"] != "Adam":
        raise RuntimeError("historical schedule seam drift")
    ordered_wins = list(map(int, admission["train_objective"]["full_model_train_bank"]["ordered_train_windows"]))
    if ordered_wins != list(map(int, historical["config"]["train_combined_wins"])) or len(ordered_wins) != 64:
        raise RuntimeError("ordered TRAIN window seam drift")
    if inventory.get("inventory") != admission["trainable_roster"]["luts"]:
        # Inventory rows contain the same exact layer/name/wire/manifest surface in
        # a flatter representation; compare a canonical projection instead.
        projected = [
            {
                "layer": int(row["layer"]),
                "name": str(row["name"]),
                "manifest_path": str(row["manifest_path"]),
                "manifest_sha256": str(row["manifest_sha256"]),
                "wire_path": str(row["wire_path"]),
                "wire_sha256": str(row["wire_sha256"]),
                "wire_shape": row["wire_shape"],
                "wire_dtype": row["wire_dtype"],
                "wire_bytes": int(row["wire_bytes"]),
            }
            for row in inventory["inventory"]
        ]
        expected = [
            {
                "layer": int(row["layer"]),
                "name": str(row["name"]),
                "manifest_path": str(row["source_manifest"]["path"]),
                "manifest_sha256": str(row["source_manifest"]["sha256"]),
                "wire_path": str(row["wire"]["source_path"]),
                "wire_sha256": str(row["wire"]["sha256"]),
                "wire_shape": row["wire"]["shape"],
                "wire_dtype": row["wire"]["dtype"],
                "wire_bytes": int(row["wire"]["bytes"]),
            }
            for row in admission["trainable_roster"]["luts"]
        ]
        if projected != expected:
            raise RuntimeError("V7 inventory/admission exact LUT roster drift")

    root.mkdir(parents=True, exist_ok=True)
    for directory in (root / "checkpoints", root / "receipts", root / "logs", root / "run"):
        directory.mkdir(parents=True, exist_ok=True)
    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device required but unavailable")
    device = torch.device(args.device)
    random.seed(1701)
    torch.manual_seed(1701)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(1701)

    sources = {
        int(row["layer"]): PlaneSource(
            torch=torch,
            layer=int(row["layer"]),
            roster=row,
            device=device,
            full_wire_hash=args.full_wire_hash,
        )
        for row in admission["trainable_roster"]["luts"]
    }
    if sorted(sources) != list(range(LAYERS)):
        raise RuntimeError("PlaneSource roster is not exact layers 0..42")
    runtime_module = load_module(runtime_path, "banana_smasher_joint_v7_runtime_adapter")
    builder = getattr(runtime_module, "build_joint_v7_runtime", None)
    if not callable(builder):
        raise RuntimeError("runtime module has no build_joint_v7_runtime")
    runtime = builder(
        plane_sources=sources,
        device=device,
        admission=admission,
        ordered_train_windows=ordered_wins,
        batch_size=BATCH,
    )
    student = runtime_field(runtime, "student")
    if student is None or not hasattr(student, "model"):
        raise RuntimeError("runtime adapter did not return a whole-model student")
    consumed = runtime_field(runtime, "plane_sources")
    if not isinstance(consumed, Mapping) or any(consumed.get(layer) is not source for layer, source in sources.items()):
        raise RuntimeError("runtime adapter did not preserve the exact 43 PlaneSource objects")
    objective = bind_objective(runtime)
    norms, outputs = expose_dense_parameters(torch, student, admission)
    lut_named = [(sources[layer].name, sources[layer].master) for layer in range(LAYERS)]
    norm_named = [(name, parameter) for name, _module, parameter in norms]
    output_named = [(name, parameter) for name, _module, parameter in outputs]
    optimizer = torch.optim.Adam(
        [
            {"params": [parameter for _name, parameter in lut_named], "lr": LUT_LR, "group_name": "all43_luts"},
            {"params": [parameter for _name, parameter in norm_named], "lr": NORM_LR, "group_name": "all_rmsnorms"},
            {"params": [parameter for _name, parameter in output_named], "lr": OUTPUT_LR, "group_name": "all43_output_log_gains"},
        ],
        foreach=False,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=[lambda step: cosine_multiplier(step)] * 3
    )
    identity = {
        "schema": SCHEMA,
        "admission": str(admission_path),
        "admission_sha256": sha256_file(admission_path),
        "inventory": str(inventory_path),
        "inventory_sha256": sha256_file(inventory_path),
        "historical_roster": str(roster_path),
        "historical_roster_sha256": sha256_file(roster_path),
        "runtime_module": str(runtime_path),
        "runtime_module_sha256": sha256_file(runtime_path),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "compact_wire_stat_identity_sha256": canonical_sha256(
            {str(layer): source._stat_identity for layer, source in sources.items()}
        ),
        "ordered_train_windows": ordered_wins,
        "frozen_policy": admission["frozen_policy"],
        "trainables": {"luts": LAYERS, "norms": NORMS, "norm_numel": NORM_NUMEL, "outputs": OUTPUTS},
        "optimizer": "Adam",
        "learning_rates": [LUT_LR, NORM_LR, OUTPUT_LR],
        "cosine_min_ratio": COSINE_MIN_RATIO,
        "updates": UPDATES,
        "batch": BATCH,
        "fast_path": {
            "persistent_payload_cache": True,
            "activation_checkpoint": bool(args.activation_checkpoint),
            "diagnostic_every": int(args.diagnostic_every),
            "frozen_cache_bytes": int(args.frozen_cache_bytes),
            "fwht_backend": args.fwht_backend,
        },
    }
    identity_sha = canonical_sha256(identity)
    checkpoints = root / "checkpoints"
    receipts = root / "receipts"
    latest = checkpoints / "LATEST.pt"
    resume_checkpoint = (
        latest if args.resume_checkpoint is None else args.resume_checkpoint.resolve()
    )
    start_update = 0
    if resume_checkpoint.is_file():
        if args.resume_checkpoint_sha256 is not None:
            require_sha(
                resume_checkpoint,
                args.resume_checkpoint_sha256,
                "explicit resume checkpoint",
            )
        checkpoint = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.get("format") != CHECKPOINT_FORMAT:
            raise RuntimeError("resume checkpoint format drift")
        saved_identity = checkpoint.get("identity")
        if checkpoint.get("identity_sha256") != identity_sha:
            if args.resume_checkpoint_sha256 is None or not isinstance(saved_identity, Mapping):
                raise RuntimeError("LATEST checkpoint identity drift")
            # An explicitly SHA-bound cutover may change only executable fast-path
            # identity. Dataset, schedule, model surface, and optimizer identity
            # must remain exact before optimizer/model state is resumed.
            code_keys = {
                "runner_sha256",
                "runtime_module_sha256",
                "fast_path",
            }
            saved_science = {
                key: value for key, value in saved_identity.items() if key not in code_keys
            }
            current_science = {
                key: value for key, value in identity.items() if key not in code_keys
            }
            if saved_science != current_science:
                raise RuntimeError("explicit resume checkpoint scientific identity drift")
        saved_luts = checkpoint["state"]["luts"]
        if list(saved_luts) != [name for name, _parameter in lut_named]:
            raise RuntimeError("checkpoint LUT key/order drift")
        for name, parameter in lut_named:
            parameter.data.copy_(saved_luts[name].to(device))
        load_tensor_state(norms, checkpoint["state"]["norms"], device)
        load_tensor_state(outputs, checkpoint["state"]["outputs"], device)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_update = int(checkpoint["next_update"])
        if not 0 <= start_update <= UPDATES:
            raise RuntimeError("checkpoint next_update outside 0..64")
    if start_update >= 1 and not gate_receipt_valid(receipts / "UPDATE_000.json", 0):
        raise RuntimeError("resume refuses without valid update0 gate")
    if start_update >= 2 and not gate_receipt_valid(receipts / "UPDATE_001.json", 1):
        raise RuntimeError("resume refuses without valid update1 auto-continuation gate")

    log_path = root / "logs" / "JOINT_V7.jsonl"
    status_path = root / "run" / "STATUS.json"

    def emit(value: Mapping[str, Any]) -> None:
        row = {**value, "unix": time.time()}
        with log_path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(json.dumps(row, sort_keys=True), flush=True)

    atomic_json(status_path, {
        "schema": SCHEMA,
        "state": "RUNNING",
        "pid": os.getpid(),
        "next_update": start_update,
        "identity_sha256": identity_sha,
    })
    emit({"event": "start", "next_update": start_update, "identity_sha256": identity_sha})

    groups = len(ordered_wins) // BATCH
    for update in range(start_update, UPDATES):
        for source in sources.values():
            source.assert_frozen()
        optimizer.zero_grad(set_to_none=True)
        group = update % groups
        wins = ordered_wins[group * BATCH : (group + 1) * BATCH]
        started = time.time()
        before_loss, objective_before, before_uses = objective_value(
            torch, objective, sources, wins, True
        )
        before_loss.backward()
        del before_loss
        gradient_coverage = {
            "luts": coverage(torch, lut_named, gradient=True),
            "norms": coverage(torch, norm_named, gradient=True),
            "outputs": coverage(torch, output_named, gradient=True),
        }
        expected_totals = {"luts": LAYERS, "norms": NORMS, "outputs": OUTPUTS}
        for name in expected_totals:
            require_authentic_coverage(
                gradient_coverage[name], surface=name, phase="gradient", update=update
            )
        snapshots = {
            "luts": [parameter.detach().clone() for _name, parameter in lut_named],
            "norms": [parameter.detach().clone() for _name, parameter in norm_named],
            "outputs": [parameter.detach().clone() for _name, parameter in output_named],
        }
        # The sole optimizer step is deliberately after the complete whole-model
        # objective and its all43/all235/all43 gradient gate.
        optimizer.step()
        scheduler.step()
        update_coverage_value = {
            "luts": update_coverage(torch, lut_named, snapshots["luts"]),
            "norms": update_coverage(torch, norm_named, snapshots["norms"]),
            "outputs": update_coverage(torch, output_named, snapshots["outputs"]),
        }
        for name in expected_totals:
            require_authentic_coverage(
                update_coverage_value[name], surface=name, phase="update", update=update
            )
        next_update = update + 1
        strict_probe_gate = update in (0, 1)
        diagnostic_performed = (
            strict_probe_gate
            or next_update == UPDATES
            or next_update % args.diagnostic_every == 0
        )
        if diagnostic_performed:
            _after_loss, objective_after, after_uses = objective_value(
                torch, objective, sources, wins, False
            )
            del _after_loss
        else:
            objective_after = None
            after_uses = {str(layer): 0 for layer in range(LAYERS)}
        gate_pass = (
            math.isfinite(objective_before)
            and (
                not diagnostic_performed
                or (
                    objective_after is not None
                    and math.isfinite(objective_after)
                    and (not strict_probe_gate or objective_after < objective_before)
                )
            )
        )
        checkpoint_payload = {
            "format": CHECKPOINT_FORMAT,
            "mechanism": "joint-v7-luts-plus-rmsnorms-plus-attention-output-gains",
            "next_update": next_update,
            "identity": identity,
            "identity_sha256": identity_sha,
            "state": {
                "luts": {name: parameter.detach().cpu().clone() for name, parameter in lut_named},
                "norms": tensor_state(norms),
                "outputs": tensor_state(outputs),
            },
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "objective": {"update": update, "before": objective_before, "after": objective_after},
            "invariants": {
                "codes_frozen": True,
                "assignments_frozen": True,
                "scales_frozen": True,
                "packed_geometry_frozen": True,
                "optimizer_steps_this_update": 1,
                "ordered_whole_model_objective": True,
            },
            "saved_unix": time.time(),
        }
        checkpoint_path = checkpoints / f"UPDATE_{next_update:03d}.pt"
        checkpoint_sha = immutable_torch_checkpoint(torch, checkpoint_path, checkpoint_payload)
        sidecar = {
            "schema": "banana-smasher-qtip2-v7-joint-checkpoint-sidecar-v1",
            "completed_update": update,
            "next_update": next_update,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "identity_sha256": identity_sha,
            "objective": {"before": objective_before, "after": objective_after},
            "gradient_coverage": gradient_coverage,
            "update_coverage": update_coverage_value,
        }
        install_json_once(checkpoints / f"UPDATE_{next_update:03d}.json", sidecar)
        update_latest(latest, checkpoint_path)
        receipt = {
            "schema": "banana-smasher-qtip2-v7-joint-update-receipt-v1",
            "status": "PASS" if gate_pass else "FAIL",
            "update": update,
            "next_update": next_update,
            "train_windows": wins,
            "batch": BATCH,
            "objective": {
                "before": objective_before,
                "after": objective_after,
                "delta": (
                    None
                    if objective_after is None
                    else objective_after - objective_before
                ),
            },
            "gradient_coverage": gradient_coverage,
            "update_coverage": update_coverage_value,
            "plane_source_uses_before": before_uses,
            "plane_source_uses_after": after_uses,
            "optimizer_steps": 1,
            "optimizer": "Adam",
            "learning_rates_after_scheduler": {
                str(group["group_name"]): float(group["lr"])
                for group in optimizer.param_groups
            },
            "cosine_min_ratio": COSINE_MIN_RATIO,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "frozen": admission["frozen_policy"],
            "strict_probe_gate": strict_probe_gate,
            "diagnostic_performed": diagnostic_performed,
            "diagnostic_every": int(args.diagnostic_every),
            "fast_path_counters": importlib.import_module(
                "banana_smasher.joint_v7_expert_base"
            ).joint_v7_fast_path_stats(),
            "gate_pass": gate_pass,
            "seconds": time.time() - started,
            "identity_sha256": identity_sha,
        }
        install_json_once(receipts / f"UPDATE_{update:03d}.json", receipt)
        emit({"event": "update", **receipt})
        atomic_json(status_path, {
            "schema": SCHEMA,
            "state": "RUNNING" if next_update < UPDATES else "TRAINING_COMPLETE",
            "pid": os.getpid(),
            "next_update": next_update,
            "last_checkpoint": str(checkpoint_path),
            "last_checkpoint_sha256": checkpoint_sha,
            "last_objective": receipt["objective"],
            "identity_sha256": identity_sha,
            "resume_checkpoint": str(resume_checkpoint),
            "resume_checkpoint_sha256": sha256_file(resume_checkpoint),
        })
        if not gate_pass:
            raise RuntimeError(f"joint update gate failed at update {update}")
        stop_after = int(os.environ.get("JOINT_STOP_AFTER_UPDATE", str(UPDATES)))
        if next_update >= stop_after and next_update < UPDATES:
            atomic_json(status_path, {
                "schema": SCHEMA,
                "state": "PAUSED_AT_PREDECLARED_UPDATE",
                "pid": os.getpid(),
                "next_update": next_update,
                "last_checkpoint": str(checkpoint_path),
                "last_checkpoint_sha256": checkpoint_sha,
                "last_objective": receipt["objective"],
                "identity_sha256": identity_sha,
            })
            emit({
                "event": "predeclared_pause",
                "next_update": next_update,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha,
                "identity_sha256": identity_sha,
            })
            return 0
        if update == 1:
            if not gate_receipt_valid(receipts / "UPDATE_000.json", 0) or not gate_receipt_valid(receipts / "UPDATE_001.json", 1):
                raise RuntimeError("update1 auto-continuation gate readback failed")
            install_json_once(receipts / "UPDATE1_AUTOCONTINUE_GATE.json", {
                "schema": "banana-smasher-qtip2-v7-update1-autocontinue-v1",
                "status": "PASS_AUTO_CONTINUE_TO_64",
                "update0_receipt": str(receipts / "UPDATE_000.json"),
                "update0_sha256": sha256_file(receipts / "UPDATE_000.json"),
                "update1_receipt": str(receipts / "UPDATE_001.json"),
                "update1_sha256": sha256_file(receipts / "UPDATE_001.json"),
                "next_update": 2,
                "terminal_update": UPDATES,
                "identity_sha256": identity_sha,
            })

    final = {
        "schema": "banana-smasher-qtip2-v7-joint-training-terminal-v1",
        "status": "PASS_TRAINING_COMPLETE",
        "updates": UPDATES,
        "next_update": UPDATES,
        "checkpoint": str(checkpoints / f"UPDATE_{UPDATES:03d}.pt"),
        "checkpoint_sha256": sha256_file(checkpoints / f"UPDATE_{UPDATES:03d}.pt"),
        "update0_gate": gate_receipt_valid(receipts / "UPDATE_000.json", 0),
        "update1_gate": gate_receipt_valid(receipts / "UPDATE_001.json", 1),
        "identity_sha256": identity_sha,
        "completed_unix": time.time(),
    }
    install_json_once(receipts / "TRAINING_COMPLETE.json", final)
    atomic_json(status_path, {**final, "state": "TRAINING_COMPLETE", "pid": os.getpid()})
    emit({"event": "training_complete", **final})
    return 0


def run_joint_v7_repair(
    *,
    admission: str | Path,
    inventory: str | Path,
    historical_roster: str | Path,
    runtime_module: str | Path,
    run_root: str | Path,
    optimizer_steps: int = UPDATES,
    device: str = "cuda",
    resume: bool = True,
    full_wire_hash: bool = False,
    diagnostic_every: int = 8,
    activation_checkpoint: bool = False,
    frozen_cache_bytes: int = 2 << 30,
    fwht_backend: str = "quack",
    resume_checkpoint: str | Path | None = None,
    resume_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Public API entry point for persistent all-surface QTIP V7 repair."""
    if (
        isinstance(optimizer_steps, bool)
        or not isinstance(optimizer_steps, int)
        or not 1 <= optimizer_steps <= UPDATES
    ):
        raise ValueError(f"optimizer_steps must be within 1..{UPDATES}")
    if (
        isinstance(diagnostic_every, bool)
        or not isinstance(diagnostic_every, int)
        or diagnostic_every <= 0
    ):
        raise ValueError("diagnostic_every must be a positive integer")
    if (
        isinstance(frozen_cache_bytes, bool)
        or not isinstance(frozen_cache_bytes, int)
        or frozen_cache_bytes <= 0
    ):
        raise ValueError("frozen_cache_bytes must be a positive integer")
    if fwht_backend != "quack":
        raise ValueError("joint V7 production repair requires fwht_backend='quack'")
    if (resume_checkpoint is None) != (resume_checkpoint_sha256 is None):
        raise ValueError(
            "resume_checkpoint and resume_checkpoint_sha256 are required together"
        )
    root = Path(run_root).expanduser().resolve()
    latest = root / "checkpoints" / "LATEST.pt"
    if latest.is_file() and not resume:
        raise FileExistsError(
            f"joint repair checkpoint exists and resume is disabled: {latest}"
        )
    argv = [
        "--admission", str(Path(admission).expanduser().resolve()),
        "--inventory", str(Path(inventory).expanduser().resolve()),
        "--historical-roster", str(Path(historical_roster).expanduser().resolve()),
        "--runtime-module", str(Path(runtime_module).expanduser().resolve()),
        "--run-root", str(root),
        "--device", device,
        "--diagnostic-every", str(diagnostic_every),
        "--frozen-cache-bytes", str(frozen_cache_bytes),
        "--fwht-backend", fwht_backend,
    ]
    if activation_checkpoint:
        argv.append("--activation-checkpoint")
    if resume_checkpoint is not None:
        argv.extend([
            "--resume-checkpoint",
            str(Path(resume_checkpoint).expanduser().resolve()),
            "--resume-checkpoint-sha256",
            str(resume_checkpoint_sha256),
        ])
    if full_wire_hash:
        argv.append("--full-wire-hash")
    previous_stop = os.environ.get("JOINT_STOP_AFTER_UPDATE")
    os.environ["JOINT_STOP_AFTER_UPDATE"] = str(optimizer_steps)
    try:
        main(argv)
    finally:
        if previous_stop is None:
            os.environ.pop("JOINT_STOP_AFTER_UPDATE", None)
        else:
            os.environ["JOINT_STOP_AFTER_UPDATE"] = previous_stop
    terminal = root / "receipts" / "TRAINING_COMPLETE.json"
    if terminal.is_file():
        result = load_json(terminal)
        if result.get("status") != "PASS_TRAINING_COMPLETE":
            raise RuntimeError("joint repair terminal did not pass")
        return {
            **result,
            "status": "PASS",
            "optimizer_steps_completed": int(result["updates"]),
            "terminal": str(terminal),
            "terminal_sha256": sha256_file(terminal),
        }
    status_path = root / "run" / "STATUS.json"
    status = load_json(status_path)
    if (
        status.get("state") != "PAUSED_AT_PREDECLARED_UPDATE"
        or int(status.get("next_update", -1)) != optimizer_steps
    ):
        raise RuntimeError(
            f"joint repair exited without requested checkpoint: {status.get('state')!r}"
        )
    return {
        **status,
        "status": "PASS",
        "optimizer_steps_completed": optimizer_steps,
        "status_receipt": str(status_path),
        "status_receipt_sha256": sha256_file(status_path),
    }


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        traceback.print_exc()
        print(json.dumps({
            "schema": SCHEMA,
            "status": "FAILED",
            "error": f"{type(exc).__name__}: {exc}",
        }, sort_keys=True), file=sys.stderr, flush=True)
        raise
