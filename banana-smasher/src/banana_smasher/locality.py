from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Iterable

REMOTE_FILESYSTEMS = frozenset(
    {
        "9p",
        "afs",
        "ceph",
        "cifs",
        "fuse.rclone",
        "fuse.sshfs",
        "glusterfs",
        "nfs",
        "nfs4",
        "smb3",
        "smbfs",
        "sshfs",
    }
)


def _unescape_mount_field(value: str) -> str:
    return (
        value.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def _mount_rows(mountinfo_path: str | Path = "/proc/self/mountinfo") -> list[tuple[Path, str, str]]:
    path = Path(mountinfo_path)
    rows: list[tuple[Path, str, str]] = []
    if path.is_file():
        for line in path.read_text().splitlines():
            before, separator, after = line.partition(" - ")
            fields = before.split()
            trailing = after.split()
            if not separator or len(fields) < 5 or len(trailing) < 2:
                continue
            rows.append(
                (
                    Path(_unescape_mount_field(fields[4])),
                    trailing[0],
                    _unescape_mount_field(trailing[1]),
                )
            )
        return rows

    mounted = subprocess.run(
        ["mount"], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    for line in mounted:
        source, marker, remainder = line.partition(" on ")
        if not marker:
            continue
        mountpoint, marker, options = remainder.partition(" (")
        if not marker:
            continue
        filesystem = options.rstrip(")").split(",", 1)[0]
        rows.append((Path(mountpoint), filesystem, source))
    return rows


def mount_for_path(
    path: str | Path, *, mountinfo_path: str | Path = "/proc/self/mountinfo"
) -> tuple[Path, str, str]:
    candidate = Path(path).expanduser().resolve(strict=False)
    probe = candidate
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    probe = probe.resolve(strict=False)
    matches: list[tuple[int, Path, str, str]] = []
    for mountpoint, filesystem, source in _mount_rows(mountinfo_path):
        try:
            probe.relative_to(mountpoint)
        except ValueError:
            continue
        matches.append((len(mountpoint.parts), mountpoint, filesystem, source))
    if not matches:
        raise ValueError(f"cannot identify filesystem for local-only path {candidate}")
    _, mountpoint, filesystem, source = max(matches, key=lambda row: row[0])
    return mountpoint, filesystem, source


def require_local_path(
    path: str | Path,
    *,
    label: str,
    mountinfo_path: str | Path = "/proc/self/mountinfo",
) -> Path:
    candidate = Path(path).expanduser().resolve(strict=False)
    mountpoint, filesystem, source = mount_for_path(
        candidate, mountinfo_path=mountinfo_path
    )
    remote = (
        filesystem.lower() in REMOTE_FILESYSTEMS
        or source.startswith("//")
        or ":/" in source
    )
    if remote:
        raise ValueError(
            f"{label} must be local; {candidate} is on {filesystem} source={source} "
            f"mounted at {mountpoint}. Run the explicit QSFP staging API first."
        )
    return candidate


def require_local_paths(
    paths: Iterable[tuple[str, str | Path]],
    *,
    mountinfo_path: str | Path = "/proc/self/mountinfo",
) -> None:
    for label, path in paths:
        require_local_path(path, label=label, mountinfo_path=mountinfo_path)


def _referenced_paths(root: Path, value: object) -> list[Path]:
    if not isinstance(value, dict):
        return []
    paths: list[Path] = []
    for item in value.values():
        if isinstance(item, str) and item:
            candidate = Path(item)
            paths.append(candidate if candidate.is_absolute() else root / candidate)
    return paths


def require_local_backpack_inputs(
    *,
    model_root: str | Path,
    bank_path: str | Path,
    teacher_manifest_path: str | Path,
    virtual_manifest_path: str | Path,
    materialization_index_path: str | Path,
    qtip2_root_map_path: str | Path,
    qtip3_root_map_path: str | Path,
    output_root: str | Path,
    mountinfo_path: str | Path = "/proc/self/mountinfo",
) -> None:
    model_root = Path(model_root).expanduser().resolve(strict=False)
    teacher_manifest_path = Path(teacher_manifest_path).expanduser().resolve(strict=False)
    virtual_manifest_path = Path(virtual_manifest_path).expanduser().resolve(strict=False)
    qtip_maps = [
        ("qtip2", Path(qtip2_root_map_path).expanduser().resolve(strict=False)),
        ("qtip3", Path(qtip3_root_map_path).expanduser().resolve(strict=False)),
    ]
    direct = [
        ("model_root", model_root),
        ("bank", bank_path),
        ("teacher_manifest", teacher_manifest_path),
        ("virtual_manifest", virtual_manifest_path),
        ("materialization_index", materialization_index_path),
        ("qtip2_root_map", qtip_maps[0][1]),
        ("qtip3_root_map", qtip_maps[1][1]),
        ("output_root", output_root),
    ]
    require_local_paths(direct, mountinfo_path=mountinfo_path)

    index_path = model_root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    for shard in sorted(set(index.get("weight_map", {}).values())):
        require_local_path(
            model_root / shard,
            label=f"model_shard:{shard}",
            mountinfo_path=mountinfo_path,
        )

    teacher = json.loads(teacher_manifest_path.read_text())
    for sidecar in _referenced_paths(teacher_manifest_path.parent, {
        str(i): row.get("path") for i, row in enumerate(teacher.get("windows", []))
        if isinstance(row, dict)
    }):
        require_local_path(
            sidecar, label="teacher_sidecar", mountinfo_path=mountinfo_path
        )

    virtual = json.loads(virtual_manifest_path.read_text())
    for source_key, source in virtual.get("source_bindings", {}).items():
        if isinstance(source, dict) and source.get("root"):
            require_local_path(
                source["root"],
                label=f"virtual_source:{source_key}",
                mountinfo_path=mountinfo_path,
            )

    for tier, map_path in qtip_maps:
        root_map = json.loads(map_path.read_text())
        for layer, root in root_map.get("layer_roots", {}).items():
            require_local_path(
                root,
                label=f"{tier}_layer_root:{layer}",
                mountinfo_path=mountinfo_path,
            )
