#!/usr/bin/env python3
"""Build the pure-plugin derivative tier.

Tiering is deliberate: pure plugin assets use this seconds-scale path; a plugin
native wheel change rebuilds only that wheel; dependency/base changes require a
full AOT image. After choosing a winner, produce one final full clean seal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from apply_plugin_derivative import build_manifest, canonical_json


def command(*argv: str, capture: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        check=True,
        capture_output=capture,
        text=True,
    )


def inspect_image(reference: str) -> dict[str, Any]:
    value = json.loads(command("docker", "image", "inspect", reference).stdout)
    if not isinstance(value, list) or len(value) != 1:
        raise RuntimeError(f"DERIVATIVE_IMAGE_INSPECT_REFUSE {reference}")
    return value[0]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_exclusive_json(path: Path, value: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return hashlib.sha256(raw).hexdigest()


def materialize_context(
    context: Path,
    repo: Path,
    manifest: dict[str, Any],
    payloads: dict[str, bytes],
    dockerfile: Path,
    apply_script: Path,
) -> None:
    context.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dockerfile, context / "Dockerfile")
    shutil.copy2(apply_script, context / "apply_plugin_derivative.py")
    (context / "manifest.json").write_bytes(canonical_json(manifest))
    assets = context / "assets"
    for row in manifest["runtime_assets"]:
        destination = assets / row["distribution_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payloads[row["source_path"]])
    inventory = manifest.get("source_inventory")
    if inventory is not None:
        destination = assets / inventory["staged_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payloads[inventory["source_path"]])
    actual = {
        path.relative_to(context).as_posix(): file_sha256(path)
        for path in sorted(context.rglob("*"))
        if path.is_file()
    }
    if any(path.suffix.lower() not in {".py", ".json", ".npy", ""} for path in context.rglob("*")):
        raise RuntimeError("DERIVATIVE_CONTEXT_REFUSE unexpected file suffix")
    if not actual:
        raise RuntimeError("DERIVATIVE_CONTEXT_REFUSE empty")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--target-commit", required=True)
    parser.add_argument("--parent-image", required=True)
    parser.add_argument("--parent-image-id", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--dockerfile",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "Dockerfile.plugin-derivative",
    )
    arguments = parser.parse_args()
    if arguments.receipt.exists():
        raise RuntimeError(f"DERIVATIVE_RECEIPT_EXISTS_REFUSE {arguments.receipt}")

    repo = arguments.repo.resolve()
    apply_script = Path(__file__).resolve().with_name("apply_plugin_derivative.py")
    parent = inspect_image(arguments.parent_image)
    if parent.get("Id") != arguments.parent_image_id:
        raise RuntimeError(
            f"DERIVATIVE_PARENT_IMAGE_REFUSE actual={parent.get('Id')} "
            f"expected={arguments.parent_image_id}"
        )
    manifest, payloads = build_manifest(
        repo,
        arguments.base_commit,
        arguments.target_commit,
        arguments.parent_image_id,
    )
    started_wall = time.time()
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="banana-smasher-plugin-derivative-") as temporary:
        context = Path(temporary)
        materialize_context(
            context,
            repo,
            manifest,
            payloads,
            arguments.dockerfile.resolve(),
            apply_script,
        )
        build_command = [
            "docker",
            "buildx",
            "build",
            "--progress=plain",
            "--platform",
            "linux/arm64",
            "--no-cache",
            "--load",
            "--file",
            str(context / "Dockerfile"),
            "--build-arg",
            f"BASE_IMAGE={arguments.parent_image}",
            "--build-arg",
            f"DERIVATIVE_MANIFEST_SHA256={manifest['manifest_sha256']}",
            "--build-arg",
            f"BANANA_SMASHER_SOURCE_COMMIT={manifest['source_commit']}",
            "--build-arg",
            f"BANANA_SMASHER_SOURCE_TREE={manifest['source_tree']}",
            "--build-arg",
            f"PARENT_IMAGE_ID={arguments.parent_image_id}",
            "--tag",
            arguments.tag,
            str(context),
        ]
        completed = subprocess.run(
            build_command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        elapsed = time.monotonic() - started
        if completed.returncode:
            raise RuntimeError(
                f"DERIVATIVE_BUILD_FAIL rc={completed.returncode}\n{completed.stdout[-12000:]}"
            )

    derived = inspect_image(arguments.tag)
    parent_config = parent.get("Config") or {}
    derived_config = derived.get("Config") or {}
    inherited = {
        "Entrypoint": parent_config.get("Entrypoint"),
        "Cmd": parent_config.get("Cmd"),
        "Env": parent_config.get("Env"),
        "WorkingDir": parent_config.get("WorkingDir"),
    }
    actual_inherited = {
        "Entrypoint": derived_config.get("Entrypoint"),
        "Cmd": derived_config.get("Cmd"),
        "Env": derived_config.get("Env"),
        "WorkingDir": derived_config.get("WorkingDir"),
    }
    if actual_inherited != inherited:
        raise RuntimeError(
            f"DERIVATIVE_RUNTIME_CONFIG_REFUSE actual={actual_inherited} expected={inherited}"
        )
    labels = derived_config.get("Labels") or {}
    expected_labels = {
        "io.banana-smasher.derivative.tier": "pure-plugin",
        "io.banana-smasher.derivative.manifest": manifest["manifest_sha256"],
        "io.banana-smasher.source.commit": manifest["source_commit"],
        "io.banana-smasher.source.tree": manifest["source_tree"],
        "io.banana-smasher.parent.image": arguments.parent_image_id,
    }
    if any(labels.get(key) != value for key, value in expected_labels.items()):
        raise RuntimeError("DERIVATIVE_LABEL_REFUSE")

    receipt = {
        "schema": "banana-smasher-plugin-derivative-build-v1",
        "status": "PASS",
        "tier": "pure-plugin-seconds",
        "started_unix": started_wall,
        "finished_unix": time.time(),
        "elapsed_seconds": elapsed,
        "build_command": build_command,
        "build_returncode": completed.returncode,
        "build_output_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
        "build_output_tail": completed.stdout[-8000:],
        "base_commit": manifest["base_commit"],
        "source_commit": manifest["source_commit"],
        "source_tree": manifest["source_tree"],
        "parent_image_reference": arguments.parent_image,
        "parent_image_id": arguments.parent_image_id,
        "image_tag": arguments.tag,
        "image_id": derived.get("Id"),
        "manifest": manifest,
        "runtime_config_inherited_exact": actual_inherited,
        "labels": expected_labels,
        "tiering": {
            "pure_plugin": "seconds-scale derivative; only .py/.json/.npy installed assets",
            "plugin_native_wheel": "rebuild only the plugin native wheel",
            "full_aot": "required for dependency, base image, C++/CUDA, pyproject, or AOT changes",
            "final_seal": "one final full clean source/AOT image after selecting the winner",
        },
    }
    receipt_sha = atomic_exclusive_json(arguments.receipt.resolve(), receipt)
    print(
        json.dumps(
            {
                "status": "PASS",
                "receipt": str(arguments.receipt.resolve()),
                "receipt_sha256": receipt_sha,
                "elapsed_seconds": elapsed,
                "image_id": derived.get("Id"),
                "manifest_sha256": manifest["manifest_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
