#!/usr/bin/env python3
"""Rebind the accepted native plugin wheel to the resolver-clean v0.2.4 closure."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

SOURCE_VERSION = "0.2.3"
TARGET_VERSION = "0.2.4"
SOURCE_DIST_INFO = f"banana_smasher_plugin-{SOURCE_VERSION}.dist-info"
TARGET_DIST_INFO = f"banana_smasher_plugin-{TARGET_VERSION}.dist-info"
REPLACEMENTS = {
    "Version: 0.2.3": "Version: 0.2.4",
    'Requires-Dist: flashinfer-cubin @ https://github.com/my-other-github-account/banana-smasher/releases/download/v0.2.3-sm121-runtime/flashinfer_cubin-0.6.17-py3-none-any.whl#sha256=d9195135e438226c2f63afad02ce30722c749b6254573e78edc8458d0465923d ; sys_platform == "linux" and platform_machine == "aarch64"':
        'Requires-Dist: flashinfer-cubin @ https://github.com/my-other-github-account/banana-smasher/releases/download/v0.2.4-sm121-runtime/flashinfer_cubin-0.6.12%2Bbanana.fi0617-py3-none-any.whl#sha256=b8d72cbe119f0f8d2d3cc116d4580bca6f77d0355bc24de9db82d6f8d76ddf6a ; sys_platform == "linux" and platform_machine == "aarch64"',
    'Requires-Dist: flashinfer-python @ https://github.com/my-other-github-account/banana-smasher/releases/download/v0.2.0-sm121-runtime/flashinfer_python-0.6.17-py3-none-any.whl#sha256=f507bee85810d4b79a7cdc7f16ca96c03761b75b7101a6f5a3feb97c658a129e ; sys_platform == "linux" and platform_machine == "aarch64"':
        'Requires-Dist: flashinfer-python @ https://github.com/my-other-github-account/banana-smasher/releases/download/v0.2.4-sm121-runtime/flashinfer_python-0.6.12%2Bbanana.fi0617-py3-none-any.whl#sha256=cf46d8f7549d8ce84fd65d01b2bb06e1204eaac7da889a22518ac84f6c01ae9a ; sys_platform == "linux" and platform_machine == "aarch64"',
    "Requires-Dist: numpy==2.3.5": "Requires-Dist: numpy==2.2.6",
}


def _digest(payload: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(payload).digest())
    return "sha256=" + encoded.rstrip(b"=").decode("ascii")


def _renamed(name: str) -> str:
    path = PurePosixPath(name)
    if path.parts and path.parts[0] == SOURCE_DIST_INFO:
        return str(PurePosixPath(TARGET_DIST_INFO, *path.parts[1:]))
    return name


def repack(source: Path, output: Path) -> None:
    members: list[tuple[ZipInfo, str, bytes]] = []
    metadata_seen = False
    with ZipFile(source) as archive:
        for original in archive.infolist():
            if original.filename.endswith("/RECORD"):
                continue
            name = _renamed(original.filename)
            payload = archive.read(original.filename)
            if name == f"{TARGET_DIST_INFO}/METADATA":
                text = payload.decode("utf-8")
                for old, new in REPLACEMENTS.items():
                    if text.count(old) != 1:
                        raise RuntimeError(f"expected exactly one METADATA field: {old}")
                    text = text.replace(old, new)
                payload = text.encode("utf-8")
                metadata_seen = True
            members.append((original, name, payload))
    if not metadata_seen:
        raise RuntimeError("source plugin METADATA was not found")

    record_name = f"{TARGET_DIST_INFO}/RECORD"
    rows = [[name, _digest(payload), str(len(payload))] for _, name, payload in members]
    rows.append([record_name, "", ""])
    record_buffer = io.StringIO(newline="")
    csv.writer(record_buffer, lineterminator="\n").writerows(rows)

    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for original, name, payload in members:
            info = ZipInfo(name, date_time=original.date_time)
            info.compress_type = ZIP_DEFLATED
            info.external_attr = original.external_attr
            info.create_system = original.create_system
            archive.writestr(info, payload)
        record_info = ZipInfo(record_name, date_time=(2026, 1, 1, 0, 0, 0))
        record_info.compress_type = ZIP_DEFLATED
        archive.writestr(record_info, record_buffer.getvalue().encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    repack(args.source, args.output)


if __name__ == "__main__":
    main()
