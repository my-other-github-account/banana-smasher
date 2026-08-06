#!/usr/bin/env python3
"""Repack FlashInfer 0.6.17 under a vLLM-0.24-compatible local version.

The Python package bytes and their runtime ``flashinfer.__version__`` remain
0.6.17. Only the distribution metadata uses ``0.6.12+banana.fi0617`` so stock
vLLM's exact dependency pin remains resolver-clean.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

SOURCE_VERSION = "0.6.17"
COMPAT_VERSION = "0.6.12+banana.fi0617"
SOURCE_DIST_INFO = f"flashinfer_python-{SOURCE_VERSION}.dist-info"
COMPAT_DIST_INFO = f"flashinfer_python-{COMPAT_VERSION}.dist-info"


def _record_digest(payload: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(payload).digest())
    return "sha256=" + encoded.rstrip(b"=").decode("ascii")


def _renamed(name: str) -> str:
    path = PurePosixPath(name)
    if path.parts and path.parts[0] == SOURCE_DIST_INFO:
        return str(PurePosixPath(COMPAT_DIST_INFO, *path.parts[1:]))
    return name


def repack(source: Path, output: Path) -> None:
    members: list[tuple[ZipInfo, str, bytes]] = []
    metadata_seen = False
    with ZipFile(source) as archive:
        for info in archive.infolist():
            if info.filename.endswith("/RECORD"):
                continue
            payload = archive.read(info.filename)
            name = _renamed(info.filename)
            if name == f"{COMPAT_DIST_INFO}/METADATA":
                text = payload.decode("utf-8")
                old = f"Version: {SOURCE_VERSION}\n"
                new = f"Version: {COMPAT_VERSION}\n"
                if text.count(old) != 1:
                    raise RuntimeError("source METADATA has unexpected Version field")
                payload = text.replace(old, new).encode("utf-8")
                metadata_seen = True
            members.append((info, name, payload))
    if not metadata_seen:
        raise RuntimeError("source wheel METADATA was not found")

    record_name = f"{COMPAT_DIST_INFO}/RECORD"
    rows = [[name, _record_digest(payload), str(len(payload))] for _, name, payload in members]
    rows.append([record_name, "", ""])
    record_buffer = io.StringIO(newline="")
    csv.writer(record_buffer, lineterminator="\n").writerows(rows)
    record_payload = record_buffer.getvalue().encode("utf-8")

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
        archive.writestr(record_info, record_payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    repack(args.source, args.output)


if __name__ == "__main__":
    main()
