from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import re
import zipfile
from pathlib import Path


DIST = "flashinfer_python"
UPSTREAM_VERSION = "0.6.17"
COMPAT_VERSION = "0.6.12+banana.617"


def _record_digest(data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return f"sha256={digest.decode('ascii')}"


def repack(input_wheel: Path, output_dir: Path) -> Path:
    old_dist_info = f"{DIST}-{UPSTREAM_VERSION}.dist-info"
    new_dist_info = f"{DIST}-{COMPAT_VERSION}.dist-info"
    files: dict[str, bytes] = {}

    with zipfile.ZipFile(input_wheel) as source:
        for info in source.infolist():
            name = info.filename.replace(old_dist_info, new_dist_info, 1)
            if name.endswith("/RECORD"):
                continue
            data = source.read(info.filename)
            if name.endswith("/METADATA"):
                text = data.decode("utf-8")
                text, count = re.subn(
                    rf"(?m)^Version: {re.escape(UPSTREAM_VERSION)}$",
                    f"Version: {COMPAT_VERSION}",
                    text,
                    count=1,
                )
                if count != 1:
                    raise RuntimeError("upstream METADATA version was not found exactly once")
                data = text.encode("utf-8")
            files[name] = data

    metadata = f"{new_dist_info}/METADATA"
    if metadata not in files:
        raise RuntimeError(f"missing expected metadata: {metadata}")

    record_path = f"{new_dist_info}/RECORD"
    rows = [[name, _record_digest(data), str(len(data))] for name, data in sorted(files.items())]
    rows.append([record_path, "", ""])
    record = io.StringIO(newline="")
    csv.writer(record, lineterminator="\n").writerows(rows)
    files[record_path] = record.getvalue().encode("utf-8")

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{DIST}-{COMPAT_VERSION}-py3-none-any.whl"
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as target:
        for name, data in sorted(files.items()):
            target.writestr(name, data)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repack upstream FlashInfer 0.6.17 with vLLM-0.24-compatible local-version metadata."
    )
    parser.add_argument("input_wheel", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(repack(args.input_wheel, args.output_dir))


if __name__ == "__main__":
    main()
