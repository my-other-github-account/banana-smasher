from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import re
import tomllib
import zipfile
from pathlib import Path


DIST = "banana_smasher_plugin"


def _record_digest(data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return f"sha256={digest.decode('ascii')}"


def repack(input_wheel: Path, pyproject: Path, output_dir: Path) -> Path:
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
    version = project["version"]
    dependencies = project["dependencies"]

    with zipfile.ZipFile(input_wheel) as source:
        old_dist_info = next(
            name.split("/", 1)[0]
            for name in source.namelist()
            if name.endswith(".dist-info/METADATA")
        )
        new_dist_info = f"{DIST}-{version}.dist-info"
        files: dict[str, bytes] = {}
        for info in source.infolist():
            name = info.filename.replace(old_dist_info, new_dist_info, 1)
            if name.endswith("/RECORD"):
                continue
            data = source.read(info.filename)
            if name.endswith("/METADATA"):
                text = data.decode("utf-8")
                text, count = re.subn(
                    r"(?m)^Version: .+$", f"Version: {version}", text, count=1
                )
                if count != 1:
                    raise RuntimeError("plugin METADATA version was not found exactly once")
                text = re.sub(r"(?m)^Requires-Dist: .+\n?", "", text)
                header, separator, body = text.rstrip("\n").partition("\n\n")
                requires = "\n".join(f"Requires-Dist: {item}" for item in dependencies)
                metadata = f"{header}\n{requires}"
                if separator:
                    metadata = f"{metadata}{separator}{body}"
                data = f"{metadata}\n".encode("utf-8")
            files[name] = data

    record_path = f"{new_dist_info}/RECORD"
    rows = [[name, _record_digest(data), str(len(data))] for name, data in sorted(files.items())]
    rows.append([record_path, "", ""])
    record = io.StringIO(newline="")
    csv.writer(record, lineterminator="\n").writerows(rows)
    files[record_path] = record.getvalue().encode("utf-8")

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{DIST}-{version}-cp312-cp312-linux_aarch64.whl"
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as target:
        for name, data in sorted(files.items()):
            target.writestr(name, data)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebind an accepted native plugin wheel to source metadata.")
    parser.add_argument("input_wheel", type=Path)
    parser.add_argument("pyproject", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(repack(args.input_wheel, args.pyproject, args.output_dir))


if __name__ == "__main__":
    main()
