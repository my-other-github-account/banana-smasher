import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


QTIP4_BASIS_SHA256 = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
QTIP4_GEOMETRY = {"B": 16, "L": 16, "V": 4}
QTIP4_PROVIDER = "qtip-native-v6@4.00"
QTIP4_PUBLIC_SCHEMA = "banana-smasher-qtip3-v7-public-api-producer-v1-cell"
QTIP4_OPTION_ROW_SCHEMA = "qtip4-v7-option-ledger-row-v1"


def _load_object(raw: bytes, label: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not an object")
    return value


def emit_qtip4_option_row(
    public_receipt_raw: bytes,
    api_receipt_raw: bytes,
    codes_sha256: str,
) -> bytes:
    """Emit one canonical CLEAN102 QTIP4 option row from authenticated receipts."""
    public = _load_object(public_receipt_raw, "public receipt")
    api = _load_object(api_receipt_raw, "API receipt")
    api_sha256 = hashlib.sha256(api_receipt_raw).hexdigest()

    if public.get("schema") != QTIP4_PUBLIC_SCHEMA or public.get("status") != "PASS":
        raise ValueError("public receipt schema/status mismatch")
    if public.get("basis_sha256") != QTIP4_BASIS_SHA256 or api.get("basis_sha256") != QTIP4_BASIS_SHA256:
        raise ValueError("QTIP4 basis mismatch")
    if public.get("bpw") != 4.0 or public.get("provider") != QTIP4_PROVIDER:
        raise ValueError("QTIP4 tier/provider mismatch")
    if public.get("geometry") != QTIP4_GEOMETRY or api.get("geometry") != QTIP4_GEOMETRY:
        raise ValueError("QTIP4 geometry mismatch")
    if public.get("backend") != "cuda" or int(public.get("cuda_decode_calls", 0)) <= 0:
        raise ValueError("CUDA-positive evidence missing")
    counters = api.get("installed_cuda_decode", {}).get("counters", {})
    if int(public.get("fallback_calls", -1)) != 0 or int(counters.get("fallback_calls", -1)) != 0:
        raise ValueError("fallback calls are nonzero")
    if public.get("api_receipt_sha256") != api_sha256:
        raise ValueError("API receipt SHA mismatch")
    if api.get("status") != "PASS":
        raise ValueError("API receipt status mismatch")
    accounting = api.get("accounting", {})
    if accounting.get("exact_code_bpw") != 4.0:
        raise ValueError("QTIP4 accounting mismatch")
    if api.get("artifacts", {}).get("codes", {}).get("sha256") != codes_sha256:
        raise ValueError("codes SHA mismatch")
    optimization = api.get("optimization", {})

    row = {
        "schema": QTIP4_OPTION_ROW_SCHEMA,
        "task_id": public["task_id"],
        "cell": public["cell"],
        "layer": public["layer"],
        "expert": public["expert"],
        "projection": public["projection"],
        "tier": "qtip4",
        "bpw": 4.0,
        "provider": public["provider"],
        "geometry": public["geometry"],
        "scale_factors": public["scale_factors"],
        "selected_factor": optimization["selected_factor"],
        "selected_scale": optimization["selected_scale"],
        "objective": public["objective"],
        "fallback_calls": 0,
        "exact_code_bits": accounting["exact_code_bits"],
        "weights": accounting["weights"],
        "codes_sha256": codes_sha256,
        "api_receipt_sha256": api_sha256,
        "public_receipt_sha256": hashlib.sha256(public_receipt_raw).hexdigest(),
    }
    return json.dumps(row, sort_keys=True, separators=(",", ":")).encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def emit_root_option_ledger(
    root: str | Path,
    *,
    expected_cells: int,
    authenticated_cells: dict[str, dict[str, Any]] | None = None,
) -> bytes:
    """Authenticate one producer root and emit its sorted canonical row ledger."""
    root_path = Path(root).resolve()
    public_paths = sorted((root_path / "outputs" / "q4").glob("L*_*/PUBLIC_CELL_RECEIPT.json"))
    if len(public_paths) != expected_cells:
        raise ValueError(f"root roster mismatch: expected {expected_cells}, observed {len(public_paths)}")
    rows: dict[str, bytes] = {}
    for public_path in public_paths:
        public_raw = public_path.read_bytes()
        public = _load_object(public_raw, "public receipt")
        cell = str(public.get("cell", ""))
        if not cell or cell in rows:
            raise ValueError(f"duplicate or empty cell: {cell!r}")
        api_path = Path(str(public.get("api_receipt", "")))
        api_raw = api_path.read_bytes()
        api = _load_object(api_raw, "API receipt")
        if authenticated_cells is None:
            codes_path = Path(str(api.get("artifacts", {}).get("codes", {}).get("path", "")))
            codes_sha256 = _sha256_file(codes_path)
        else:
            authority = authenticated_cells.get(cell)
            if not isinstance(authority, dict):
                raise ValueError(f"cell absent from authenticated physical ledger: {cell}")
            if authority.get("public_receipt_sha256") != hashlib.sha256(public_raw).hexdigest():
                raise ValueError(f"authenticated public receipt SHA mismatch: {cell}")
            if authority.get("cell_receipt_sha256") != hashlib.sha256(api_raw).hexdigest():
                raise ValueError(f"authenticated API receipt SHA mismatch: {cell}")
            codes_sha256 = str(authority.get("codes_sha256", ""))
            if api.get("artifacts", {}).get("codes", {}).get("sha256") != codes_sha256:
                raise ValueError(f"authenticated codes SHA mismatch: {cell}")
        rows[cell] = emit_qtip4_option_row(public_raw, api_raw, codes_sha256)
    return b"".join(rows[cell] + b"\n" for cell in sorted(rows))


def _read_authenticated_cells(path: str) -> dict[str, dict[str, Any]]:
    raw = sys.stdin.buffer.read() if path == "-" else Path(path).read_bytes()
    rows: dict[str, dict[str, Any]] = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        row = _load_object(line, "authenticated physical ledger row")
        cell = str(row.get("cell_id", ""))
        if not cell or cell in rows:
            raise ValueError(f"duplicate or empty authenticated cell: {cell!r}")
        rows[cell] = row
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--expected-cells", required=True, type=int)
    parser.add_argument("--authenticated-ledger")
    args = parser.parse_args(argv)
    authenticated = (
        _read_authenticated_cells(args.authenticated_ledger)
        if args.authenticated_ledger is not None
        else None
    )
    raw = emit_root_option_ledger(
        args.root,
        expected_cells=args.expected_cells,
        authenticated_cells=authenticated,
    )
    os.write(1, raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
