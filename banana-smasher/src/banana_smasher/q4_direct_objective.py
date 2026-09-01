"""Build Q4-only objectives from sealed direct reconstruction errors.

Authority formula (binary64): for each physical Q4 cell,
``mse = direct_error.sse / accounting.weights``.  The same cell-local
reconstruction distortion is supplied to every fixed corpus class; no probe,
fit, Q3 value, or fallback participates in the Q4 authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Iterable

_CLASSES = ("agentic", "chat", "code", "multilingual", "prose", "reasoning")


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_cell(value: str) -> str:
    value = value.replace("/", ":")
    if "_" in value:
        head, projection = value.rsplit("_", 1)
        value = head + ":" + projection
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(f"invalid cell id: {value}")
    return ":".join(parts)


def _atomic_jsonl(path: Path, rows: Iterable[dict]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    count = 0
    digest = hashlib.sha256()
    try:
        with os.fdopen(fd, "wb") as handle:
            for row in rows:
                raw = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
                handle.write(raw)
                digest.update(raw)
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    except BaseException:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise
    return count, digest.hexdigest()


def _receipt_index(
    roots: Iterable[Path], embedded_censuses: Iterable[Path]
) -> dict[str, tuple[Path, dict]]:
    index: dict[str, tuple[Path, dict]] = {}
    for root in roots:
        for path in sorted(root.rglob("CELL_RECEIPT.json")):
            raw = path.read_bytes()
            digest = _sha_bytes(raw)
            if digest in index:
                raise ValueError(f"duplicate receipt sha256: {digest}")
            index[digest] = (path, json.loads(raw))
    for census_path in embedded_censuses:
        census = json.loads(census_path.read_bytes())
        for relative_path, captured in census.get("json_receipts", {}).items():
            if not relative_path.endswith("/CELL_RECEIPT.json"):
                continue
            stat = captured["stat"]
            digest = stat["sha256"]
            receipt_path = Path(stat["path"])
            if digest in index:
                raise ValueError(f"duplicate receipt sha256: {digest}")
            index[digest] = (receipt_path, captured["object"])
    return index


def build_direct_objective_ledger(
    physical_ledger: str | Path,
    receipt_roots: Iterable[str | Path],
    output: str | Path,
    basis_sha256: str,
    *,
    expected_rows: int = 22016,
    embedded_censuses: Iterable[str | Path] = (),
) -> dict:
    """Seal one direct-MSE objective row for every physical Q4 cell."""
    physical_ledger = Path(physical_ledger)
    output = Path(output)
    index = _receipt_index(
        (Path(root) for root in receipt_roots),
        (Path(path) for path in embedded_censuses),
    )
    seen: set[str] = set()
    rows: list[dict] = []
    with physical_ledger.open() as handle:
        for line_number, line in enumerate(handle, 1):
            source_row = json.loads(line)
            if int(source_row.get("fallback_calls", 0)) != 0:
                raise ValueError(f"fallback physical row at line {line_number}")
            cell_id = _canonical_cell(source_row["cell"])
            if cell_id in seen:
                raise ValueError(f"duplicate cell: {cell_id}")
            seen.add(cell_id)
            receipt_sha = source_row["api_receipt_sha256"]
            if receipt_sha not in index:
                raise ValueError(f"missing sealed receipt for {cell_id}: {receipt_sha}")
            receipt_path, receipt = index[receipt_sha]
            if receipt.get("status") != "PASS" or receipt.get("basis_sha256") != basis_sha256:
                raise ValueError(f"receipt status/basis mismatch: {cell_id}")
            fallback_calls = int(receipt.get("installed_cuda_decode", {}).get("counters", {}).get("fallback_calls", receipt.get("cuda", {}).get("fallback_calls", 0)))
            if fallback_calls != 0:
                raise ValueError(f"fallback receipt: {cell_id}")
            weights = int(receipt["accounting"]["weights"])
            sse = float(receipt["direct_error"]["sse"])
            mse = float(receipt["direct_error"]["mse"])
            derived = sse / weights
            if weights <= 0 or not (math.isfinite(sse) and math.isfinite(mse) and sse >= 0 and mse >= 0):
                raise ValueError(f"invalid direct error: {cell_id}")
            if not math.isclose(mse, derived, rel_tol=2e-12, abs_tol=1e-18):
                raise ValueError(f"direct MSE closure mismatch: {cell_id}")
            codes_sha = receipt["artifacts"]["codes"]["sha256"]
            if source_row.get("codes_sha256") != codes_sha:
                raise ValueError(f"codes identity mismatch: {cell_id}")
            rows.append({
                "schema": "banana-smasher-q4-direct-objective-row-v1",
                "cell_id": cell_id,
                "tier": "qtip4_v7",
                "basis_sha256": basis_sha256,
                "formula": "float64(direct_error.sse) / int(accounting.weights)",
                "direct_reconstruction_sse": sse,
                "direct_reconstruction_mse": mse,
                "weights": weights,
                "prediction_by_class": {name: mse for name in _CLASSES},
                "source_weight_sha256": receipt["source"]["sha256"],
                "decoded_sha256": receipt["artifacts"]["decoded"]["sha256"],
                "codes_sha256": codes_sha,
                "cell_receipt_path": str(receipt_path),
                "cell_receipt_sha256": receipt_sha,
                "fallback_calls": 0,
            })
    if len(rows) != expected_rows:
        raise ValueError(f"row count mismatch: {len(rows)} != {expected_rows}")
    count, ledger_sha = _atomic_jsonl(output, rows)
    return {
        "schema": "banana-smasher-q4-direct-objective-terminal-v1",
        "status": "PASS",
        "basis_sha256": basis_sha256,
        "formula": "float64(direct_error.sse) / int(accounting.weights)",
        "rows": count,
        "missing": 0,
        "duplicates": 0,
        "fallback_rows": 0,
        "ledger_path": str(output.resolve()),
        "ledger_sha256": ledger_sha,
        "physical_ledger_sha256": _sha_bytes(physical_ledger.read_bytes()),
    }


def rewrite_q4_predictions(
    expanded_ledger: str | Path,
    objective_ledger: str | Path,
    output: str | Path,
    *,
    expected_q4_rows: int = 22016,
) -> dict:
    authority: dict[str, dict] = {}
    with Path(objective_ledger).open() as handle:
        for line in handle:
            row = json.loads(line)
            cell_id = row["cell_id"]
            if cell_id in authority:
                raise ValueError(f"duplicate objective cell: {cell_id}")
            authority[cell_id] = row
    if len(authority) != expected_q4_rows:
        raise ValueError(f"objective row count mismatch: {len(authority)} != {expected_q4_rows}")
    rewritten = 0
    used: set[str] = set()
    rows: list[dict] = []
    with Path(expanded_ledger).open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("tier") == "qtip4_v7":
                cell_id = _canonical_cell(row["cell_id"])
                direct = authority.get(cell_id)
                if direct is None:
                    raise ValueError(f"missing Q4 objective: {cell_id}")
                row["prediction_by_class"] = direct["prediction_by_class"]
                row["prediction_producer"] = {
                    "derivation": "sealed-q4-direct-reconstruction-mse-v1",
                    "path": str(Path(objective_ledger).resolve()),
                    "sha256": _sha_bytes(Path(objective_ledger).read_bytes()),
                    "cell_receipt_sha256": direct["cell_receipt_sha256"],
                    "source_weight_sha256": direct["source_weight_sha256"],
                    "decoded_sha256": direct["decoded_sha256"],
                    "formula": direct["formula"],
                }
                rewritten += 1
                used.add(cell_id)
            rows.append(row)
    if rewritten != expected_q4_rows or used != set(authority):
        raise ValueError(f"Q4 rewrite closure mismatch: rewritten={rewritten}, used={len(used)}")
    count, ledger_sha = _atomic_jsonl(Path(output), rows)
    return {"status": "PASS", "rows": count, "qtip4_rows_rewritten": rewritten, "ledger_sha256": ledger_sha}


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--physical-ledger", required=True)
    build.add_argument("--receipt-root", action="append", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--basis-sha256", required=True)
    build.add_argument("--expected-rows", type=int, default=22016)
    build.add_argument("--embedded-receipt-census", action="append", default=[])
    rewrite = sub.add_parser("rewrite")
    rewrite.add_argument("--expanded-ledger", required=True)
    rewrite.add_argument("--objective-ledger", required=True)
    rewrite.add_argument("--output", required=True)
    rewrite.add_argument("--expected-q4-rows", type=int, default=22016)
    args = parser.parse_args()
    if args.command == "build":
        result = build_direct_objective_ledger(
            args.physical_ledger,
            args.receipt_root,
            args.output,
            args.basis_sha256,
            expected_rows=args.expected_rows,
            embedded_censuses=args.embedded_receipt_census,
        )
    else:
        result = rewrite_q4_predictions(args.expanded_ledger, args.objective_ledger, args.output, expected_q4_rows=args.expected_q4_rows)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
