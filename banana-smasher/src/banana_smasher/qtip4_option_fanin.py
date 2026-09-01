import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from banana_smasher.qtip4_option_ledger import (
    QTIP4_BASIS_SHA256,
    QTIP4_OPTION_ROW_SCHEMA,
)


def _load_lines(raw: bytes, label: str) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{label} row {line_number} is not an object")
        rows.append(value)
    return rows


def _canonical(value: dict[str, Any], *, indent: int | None = None) -> bytes:
    if indent is None:
        return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return (json.dumps(value, sort_keys=True, indent=indent) + "\n").encode()


def fanin_qtip4_option_ledgers(
    physical_ledger_raw: bytes,
    option_ledgers_raw: list[bytes],
    *,
    expected_cells: int,
) -> bytes:
    """Select exactly one option row bound to every authenticated physical cell."""
    physical_rows = _load_lines(physical_ledger_raw, "physical ledger")
    physical: dict[str, dict[str, Any]] = {}
    for row in physical_rows:
        cell = str(row.get("cell_id", ""))
        if not cell or cell in physical:
            raise ValueError(f"duplicate or empty physical cell: {cell!r}")
        if row.get("basis_sha256") != QTIP4_BASIS_SHA256:
            raise ValueError(f"physical basis mismatch: {cell}")
        sealed_base = row.get("authority") == "sealed-base-frontier"
        fallback_clean = row.get("fallback_calls") == 0 or (
            sealed_base and "fallback_calls" not in row
        )
        if row.get("errors") or not fallback_clean:
            raise ValueError(f"physical cell is not clean: {cell}")
        physical[cell] = row
    if len(physical) != expected_cells:
        raise ValueError(
            f"physical roster mismatch: expected {expected_cells}, observed {len(physical)}"
        )

    selected: dict[str, dict[str, Any]] = {}
    for raw in option_ledgers_raw:
        for row in _load_lines(raw, "option ledger"):
            if row.get("schema") != QTIP4_OPTION_ROW_SCHEMA:
                raise ValueError("option row schema mismatch")
            cell = str(row.get("cell", ""))
            authority = physical.get(cell)
            if authority is None:
                raise ValueError(f"unknown option cell: {cell}")
            if row.get("tier") != "qtip4" or row.get("bpw") != 4.0:
                raise ValueError(f"non-QTIP4/U16 option row: {cell}")
            if int(row.get("fallback_calls", -1)) != 0:
                raise ValueError(f"option row fallback mismatch: {cell}")
            codes_match = (
                row.get("codes_sha256") == authority.get("codes_sha256")
                if authority.get("codes_sha256") is not None
                else authority.get("authority") == "sealed-base-frontier"
            )
            bindings_match = (
                row.get("public_receipt_sha256") == authority.get("public_receipt_sha256")
                and row.get("api_receipt_sha256") == authority.get("cell_receipt_sha256")
                and codes_match
            )
            if not bindings_match:
                continue
            if cell in selected:
                raise ValueError(f"duplicate authenticated option row: {cell}")
            selected[cell] = row
    missing = sorted(set(physical) - set(selected))
    if missing:
        raise ValueError(f"missing option rows: {len(missing)} first={missing[:3]}")
    return b"".join(_canonical(selected[cell]) for cell in sorted(selected))


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-ledger", required=True)
    parser.add_argument("--physical-sha256", required=True)
    parser.add_argument("--option-ledger", action="append", default=[])
    parser.add_argument("--appended-ledger", action="append", default=[])
    parser.add_argument("--expected-cells", type=int, default=22016)
    parser.add_argument("--canonical-pin", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    physical_path = Path(args.physical_ledger)
    physical_raw = physical_path.read_bytes()
    if _sha256(physical_raw) != args.physical_sha256:
        raise ValueError("physical ledger SHA mismatch")
    source_paths = [Path(value) for value in args.option_ledger]
    appended_paths = [Path(value) for value in args.appended_ledger]
    source_raw = [path.read_bytes() for path in source_paths]
    appended_raw = [path.read_bytes() for path in appended_paths]
    ledger_raw = fanin_qtip4_option_ledgers(
        physical_raw, source_raw + appended_raw, expected_cells=args.expected_cells
    )
    rows = len(ledger_raw.splitlines())
    appended_rows = sum(len(raw.splitlines()) for raw in appended_raw)
    prior_rows = rows - appended_rows
    output_dir = Path(args.output_dir)
    ledger_path = output_dir / "QTIP4_V7_OPTION_LEDGER_22016.jsonl"
    _atomic_write(ledger_path, ledger_raw)

    source_evidence = sorted(
        [
            {"sha256": _sha256(raw), "rows": len(raw.splitlines())}
            for raw in source_raw + appended_raw
        ],
        key=lambda row: row["sha256"],
    )
    frontier = {
        "schema": "banana-smasher-qtip4-v7-option-frontier-v1",
        "status": "PASS",
        "basis_sha256": QTIP4_BASIS_SHA256,
        "canonical_git_pin": args.canonical_pin,
        "expected_cells": args.expected_cells,
        "physical_cells": len(physical_raw.splitlines()),
        "option_ledger_rows": rows,
        "unique_option_cells": rows,
        "duplicates": 0,
        "missing": 0,
        "unknown": 0,
        "fallback_rows": 0,
        "u16_rows": 0,
        "physical_ledger_sha256": _sha256(physical_raw),
        "option_ledger_sha256": _sha256(ledger_raw),
        "source_option_ledgers": source_evidence,
    }
    frontier_raw = _canonical(frontier, indent=2)
    frontier_path = output_dir / "Q4_FRONTIER.json"
    _atomic_write(frontier_path, frontier_raw)

    append_receipt = {
        "schema": "qtip4-v7-option-ledger-append-receipt-v1",
        "status": "PASS",
        "canonical_git_pin": args.canonical_pin,
        "prior_rows": prior_rows,
        "appended_rows": appended_rows,
        "final_rows": rows,
        "exact_once": True,
        "appended_ledger_sha256": sorted(_sha256(raw) for raw in appended_raw),
        "final_option_ledger_sha256": _sha256(ledger_raw),
    }
    append_raw = _canonical(append_receipt, indent=2)
    append_path = output_dir / "OPTION_LEDGER_APPEND_RECEIPT.json"
    _atomic_write(append_path, append_raw)

    terminal = {
        "schema": "banana-smasher-qtip4-v7-22016-terminal-v1",
        "status": "PASS",
        "basis_sha256": QTIP4_BASIS_SHA256,
        "canonical_git_pin": args.canonical_pin,
        "physical_cells": len(physical_raw.splitlines()),
        "option_rows": rows,
        "expected_cells": args.expected_cells,
        "duplicates": 0,
        "missing": 0,
        "fallback_rows": 0,
        "u16_rows": 0,
        "physical_ledger_sha256": _sha256(physical_raw),
        "option_ledger_sha256": _sha256(ledger_raw),
        "frontier_sha256": _sha256(frontier_raw),
        "append_receipt_sha256": _sha256(append_raw),
    }
    terminal_raw = _canonical(terminal, indent=2)
    terminal_path = output_dir / "QTIP4_V7_22016_TERMINAL.json"
    _atomic_write(terminal_path, terminal_raw)
    print(json.dumps(terminal, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
