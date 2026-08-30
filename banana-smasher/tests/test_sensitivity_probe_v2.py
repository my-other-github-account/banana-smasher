"""Tests for the PROBE_MANIFEST_V2 sensitivity probe materializer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from banana_smasher.sensitivity_probe_v2 import (
    authenticate_target_units,
    materialize_sensitivity_candidate_v2,
    probe_cell_specs,
)

BASIS = "9" * 64


def _canon(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _producer(root: str, cell_id: str, unit_sha: str, receipt_sha: str, root_map: Path) -> dict:
    layer, expert, projection = cell_id.split(":")
    return {
        "artifact_sha256": unit_sha,
        "sha256": receipt_sha,
        "path": f"{root}/{layer}/{expert}_{projection}/QTIP_SOLVE_RECEIPT.json",
        "root_map_path": str(root_map),
        "root_map_sha256": _sha_bytes(root_map.read_bytes()) if root_map.exists() else "",
    }


@pytest.fixture
def world(tmp_path: Path):
    """Two-cell baseline: L000:E000:down=qtip2, L001:E001:fused13=qtip3."""

    src = tmp_path / "sources"
    baseline_units = tmp_path / "materialized"
    cells = {
        "L000:E000:down": ("qtip2", "qtip3"),
        "L001:E001:fused13": ("qtip3", "qtip2"),
    }

    def write_unit(base: Path, cell_id: str, payload: bytes) -> tuple[str, str]:
        layer, expert, projection = cell_id.split(":")
        d = base / layer / f"{expert}_{projection}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "QTIP_UNIT.pt").write_bytes(payload)
        receipt = _canon({"cell_id": cell_id, "payload_sha256": _sha_bytes(payload)})
        (d / "QTIP_SOLVE_RECEIPT.json").write_bytes(receipt)
        return _sha_bytes(payload), _sha_bytes(receipt)

    # Unit file size must equal the ledger's physical_bytes (1000 incumbent /
    # 4000 target) — the worker's byte gate depends on that identity, and it
    # holds on the real artifacts (probe-000 target is 3,164,761 B on disk and
    # in the ledger).
    incumbent = {c: write_unit(baseline_units, c, (f"incumbent-{c}".encode() * 200)[:1000]) for c in cells}
    target = {c: write_unit(src, c, (f"TARGET-{c}".encode() * 800)[:4000]) for c in cells}

    def root_map(name: str, root: Path) -> Path:
        p = tmp_path / name
        p.write_bytes(_canon({
            "schema": "banana-smasher-layer-root-map-v1",
            "basis_sha256": BASIS,
            "tier": "qtip3",
            "layer_roots": {"0": str(root), "1": str(root)},
        }))
        return p

    baseline_map = root_map("BASELINE_ROOT_MAP.json", baseline_units)
    target_map = root_map("TARGET_ROOT_MAP.json", src)

    ledger = tmp_path / "LEDGER.jsonl"
    with ledger.open("wb") as fh:
        for cell_id, (source_tier, target_tier) in cells.items():
            for tier in ("qtip2", "qtip3", "native_mxfp4"):
                is_target = tier == target_tier
                base = src if is_target else baseline_units
                unit_sha, receipt_sha = (target if is_target else incumbent)[cell_id]
                fh.write(_canon({
                    "basis_sha256": BASIS,
                    "cell_id": cell_id,
                    "tier": tier,
                    "physical_bytes": 4000 if is_target else 1000,
                    "physical_producer": _producer(
                        str(base), cell_id, unit_sha, receipt_sha,
                        target_map if is_target else baseline_map,
                    ),
                }))

    assignment = {"0": {"0": {"down": "qtip2"}}, "1": {"1": {"fused13": "qtip3"}}}
    assignment_raw = _canon(assignment)
    index_rows = [
        {"cell_id": c, "tier": t[0], "source_key": t[0], "physical_bytes": 1000,
         "physical_artifact_sha256": incumbent[c][0],
         "physical_receipt_path": "x", "physical_receipt_sha256": incumbent[c][1]}
        for c, t in cells.items()
    ]
    index_raw = b"".join(_canon(r) for r in sorted(index_rows, key=lambda r: r["cell_id"]))
    bdir = tmp_path / "baseline"
    bdir.mkdir()
    (bdir / "ASSIGNMENT.json").write_bytes(assignment_raw)
    (bdir / "MATERIALIZATION_INDEX.jsonl").write_bytes(index_raw)
    manifest = {
        "schema": "banana-smasher-backpack-virtual-assignment-v1",
        "status": "PASS_LOGICAL_FULL_WIRE",
        "basis_sha256": BASIS,
        "assignment": {"file": "ASSIGNMENT.json", "bytes": len(assignment_raw),
                       "rows": 2, "sha256": _sha_bytes(assignment_raw)},
        "materialization_index": {"file": "MATERIALIZATION_INDEX.jsonl", "bytes": len(index_raw),
                                  "rows": 2, "sha256": _sha_bytes(index_raw)},
        "tier_counts": {"qtip2": 1, "qtip3": 1, "native_mxfp4": 0},
        "byte_accounting": {"payload_bytes": 2000, "assigned_expert_bytes": 2000,
                            "assigned_package_bytes": 2000,
                            "tier_payload_bytes": {"qtip2": 1000, "qtip3": 1000, "native_mxfp4": 0}},
        "expert_parameter_denominator": 16000,
        "whole_model_accounting": {"expert_physical_wire_bytes": 2000, "fixed_nonexpert_bytes": 500,
                                   "padding_bytes": 0, "shipping_bytes_cap": 100000,
                                   "logical_base_parameters": 32000},
    }
    (bdir / "BACKPACK_VIRTUAL_MANIFEST.json").write_bytes(_canon(manifest))

    def probe(pid, role, cell_ids, source_tier, target_tier, predicted=-1e-6):
        prods = []
        for c in cell_ids:
            is_target = cells[c][1] == target_tier
            base = src if is_target else baseline_units
            unit_sha, receipt_sha = (target if is_target else incumbent)[c]
            prods.append(_producer(str(base), c, unit_sha, receipt_sha,
                                   target_map if is_target else baseline_map))
        p = {
            "probe_id": pid, "role": role, "cell_id": cell_ids[0],
            "source_tier": source_tier, "target_tier": target_tier,
            "predicted_delta_mean_kld": predicted,
            "target_physical_producer": prods[0] if len(cell_ids) == 1 else prods,
        }
        if len(cell_ids) > 1:
            p["cells"] = list(cell_ids)
        return p

    return {
        "dir": tmp_path, "baseline": bdir / "BACKPACK_VIRTUAL_MANIFEST.json", "ledger": ledger,
        "probe": probe, "target_map": target_map, "baseline_map": baseline_map,
        "target": target, "incumbent": incumbent,
    }


def test_null_control_is_accepted_and_byte_neutral(world):
    """v1 refused same-tier swaps outright; v2 must materialize them."""
    p = world["probe"]("n0", "null_control", ["L000:E000:down"], "qtip2", "qtip2", predicted=0.0)
    c = materialize_sensitivity_candidate_v2(
        world["baseline"], world["ledger"], p, output_root=world["dir"] / "null")
    assert c["is_null_control"] is True
    assert c["shipping_delta_bytes"] == 0
    manifest = json.loads(Path(c["manifest_path"]).read_text())
    assert manifest["tier_counts"] == {"qtip2": 1, "qtip3": 1, "native_mxfp4": 0}


def test_terminal_schema_is_pinned_for_the_exact64_runtime(world):
    """backpack_runtime_exact64 compares this schema string literally under
    diagnostic_nonshipping; moving it to -v2 breaks every probe at bind time."""
    p = world["probe"]("s0", "treatment", ["L000:E000:down"], "qtip2", "qtip3")
    c = materialize_sensitivity_candidate_v2(
        world["baseline"], world["ledger"], p, output_root=world["dir"] / "s0")
    terminal = json.loads(Path(c["terminal_path"]).read_text())
    assert terminal["schema"] == "banana-smasher-sensitivity-virtual-terminal-v1"
    assert terminal["status"] == "PASS"
    assert terminal["virtual_manifest_sha256"] == c["manifest_sha256"]
    assert Path(terminal["virtual_manifest_path"]).resolve() == Path(c["manifest_path"]).resolve()


def test_treatment_swap_rebinds_target_producer(world):
    p = world["probe"]("t0", "treatment", ["L000:E000:down"], "qtip2", "qtip3")
    c = materialize_sensitivity_candidate_v2(
        world["baseline"], world["ledger"], p, output_root=world["dir"] / "t0")
    assert c["shipping_delta_bytes"] == 3000
    assert c["target_units"][0]["artifact_sha256"] == world["target"]["L000:E000:down"][0]
    rows = [json.loads(l) for l in Path(c["index_path"]).read_text().splitlines() if l.strip()]
    row = next(r for r in rows if r["cell_id"] == "L000:E000:down")
    assert row["tier"] == "qtip3"
    assert row["physical_artifact_sha256"] == world["target"]["L000:E000:down"][0]


def test_additivity_joint_moves_two_cells(world):
    p = world["probe"]("a0", "additivity_joint",
                       ["L000:E000:down", "L001:E001:fused13"], "qtip2", "qtip3")
    # both cells must legitimately be qtip2->qtip3; second cell is qtip3 in the
    # baseline, so this must fail closed.
    with pytest.raises(ValueError, match="source tier mismatch"):
        materialize_sensitivity_candidate_v2(
            world["baseline"], world["ledger"], p, output_root=world["dir"] / "a0")


def test_manifest_producer_must_match_ledger_producer(world):
    """The run6989 defect class: a manifest producer that drifts from the ledger."""
    p = world["probe"]("t1", "treatment", ["L000:E000:down"], "qtip2", "qtip3")
    p["target_physical_producer"] = dict(p["target_physical_producer"], artifact_sha256="0" * 64)
    with pytest.raises(ValueError, match="producer artifact_sha256 divergence"):
        materialize_sensitivity_candidate_v2(
            world["baseline"], world["ledger"], p, output_root=world["dir"] / "t1")


def test_authenticate_target_units_catches_incumbent_payload(world):
    """Binding the BASELINE root map for a real swap must fail before GPU time.

    This is exactly run6989: the sealed baseline slot holds the incumbent QTIP2
    payload, and the QTIP3 decoder refused it 441 s in. v2 refuses at t=0.
    """
    p = world["probe"]("t2", "treatment", ["L000:E000:down"], "qtip2", "qtip3")
    c = materialize_sensitivity_candidate_v2(
        world["baseline"], world["ledger"], p, output_root=world["dir"] / "t2")
    bad = world["baseline_map"]
    with pytest.raises(RuntimeError, match="TARGET_UNIT_PAYLOAD_RED"):
        authenticate_target_units(c, root_map_path=bad,
                                  root_map_sha256=_sha_bytes(bad.read_bytes()))
    good = world["target_map"]
    proofs = authenticate_target_units(c, root_map_path=good,
                                       root_map_sha256=_sha_bytes(good.read_bytes()))
    assert proofs[0]["unit_sha256"] == world["target"]["L000:E000:down"][0]


def test_authenticate_rejects_root_map_identity_drift(world):
    p = world["probe"]("t3", "treatment", ["L000:E000:down"], "qtip2", "qtip3")
    c = materialize_sensitivity_candidate_v2(
        world["baseline"], world["ledger"], p, output_root=world["dir"] / "t3")
    with pytest.raises(RuntimeError, match="TARGET_ROOT_MAP_IDENTITY_RED"):
        authenticate_target_units(c, root_map_path=world["target_map"], root_map_sha256="0" * 64)


def test_probe_cell_specs_length_mismatch_fails(world):
    p = world["probe"]("t4", "treatment", ["L000:E000:down"], "qtip2", "qtip3")
    p["cells"] = ["L000:E000:down", "L001:E001:fused13"]
    with pytest.raises(ValueError, match="cells/producers length mismatch"):
        probe_cell_specs(p)


def test_non_empty_candidate_dir_is_refused(world):
    p = world["probe"]("t5", "treatment", ["L000:E000:down"], "qtip2", "qtip3")
    out = world["dir"] / "t5"
    out.mkdir()
    (out / "stale.json").write_text("{}")
    with pytest.raises(FileExistsError):
        materialize_sensitivity_candidate_v2(
            world["baseline"], world["ledger"], p, output_root=out)


def test_pool_binding_precedence(world, tmp_path, monkeypatch):
    """Pool binds for real swaps; the sealed baseline still binds for nulls.

    A null control must reproduce the baseline exactly, so it must never be
    pointed at the recovered-unit pool even when one is supplied.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "spw2",
        Path(__file__).resolve().parents[1] / "tools" / "sensitivity_probe_worker_v2.py",
    )
    spw2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(spw2)

    pool_q3 = tmp_path / "POOL_Q3.json"
    pool_q3.write_bytes(_canon({"layer_roots": {"0": "/pool"}}))
    baseline_q2 = world["baseline_map"]
    baseline_q3 = world["target_map"]

    treat = world["probe"]("pb0", "treatment", ["L000:E000:down"], "qtip2", "qtip3")
    c = materialize_sensitivity_candidate_v2(
        world["baseline"], world["ledger"], treat, output_root=world["dir"] / "pb0")
    bound, _sha_, q2, q3 = spw2.resolve_target_root_map(
        c, baseline_q2, baseline_q3, pool_q2=None, pool_q3=pool_q3)
    assert bound == pool_q3 and q3 == pool_q3, "real swap must bind the pool"

    null = world["probe"]("pb1", "null_control", ["L000:E000:down"], "qtip2", "qtip2", predicted=0.0)
    cn = materialize_sensitivity_candidate_v2(
        world["baseline"], world["ledger"], null, output_root=world["dir"] / "pb1")
    bound_n, _s, q2n, _q3n = spw2.resolve_target_root_map(
        cn, baseline_q2, baseline_q3, pool_q2=pool_q3, pool_q3=pool_q3)
    assert bound_n == baseline_q2 and q2n == baseline_q2, "null must bind the sealed baseline"
