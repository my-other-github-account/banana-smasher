"""GLM Q1 existing-producer integration; no launcher or replacement producer.

Plans are read-only and refer to protected historical roots. They never admit
historical fit lineage or grant a host claim. The physical owner must supply a
fresh identity/CAS and authorize the clean boundary before using the same QTIP
controller with a fresh generation of configs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .hf_moe import _derive_routed_scope, _sha256

OWNER = "t_024b05d4"
HOST_LAYERS = {"spark-3": (3, 17), "spark-5-work": (17, 31), "spark-7": (31, 45)}


def producer_plan(
    host: str, model_root: Path, *, intended_basis: str
) -> dict[str, Any]:
    """Bind all 288 experts and both QTIP projections to one existing producer."""
    if host not in HOST_LAYERS:
        raise ValueError(f"unknown protected GLM producer host: {host}")
    root = Path(model_root)
    index = root / "model.safetensors.index.json"
    if _sha256(index) != intended_basis:
        raise ValueError("GLM producer source basis mismatch")
    config_path = root / "config.json"
    config = json.loads(config_path.read_text())
    names = set(json.loads(index.read_text())["weight_map"])
    _, routed, geometry = _derive_routed_scope(config, sorted(names))
    if (
        geometry["expected_model_layers"],
        geometry["dense_prefix_layers"],
        geometry["routed_experts"],
    ) != (45, 3, 288):
        raise ValueError(
            "GLM producer allocation requires L003..L044 and all 288 experts"
        )
    prefix = "model.language_model.layers."
    expected = {
        f"{prefix}{layer}.mlp.experts.{expert}.{projection}_proj.weight"
        for layer in range(3, 45)
        for expert in range(288)
        for projection in ("gate", "up", "down")
    }
    if routed != expected:
        raise ValueError(
            f"GLM routed source mismatch: missing={sorted(expected - routed)[:8]} "
            f"unexpected={sorted(routed - expected)[:8]}"
        )
    start, stop = HOST_LAYERS[host]
    producer_root = (
        Path("/home/dnola/missions")
        / f"GLM_Q1_CHAMPION_{OWNER}_{host.replace('-', '_')}"
    )
    cells = []
    for layer in range(start, stop):
        for projection, projections in (
            ("fused13", ("gate", "up")),
            ("down", ("down",)),
        ):
            for expert in range(288):
                cell = f"L{layer:03d}_E{expert:03d}_{projection}"
                cells.append(
                    {
                        "id": cell,
                        "layer": layer,
                        "expert": expert,
                        "projection": projection,
                        "source_names": [
                            f"{prefix}{layer}.mlp.experts.{expert}.{p}_proj.weight"
                            for p in projections
                        ],
                        "historical_receipt": str(
                            producer_root
                            / "solve"
                            / f"L{layer:03d}"
                            / f"E{expert:03d}_{projection}"
                            / "QTIP_SOLVE_RECEIPT.json"
                        ),
                    }
                )
    return {
        "schema": "banana-smasher.glm-q1-producer-plan.v1",
        "status": "PLAN_ONLY",
        "host": host,
        "owner": OWNER,
        "producer_root": str(producer_root),
        "layers": list(range(start, stop)),
        "experts": 288,
        "cells": cells,
        "expected_cells": len(cells),
        "source_model_index_sha256": intended_basis,
        "config_sha256": _sha256(config_path),
        "launch_authorized": False,
        "historical_label": "partial K1/256-expert-plan diagnostic",
        "historical_q2_label": "Q2 on routed E000–E255 + native FP8 E256–E287, native nonrouted rest; hybrid diagnostic",
        "historical_heldout_admitted": False,
        "target": {
            "tier": "q1",
            "scope": "routed_only",
            "native_rest": True,
            "routed_source_projections": len(expected),
            "uniform_cells": 24192,
        },
        "native_names": sorted(names - expected),
    }


def validate_adoption(plan, claim, shards, process, *, now):
    """Validate fresh owner observations; never launch or rewrite controls."""
    pid, ticks = process.get("pid"), process.get("start_ticks")
    if (
        process.get("alive") is not True
        or type(pid) is not int
        or pid <= 0
        or type(ticks) is not int
        or ticks <= 0
        or process.get("host") != plan["host"]
        or process.get("root") != plan["producer_root"]
        or process.get("layers") != plan["layers"]
    ):
        raise ValueError("producer identity requires owner reconciliation")
    if (
        claim.get("task_id") != OWNER
        or claim.get("host") != plan["host"]
        or claim.get("workload_pid") != pid
        or claim.get("start_ticks") != ticks
        or claim.get("expiry_unix", 0) <= now
        or claim.get("source_model_index_sha256") != plan["source_model_index_sha256"]
    ):
        raise ValueError("host claim requires identity-bound CAS reconciliation")
    rows = shards.get("rows", [])
    if (
        shards.get("intended_basis") != plan["source_model_index_sha256"]
        or len(rows) != len(plan["layers"])
        or sorted(row.get("layer", -1) for row in rows) != plan["layers"]
        or any(
            row.get("owner") != OWNER
            or row.get("pid") != pid
            or row.get("startticks") != ticks
            or row.get("range") != [0, 288]
            or row.get("basis") != plan["source_model_index_sha256"]
            or row.get("state") != "CLAIMED"
            or row.get("projections") != ["fused13", "down"]
            for row in rows
        )
    ):
        raise ValueError("shards require full288 identity-bound CAS reconciliation")
    return {
        "status": "ADOPT_EXISTING",
        "pid": pid,
        "start_ticks": ticks,
        "host": plan["host"],
        "producer_root": plan["producer_root"],
        "launch_authorized": False,
        "heldout_admitted": False,
    }


def validate_fanin_roster(model_root, intended_basis, rows):
    """Check exact host/cell coverage, not self-declared totals or PRE quality.

    Artifact hashes, clean fit ancestry and runtime parity must be independently
    admitted by the owner; this metadata gate grants none of those properties.
    """
    expected = {
        (host, cell["id"])
        for host in HOST_LAYERS
        for cell in producer_plan(host, model_root, intended_basis=intended_basis)[
            "cells"
        ]
    }
    actual = [(row["host"], row["id"]) for row in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError(
            f"GLM fan-in roster mismatch: expected={len(expected)} actual={len(actual)} "
            f"missing={len(expected - set(actual))} unexpected={len(set(actual) - expected)}"
        )
    return {
        "status": "ROSTER_COMPLETE_ONLY",
        "expected_cells": len(expected),
        "source_model_index_sha256": intended_basis,
        "heldout_admitted": False,
        "physical_hashes_verified": False,
        "launch_authorized": False,
    }
