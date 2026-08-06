from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import banana_smasher
from banana_smasher import (
    BackpackPlan,
    build_backpack,
    generate_backpack_candidates,
    inspect_backpack,
    list_backpack_family_bindings,
    price_backpack_selection,
    verify_pack,
)
from test_backpack_fixed_assignment_provider import _fixed_assignment_plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    workdir = args.workdir.resolve()
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)

    plan, manifest = _fixed_assignment_plan(workdir)
    parsed = BackpackPlan.from_mapping(plan)
    probe = workdir / "probe"
    inspected = inspect_backpack(parsed, run_root=probe)
    candidates = generate_backpack_candidates(parsed, run_root=probe)
    assignment = [
        {"cell_id": cell_id, "tier": "fixture-qtip2"}
        for cell_id in inspected["cell_ids"]
    ]
    price = price_backpack_selection(
        parsed,
        assignment=assignment,
        candidates=candidates,
    )
    plan["target"] = {
        "exact_bytes": inspected["fixed_total_bytes"] + price["payload_bytes"]
    }
    result = build_backpack(plan, run_root=workdir / "run")
    verification = verify_pack(Path(result["final_pack"]))
    if result["status"] != "PASS" or verification["status"] != "PASS":
        raise RuntimeError("public fixed-assignment Backpack proof did not verify")
    print(
        json.dumps(
            {
                "status": "PASS",
                "package_file": str(Path(banana_smasher.__file__).resolve()),
                "providers": [
                    row.provider for row in list_backpack_family_bindings()
                ],
                "fixed_assignment_sha256": parsed.tiers[0]["fixed_assignment"][
                    "sha256"
                ],
                "provider_activation_bytes": price["activation_bytes"],
                "manifest_bytes": manifest.stat().st_size,
                "final_pack": result["final_pack"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
