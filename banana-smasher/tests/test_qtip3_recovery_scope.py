from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from banana_smasher.qtip3_api_producer import (
    BASIS,
    EXPECTED_CELLS,
    LAYERS,
    Qtip3ApiPlan,
    verify_basis,
    verify_driver_authority,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_recovery_scope_is_exactly_l034_l042() -> None:
    assert LAYERS == tuple(range(34, 43))
    assert EXPECTED_CELLS == 9 * 256 * 2


def test_plan_rejects_any_other_layer_scope(tmp_path: Path) -> None:
    authority = tmp_path / "CARD_AUTHORITY.md"
    authority.write_text("HOST_ALLOCATION t_test spark-6 qtip3-l034-l042-regeneration\n")
    index = tmp_path / "model.safetensors.index.json"
    index.write_text("{}")
    with pytest.raises(ValueError, match="missing scope is fixed"):
        Qtip3ApiPlan(
            task_id="t_test",
            board_run_id=1,
            host="spark-6",
            allocation=authority.read_text().strip(),
            intended_basis_sha256=BASIS,
            driver_goals_path=authority,
            driver_goals_sha256=_sha(authority),
            claim_path=tmp_path / "HOST_CLAIM.json",
            shards_path=tmp_path / "SHARDS.json",
            mission_root=tmp_path / "mission",
            model_index_path=index,
            tlut_path=tmp_path / "qtip_tlut.npy",
            layers=(34,),
        )


def test_basis_and_exact_allocation_are_fail_closed(tmp_path: Path) -> None:
    authority = tmp_path / "CARD_AUTHORITY.md"
    allocation = "HOST_ALLOCATION t_test spark-6 qtip3-l034-l042-regeneration"
    authority.write_text(allocation + "\n")
    index = tmp_path / "model.safetensors.index.json"
    index.write_bytes(b"basis")
    plan = Qtip3ApiPlan(
        task_id="t_test",
        board_run_id=1,
        host="spark-6",
        allocation=allocation,
        intended_basis_sha256=_sha(index),
        driver_goals_path=authority,
        driver_goals_sha256=_sha(authority),
        claim_path=tmp_path / "HOST_CLAIM.json",
        shards_path=tmp_path / "SHARDS.json",
        mission_root=tmp_path / "mission",
        model_index_path=index,
        tlut_path=tmp_path / "qtip_tlut.npy",
    )
    assert verify_basis(plan)["status"] == "PASS"
    assert verify_driver_authority(plan)["allocation"] == allocation

    authority.write_text("prose mentioning t_test and spark-6 is not authority\n")
    with pytest.raises(RuntimeError, match="DRIVER_GOALS_SHA_REFUSED"):
        verify_driver_authority(plan)


def test_public_qtip_import_does_not_require_scipy() -> None:
    script = r'''
import importlib.abc
import sys
class BlockScipy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "scipy" or fullname.startswith("scipy."):
            raise ModuleNotFoundError("blocked optional scipy", name=fullname)
        return None
sys.meta_path.insert(0, BlockScipy())
from banana_smasher.qtip3_api_producer import LAYERS
assert LAYERS == tuple(range(34, 43))
'''
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    subprocess.run([sys.executable, "-c", script], env=env, check=True)
