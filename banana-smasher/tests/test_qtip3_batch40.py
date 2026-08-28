from __future__ import annotations

import hashlib
import importlib.util
import sys
import types
from pathlib import Path


def load_producer(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    package = types.ModuleType("banana_smasher")
    package.__path__ = [str(root / "src" / "banana_smasher")]
    monkeypatch.setitem(sys.modules, "banana_smasher", package)
    api_stub = types.ModuleType("banana_smasher.qtip25_native_v4_api")
    api_stub.build_qtip_native_cell = lambda *args, **kwargs: {}
    monkeypatch.setitem(sys.modules, api_stub.__name__, api_stub)
    spec = importlib.util.spec_from_file_location(
        "banana_smasher.qtip3_api_producer",
        root / "src" / "banana_smasher" / "qtip3_api_producer.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_public_batch_default_groups_40_cells_without_reordering(tmp_path, monkeypatch):
    producer = load_producer(monkeypatch)
    monkeypatch.setattr(producer, "LAYERS", (2,))
    monkeypatch.setattr(producer, "EXPERTS", tuple(range(20)))

    basis_file = tmp_path / "model.index"
    basis_file.write_bytes(b"basis")
    basis = hashlib.sha256(basis_file.read_bytes()).hexdigest()
    driver = tmp_path / "authority"
    allocation = "HOST_ALLOCATION t_test spark-4 qtip3-batch40-test"
    driver.write_text(allocation + "\n")
    control = tmp_path / "control.npy"
    control.write_bytes(b"control")
    source = tmp_path / "source.npy"
    source.write_bytes(b"source")
    tlut = tmp_path / "tlut.npy"
    tlut.write_bytes(b"tlut")
    mission = tmp_path / "mission"
    (mission / "receipts").mkdir(parents=True)
    (mission / "receipts" / "ADMISSION.json").write_text("{}\n")

    plan = producer.Qtip3ApiPlan(
        task_id="t_test",
        board_run_id=1,
        host="spark-4",
        allocation=allocation,
        intended_basis_sha256=basis,
        driver_goals_path=driver,
        driver_goals_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),
        claim_path=tmp_path / "claim.json",
        shards_path=mission / "SHARDS.json",
        mission_root=mission,
        model_index_path=basis_file,
        tlut_path=tlut,
        layers=(2,),
    )
    cells = [
        producer.CellSpec(
            layer=2,
            expert=expert,
            projection=projection,
            source=source,
            control=control,
            output=mission / "outputs" / f"E{expert:03d}_{projection}",
        )
        for expert in range(20)
        for projection in producer.PROJECTIONS
    ]
    batches = []

    def batch_api(rows, *_args, **_kwargs):
        batches.append([Path(row["output"]).name for row in rows])
        return [
            {
                "status": "PASS",
                "backend": "cuda",
                "codec_version": "v6",
                "provider": "qtip-native-v6@3.00",
                "geometry": {"B": 12, "L": 16, "V": 4},
                "installed_cuda_decode": {
                    "counters": {"cuda_decode_calls": 1, "fallback_calls": 0}
                },
                "receipt_sha256": f"receipt-{index}",
            }
            for index, _row in enumerate(rows)
        ]

    terminal = producer.run_cells_batched(
        plan,
        producer.Qtip3ApiConfig(),
        reversed(cells),
        batch_api=batch_api,
    )

    expected_order = [f"E{expert:03d}_{projection}" for expert in range(20) for projection in producer.PROJECTIONS]
    assert batches == [expected_order]
    assert terminal["batch_size"] == 40
    assert terminal["cells"] == 40
    assert terminal["new_cells"] == 40
    assert terminal["new_batches"] == 1


def test_public_batch_can_stop_after_one_novel_batch(tmp_path, monkeypatch):
    producer = load_producer(monkeypatch)
    monkeypatch.setattr(producer, "LAYERS", (2,))
    monkeypatch.setattr(producer, "EXPERTS", tuple(range(40)))
    basis_file = tmp_path / "model.index"
    basis_file.write_bytes(b"basis")
    basis = hashlib.sha256(basis_file.read_bytes()).hexdigest()
    driver = tmp_path / "authority"
    allocation = "HOST_ALLOCATION t_test spark-4 qtip3-batch40-test"
    driver.write_text(allocation + "\n")
    source = tmp_path / "source.npy"
    source.write_bytes(b"source")
    control = tmp_path / "control.npy"
    control.write_bytes(b"control")
    tlut = tmp_path / "tlut.npy"
    tlut.write_bytes(b"tlut")
    mission = tmp_path / "mission"
    (mission / "receipts").mkdir(parents=True)
    (mission / "receipts" / "ADMISSION.json").write_text("{}\n")
    plan = producer.Qtip3ApiPlan(
        task_id="t_test", board_run_id=1, host="spark-4", allocation=allocation,
        intended_basis_sha256=basis, driver_goals_path=driver,
        driver_goals_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),
        claim_path=tmp_path / "claim.json", shards_path=mission / "SHARDS.json",
        mission_root=mission, model_index_path=basis_file, tlut_path=tlut, layers=(2,),
    )
    cells = [
        producer.CellSpec(layer=2, expert=expert, projection=projection, source=source,
                          control=control, output=mission / "outputs" / f"E{expert:03d}_{projection}")
        for expert in range(40) for projection in producer.PROJECTIONS
    ]
    calls = []

    def batch_api(rows, *_args, **_kwargs):
        calls.append(len(rows))
        return [
            {"status": "PASS", "backend": "cuda", "codec_version": "v6",
             "provider": "qtip-native-v6@3.00", "geometry": {"B": 12, "L": 16, "V": 4},
             "installed_cuda_decode": {"counters": {"cuda_decode_calls": 1, "fallback_calls": 0}},
             "receipt_sha256": f"receipt-{index}"}
            for index, _ in enumerate(rows)
        ]

    terminal = producer.run_cells_batched(
        plan, producer.Qtip3ApiConfig(), cells, batch_api=batch_api, max_new_batches=1
    )
    assert calls == [40]
    assert terminal["bounded_partial"] is True
    assert terminal["new_batches"] == 1
    assert terminal["new_cells"] == 40
    assert terminal["cells"] == 40


def test_public_batch_accepts_an_exact_irregular_cell_roster(tmp_path, monkeypatch):
    producer = load_producer(monkeypatch)
    monkeypatch.setattr(producer, "LAYERS", (0, 3))
    monkeypatch.setattr(producer, "EXPERTS", tuple(range(5)))
    basis_file = tmp_path / "model.index"
    basis_file.write_bytes(b"basis")
    basis = hashlib.sha256(basis_file.read_bytes()).hexdigest()
    driver = tmp_path / "authority"
    allocation = "HOST_ALLOCATION t_test spark-8 exact-roster-test"
    driver.write_text(allocation + "\n")
    control = tmp_path / "control.npy"; control.write_bytes(b"control")
    source = tmp_path / "source.npy"; source.write_bytes(b"source")
    tlut = tmp_path / "tlut.npy"; tlut.write_bytes(b"tlut")
    mission = tmp_path / "mission"
    (mission / "receipts").mkdir(parents=True)
    (mission / "receipts" / "ADMISSION.json").write_text("{}\n")
    roster = ((0, 0, "down"), (0, 4, "fused13"), (3, 2, "down"))
    plan = producer.Qtip3ApiPlan(
        task_id="t_test", board_run_id=1, host="spark-8", allocation=allocation,
        intended_basis_sha256=basis, driver_goals_path=driver,
        driver_goals_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),
        claim_path=tmp_path / "claim.json", shards_path=mission / "SHARDS.json",
        mission_root=mission, model_index_path=basis_file, tlut_path=tlut,
        layers=(0, 3), cell_roster=roster)
    cells = [producer.CellSpec(layer=l, expert=e, projection=p, source=source,
             control=control, output=mission / "outputs" / f"L{l:03d}_E{e:03d}_{p}")
             for l, e, p in roster]
    def batch_api(rows, *_args, **_kwargs):
        return [{"status": "PASS", "backend": "cuda", "codec_version": "v6",
                 "provider": "qtip-native-v6@3.00", "geometry": {"B": 12, "L": 16, "V": 4},
                 "installed_cuda_decode": {"counters": {"cuda_decode_calls": 1, "fallback_calls": 0}},
                 "receipt_sha256": f"receipt-{i}"} for i, _ in enumerate(rows)]
    terminal = producer.run_cells_batched(plan, producer.Qtip3ApiConfig(), cells, batch_api=batch_api)
    assert plan.expected_cells == 3
    assert terminal["cells"] == 3
    assert terminal["cell_roster"] == [list(row) for row in roster]


def test_load_cell_roster_from_physical_preflight(tmp_path, monkeypatch):
    producer = load_producer(monkeypatch)
    path = tmp_path / "preflight.json"
    path.write_text('{"basis_sha256":"' + producer.BASIS + '","missing":['
                    '{"cell_id":"L003:E004:down"},'
                    '{"cell_id":"L000:E000:fused13"}]}')
    assert producer.load_cell_roster(path, intended_basis_sha256=producer.BASIS) == (
        (0, 0, "fused13"), (3, 4, "down"))


def test_admission_adopts_released_same_task_irregular_shard(tmp_path, monkeypatch):
    producer = load_producer(monkeypatch)
    monkeypatch.setattr(producer, "LAYERS", (0, 3))
    monkeypatch.setattr(producer, "EXPERTS", tuple(range(5)))
    basis_file = tmp_path / "model.index"; basis_file.write_bytes(b"basis")
    basis = hashlib.sha256(basis_file.read_bytes()).hexdigest()
    driver = tmp_path / "authority"
    allocation = "HOST_ALLOCATION t_test spark-8 exact-roster-test"
    driver.write_text(allocation + "\n")
    claim = tmp_path / "claim.json"
    claim.write_text('{"status":"RELEASED","state":"RELEASED","controller_pid":null,"workload_pid":null}\n')
    mission = tmp_path / "mission"; mission.mkdir()
    shard = mission / "SHARDS.json"
    shard.write_text('{"status":"RELEASED","state":"RELEASED","task_id":"t_test",'
                     '"scope_layers":[0,3],"missing_members":3}\n')
    tlut = mission / "inputs" / "qtip_tlut.npy"; tlut.parent.mkdir(); tlut.write_bytes(b"tlut")
    plan = producer.Qtip3ApiPlan(
        task_id="t_test", board_run_id=2, host="spark-8", allocation=allocation,
        intended_basis_sha256=basis, driver_goals_path=driver,
        driver_goals_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),
        claim_path=claim, shards_path=shard, mission_root=mission,
        model_index_path=basis_file, tlut_path=tlut, layers=(0, 3),
        cell_roster=((0, 0, "down"), (0, 4, "fused13"), (3, 2, "down")),
        expected_claim_sha256=hashlib.sha256(claim.read_bytes()).hexdigest())
    receipt = producer.admit_host_and_shard(plan, gpu_probe=lambda: (), pid=1)
    assert receipt["status"] == "PASS"
    assert receipt["cells"] == 3


def test_progress_counter_is_monotone_within_run_but_resets_for_new_run(tmp_path, monkeypatch):
    producer = load_producer(monkeypatch)
    path = tmp_path / "PROGRESS.json"
    producer._write_progress_monotone(
        path,
        {"board_run_id": 10, "accepted_cells": 3160, "last_cell": "old"},
    )

    producer._write_progress_monotone(
        path,
        {"board_run_id": 11, "accepted_cells": 40, "last_cell": "new"},
    )

    import json

    assert json.loads(path.read_text()) == {
        "board_run_id": 11,
        "accepted_cells": 40,
        "last_cell": "new",
    }
