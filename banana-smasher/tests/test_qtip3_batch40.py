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
