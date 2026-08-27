import ast
import inspect
from pathlib import Path


def test_qtip3_public_batch_entrypoint_has_complete_runtime_import_closure():
    from banana_smasher import qtip25_native_v4_api, qtip3_api_producer

    assert qtip3_api_producer.build_qtip_native_cell is qtip25_native_v4_api.build_qtip_native_cell
    assert callable(qtip25_native_v4_api.build_qtip_native_cells)
    assert inspect.signature(qtip3_api_producer.run_cells_batched).parameters["batch_size"].default == 40


def test_qtip3_regenerate_entrypoint_accepts_authorized_routed_scope_and_batches_40():
    source = Path(__file__).parents[1] / "src" / "banana_smasher" / "qtip3_regenerate.py"
    tree = ast.parse(source.read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    run = next(node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "run_cells_batched")
    assert next(keyword.value.value for keyword in run.keywords if keyword.arg == "batch_size") == 40
    assert "QTIP3 recovery scope must be exactly" not in source.read_text()
    assert "QTIP3_CELL_ROSTER_PATH" in source.read_text()
    assert "load_cell_roster" in source.read_text()
