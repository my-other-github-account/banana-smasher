from pathlib import Path
import hashlib
import re


def test_sealed_pre_binding_tracks_actual_builder_bytes() -> None:
    root = Path(__file__).parents[1]
    builder = root / "assets" / "builder_B2_PUBLISHED_PRE.py"
    binding = (root / "sealed_pre_forward.py").read_text()
    expected = re.search(r'^BUILDER_SHA256 = "([0-9a-f]{64})"$', binding, re.MULTILINE)
    assert expected is not None
    assert expected.group(1) == hashlib.sha256(builder.read_bytes()).hexdigest()


def test_sealed_builder_exposes_authentic_ordered_tap_contract() -> None:
    source = (Path(__file__).parents[1] / "assets" / "builder_B2_PUBLISHED_PRE.py").read_text()
    assert 'ap.add_argument("--tap-fixture"' in source
    actual_index = 'actual_basis_index = os.path.join(a.meta_dir, "model.safetensors.index.json")'
    actual_gate = 'sha256(actual_basis_index) != a.intended_basis_sha256'
    actual_error = 'raise RuntimeError("BASIS_GATE_ACTUAL_META_INDEX_MISMATCH")'
    weight_map_load = 'wm = json.load(open(actual_basis_index))["weight_map"]'
    assert actual_index in source
    assert actual_gate in source
    assert actual_error in source
    assert weight_map_load in source
    assert source.index(actual_gate) < source.index("import transformers")
    assert source.index(actual_gate) < source.index(weight_map_load)
    assert 'taps[f"L{L:03d}"] = tensor_tap(hidden[0])' in source
    assert '"source_builder_sha256": sha256(__file__)' in source
    assert '"public_api"' not in source


def test_law4_product_tap_runs_through_public_validate() -> None:
    source = (Path(__file__).parents[1] / "resident_full64_accept.py").read_text()
    assert "def _law4_public_product_taps(" in source
    assert "measurement = api.validate(engine, (window,)" in source
    assert '"method": "ResidentRepairAPI.validate"' in source