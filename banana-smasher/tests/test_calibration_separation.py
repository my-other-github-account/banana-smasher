import importlib.util
from pathlib import Path
import pytest


def test_external_calibration_rejects_eval_content_even_with_new_ids():
    path = Path(__file__).parents[1] / 'src/banana_smasher/calibration_separation.py'
    assert path.is_file(), 'missing content/ID-disjoint calibration gate'
    spec = importlib.util.spec_from_file_location('calibration_separation', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    evaluation = [{'item_id': 1, 'token_ids': [1, 2, 3, 4]}]
    with pytest.raises(ValueError, match='content'):
        mod.validate_separation([{'item_id': 'external:1', 'token_ids': [9, 1, 2, 3, 4, 8]}], evaluation)
    with pytest.raises(ValueError, match='ID'):
        mod.validate_separation([{'item_id': 1, 'token_ids': [8, 9, 10]}], evaluation)
    result = mod.validate_separation([{'item_id': 'external:1', 'token_ids': [1, 9, 10]}], evaluation)
    assert result['id_overlap'] == []
    assert result['maximum_contiguous_match_tokens'] == 1
    assert result['full_window_overlap'] == []
