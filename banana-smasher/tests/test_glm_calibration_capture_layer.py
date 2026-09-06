import hashlib
import json
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("target", [3,4,2,45])
def test_capture_exact_requested_next_layer(tmp_path, monkeypatch, target):
    def put(name, obj):
        p = tmp_path / name
        p.write_text(json.dumps(obj))
        return p
    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()
    index = put('model.safetensors.index.json', {})
    put('config.json', {})
    put('tokenizer.json', {})
    ledger = put('ledger.json', {'rows':[{'item_id':'independent:0', 'token_ids':[1]*1024}]})
    separation = put('separation.json', {'status':'test only'})
    lock = put('lock.json', {})
    visited = []
    class Gate(torch.nn.Module):
        def forward(self, x):
            return None, torch.ones(1024, 1), torch.zeros(1024, 1, dtype=torch.long)
    class Layer(torch.nn.Module):
        def __init__(self, number):
            super().__init__()
            self.number = number
            self.mlp = SimpleNamespace(gate=Gate())
        def forward(self, hidden, **kwargs):
            visited.append(self.number)
            self.mlp.gate(hidden.reshape(1024, 2))
            return hidden + 1, torch.zeros(1024, 1, dtype=torch.long)
    language = SimpleNamespace(config=SimpleNamespace(hc_mult=1), embed_tokens=torch.nn.Embedding(2,2), layers=[Layer(i) for i in range(6)])
    class Executor:
        device = 'cpu'
        def __init__(self, **kwargs):
            assert kwargs['corpus_rows'][0]['item_id']=='independent:0'
        def _language_model(self):
            return language
        def _materialize(self, module):
            pass
        def _dematerialize(self, module):
            pass
    module = ModuleType('banana_smasher.hf_sharded_balanced64_executor')
    module.PackageHFShardedExecutor = Executor
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(__version__='fixture'))
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda n:'fixture CPU')
    output = tmp_path/'output'
    spec = put('spec.json', dict(output=str(output), model_root=str(tmp_path), canonical_commit='test-only', calibration_ledger=str(ledger), calibration_ledger_sha256=sha(ledger), separation_receipt=str(separation), separation_receipt_sha256=sha(separation), suite_lock=str(lock), intended_basis=sha(index), windows=1, layer=target, clean_fit_admission=True))
    script = Path(__file__).parents[2]/'tools/glm_calibration_capture.py'
    monkeypatch.setattr(sys, 'argv', [str(script),str(spec)])
    if target not in (3,4):
        with pytest.raises(AssertionError, match="outside routed scope"):
            runpy.run_path(str(script),run_name="__main__")
        assert visited == []
        return
    runpy.run_path(str(script),run_name='__main__')
    assert visited == list(range(target+1))
    result = json.loads((output/'CAPTURE_RESULT.json').read_text())
    assert result['status']=='PASS_NATIVE_CAPTURE' and result['layer']==target
    captures = list((output/'captures').glob('*.pt'))
    assert [p.name for p in captures]==[f'xmoe_L{target:03d}_win0000.pt']
    assert torch.load(captures[0],weights_only=True)['layer']==target
