import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "tools" / "sensitivity_probe_worker_v3.py"
MANIFEST_SHA = "17a973ee4becbad9c7e456ab95cb3fd5b4866712a069a04a8d0a13306e3a1445"

def constants():
    tree = ast.parse(WORKER.read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try: out[node.targets[0].id] = ast.literal_eval(node.value)
            except Exception: pass
    return out

def test_complete20_manifest_contract_is_hard_pinned():
    c = constants()
    assert c["PROBE_MANIFEST_V3_COMPLETE20_SHA"] == MANIFEST_SHA
    assert c["PROBE_MANIFEST_V3_COMPLETE20_COUNT"] == 20
    assert c["BASELINE_W28_KLD"] == 0.09936928004026413
    assert c["CHECKPOINT_SHA"].startswith("f9bffe04")

def test_worker_rejects_unpinned_manifest_before_model_load():
    src = WORKER.read_text()
    assert "if sha(args.probe_manifest) != PROBE_MANIFEST_V3_COMPLETE20_SHA" in src
    assert src.index("PROBE_MANIFEST_V3_COMPLETE20_SHA_RED") < src.index("model_root = Path")
