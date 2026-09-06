"""The shipped solver must admit its own pinned public runner, not arbitrary code."""
import hashlib
from pathlib import Path
from banana_smasher import solver_qtip_profile as solver


def test_shipped_public_runner_matches_trusted_package_anchor():
    runner = Path(solver.__file__).with_name('qtip_runner.py')
    digest = hashlib.sha256(runner.read_bytes()).hexdigest()
    assert digest == solver._TRUSTED_PUBLIC_QTIP_RUNNER_SHA256
    assert callable(solver._load_public_qtip_runner(runner, digest).build_qtip)
