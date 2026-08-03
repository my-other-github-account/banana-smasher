from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLIENT = ROOT / "tools" / "bounded_top1_route.py"
CONTROLLER = ROOT / "tools" / "claim_bound_bounded_top1.py"


def test_operational_tools_parse_as_python() -> None:
    ast.parse(CLIENT.read_text())
    ast.parse(CONTROLLER.read_text())


def test_bounded_client_is_top1_only_and_config_driven() -> None:
    source = CLIENT.read_text()
    assert "top1_matches" in source
    assert '"kld_computed": False' in source
    assert '"kld_published": False' in source
    assert "torch.nn.functional" not in source
    assert "kl_div" not in source
    assert ("/ho" + "me/dnola") not in source
    assert ("t_" + "fdd0d107") not in source
    assert "98efab455cf08dfb" not in source


def test_client_enforces_bulk_route_and_drift_gates() -> None:
    source = CLIENT.read_text()
    assert '"one_loopback_bulk_request_per_window"' in source
    assert '"CANARY_DRIFT' in source
    assert '"SEALED_CANARY_DRIFT' in source
    assert '"FINAL_VECTOR_DRIFT' in source
    assert "requests_sent += 1" in source
    assert "expected_route_requests = initial_route_requests + requests_sent" in source


def test_controller_stops_consumer_before_route_on_failure() -> None:
    source = CONTROLLER.read_text()
    failure = source.index("except BaseException as error:")
    consumer_stop = source.index("consumer_dead = terminate_exact(consumer", failure)
    route_stop = source.index("route_dead = terminate_exact(route", failure)
    assert consumer_stop < route_stop
    assert '"route_dead_verified": route_dead' in source
    assert '"consumer_dead_verified": consumer_dead' in source


def test_controller_is_config_driven() -> None:
    source = CONTROLLER.read_text()
    assert ("/ho" + "me/dnola") not in source
    assert ("t_" + "fdd0d107") not in source
    assert "98efab455cf08dfb" not in source
    assert "ROUTE_BINARY_DRIFT" in source
    assert "BASIS_GATE_MISMATCH" in source
    assert "CANDIDATE_MANIFEST_DRIFT" in source
