from __future__ import annotations

import ast
import inspect
from pathlib import Path

import banana_smasher
from banana_smasher import anchor, backpack, backpack_exact64, metrics
from banana_smasher.cli import _parser
from banana_smasher.hf_deepseek_v4_backpack_adapter import DeepseekV4BackpackRuntime
from banana_smasher import qtip_v7_joint_workflow
from banana_smasher.production_rails import ProductionRails
from banana_smasher.official_k2_resident import OfficialK2PackedResidentAdapter
from banana_smasher.qtip_v7_routes import _load_qtip2_v7_member_roster
from banana_smasher.resident_repair_api import ResidentRepairAPI


def _command_names(parser) -> set[str]:
    names: set[str] = set()
    pending = [parser]
    while pending:
        current = pending.pop()
        for action in current._actions:
            choices = getattr(action, "choices", None)
            if not isinstance(choices, dict):
                continue
            names.update(choices)
            pending.extend(choices.values())
    return names


def test_resident_api_is_the_only_public_score_and_train_surface() -> None:
    forbidden_package_exports = {
        "score_bank",
        "score_backpack",
        "run_backpack_exact64",
        "run_backpack_train8",
        "repair_backpack",
        "build_backpack",
        "train_joint",
    }
    assert forbidden_package_exports.isdisjoint(banana_smasher.__all__)
    assert not hasattr(anchor, "score_bank")
    assert not hasattr(backpack, "score_backpack")
    assert not hasattr(backpack, "repair_backpack")
    assert not hasattr(backpack, "build_backpack")
    assert not hasattr(qtip_v7_joint_workflow, "train_joint")
    assert not hasattr(backpack_exact64, "score_backpack_exact64")
    assert not hasattr(metrics, "score_candidate")
    assert not hasattr(OfficialK2PackedResidentAdapter, "score")

    commands = _command_names(_parser())
    assert {
        "score",
        "backpack-exact64",
        "train",
        "update",
        "qtip-v7-joint-repair",
    }.isdisjoint(commands)
    top_level = next(
        action.choices
        for action in _parser()._actions
        if isinstance(getattr(action, "choices", None), dict)
    )
    backpack_commands = next(
        action.choices
        for action in top_level["backpack"]._actions
        if isinstance(getattr(action, "choices", None), dict)
    )
    assert "build" not in backpack_commands


def test_resident_public_api_and_cli_have_no_callable_slow_path_fallback() -> None:
    forbidden = {"fallback", "offline", "replay", "staged", "reload", "rate_low", "slow"}
    public_callables = {
        name.lower()
        for cls in (ResidentRepairAPI, ProductionRails)
        for name, value in vars(cls).items()
        if not name.startswith("_") and callable(value)
    }
    assert all(token not in name for name in public_callables for token in forbidden)

    resident_parser = next(
        action.choices["resident"]
        for action in _parser()._actions
        if isinstance(getattr(action, "choices", None), dict)
    )
    arm_parser = next(
        action.choices["arm"]
        for action in resident_parser._actions
        if isinstance(getattr(action, "choices", None), dict)
    )
    option_names = {
        option.lower()
        for action in arm_parser._actions
        for option in action.option_strings
    }
    assert all(token not in option for option in option_names for token in forbidden)

    resident_sources = "\n".join(
        Path(inspect.getsourcefile(cls)).read_text()
        for cls in (ResidentRepairAPI, ProductionRails)
    )
    assert "RATE_LOW" not in resident_sources


def test_qtip_v7_runtime_has_no_layer_number_special_case() -> None:
    sources = (
        inspect.getsource(DeepseekV4BackpackRuntime),
        inspect.getsource(_load_qtip2_v7_member_roster),
    )
    for source in sources:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            direct_layer_operand = any(
                isinstance(operand, ast.Name) and operand.id == "layer"
                for operand in (node.left, *node.comparators)
            )
            integer_literals = {
                operand.value
                for operand in (node.left, *node.comparators)
                if isinstance(operand, ast.Constant)
                and isinstance(operand.value, int)
                and not isinstance(operand.value, bool)
            }
            assert not (direct_layer_operand and integer_literals), ast.unparse(node)
    assert "L034" not in "\n".join(sources)


def test_all_routing_implementations_reject_layer_number_conditionals() -> None:
    repository = Path(__file__).resolve().parents[2]
    implementations = (
        repository
        / "banana-smasher/src/banana_smasher/hf_deepseek_v4_backpack_adapter.py",
        repository / "banana-smasher/src/banana_smasher/qtip_v7_routes.py",
        repository / "runtime/v7/runner/fast_two_node_v7.py",
        repository
        / "runtime/v7/vendor/site/banana_smasher/update_backends/joint_v7_repair.py",
    )
    violations = []
    exceptional_tokens = []
    for path in implementations:
        source = path.read_text()
        tree = ast.parse(source, filename=str(path))
        if "L034" in source or "l034" in source:
            exceptional_tokens.append(str(path.relative_to(repository)))
        for node in ast.walk(tree):
            condition = None
            if isinstance(node, (ast.If, ast.IfExp, ast.While)):
                condition = node.test
            elif isinstance(node, ast.Match):
                condition = node
            elif (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Dict)
            ):
                condition = node
            if condition is None:
                continue
            comprehension_bound_names = {
                target.id
                for child in ast.walk(condition)
                if isinstance(child, ast.comprehension)
                for target in ast.walk(child.target)
                if isinstance(target, ast.Name)
            }
            layer_operands = {
                child.id
                for child in ast.walk(condition)
                if isinstance(child, ast.Name)
                and child.id.lower() == "layer"
                and child.id not in comprehension_bound_names
            }
            layer_operands.update(
                child.attr
                for child in ast.walk(condition)
                if isinstance(child, ast.Attribute)
                and child.attr.lower() == "layer"
            )
            integers = {
                child.value
                for child in ast.walk(condition)
                if isinstance(child, ast.Constant)
                and isinstance(child.value, int)
                and not isinstance(child.value, bool)
            }
            if layer_operands and integers:
                violations.append(
                    f"{path.relative_to(repository)}: {ast.unparse(condition)}"
                )
    assert violations == []
    assert exceptional_tokens == []


def test_joint_runtime_public_interface_forwards_pinned_member_roster() -> None:
    repository = Path(__file__).resolve().parents[2]
    backend_path = (
        repository
        / "runtime/v7/vendor/site/banana_smasher/update_backends/joint_v7_repair.py"
    )
    cli_path = repository / "runtime/v7/vendor/site/banana_smasher/cli.py"
    backend_tree = ast.parse(backend_path.read_text(), filename=str(backend_path))
    cli_tree = ast.parse(cli_path.read_text(), filename=str(cli_path))

    wrapper = next(
        node
        for node in ast.walk(backend_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "run_joint_v7_repair"
    )
    wrapper_arguments = {argument.arg for argument in wrapper.args.kwonlyargs}
    assert {"member_roster", "member_roster_sha256"} <= wrapper_arguments

    cli_literals = {
        node.value
        for node in ast.walk(cli_tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert {"--member-roster", "--member-roster-sha256"} <= cli_literals
    wrapper_calls = [
        node
        for node in ast.walk(cli_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_joint_v7_repair"
    ]
    assert len(wrapper_calls) == 1
    forwarded = {keyword.arg for keyword in wrapper_calls[0].keywords}
    assert {"member_roster", "member_roster_sha256"} <= forwarded
