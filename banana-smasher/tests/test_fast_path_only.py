from __future__ import annotations

import ast
import inspect
from pathlib import Path

import banana_smasher
from banana_smasher import anchor, backpack, backpack_exact64, metrics
from banana_smasher.cli import _parser
from banana_smasher.hf_deepseek_v4_backpack_adapter import DeepseekV4BackpackRuntime
from banana_smasher import qtip_v7_joint_workflow
from banana_smasher.qtip_v7_routes import _load_qtip2_v7_member_roster


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
            if not isinstance(node, ast.Compare):
                continue
            operands = (node.left, *node.comparators)
            layer_operands = {
                operand.id
                for operand in operands
                if isinstance(operand, ast.Name)
                and "layer" in operand.id.lower()
                and operand.id != "split_layer"
            }
            layer_operands.update(
                operand.attr
                for operand in operands
                if isinstance(operand, ast.Attribute)
                and "layer" in operand.attr.lower()
            )
            integers = {
                operand.value
                for operand in operands
                if isinstance(operand, ast.Constant)
                and isinstance(operand.value, int)
                and not isinstance(operand.value, bool)
            }
            if layer_operands and integers:
                violations.append(
                    f"{path.relative_to(repository)}: {ast.unparse(node)}"
                )
    assert violations == []
    assert exceptional_tokens == []
