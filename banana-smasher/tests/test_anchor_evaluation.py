from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from banana_smasher.anchor import (
    AnchorEvaluationError,
    aggregate_scores,
    build_bank_manifest,
    compare_training_rails,
    create_balanced_subset,
    emit_solver_row,
    format_status,
    import_producer,
    materialize_bank,
    register_bank,
    resolve_bank_identities,
    score_bank,
    status_report,
    validate_bank_manifest,
)
from banana_smasher.cli import main


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return path


def _identity(payload: bytes, uri: str) -> dict[str, str]:
    return {
        "status": "resolved",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "uri": uri,
    }


def _parent(tmp_path: Path, *, role: str = "train512") -> tuple[Path, dict]:
    rows = [
        {"window_id": 0, "class": "alpha", "payload": "a0", "rank": 9},
        {"window_id": 1, "class": "beta", "payload": "b1", "rank": 8},
        {"window_id": 2, "class": "alpha", "payload": "a2", "rank": 1},
        {"window_id": 3, "class": "beta", "payload": "b3", "rank": 2},
        {"window_id": 4, "class": "alpha", "payload": "a4", "rank": 4},
        {"window_id": 5, "class": "beta", "payload": "b5", "rank": 3},
    ]
    path = _write_jsonl(tmp_path / f"{role}.jsonl", rows)
    corpus = _identity(path.read_bytes(), path.name)
    identities = {
        "corpus": corpus,
        "tokenizer": {"status": "resolved", "sha256": "1" * 64, "uri": "tokenizer://fixture"},
        "teacher": {"status": "unresolved", "reason": "fixture producer supplied at score time"},
        "scorer": {"status": "resolved", "sha256": "2" * 64, "uri": "scorer://fixture"},
    }
    manifest = build_bank_manifest(
        bank_id=role,
        role=role,
        windows=[{"id": row["window_id"], "class": row["class"]} for row in rows],
        parent_corpus=corpus,
        identities=identities,
        split_lineage={"split": role, "parent_bank_id": None},
        creation={"method": "fixture", "config": {}},
        relationships=[],
    )
    return path, manifest


def test_manifest_hashes_are_stable_and_tampering_fails(tmp_path: Path) -> None:
    _, manifest = _parent(tmp_path)

    assert validate_bank_manifest(manifest)["status"] == "PASS"
    rebuilt = json.loads(json.dumps(manifest))
    assert rebuilt["content_hashes"] == manifest["content_hashes"]

    rebuilt["windows"][0]["class"] = "beta"
    with pytest.raises(AnchorEvaluationError, match="class_counts|class_map_sha256"):
        validate_bank_manifest(rebuilt)

    extra = json.loads(json.dumps(manifest))
    extra["unexpected"] = "not in the versioned schema"
    with pytest.raises(AnchorEvaluationError, match="manifest fields mismatch"):
        validate_bank_manifest(extra)

    identity_extra = json.loads(json.dumps(manifest))
    identity_extra["identities"]["scorer"]["secret"] = "must not be admitted"
    with pytest.raises(AnchorEvaluationError, match="resolved identity.*exactly"):
        validate_bank_manifest(identity_extra)

    bad_relationship = json.loads(json.dumps(manifest))
    bad_relationship["relationships"] = [{}]
    with pytest.raises(AnchorEvaluationError, match=r"relationships\[0\]"):
        validate_bank_manifest(bad_relationship)


def test_balanced_selection_is_config_driven_and_deterministic(tmp_path: Path) -> None:
    parent_path, parent = _parent(tmp_path)
    config = {
        "bank_id": "train-panel",
        "role": "train_balanced64",
        "quotas": {"alpha": 2, "beta": 1},
        "seed": "fixture-seed",
        "ranking_field": "rank",
        "tier_menus": {"uniform": ["small", "large"]},
    }

    first = create_balanced_subset(parent, parent_path, config)
    second = create_balanced_subset(parent, parent_path, config)

    assert first == second
    assert first["class_counts"] == {"alpha": 2, "beta": 1}
    assert [row["id"] for row in first["windows"]] == [2, 4, 3]
    assert first["creation"]["config"]["tier_menus"] == config["tier_menus"]
    assert first["split_lineage"]["parent_bank_id"] == "train512"


def test_materialize_verifies_parent_membership_labels_and_disjointness(tmp_path: Path) -> None:
    parent_path, parent = _parent(tmp_path)
    panel = create_balanced_subset(
        parent,
        parent_path,
        {
            "bank_id": "panel",
            "role": "train_balanced64",
            "quotas": {"alpha": 1, "beta": 1},
            "seed": "fixture",
            "ranking_field": "rank",
        },
    )
    output = tmp_path / "materialized.jsonl"

    receipt = materialize_bank(panel, parent_path, output)

    assert receipt["status"] == "PASS"
    assert receipt["expected_count"] == receipt["materialized_count"] == 2
    assert [json.loads(line)["window_id"] for line in output.read_text().splitlines()] == [2, 3]

    overlapping = build_bank_manifest(
        bank_id="overlap",
        role="holdout_balanced64",
        windows=[{"id": 2, "class": "alpha"}],
        parent_corpus=parent["parent_corpus"],
        identities=parent["identities"],
        split_lineage={"split": "holdout", "parent_bank_id": "train512"},
        creation={"method": "fixture", "config": {}},
        relationships=[],
    )
    with pytest.raises(AnchorEvaluationError, match="disjointness.*2"):
        materialize_bank(panel, parent_path, output, disjoint_manifests=[overlapping])


def _resolved_file(path: Path) -> dict[str, str]:
    return _identity(path.read_bytes(), path.name)


def _score_fixture(tmp_path: Path) -> tuple[dict, Path, Path]:
    _, full = _parent(tmp_path)
    windows = full["windows"][:4]
    manifest = build_bank_manifest(
        bank_id="score-panel",
        role="train_balanced64",
        windows=windows,
        parent_corpus=full["parent_corpus"],
        identities=full["identities"],
        split_lineage={"split": "fixture", "parent_bank_id": "train512"},
        creation={"method": "fixture", "config": {}},
        relationships=[{"bank_id": "train512", "relation": "ordered_subset_of"}],
    )
    teacher = _write_jsonl(
        tmp_path / "teacher.jsonl",
        [
            {"window_id": row["id"], "logits": [2.0, 0.0]}
            for row in windows
        ],
    )
    candidate = _write_jsonl(
        tmp_path / "candidate.jsonl",
        [
            {"window_id": row["id"], "logits": [1.0 + index / 10.0, 0.0]}
            for index, row in enumerate(windows)
        ],
    )
    return manifest, teacher, candidate


def test_scoring_is_exactly_bound_and_resumable(tmp_path: Path) -> None:
    manifest, teacher, candidate = _score_fixture(tmp_path)
    output = tmp_path / "raw.jsonl"
    candidate_identity = {"status": "resolved", "sha256": "3" * 64, "uri": "candidate://tier-a"}
    teacher_identity = {"status": "resolved", "sha256": "5" * 64, "uri": "teacher://fixture-model"}

    first = score_bank(
        manifest,
        teacher,
        candidate,
        output,
        candidate_id="tier-a",
        candidate_identity=candidate_identity,
        teacher_identity=teacher_identity,
        basis_sha256="4" * 64,
    )
    second = score_bank(
        manifest,
        teacher,
        candidate,
        output,
        candidate_id="tier-a",
        candidate_identity=candidate_identity,
        teacher_identity=teacher_identity,
        basis_sha256="4" * 64,
    )

    assert first["coverage"] == "4/4"
    assert first["new_rows"] == 4
    assert first["bindings"]["teacher_sha256"] == "5" * 64
    assert first["bindings"]["teacher_producer_sha256"] == hashlib.sha256(teacher.read_bytes()).hexdigest()
    assert second["new_rows"] == 0
    assert second["resumed_rows"] == 4
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 4
    assert all(row["kld"] >= 0 for row in rows)
    assert rows[0]["bindings"]["basis_sha256"] == "4" * 64


def test_scoring_fails_loudly_when_a_producer_is_missing(tmp_path: Path) -> None:
    manifest, teacher, candidate = _score_fixture(tmp_path)
    with pytest.raises(AnchorEvaluationError, match="missing candidate producer.*materialize or import"):
        score_bank(
            manifest,
            teacher,
            tmp_path / "absent.jsonl",
            tmp_path / "raw.jsonl",
            candidate_id="tier-a",
            candidate_identity={"status": "resolved", "sha256": "3" * 64, "uri": "candidate://tier-a"},
            teacher_identity=_resolved_file(teacher),
            basis_sha256="4" * 64,
        )


def test_scoring_requires_resolved_corpus_tokenizer_and_scorer(tmp_path: Path) -> None:
    manifest, teacher, candidate = _score_fixture(tmp_path)
    manifest["identities"]["scorer"] = {
        "status": "unresolved",
        "reason": "fixture unresolved scorer",
    }
    manifest["content_hashes"] = build_bank_manifest(
        bank_id=manifest["bank_id"],
        role=manifest["role"],
        windows=manifest["windows"],
        parent_corpus=manifest["parent_corpus"],
        identities=manifest["identities"],
        split_lineage=manifest["split_lineage"],
        creation=manifest["creation"],
        relationships=manifest["relationships"],
        dataset_fields=manifest["dataset_fields"],
    )["content_hashes"]
    with pytest.raises(AnchorEvaluationError, match="resolve bank identities.*scorer"):
        score_bank(
            manifest,
            teacher,
            candidate,
            tmp_path / "raw.jsonl",
            candidate_id="tier-a",
            candidate_identity={"status": "resolved", "sha256": "3" * 64, "uri": "candidate://tier-a"},
            teacher_identity=_resolved_file(teacher),
            basis_sha256="4" * 64,
        )

    resolved = resolve_bank_identities(
        manifest,
        {"scorer": {"status": "resolved", "sha256": "9" * 64, "uri": "scorer://exact"}},
    )
    assert resolved["identities"]["scorer"]["status"] == "resolved"
    assert validate_bank_manifest(resolved)["status"] == "PASS"


def test_aggregation_labels_measurements_and_parent_estimates(tmp_path: Path) -> None:
    manifest, teacher, candidate = _score_fixture(tmp_path)
    raw = tmp_path / "raw.jsonl"
    score_bank(
        manifest,
        teacher,
        candidate,
        raw,
        candidate_id="tier-a",
        candidate_identity={"status": "resolved", "sha256": "3" * 64, "uri": "candidate://tier-a"},
        teacher_identity=_resolved_file(teacher),
        basis_sha256="4" * 64,
    )
    calibration = {
        "schema": "banana-smasher-anchor-calibration-v1",
        "correction_factors": {"alpha": 2.0, "beta": 0.5},
        "parent_class_counts": {"alpha": 3, "beta": 1},
        "source": {"status": "resolved", "sha256": "5" * 64, "uri": "calibration://fixture"},
    }

    aggregate = aggregate_scores(
        manifest,
        raw,
        tmp_path / "aggregate.json",
        candidate_id="tier-a",
        calibration=calibration,
    )

    assert aggregate["measured"]["label"] == "measured_on_bank"
    assert aggregate["measured"]["global_mean_kld"] == pytest.approx(
        sum(json.loads(line)["kld"] for line in raw.read_text().splitlines()) / 4
    )
    assert aggregate["parent_estimates"]["label"] == "estimated_parent_not_measured"
    expected = (
        3 * aggregate["parent_estimates"]["per_class_mean_kld"]["alpha"]
        + aggregate["parent_estimates"]["per_class_mean_kld"]["beta"]
    ) / 4
    assert aggregate["parent_estimates"]["global_mean_kld"] == pytest.approx(expected)


def test_compare_returns_explicit_retain_or_escalate_policy() -> None:
    panel = {
        "schema": "banana-smasher-anchor-aggregate-v1",
        "bank_id": "panel",
        "bank_role": "train_balanced64",
        "candidate_id": "tier-a",
        "evaluation_contract": {"dimensions": [1, 2], "tier_menus": {"uniform": ["tier-a"]}, "basis_sha256": "4" * 64, "candidate_sha256": "3" * 64, "teacher_sha256": "5" * 64, "scorer_sha256": "2" * 64},
        "measured": {"global_mean_kld": 1.0, "per_class_mean_kld": {"alpha": 1.0, "beta": 2.0}},
    }
    parent = {
        "schema": "banana-smasher-anchor-aggregate-v1",
        "bank_id": "parent",
        "bank_role": "train512",
        "candidate_id": "tier-a",
        "evaluation_contract": {"dimensions": [1, 2], "tier_menus": {"uniform": ["tier-a"]}, "basis_sha256": "4" * 64, "candidate_sha256": "3" * 64, "teacher_sha256": "5" * 64, "scorer_sha256": "2" * 64},
        "measured": {"global_mean_kld": 1.02, "per_class_mean_kld": {"alpha": 1.01, "beta": 2.2}},
    }

    strict = compare_training_rails(
        panel,
        parent,
        {"max_abs_global_relative_pct": 3.0, "max_abs_class_relative_pct": 5.0},
    )
    loose = compare_training_rails(
        panel,
        parent,
        {"max_abs_global_relative_pct": 3.0, "max_abs_class_relative_pct": 10.0},
    )

    assert strict["policy_result"] == "escalate_to_full_parent"
    assert loose["policy_result"] == "retain_panel"
    assert strict["per_class"]["beta"]["relative_error_pct"] == pytest.approx(-100 / 11)

    incompatible = json.loads(json.dumps(parent))
    incompatible["evaluation_contract"]["tier_menus"] = {"uniform": ["other-tier"]}
    with pytest.raises(AnchorEvaluationError, match="same dimensions and tier menus"):
        compare_training_rails(panel, incompatible, {"max_abs_global_relative_pct": 3.0, "max_abs_class_relative_pct": 10.0})


def test_solver_rows_fail_closed_for_holdout_roles(tmp_path: Path) -> None:
    _, parent = _parent(tmp_path, role="holdout512")
    aggregate = {
        "schema": "banana-smasher-anchor-aggregate-v1",
        "bank_id": "holdout512",
        "bank_role": "holdout512",
        "candidate_id": "tier-a",
        "measured": {"global_mean_kld": 1.0, "per_class_mean_kld": {"alpha": 1.0, "beta": 2.0}},
    }
    with pytest.raises(AnchorEvaluationError, match="holdout.*solver"):
        emit_solver_row(parent, aggregate, tmp_path / "solver.json")

    row = emit_solver_row(
        parent,
        aggregate,
        tmp_path / "diagnostic.json",
        diagnostic_override=True,
    )
    assert row["diagnostic_only"] is True
    assert math.isfinite(row["measured_global_mean_kld"])


def test_run_root_status_reports_every_completion_surface(tmp_path: Path) -> None:
    parent_path, manifest = _parent(tmp_path)
    run_root = tmp_path / "run"
    register_bank(run_root, manifest)
    materialize_bank(manifest, parent_path, run_root / "banks" / "train512.jsonl")
    producer = _write_jsonl(
        tmp_path / "teacher.jsonl",
        [
            {"window_id": row["id"], "logits": [2.0, 0.0]}
            for row in manifest["windows"]
        ],
    )
    imported = import_producer(
        run_root,
        manifest,
        producer,
        kind="teacher",
        expected_sha256=hashlib.sha256(producer.read_bytes()).hexdigest(),
    )

    status = status_report(run_root)
    row = status["banks"][0]
    assert imported["relative_path"] == "producers/teacher/train512.jsonl"
    assert row["bank_production"] == "6/6"
    assert row["teacher_coverage"] == "6/6"
    assert row["candidate_coverage"] == "0 candidates"
    assert row["scoring"] == "0 complete"
    assert row["aggregation"] == "0 complete"
    assert row["provenance"].startswith("UNRESOLVED")
    grid = format_status(status)
    assert "BANK" in grid
    assert "train512" in grid
    assert "TEACHER" in grid


def test_import_producer_hash_admits_and_copies_the_same_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, manifest = _parent(tmp_path)
    source = _write_jsonl(
        tmp_path / "producer.jsonl",
        [{"window_id": row["id"], "logits": [2.0, 0.0]} for row in manifest["windows"]],
    )
    admitted = source.read_bytes()
    replacement = b"".join(
        (json.dumps({"window_id": row["id"], "logits": [9.0, 0.0]}, sort_keys=True) + "\n").encode()
        for row in manifest["windows"]
    )
    original_read_bytes = Path.read_bytes
    swapped = False

    def swap_after_first_read(path: Path) -> bytes:
        nonlocal swapped
        payload = original_read_bytes(path)
        if path == source and not swapped:
            swapped = True
            path.write_bytes(replacement)
        return payload

    monkeypatch.setattr(Path, "read_bytes", swap_after_first_read)
    receipt = import_producer(
        tmp_path / "run",
        manifest,
        source,
        kind="teacher",
        expected_sha256=hashlib.sha256(admitted).hexdigest(),
    )

    destination = tmp_path / "run" / receipt["relative_path"]
    assert original_read_bytes(destination) == admitted


def test_cli_runs_minimal_manifest_to_solver_chain(tmp_path: Path, capsys) -> None:
    parent_path, manifest = _parent(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    run_root = tmp_path / "run"
    teacher = _write_jsonl(
        tmp_path / "teacher.jsonl",
        [
            {"window_id": row["id"], "logits": [[2.0, 0.0]] * 1024}
            for row in manifest["windows"]
        ],
    )
    candidate = _write_jsonl(
        tmp_path / "candidate.jsonl",
        [
            {"window_id": row["id"], "logits": [[1.0, 0.0]] * 1024}
            for row in manifest["windows"]
        ],
    )
    teacher_sha = hashlib.sha256(teacher.read_bytes()).hexdigest()
    candidate_producer_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()

    commands = [
        ["anchor", "register", "--run-root", str(run_root), "--manifest", str(manifest_path)],
        ["anchor", "materialize", "--run-root", str(run_root), "--bank", "train512", "--parent", str(parent_path)],
        ["anchor", "import-producer", "--run-root", str(run_root), "--bank", "train512", "--kind", "teacher", "--source", str(teacher), "--sha256", teacher_sha],
        ["anchor", "import-producer", "--run-root", str(run_root), "--bank", "train512", "--kind", "candidate", "--candidate-id", "tier-a", "--source", str(candidate), "--sha256", candidate_producer_sha],
        ["anchor", "score", "--run-root", str(run_root), "--bank", "train512", "--candidate-id", "tier-a", "--teacher-sha256", teacher_sha, "--teacher-uri", "producer://teacher", "--candidate-sha256", "3" * 64, "--candidate-uri", "candidate://tier-a", "--basis-sha256", "4" * 64],
        ["anchor", "aggregate", "--run-root", str(run_root), "--bank", "train512", "--candidate-id", "tier-a"],
        ["anchor", "solver-row", "--run-root", str(run_root), "--bank", "train512", "--candidate-id", "tier-a"],
        ["anchor", "status", "--run-root", str(run_root), "--format", "json"],
    ]
    outputs = []
    for command in commands:
        assert main(command) == 0
        outputs.append(json.loads(capsys.readouterr().out))

    assert outputs[4]["coverage"] == "6/6"
    raw_rows = [
        json.loads(line)
        for line in (run_root / "scores" / "tier-a" / "train512" / "raw.jsonl")
        .read_text()
        .splitlines()
    ]
    assert {row["position_count"] for row in raw_rows} == {1024}
    assert all(0 < row["kld"] < 1 for row in raw_rows)
    assert outputs[5]["measured"]["label"] == "measured_on_bank"
    assert outputs[6]["diagnostic_only"] is False
    assert outputs[7]["banks"][0]["scoring"] == "1 complete"
    assert outputs[7]["banks"][0]["aggregation"] == "1 complete"


def test_cli_rejects_path_unsafe_anchor_identifiers(tmp_path: Path, capsys) -> None:
    _, manifest = _parent(tmp_path)
    run_root = tmp_path / "run"
    register_bank(run_root, manifest)

    assert main(
        [
            "anchor",
            "score",
            "--run-root",
            str(run_root),
            "--bank",
            "train512",
            "--candidate-id",
            "../../../escape",
            "--teacher-sha256",
            "5" * 64,
            "--teacher-uri",
            "teacher://fixture",
            "--candidate-sha256",
            "3" * 64,
            "--candidate-uri",
            "candidate://fixture",
            "--basis-sha256",
            "4" * 64,
        ]
    ) == 2
    error = json.loads(capsys.readouterr().err)
    assert "path-safe identifier" in error["error"]
    assert not (tmp_path / "escape").exists()


def test_public_four_bank_bundle_is_valid_and_semantically_separated() -> None:
    root = Path(__file__).parents[1] / "anchor_banks"
    manifests = {
        path.stem: json.loads(path.read_text())
        for path in sorted(root.glob("*.bank.json"))
    }
    bundle = json.loads((root / "FOUR_BANK_PROVENANCE.json").read_text())

    assert {manifest["role"] for manifest in manifests.values()} == {
        "train_balanced64",
        "train512",
        "holdout_balanced64",
        "holdout512",
    }
    for manifest in manifests.values():
        assert validate_bank_manifest(manifest)["status"] == "PASS"
    train_parent = next(value for value in manifests.values() if value["role"] == "train512")
    assert train_parent["class_counts"] == {
        "agentic": 154,
        "chat": 52,
        "code": 76,
        "multilingual": 76,
        "prose": 78,
        "reasoning": 76,
    }
    train_panel = next(value for value in manifests.values() if value["role"] == "train_balanced64")
    quick_holdout = next(value for value in manifests.values() if value["role"] == "holdout_balanced64")
    assert set(row["id"] for row in train_panel["windows"]).isdisjoint(
        row["id"] for row in quick_holdout["windows"]
    )
    assert bundle["usage_policy"]["holdout_solver_default"] == "REJECT"
    assert "19/7/11/7/9/11" not in json.dumps(bundle)
