from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .contract import (
    PackValidationError,
    export_pack,
    refresh_serving_metadata,
    verify_pack,
    verify_serve_compatibility,
)
from .repack import repack_to_safetensors
from .repair import load_repair_bundle
from .validation import ValidationError, validate_artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smash",
        description="Fail-closed bs-pack lifecycle and exact Backpack tooling.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export", help="export quantizer output to bs-pack")
    export.add_argument("--source-root", type=Path, required=True)
    export.add_argument(
        "--runtime-floor-bytes",
        type=int,
        help="required for p1016: documented runtime residency added to the memory preflight",
    )
    export.add_argument(
        "--serving-model-root",
        type=Path,
        help="base-model directory providing full config and tokenizer metadata",
    )
    export.add_argument(
        "--refresh-metadata",
        action="store_true",
        help="refresh serving config/tokenizer metadata in an existing pack without tensors",
    )
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--model-id", required=True)
    export.add_argument("--instance-id", required=True)
    export.add_argument(
        "--link-mode", choices=("hardlink", "copy", "auto"), default="hardlink"
    )
    export.add_argument(
        "--safetensors",
        action="store_true",
        help="also repack all planes into bs-pack.safetensors and verify payload identity",
    )
    export.add_argument(
        "--drop-planes",
        action="store_true",
        help="drop .npy planes only after a verified safetensors repack",
    )
    export.add_argument(
        "--repair-checkpoint",
        type=Path,
        help="weights-only bs-basic-repair-v1 checkpoint to materialize",
    )
    export.add_argument("--repair-checkpoint-sha256")
    export.add_argument("--active-overlay", type=Path)
    export.add_argument("--active-overlay-sha256")
    export.add_argument("--assignment", type=Path)
    export.add_argument("--assignment-sha256")
    export.add_argument("--repair-update", type=int)

    verify = subparsers.add_parser("verify", help="verify manifest, schema, and bytes")
    verify.add_argument("pack", type=Path)

    serve = subparsers.add_parser(
        "serve-check", help="verify pack/kernel-cache compatibility before vllm serve"
    )
    serve.add_argument("pack", type=Path)
    serve.add_argument("--kernel-cache", type=Path, required=True)
    serve.add_argument("--architecture", required=True)

    validate = subparsers.add_parser(
        "validate", help="run a banked-teacher KLD validation ceremony"
    )
    validate.add_argument("artifact", type=Path)
    validate.add_argument("--bank", required=True)
    validate.add_argument("--check-exposure", action="store_true")
    validate.add_argument("--receipt", type=Path)
    validate.add_argument("--bank-teacher-logits", type=Path)

    knapsack = subparsers.add_parser(
        "knapsack",
        help="solve a manifest-bound tier menu under an exact integer byte envelope",
    )
    knapsack.add_argument("--run-root", type=Path, required=True)
    knapsack.add_argument("--envelope-bytes", type=int, required=True)
    knapsack.add_argument("--output", type=Path)
    knapsack.add_argument("--receipt", type=Path)

    backpack_dimensions = subparsers.add_parser(
        "backpack-dimensions",
        help="join explicit per-candidate dimensions without aggregate inference",
    )
    backpack_dimensions.add_argument("--ledger", type=Path, required=True)
    backpack_dimensions.add_argument("--dimensions", type=Path, required=True)
    backpack_dimensions.add_argument("--class-ceilings", type=Path, required=True)
    backpack_dimensions.add_argument("--basis-sha256", required=True)
    backpack_dimensions.add_argument("--output", type=Path, required=True)
    backpack_dimensions.add_argument("--receipt", type=Path, required=True)

    fixed_d4 = subparsers.add_parser(
        "fixed-d4", help="persist exact fixed-D4 assignments as executable wire"
    )
    fixed_d4_commands = fixed_d4.add_subparsers(
        dest="fixed_d4_command", required=True
    )
    fixed_d4_materialize = fixed_d4_commands.add_parser(
        "materialize", help="materialize one basis-bound fixed-D4 layer"
    )
    fixed_d4_materialize.add_argument("--manifest", type=Path, required=True)
    fixed_d4_materialize.add_argument("--output", type=Path, required=True)
    fixed_d4_materialize.add_argument("--basis-sha256", required=True)

    anchor = subparsers.add_parser(
        "anchor", help="reproducible four-bank anchor evaluation workflow"
    )
    anchor_commands = anchor.add_subparsers(dest="anchor_command", required=True)

    anchor_validate = anchor_commands.add_parser(
        "validate", help="validate a versioned bank manifest"
    )
    anchor_validate.add_argument("--manifest", type=Path, required=True)

    anchor_resolve = anchor_commands.add_parser(
        "resolve", help="write a new manifest with exact resolved identities"
    )
    anchor_resolve.add_argument("--manifest", type=Path, required=True)
    anchor_resolve.add_argument("--identities", type=Path, required=True)
    anchor_resolve.add_argument("--output", type=Path, required=True)

    anchor_register = anchor_commands.add_parser(
        "register", help="register an immutable bank manifest in one run root"
    )
    anchor_register.add_argument("--run-root", type=Path, required=True)
    anchor_register.add_argument("--manifest", type=Path, required=True)

    anchor_materialize = anchor_commands.add_parser(
        "materialize", help="materialize a registered bank from its declared parent"
    )
    anchor_materialize.add_argument("--run-root", type=Path, required=True)
    anchor_materialize.add_argument("--bank", required=True)
    anchor_materialize.add_argument("--parent", type=Path, required=True)
    anchor_materialize.add_argument("--disjoint-bank", action="append", default=[])

    anchor_select = anchor_commands.add_parser(
        "select", help="create and register a deterministic balanced training subset"
    )
    anchor_select.add_argument("--run-root", type=Path, required=True)
    anchor_select.add_argument("--parent-bank", required=True)
    anchor_select.add_argument("--parent", type=Path, required=True)
    anchor_select.add_argument("--config", type=Path, required=True)

    anchor_import = anchor_commands.add_parser(
        "import-producer", help="hash-admit exact teacher or candidate producer rows"
    )
    anchor_import.add_argument("--run-root", type=Path, required=True)
    anchor_import.add_argument("--bank", required=True)
    anchor_import.add_argument("--kind", choices=("teacher", "candidate"), required=True)
    anchor_import.add_argument("--source", type=Path, required=True)
    anchor_import.add_argument("--sha256", required=True)
    anchor_import.add_argument("--candidate-id")

    anchor_candidate = anchor_commands.add_parser(
        "materialize-candidate",
        help="run a model/config producer and import one exact 64-row candidate",
    )
    anchor_candidate.add_argument("--run-root", type=Path, required=True)
    anchor_candidate.add_argument("--bank", required=True)
    anchor_candidate.add_argument("--candidate-id", required=True)
    anchor_candidate.add_argument("--model", type=Path, required=True)
    anchor_candidate.add_argument("--config", type=Path, required=True)
    anchor_candidate.add_argument("--basis-sha256", required=True)

    anchor_score = anchor_commands.add_parser(
        "score", help="score exact producer rows with resumable per-window KLD"
    )
    anchor_score.add_argument("--run-root", type=Path, required=True)
    anchor_score.add_argument("--bank", required=True)
    anchor_score.add_argument("--candidate-id", required=True)
    anchor_score.add_argument("--teacher", type=Path)
    anchor_score.add_argument("--candidate-producer", type=Path)
    anchor_score.add_argument("--teacher-sha256", required=True)
    anchor_score.add_argument("--teacher-uri", required=True)
    anchor_score.add_argument("--candidate-sha256", required=True)
    anchor_score.add_argument("--candidate-uri", required=True)
    anchor_score.add_argument("--basis-sha256", required=True)

    anchor_aggregate = anchor_commands.add_parser(
        "aggregate", help="aggregate raw KLD and optionally estimate a declared parent"
    )
    anchor_aggregate.add_argument("--run-root", type=Path, required=True)
    anchor_aggregate.add_argument("--bank", required=True)
    anchor_aggregate.add_argument("--candidate-id", required=True)
    anchor_aggregate.add_argument("--calibration", type=Path)

    anchor_compare = anchor_commands.add_parser(
        "compare", help="compare train balanced panel with its full training parent"
    )
    anchor_compare.add_argument("--run-root", type=Path, required=True)
    anchor_compare.add_argument("--panel-bank", required=True)
    anchor_compare.add_argument("--parent-bank", required=True)
    anchor_compare.add_argument("--candidate-id", required=True)
    anchor_compare.add_argument("--thresholds", type=Path, required=True)
    anchor_compare.add_argument("--output", type=Path)

    anchor_solver = anchor_commands.add_parser(
        "solver-row", help="emit a solver-ready row from the training rail"
    )
    anchor_solver.add_argument("--run-root", type=Path, required=True)
    anchor_solver.add_argument("--bank", required=True)
    anchor_solver.add_argument("--candidate-id", required=True)
    anchor_solver.add_argument("--output", type=Path)
    anchor_solver.add_argument("--diagnostic-override", action="store_true")

    anchor_status = anchor_commands.add_parser(
        "status", help="report production, coverage, scoring and provenance grid"
    )
    anchor_status.add_argument("--run-root", type=Path, required=True)
    anchor_status.add_argument("--format", choices=("human", "json"), default="human")

    return parser


def _emit(value: dict[str, Any], *, stream: Any | None = None) -> None:
    if stream is None:
        stream = sys.stdout
    stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"missing JSON input {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON input {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON input {path} must contain an object")
    return value


def _run_anchor(args: argparse.Namespace) -> dict[str, Any] | str:
    from .anchor import (
        _atomic_write,
        _canonical_bytes,
        _safe_component,
        aggregate_scores,
        compare_training_rails,
        create_balanced_subset,
        emit_solver_row,
        format_status,
        import_producer,
        load_registered_bank,
        materialize_candidate_producer,
        materialize_bank,
        register_bank,
        resolve_bank_identities,
        score_bank,
        status_report,
        validate_bank_manifest,
    )

    command = args.anchor_command
    if command == "validate":
        return validate_bank_manifest(_load_json_object(args.manifest))
    if command == "resolve":
        resolved = resolve_bank_identities(
            _load_json_object(args.manifest), _load_json_object(args.identities)
        )
        _atomic_write(args.output, _canonical_bytes(resolved))
        return validate_bank_manifest(resolved)
    if command == "register":
        return register_bank(args.run_root, _load_json_object(args.manifest))
    if command == "materialize":
        manifest = load_registered_bank(args.run_root, args.bank)
        disjoint = [
            load_registered_bank(args.run_root, bank_id)
            for bank_id in args.disjoint_bank
        ]
        return materialize_bank(
            manifest,
            args.parent,
            args.run_root / "banks" / f"{manifest['bank_id']}.jsonl",
            disjoint_manifests=disjoint,
        )
    if command == "select":
        parent = load_registered_bank(args.run_root, args.parent_bank)
        manifest = create_balanced_subset(
            parent, args.parent, _load_json_object(args.config)
        )
        return {
            **register_bank(args.run_root, manifest),
            "bank_manifest": manifest,
        }
    if command == "import-producer":
        manifest = load_registered_bank(args.run_root, args.bank)
        return import_producer(
            args.run_root,
            manifest,
            args.source,
            kind=args.kind,
            expected_sha256=args.sha256,
            candidate_id=args.candidate_id,
        )
    if command == "materialize-candidate":
        manifest = load_registered_bank(args.run_root, args.bank)
        return materialize_candidate_producer(
            args.run_root,
            manifest,
            candidate_id=args.candidate_id,
            model_root=args.model,
            producer_config=args.config,
            basis_sha256=args.basis_sha256,
        )
    if command == "score":
        manifest = load_registered_bank(args.run_root, args.bank)
        candidate_id = _safe_component(args.candidate_id, "candidate_id")
        teacher_path = args.teacher or (
            args.run_root / "producers" / "teacher" / f"{manifest['bank_id']}.jsonl"
        )
        candidate_path = args.candidate_producer or (
            args.run_root
            / "producers"
            / "candidate"
            / candidate_id
            / f"{manifest['bank_id']}.jsonl"
        )
        output = (
            args.run_root
            / "scores"
            / candidate_id
            / manifest["bank_id"]
            / "raw.jsonl"
        )
        return score_bank(
            manifest,
            teacher_path,
            candidate_path,
            output,
            candidate_id=candidate_id,
            candidate_identity={
                "status": "resolved",
                "sha256": args.candidate_sha256,
                "uri": args.candidate_uri,
            },
            teacher_identity={
                "status": "resolved",
                "sha256": args.teacher_sha256,
                "uri": args.teacher_uri,
            },
            basis_sha256=args.basis_sha256,
        )
    if command == "aggregate":
        manifest = load_registered_bank(args.run_root, args.bank)
        candidate_id = _safe_component(args.candidate_id, "candidate_id")
        raw = (
            args.run_root
            / "scores"
            / candidate_id
            / manifest["bank_id"]
            / "raw.jsonl"
        )
        output = (
            args.run_root
            / "aggregates"
            / candidate_id
            / f"{manifest['bank_id']}.json"
        )
        calibration = (
            _load_json_object(args.calibration) if args.calibration is not None else None
        )
        return aggregate_scores(
            manifest,
            raw,
            output,
            candidate_id=candidate_id,
            calibration=calibration,
        )
    if command == "compare":
        candidate_id = _safe_component(args.candidate_id, "candidate_id")
        panel_bank = _safe_component(args.panel_bank, "panel_bank")
        parent_bank = _safe_component(args.parent_bank, "parent_bank")
        panel = _load_json_object(
            args.run_root
            / "aggregates"
            / candidate_id
            / f"{panel_bank}.json"
        )
        parent = _load_json_object(
            args.run_root
            / "aggregates"
            / candidate_id
            / f"{parent_bank}.json"
        )
        result = compare_training_rails(
            panel, parent, _load_json_object(args.thresholds)
        )
        output = args.output or (
            args.run_root / "comparisons" / f"{candidate_id}.json"
        )
        _atomic_write(output, _canonical_bytes(result))
        return result
    if command == "solver-row":
        manifest = load_registered_bank(args.run_root, args.bank)
        candidate_id = _safe_component(args.candidate_id, "candidate_id")
        aggregate = _load_json_object(
            args.run_root
            / "aggregates"
            / candidate_id
            / f"{manifest['bank_id']}.json"
        )
        output = args.output or (
            args.run_root
            / "solver_rows"
            / f"{candidate_id}--{manifest['bank_id']}.json"
        )
        return emit_solver_row(
            manifest,
            aggregate,
            output,
            diagnostic_override=args.diagnostic_override,
        )
    if command == "status":
        status = status_report(args.run_root)
        return format_status(status) if args.format == "human" else status
    raise ValueError(f"unsupported anchor command {command!r}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    tokens = list(sys.argv[1:] if argv is None else argv)
    reported_command = tokens[0] if tokens else None
    if reported_command == "validate-pack":
        # Compatibility spelling for reproducibility automation. Keep the five
        # primary lifecycle verbs and their help surface stable.
        tokens[0] = "verify"
    args = parser.parse_args(tokens)
    try:
        if args.command == "export":
            if args.refresh_metadata:
                if args.serving_model_root is None:
                    raise ValueError("--refresh-metadata requires --serving-model-root")
                if args.safetensors or args.drop_planes:
                    raise ValueError(
                        "--refresh-metadata cannot repack or drop tensor planes"
                    )
                result = {
                    **refresh_serving_metadata(
                        args.output,
                        serving_model_root=args.serving_model_root,
                        link_mode=args.link_mode,
                        runtime_floor_bytes=args.runtime_floor_bytes,
                    ),
                    "command": "export",
                    "output": str(args.output.resolve()),
                }
            else:
                if args.drop_planes and not args.safetensors:
                    raise ValueError("--drop-planes requires --safetensors")
                repair_values = {
                    "checkpoint": args.repair_checkpoint,
                    "checkpoint_sha256": args.repair_checkpoint_sha256,
                    "active_overlay": args.active_overlay,
                    "active_overlay_sha256": args.active_overlay_sha256,
                    "assignment": args.assignment,
                    "assignment_sha256": args.assignment_sha256,
                    "update": args.repair_update,
                }
                supplied_repair = [
                    name for name, value in repair_values.items() if value is not None
                ]
                if supplied_repair and len(supplied_repair) != len(repair_values):
                    missing = sorted(set(repair_values) - set(supplied_repair))
                    raise ValueError(
                        "repair materialization requires every bound input; "
                        f"missing={missing}"
                    )
                repair = (
                    load_repair_bundle(**repair_values) if supplied_repair else None
                )
                manifest = export_pack(
                    source_root=args.source_root,
                    output=args.output,
                    model_id=args.model_id,
                    instance_id=args.instance_id,
                    link_mode=args.link_mode,
                    repair=repair,
                    serving_model_root=args.serving_model_root,
                    runtime_floor_bytes=args.runtime_floor_bytes,
                )
                receipt = verify_pack(args.output)
                result = {
                    **receipt,
                    "command": "export",
                    "output": str(args.output.resolve()),
                    "file_count": len(manifest["files"]),
                }
                if args.safetensors:
                    result["repack"] = repack_to_safetensors(
                        args.output,
                        drop_planes=args.drop_planes,
                    )
        elif args.command == "verify":
            result = {
                **verify_pack(args.pack),
                "command": reported_command or "verify",
            }
        elif args.command == "serve-check":
            result = {
                **verify_serve_compatibility(
                    args.pack,
                    args.kernel_cache,
                    architecture=args.architecture,
                ),
                "command": "serve-check",
            }
        elif args.command == "validate":
            result = {
                **validate_artifact(
                    args.artifact,
                    bank=args.bank,
                    check_exposure=args.check_exposure,
                    receipt_path=args.receipt,
                    bank_teacher_logits=args.bank_teacher_logits,
                ),
                "command": "validate",
            }
        elif args.command == "knapsack":
            from .knapsack import run_knapsack

            result = run_knapsack(
                run_root=args.run_root,
                envelope_bytes=args.envelope_bytes,
                output=args.output,
                receipt=args.receipt,
            )
        elif args.command == "backpack-dimensions":
            from .backpack_dimensions import build_dynamic_dimensions

            result = build_dynamic_dimensions(
                ledger=args.ledger,
                dimensions=args.dimensions,
                class_ceilings=args.class_ceilings,
                basis_sha256=args.basis_sha256,
                output=args.output,
                receipt=args.receipt,
            )
        elif args.command == "fixed-d4":
            from .fixed_d4 import materialize_fixed_d4

            result = materialize_fixed_d4(
                args.manifest,
                args.output,
                basis_sha256=args.basis_sha256,
            )
        elif args.command == "anchor":
            result = _run_anchor(args)
        else:  # pragma: no cover - argparse guarantees the choices
            parser.error(f"unsupported command {args.command!r}")
            return 2
    except (
        PackValidationError,
        ValidationError,
        FileExistsError,
        OSError,
        ValueError,
    ) as exc:
        _emit(
            {
                "status": "FAIL",
                "command": reported_command or args.command,
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
            stream=sys.stderr,
        )
        return 2
    if isinstance(result, str):
        sys.stdout.write(result)
    else:
        _emit(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
