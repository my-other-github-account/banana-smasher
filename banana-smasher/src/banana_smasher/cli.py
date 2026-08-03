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
        description="Fail-closed bs-pack lifecycle and exact QTIP solve tools.",
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

    solve = subparsers.add_parser(
        "solve", help="solve a manifest-declared QTIP ring with exact search"
    )
    solve.add_argument("--source-root", type=Path, required=True)
    solve.add_argument("--root", type=Path, required=True)
    solve.add_argument("--layers", required=True)
    solve.add_argument(
        "--tier",
        required=True,
        help="use qtip with --bpw, or the qtip2/qtip3 compatibility aliases",
    )
    solve.add_argument(
        "--bpw",
        help="QTIP target in 0.25 increments; valid only with --tier qtip",
    )
    solve.add_argument(
        "--all-cells",
        action="store_true",
        required=True,
        help="solve every ordered expert/projection cell for each selected layer",
    )
    solve.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="validate and skip existing hash-bound PASS units (default)",
    )
    solve.add_argument(
        "--kernel-cache-root",
        type=Path,
        help="receipt-bound QTIP kernel cache produced by smash kernels build",
    )

    qtip_configs = subparsers.add_parser(
        "qtip-configs",
        help="materialize hash-bound local QTIP configs from an open-tier run manifest",
    )
    qtip_configs.add_argument("--manifest", type=Path, required=True)
    qtip_configs.add_argument("--tier", required=True)
    qtip_configs.add_argument("--layers", required=True)
    qtip_configs.add_argument("--output", type=Path, required=True)

    kernels = subparsers.add_parser(
        "kernels", help="manage SHA-pinned compiled kernel caches"
    )
    kernel_subparsers = kernels.add_subparsers(
        dest="kernel_command", required=True
    )
    kernel_build = kernel_subparsers.add_parser(
        "build", help="AOT-compile a packaged ring before solving"
    )
    kernel_build.add_argument("--tier", required=True)
    kernel_build.add_argument("--bpw", required=True)
    kernel_build.add_argument("--cache-root", type=Path)

    return parser


def _emit(value: dict[str, Any], *, stream: Any | None = None) -> None:
    if stream is None:
        stream = sys.stdout
    stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _parse_layers(value: str) -> list[int]:
    """Parse comma-separated layers and inclusive ranges without campaign defaults."""
    result: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            raise ValueError("empty layer selector")
        if "-" in token:
            lower_text, upper_text = token.split("-", 1)
            lower, upper = int(lower_text), int(upper_text)
            if lower < 0 or upper < lower:
                raise ValueError(f"invalid layer range {token!r}")
            result.extend(range(lower, upper + 1))
        else:
            layer = int(token)
            if layer < 0:
                raise ValueError(f"invalid layer {layer}")
            result.append(layer)
    if len(set(result)) != len(result):
        raise ValueError(f"duplicate layer selection: {value!r}")
    return result


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
        elif args.command == "qtip-configs":
            from .qtip_materialize import materialize_qtip_configs

            result = materialize_qtip_configs(
                args.manifest,
                tier=args.tier,
                layers=_parse_layers(args.layers),
                output_root=args.output,
            )
        elif args.command == "kernels":
            if args.kernel_command != "build":
                raise ValueError(f"unsupported kernels command: {args.kernel_command}")
            if args.tier != "qtip":
                raise ValueError("kernel builds use the unified --tier qtip surface")
            from .qtip_kernel_cache import build_qtip_kernels

            result = build_qtip_kernels(args.bpw, cache_root=args.cache_root)
        elif args.command == "solve":
            compatibility_bpw = {"qtip2": "2.00", "qtip3": "3.00"}.get(
                args.tier
            )
            if compatibility_bpw is not None:
                if args.bpw is not None:
                    raise ValueError(
                        f"compatibility alias --tier {args.tier} cannot be combined with --bpw"
                    )
                selected_bpw = compatibility_bpw
            elif args.tier == "qtip":
                if args.bpw is None:
                    raise ValueError("--tier qtip requires --bpw")
                selected_bpw = args.bpw
            else:
                raise ValueError(
                    "exact QTIP solve requires --tier qtip --bpw, qtip2, or qtip3"
                )

            from .qtip_materialize import (
                ensure_qtip_configs,
                require_qtip_ring_manifest,
            )
            from .qtip_rings import canonical_qtip_tier
            from .solver_qtip_profile import main_many as qtip_profile_main_many

            selected_tier = canonical_qtip_tier(selected_bpw)
            selected_layers = _parse_layers(args.layers)
            materialization = ensure_qtip_configs(
                args.source_root,
                tier=selected_tier,
                layers=selected_layers,
            )
            selected_tier = require_qtip_ring_manifest(
                args.source_root, selected_bpw
            )
            layer_receipts = [
                qtip_profile_main_many(
                    args.source_root,
                    args.root,
                    layer,
                    tier=selected_tier,
                    all_cells=True,
                    profile_mode=False,
                    **(
                        {"resume": args.resume, "resume_flag_explicit": True}
                        if "--resume" in tokens
                        else {}
                    ),
                    **(
                        {"kernel_cache_root": args.kernel_cache_root}
                        if args.kernel_cache_root is not None
                        else {}
                    ),
                )
                for layer in selected_layers
            ]
            result = {
                "schema": "banana-smasher-qtip-all-cells-solve-v1",
                "status": "PASS",
                "command": "solve",
                "tier": selected_tier,
                "bpw": selected_bpw,
                "layers": [row["layer"] for row in layer_receipts],
                "layer_receipts": layer_receipts,
                "config_materialization": materialization,
            }
        else:  # pragma: no cover - argparse guarantees the choices
            parser.error(f"unsupported command {args.command!r}")
            return 2
    except (
        PackValidationError,
        ValidationError,
        FileExistsError,
        OSError,
        RuntimeError,
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
    _emit(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
