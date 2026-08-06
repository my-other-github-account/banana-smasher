from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from decimal import Decimal, ROUND_HALF_UP, localcontext
from typing import Any

BPW_ACCOUNTING_SCHEMA = "banana-smasher.bpw-accounting.v1"
WIRE_SCOPE = "whole_shipped_model_weights"
BASE_PARAMETER_SCOPE = "canonical_base_model_logical_parameters"
AUXILIARY_PARAMETER_SCOPE = "auxiliary_model_logical_parameters"
COMPARISON_BPW_SCOPE = (
    "whole_shipped_model_weights/canonical_base_model_logical_parameters"
)
INCLUDING_AUXILIARY_BPW_SCOPE = (
    "whole_shipped_model_weights/all_shipped_model_logical_parameters"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class BpwAccountingError(ValueError):
    """Raised when BPW inputs or comparison bases are inconsistent."""


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BpwAccountingError(f"{label} must be a positive integer")
    return value


def _ratio(numerator: int, denominator: int) -> str:
    with localcontext() as context:
        context.prec = 100
        return format(Decimal(numerator) / Decimal(denominator), "f")


def _publication_bpw(value: str, decimal_places: int) -> str:
    quantum = Decimal(1).scaleb(-decimal_places)
    return format(Decimal(value).quantize(quantum, rounding=ROUND_HALF_UP), "f")


def build_bpw_accounting(
    *,
    weight_bytes: int,
    base_model_parameters: int,
    base_parameter_inventory_sha256: str,
    auxiliary_model_parameters: Mapping[str, int] | None = None,
    publication_decimal_places: int = 1,
) -> dict[str, Any]:
    """Build canonical, JSON-safe whole-model BPW accounting.

    ``comparison`` always divides the complete shipped model-weight bytes by the
    canonical logical parameter count of the base model.  This is the only BPW
    used for public comparison tables and model labels.  Auxiliary model
    parameters are reported separately and affect only ``including_auxiliary``.

    The base parameter count must come from a canonical logical tensor
    inventory.  Packed-container element counts such as Hugging Face
    ``safetensors.total`` are storage metadata and are not valid substitutes.
    """

    weight_bytes = _positive_integer(weight_bytes, "weight_bytes")
    base_model_parameters = _positive_integer(
        base_model_parameters, "base_model_parameters"
    )
    if not isinstance(base_parameter_inventory_sha256, str) or not _SHA256_RE.fullmatch(
        base_parameter_inventory_sha256
    ):
        raise BpwAccountingError(
            "base_parameter_inventory_sha256 must be a lowercase SHA-256 digest"
        )
    if (
        isinstance(publication_decimal_places, bool)
        or not isinstance(publication_decimal_places, int)
        or not 0 <= publication_decimal_places <= 6
    ):
        raise BpwAccountingError(
            "publication_decimal_places must be an integer from 0 through 6"
        )

    auxiliaries: dict[str, dict[str, Any]] = {}
    for name, raw_count in sorted((auxiliary_model_parameters or {}).items()):
        if not isinstance(name, str) or not name.strip():
            raise BpwAccountingError("auxiliary model names must be non-empty strings")
        count = _positive_integer(raw_count, f"auxiliary_model_parameters[{name!r}]")
        auxiliaries[name] = {
            "scope": AUXILIARY_PARAMETER_SCOPE,
            "logical_parameters": count,
        }

    all_parameters = base_model_parameters + sum(
        row["logical_parameters"] for row in auxiliaries.values()
    )
    comparison = _ratio(weight_bytes * 8, base_model_parameters)
    including_auxiliary = _ratio(weight_bytes * 8, all_parameters)
    publication = _publication_bpw(comparison, publication_decimal_places)

    return {
        "schema": BPW_ACCOUNTING_SCHEMA,
        "wire": {
            "scope": WIRE_SCOPE,
            "bytes": weight_bytes,
            "decimal_gb": format(Decimal(weight_bytes) / Decimal(1_000_000_000), "f"),
        },
        "parameters": {
            "base_model": {
                "scope": BASE_PARAMETER_SCOPE,
                "logical_parameters": base_model_parameters,
                "inventory_sha256": base_parameter_inventory_sha256,
            },
            "auxiliary_models": auxiliaries,
            "all_shipped_model_logical_parameters": all_parameters,
        },
        "bpw": {
            "comparison": comparison,
            "comparison_scope": COMPARISON_BPW_SCOPE,
            "including_auxiliary": including_auxiliary,
            "including_auxiliary_scope": INCLUDING_AUXILIARY_BPW_SCOPE,
        },
        "publication": {
            "source": "comparison",
            "decimal_places": publication_decimal_places,
            "bpw": publication,
            "label": f"{publication}bpw",
        },
    }


def verify_bpw_accounting(accounting: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute a BPW document and reject altered or ambiguous fields."""

    try:
        wire = accounting["wire"]
        parameters = accounting["parameters"]
        base = parameters["base_model"]
        auxiliary_rows = parameters["auxiliary_models"]
        publication = accounting["publication"]
        auxiliary = {
            name: row["logical_parameters"] for name, row in auxiliary_rows.items()
        }
        expected = build_bpw_accounting(
            weight_bytes=wire["bytes"],
            base_model_parameters=base["logical_parameters"],
            base_parameter_inventory_sha256=base["inventory_sha256"],
            auxiliary_model_parameters=auxiliary,
            publication_decimal_places=publication["decimal_places"],
        )
    except (KeyError, TypeError, AttributeError) as exc:
        raise BpwAccountingError("malformed BPW accounting document") from exc
    if dict(accounting) != expected:
        raise BpwAccountingError("BPW accounting document does not match canonical arithmetic")
    return expected


def require_comparable_bpw(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Require rows to share one canonical base-model comparison denominator."""

    verified = [verify_bpw_accounting(row) for row in rows]
    if not verified:
        raise BpwAccountingError("at least one BPW accounting row is required")
    first_base = verified[0]["parameters"]["base_model"]
    for row in verified[1:]:
        base = row["parameters"]["base_model"]
        if base["inventory_sha256"] != first_base["inventory_sha256"]:
            raise BpwAccountingError("base-model parameter inventory mismatch")
        if base["logical_parameters"] != first_base["logical_parameters"]:
            raise BpwAccountingError("base-model parameter count mismatch")
    return {
        "wire_scope": WIRE_SCOPE,
        "bpw_scope": COMPARISON_BPW_SCOPE,
        "base_model_parameters": first_base["logical_parameters"],
        "base_parameter_inventory_sha256": first_base["inventory_sha256"],
    }
