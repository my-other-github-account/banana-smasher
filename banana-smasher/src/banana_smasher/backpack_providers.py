"""Public Backpack family providers and receipt-backed wire pricing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

ProviderCallable = Callable[..., Any]


@dataclass(frozen=True)
class BackpackWireArtifact:
    """One physical shared-family artifact bound by identity and hash."""

    key: str
    bytes: int
    path: str | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class BackpackFamilyActivation:
    """A shared wire cost charged once when any dependent option is selected."""

    key: str
    bytes: int
    artifacts: tuple[BackpackWireArtifact, ...]


@dataclass(frozen=True)
class BackpackWirePrice:
    """Exact per-cell payload plus deduplicable shared activation costs."""

    cell_payload_bytes: int
    activations: tuple[BackpackFamilyActivation, ...]
    receipt: str | None = None
    receipt_sha256: str | None = None

    @property
    def activation_bytes(self) -> int:
        return sum(activation.bytes for activation in self.activations)

    @property
    def full_wire_bytes(self) -> int:
        return self.cell_payload_bytes + self.activation_bytes

    @property
    def activation_artifacts(self) -> tuple[dict[str, Any], ...]:
        """Return the solver-facing shared-artifact declarations."""

        return tuple(
            {
                "id": activation.key,
                "bytes": activation.bytes,
                **(
                    {"path": activation.artifacts[0].path}
                    if activation.artifacts and activation.artifacts[0].path is not None
                    else {}
                ),
                **(
                    {"sha256": activation.artifacts[0].sha256}
                    if activation.artifacts and activation.artifacts[0].sha256 is not None
                    else {}
                ),
            }
            for activation in self.activations
        )


@dataclass(frozen=True)
class BackpackFamilyProvider:
    """Immutable public bindings for one selectable Backpack family."""

    provider_id: str
    kind: str
    runtime_family: str
    generate: ProviderCallable
    materialize: ProviderCallable
    price: Callable[[Mapping[str, Any] | str | Path], BackpackWirePrice]
    predict: ProviderCallable
    verify: ProviderCallable
    rate_num: int | None = None
    rate_den: int | None = None
    transition_bits: int | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_price_receipt(
    value: Mapping[str, Any] | str | Path,
) -> tuple[Mapping[str, Any], str | None, str | None]:
    if isinstance(value, Mapping):
        return value, None, None
    path = Path(value).expanduser().resolve()
    payload = json.loads(path.read_text())
    if not isinstance(payload, Mapping):
        raise ValueError("candidate price receipt must contain a JSON object")
    return payload, str(path), _sha256_file(path)


def price_backpack_candidate(
    receipt: Mapping[str, Any] | str | Path,
) -> BackpackWirePrice:
    """Price actual candidate bytes and each declared shared artifact once."""

    payload, receipt_path, receipt_sha256 = _load_price_receipt(receipt)
    cell_bytes = payload.get("cell_payload_bytes", payload.get("physical_bytes"))
    if isinstance(cell_bytes, bool) or not isinstance(cell_bytes, int) or cell_bytes < 0:
        raise ValueError("candidate receipt requires non-negative cell_payload_bytes")
    raw_artifacts = payload.get("activation_artifacts", ())
    if not isinstance(raw_artifacts, (list, tuple)):
        raise ValueError("activation_artifacts must be an array")
    activations: list[BackpackFamilyActivation] = []
    seen: set[str] = set()
    for index, row in enumerate(raw_artifacts):
        if not isinstance(row, Mapping):
            raise ValueError(f"activation_artifacts[{index}] must be an object")
        key = row.get("id", row.get("key"))
        size = row.get("bytes")
        if not isinstance(key, str) or not key:
            raise ValueError(f"activation_artifacts[{index}].id must be non-empty")
        if key in seen:
            raise ValueError(f"duplicate activation artifact {key!r}")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(
                f"activation_artifacts[{index}].bytes must be a non-negative integer"
            )
        artifact_path = row.get("path")
        artifact_sha = row.get("sha256")
        if artifact_path is not None:
            if not isinstance(artifact_path, str) or not artifact_path:
                raise ValueError(f"activation_artifacts[{index}].path must be non-empty")
            path = Path(artifact_path).expanduser().resolve()
            if path.stat().st_size != size:
                raise ValueError(f"activation artifact {key!r} byte count mismatch")
            actual_sha = _sha256_file(path)
            if artifact_sha != actual_sha:
                raise ValueError(f"activation artifact {key!r} SHA-256 mismatch")
        elif artifact_sha is not None and not isinstance(artifact_sha, str):
            raise ValueError(f"activation_artifacts[{index}].sha256 must be a string")
        artifact = BackpackWireArtifact(
            key=key,
            bytes=size,
            path=artifact_path,
            sha256=artifact_sha,
        )
        activations.append(
            BackpackFamilyActivation(key=key, bytes=size, artifacts=(artifact,))
        )
        seen.add(key)
    return BackpackWirePrice(
        cell_payload_bytes=cell_bytes,
        activations=tuple(activations),
        receipt=receipt_path,
        receipt_sha256=receipt_sha256,
    )


def bind_native_mxfp4_backpack_candidate(
    _run_root: object, *, tier: Mapping[str, Any], cell: Mapping[str, Any], **_: Any
) -> dict[str, Any]:
    """Declare an unchanged source cell with zero incremental alternate payload."""

    return {
        "schema": "banana-smasher-backpack-candidate-cell-v1",
        "status": "PASS",
        "tier": tier["id"],
        "family": "native_mxfp4",
        "cell_id": cell["cell_id"],
        "cell_payload_bytes": 0,
        "physical_bytes": 0,
        "activation_artifacts": [],
        "no_swap": True,
    }


def _materialize_record_payload(
    payloads: dict[tuple[int, str], dict[str, list[np.ndarray]]],
    *,
    family: str,
    cell: Mapping[str, Any],
    artifact_root: str | Path,
) -> None:
    """Append one public provider candidate to the canonical layer payloads."""

    root = Path(artifact_root)
    if family == "native_mxfp4":
        field_names = ("packed", "scales", "expert_ids", "tensor_offsets")
        byte_fields = ("packed", "scales")
        source_names = {
            "packed": "wire.bin",
            "scales": "scales.npy",
            "expert_ids": "expert_ids.npy",
            "tensor_offsets": "tensor_offsets.npy",
        }
    elif family == "qtip_native_v4":
        field_names = (
            "codes",
            "SU",
            "SV",
            "Wscale",
            "expert_ids",
            "record_tiers",
            "record_geometry",
            "record_projections",
            "record_boundaries",
        )
        byte_fields = ("codes", "SU", "SV", "Wscale")
        source_names = {
            "codes": "wire.bin",
            **{
                name: f"{name}.npy"
                for name in field_names
                if name not in {"codes", "record_boundaries"}
            },
        }
    else:
        field_names = (
            "codes",
            "codebooks",
            "scales",
            "expert_ids",
            "tensor_offsets",
            "record_tiers",
            "record_geometry",
            "record_projections",
            "record_boundaries",
        )
        byte_fields = ("codes", "scales", "codebooks")
        source_names = {
            "codes": "wire.bin",
            **{name: f"{name}.npy" for name in field_names if name != "codes"},
        }
    bucket = payloads.setdefault(
        (int(cell["layer"]), family), {name: [] for name in field_names}
    )
    prior_bytes = np.asarray(
        [sum(array.nbytes for array in bucket[name]) for name in byte_fields],
        dtype=np.int64,
    )
    prior_records = sum(array.size for array in bucket["expert_ids"])
    if family != "native_mxfp4" and bucket["expert_ids"]:
        bucket["record_boundaries"].append(
            np.full((1, 3), prior_records, dtype=np.int64)
        )
    for name in field_names:
        if name in {"tensor_offsets", "record_boundaries"}:
            continue
        source = root / source_names[name]
        value = (
            np.frombuffer(source.read_bytes(), dtype=np.uint8)
            if source.name == "wire.bin"
            else np.asarray(np.load(source, allow_pickle=False))
        )
        if name in {"record_geometry", "record_tiers", "record_projections"}:
            value = value.reshape(value.shape[0], -1)
        elif name != "codebooks":
            value = value.reshape(-1)
        bucket[name].append(value)
    if family != "qtip_native_v4":
        offsets = np.asarray(
            np.load(root / "tensor_offsets.npy", allow_pickle=False), dtype=np.int64
        ).reshape(-1, len(byte_fields))
        adjusted = offsets + prior_bytes
        bucket["tensor_offsets"].append(
            adjusted if not bucket["tensor_offsets"] else adjusted[1:]
        )


def _materialize_provider_assignment(
    payloads: dict[tuple[int, str], dict[str, list[np.ndarray]]],
    *,
    tier: Mapping[str, Any],
    cell: Mapping[str, Any],
    artifact_root: str | Path,
) -> None:
    """Materialize one solved provider assignment into canonical payloads."""

    provider = backpack_provider_from_declaration(tier)
    if provider.kind == "native_mxfp4":
        family = "native_mxfp4"
    elif provider.kind == "fixed_d4":
        family = "truevq_d4"
    elif provider.kind == "vector_vq":
        family = f"truevq_d{int(tier['dimension'])}"
    else:
        family = provider.runtime_family
    _materialize_record_payload(
        payloads, family=family, cell=cell, artifact_root=artifact_root
    )


def materialize_backpack_assignment(
    source: str | Path,
    *,
    plan: Any,
    cells: Sequence[Mapping[str, Any]],
    assignment: Sequence[Mapping[str, Any]],
    artifact_roots: Mapping[str, Path],
) -> None:
    """Materialize a solved assignment through the canonical public writer."""

    from .backpack import materialize_backpack_source

    materialize_backpack_source(
        Path(source),
        plan=plan,
        cells=cells,
        assignment=assignment,
        artifact_roots=artifact_roots,
    )


def predict_backpack_candidate(
    features: Any,
    classes: Any,
    teacher_weights: Any,
    candidate_weights: Any,
) -> dict[str, Any]:
    """Run the same six-class candidate prediction used by a Backpack plan."""

    from .backpack import _anchor_metrics

    return _anchor_metrics(features, classes, teacher_weights, candidate_weights)


def verify_backpack_candidate(
    receipt: object,
    *,
    tier: Mapping[str, Any],
    cell: Mapping[str, Any],
    geometry_by_identity: Mapping[tuple[int, int, str], tuple[int, int, int]] | None = None,
) -> bool:
    """Verify one candidate with the canonical resume/status validator."""

    receipt_value: object = receipt
    if isinstance(receipt, Mapping):
        receipt_value = receipt.get("receipt")
    if not isinstance(receipt_value, (str, Path)):
        return False
    from .backpack import _validate_candidate_receipt

    return _validate_candidate_receipt(
        str(receipt_value),
        tier=tier,
        cell=cell,
        geometry_by_identity=geometry_by_identity,
    )


def generate_backpack_candidate(
    run_root: str | Path,
    *,
    tier: Mapping[str, Any],
    cell: Mapping[str, Any],
    geometry_by_identity: Mapping[tuple[int, int, str], tuple[int, int, int]] | None = None,
) -> dict[str, Any]:
    """Generate one declared tier through the shared public candidate seam."""

    provider = backpack_provider_from_declaration(tier)
    if provider.kind == "qtip_ring":
        if geometry_by_identity is None:
            raise ValueError("QTIP candidate generation requires exact ring geometry")
        return provider.generate(
            run_root,
            tier=tier,
            cell=cell,
            geometry_by_identity=geometry_by_identity,
        )
    return provider.generate(run_root, tier=tier, cell=cell)


def native_mxfp4_backpack_provider() -> BackpackFamilyProvider:
    from .backpack import generate_native_mxfp4_backpack_candidate

    return BackpackFamilyProvider(
        provider_id="native-mxfp4",
        kind="native_mxfp4",
        runtime_family="native_mxfp4",
        generate=generate_native_mxfp4_backpack_candidate,
        materialize=_materialize_provider_assignment,
        price=price_backpack_candidate,
        predict=predict_backpack_candidate,
        verify=verify_backpack_candidate,
    )


def qtip_ring_backpack_provider(bpw: object) -> BackpackFamilyProvider:
    from .backpack import generate_qtip_backpack_candidate
    from .qtip_rings import resolve_qtip_ring

    ring = resolve_qtip_ring(bpw)
    runtime_family = "qtip2" if float(ring.canonical_bpw) < 3.0 else "qtip3"
    return BackpackFamilyProvider(
        provider_id=ring.tier,
        kind="qtip_ring",
        runtime_family=runtime_family,
        generate=generate_qtip_backpack_candidate,
        materialize=_materialize_provider_assignment,
        price=price_backpack_candidate,
        predict=predict_backpack_candidate,
        verify=verify_backpack_candidate,
    )


def qtip_native_v4_backpack_provider(bpw: object) -> BackpackFamilyProvider:
    """Return one homogeneous native-V4 provider for an exact quarter rate."""

    from .backpack import generate_native_v4_backpack_candidate
    from .qtip25_native_v4 import native_v4_geometry

    geometry = native_v4_geometry(bpw)
    canonical_bpw = geometry.rate_num / geometry.rate_den
    return BackpackFamilyProvider(
        provider_id=f"qtip-native-v4@{canonical_bpw:.2f}",
        kind="qtip_native_v4",
        runtime_family="qtip_native_v4",
        generate=generate_native_v4_backpack_candidate,
        materialize=_materialize_provider_assignment,
        price=price_backpack_candidate,
        predict=predict_backpack_candidate,
        verify=verify_backpack_candidate,
        rate_num=geometry.rate_num,
        rate_den=geometry.rate_den,
        transition_bits=geometry.B,
    )


def periodic_qtip3_backpack_provider(
    provider_id: str = "periodic-qtip3@3.00",
) -> BackpackFamilyProvider:
    """Return the homogeneous fixed-assignment Periodic QTIP3 provider."""

    if provider_id not in {"periodic-qtip3@3.00", "qtip-native-v6@3.00"}:
        raise ValueError(f"unsupported Periodic QTIP3 identity {provider_id!r}")
    return BackpackFamilyProvider(
        provider_id=provider_id,
        kind="fixed_qtip",
        runtime_family="periodic_qtip3",
        generate=generate_backpack_candidate,
        materialize=_materialize_provider_assignment,
        price=price_backpack_candidate,
        predict=predict_backpack_candidate,
        verify=verify_backpack_candidate,
        rate_num=3,
        rate_den=1,
        transition_bits=12,
    )


def vector_vq_backpack_provider(
    *, dimension: int, codebook_size: int
) -> BackpackFamilyProvider:
    """Return the public fixture/small-model D4 or D8 vector-VQ provider."""

    if dimension not in {4, 8}:
        raise ValueError("vector-VQ provider dimension must be 4 or 8")
    if (
        isinstance(codebook_size, bool)
        or not isinstance(codebook_size, int)
        or codebook_size < 2
        or codebook_size > 65536
        or codebook_size & (codebook_size - 1)
    ):
        raise ValueError("vector-VQ provider codebook_size must be a power of two")
    from .backpack import generate_vector_vq_backpack_candidate

    return BackpackFamilyProvider(
        provider_id=f"d{dimension}-k{codebook_size}",
        kind="vector_vq",
        runtime_family=f"truevq_d{dimension}",
        generate=generate_vector_vq_backpack_candidate,
        materialize=_materialize_provider_assignment,
        price=price_backpack_candidate,
        predict=predict_backpack_candidate,
        verify=verify_backpack_candidate,
    )


def _fixed_d4_candidate_tier(tier: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a fixed-D4 declaration for the shared vector candidate path."""

    codebook_size = tier.get("codebook_size")
    if codebook_size not in {2048, 4096}:
        provider = str(tier.get("provider", ""))
        if "2048" in provider:
            codebook_size = 2048
        elif "4096" in provider:
            codebook_size = 4096
        else:
            raise ValueError("fixed D4 candidate requires K2048 or K4096")
    normalized = {
        **tier,
        "provider": f"d4-k{codebook_size}",
        "family": "vector_vq",
        "dimension": 4,
        "codebook_size": codebook_size,
    }
    normalized.pop("bits", None)
    normalized.pop("bpw", None)
    return normalized


def _fixed_d4_generate(*args: Any, **kwargs: Any) -> Any:
    """Dispatch plan candidates or the full-model fixed-D4 source adapter."""

    if "tier" in kwargs and "cell" in kwargs:
        from .backpack import generate_fixed_d4_backpack_candidate

        return generate_fixed_d4_backpack_candidate(
            *args,
            **{**kwargs, "tier": _fixed_d4_candidate_tier(kwargs["tier"])},
        )
    from .fixed_d4 import prepare_fixed_d4_solve_config

    return prepare_fixed_d4_solve_config(*args, **kwargs)


def _fixed_d4_materialize(*args: Any, **kwargs: Any) -> Any:
    """Dispatch one plan assignment or a full fixed-D4 solve manifest."""

    if "tier" in kwargs and "cell" in kwargs and "artifact_root" in kwargs:
        return _materialize_provider_assignment(*args, **kwargs)
    from .fixed_d4 import materialize_fixed_d4

    return materialize_fixed_d4(*args, **kwargs)


def _fixed_d4_predict(*args: Any, **kwargs: Any) -> Any:
    """Dispatch plan-local prediction or public layerwise model prediction."""

    if len(args) == 4 and not kwargs:
        return predict_backpack_candidate(*args)
    from .fixed_d4 import produce_fixed_d4_layerwise_logits

    return produce_fixed_d4_layerwise_logits(*args, **kwargs)


def _fixed_d4_verify(*args: Any, **kwargs: Any) -> Any:
    """Dispatch candidate-receipt or full-model fixed-D4 verification."""

    if "tier" in kwargs and "cell" in kwargs:
        return verify_backpack_candidate(
            *args,
            **{**kwargs, "tier": _fixed_d4_candidate_tier(kwargs["tier"])},
        )
    from .fixed_d4 import verify_fixed_d4_model

    return verify_fixed_d4_model(*args, **kwargs)


def fixed_d4_backpack_provider(codebook_size: int) -> BackpackFamilyProvider:
    if codebook_size not in {2048, 4096}:
        raise ValueError("fixed D4 provider requires K2048 or K4096")

    return BackpackFamilyProvider(
        provider_id=f"d4-k{codebook_size}",
        kind="fixed_d4",
        runtime_family="truevq_d4",
        generate=_fixed_d4_generate,
        materialize=_fixed_d4_materialize,
        price=price_backpack_candidate,
        predict=_fixed_d4_predict,
        verify=_fixed_d4_verify,
    )


def builtin_backpack_family_providers() -> dict[str, BackpackFamilyProvider]:
    """Return the stock native, QTIP2/2.5/3, and fixed-D4 menu."""

    providers = (
        native_mxfp4_backpack_provider(),
        qtip_ring_backpack_provider(2.0),
        qtip_ring_backpack_provider(2.5),
        qtip_ring_backpack_provider(3.0),
        periodic_qtip3_backpack_provider(),
        periodic_qtip3_backpack_provider("qtip-native-v6@3.00"),
        fixed_d4_backpack_provider(2048),
        fixed_d4_backpack_provider(4096),
    )
    return {provider.provider_id: provider for provider in providers}


BQ23_PROVIDER_IDS = (
    "native-mxfp4",
    "qtip@2.00",
    "qtip@3.00",
    "d4-k2048",
    "d4-k4096",
)


def bq23_backpack_family_providers() -> dict[str, BackpackFamilyProvider]:
    """Return the canonical dynamic QTIP2/QTIP3 plus fixed-D4 BQ23 menu."""

    providers = builtin_backpack_family_providers()
    return {provider_id: providers[provider_id] for provider_id in BQ23_PROVIDER_IDS}


def backpack_provider_from_declaration(
    declaration: str | Mapping[str, Any],
) -> BackpackFamilyProvider:
    """Resolve a declarative tier to the public provider it actually executes."""

    if isinstance(declaration, str):
        if declaration.startswith("qtip-native-v4@"):
            return qtip_native_v4_backpack_provider(declaration.rsplit("@", 1)[1])
        aliases = {
            "native_mxfp4": "native-mxfp4",
            "d4_k2048": "d4-k2048",
            "d4_k4096": "d4-k4096",
            "qtip2": "qtip@2.00",
            "qtip2.5": "qtip@2.50",
            "qtip3": "qtip@3.00",
        }
        provider_id = aliases.get(declaration, declaration)
        try:
            return builtin_backpack_family_providers()[provider_id]
        except KeyError as exc:
            raise ValueError(f"unknown Backpack family provider {declaration!r}") from exc
    if not isinstance(declaration, Mapping):
        raise TypeError("provider declaration must be a provider id or mapping")
    explicit = declaration.get("provider")
    if explicit in {"periodic-qtip3@3.00", "qtip-native-v6@3.00"}:
        return periodic_qtip3_backpack_provider(str(explicit))
    if explicit in {"qtip_native_v4", "qtip-native-v4"}:
        return qtip_native_v4_backpack_provider(declaration.get("bpw"))
    if isinstance(explicit, str) and explicit.startswith("qtip-native-v4@"):
        provider = qtip_native_v4_backpack_provider(explicit.rsplit("@", 1)[1])
        requested = qtip_native_v4_backpack_provider(declaration.get("bpw"))
        if provider.provider_id != requested.provider_id:
            raise ValueError("native V4 provider id does not match declared bpw")
        return provider
    if explicit in {"native_mxfp4", "native-mxfp4"}:
        return native_mxfp4_backpack_provider()
    if explicit in {"d4_k2048", "d4-k2048"}:
        return fixed_d4_backpack_provider(2048)
    if explicit in {"d4_k4096", "d4-k4096"}:
        return fixed_d4_backpack_provider(4096)
    if explicit in {"qtip2", "qtip@2.00"}:
        return qtip_ring_backpack_provider(2.0)
    if explicit in {"qtip2.5", "qtip@2.50"}:
        return qtip_ring_backpack_provider(2.5)
    if explicit in {"qtip3", "qtip@3.00"}:
        return qtip_ring_backpack_provider(3.0)
    kind = declaration.get("kind", declaration.get("family"))
    if kind in {"qtip", "qtip_ring"}:
        return qtip_ring_backpack_provider(declaration.get("bpw"))
    if kind == "qtip_native_v4":
        return qtip_native_v4_backpack_provider(declaration.get("bpw"))
    if kind == "native_mxfp4":
        return native_mxfp4_backpack_provider()
    if kind == "fixed_d4":
        size = declaration.get("codebook_size")
        if isinstance(size, bool) or not isinstance(size, int):
            raise ValueError("fixed D4 provider declaration requires codebook_size")
        return fixed_d4_backpack_provider(size)
    if kind == "vector_vq":
        dimension = declaration.get("dimension")
        if dimension not in {4, 8}:
            raise ValueError("vector-VQ provider declaration requires dimension 4 or 8")
        size = declaration.get("codebook_size")
        bits = declaration.get("bits")
        if size is None and isinstance(bits, int) and not isinstance(bits, bool):
            size = 1 << bits
        if size is None and isinstance(declaration.get("bpw"), (int, float)):
            width = float(declaration["bpw"]) * int(dimension)
            if width.is_integer():
                size = 1 << int(width)
        if isinstance(size, bool) or not isinstance(size, int):
            raise ValueError("vector-VQ provider declaration requires exact codebook geometry")
        return vector_vq_backpack_provider(
            dimension=int(dimension), codebook_size=size
        )
    raise ValueError(f"unsupported Backpack family declaration {dict(declaration)!r}")


def resolve_backpack_family_provider(
    declaration: str | Mapping[str, Any],
    *,
    providers: Mapping[str, BackpackFamilyProvider] | None = None,
) -> BackpackFamilyProvider:
    """Resolve a supplied provider first, then the immutable built-in declaration."""

    if isinstance(declaration, str) and providers is not None and declaration in providers:
        return providers[declaration]
    if isinstance(declaration, Mapping):
        provider_id = declaration.get("provider")
        if isinstance(provider_id, str) and providers is not None and provider_id in providers:
            return providers[provider_id]
    return backpack_provider_from_declaration(declaration)


def activation_artifacts_for_options(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Return a consistent, deduplicated activation set for grouped cell rows."""

    registry: dict[str, dict[str, Any]] = {}
    for row in rows:
        for artifact in price_backpack_candidate(row).activation_artifacts:
            key = str(artifact["id"])
            prior = registry.get(key)
            if prior is not None and prior != artifact:
                raise ValueError(f"activation artifact {key!r} has inconsistent declarations")
            registry[key] = artifact
    return tuple(registry[key] for key in sorted(registry))
