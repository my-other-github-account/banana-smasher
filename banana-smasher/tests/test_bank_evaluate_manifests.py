from __future__ import annotations

import json
from pathlib import Path

from banana_smasher.durability import canonical_sha256, tree_identity


ROOT = Path(__file__).parents[1]
SCHEMAS = (
    "bs-teacher-bank-v1.schema.json",
    "bs-real-axis-windows-v1.schema.json",
    "bs-real-axis-runtime-v1.schema.json",
    "bs-real-axis-instrument-v1.schema.json",
    "bs-paired-real-axis-evaluation-v1.schema.json",
)
CANONICAL_PACK_REQUIRED = {
    "schema",
    "schema_version",
    "source_format",
    "model_id",
    "instance_id",
    "quant_method",
    "layers",
    "experts_per_layer",
    "expert_partitions",
    "tier_codes",
    "tensor_layout_sha256",
    "tensor_index",
    "files",
    "link_mode_requested",
    "links",
    "container",
    "provenance",
}


def test_portable_schemas_are_nested_json_contracts() -> None:
    for filename in SCHEMAS:
        schema = json.loads((ROOT / "schema" / filename).read_text())
        assert schema["$schema"].endswith("2020-12/schema")
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert schema["required"]

    bank = json.loads((ROOT / "schema/bs-teacher-bank-v1.schema.json").read_text())
    for name in ("model", "corpus", "instrument", "population"):
        assert bank["properties"][name]["additionalProperties"] is False
        assert bank["properties"][name]["required"]

    evaluation = json.loads(
        (ROOT / "schema/bs-paired-real-axis-evaluation-v1.schema.json").read_text()
    )
    assert evaluation["properties"]["mode"] == {"const": "paired_real_axis"}
    assert set(evaluation["properties"]["arms"]["properties"]) == {
        "candidate",
        "reference",
    }


def test_pack_schema_adds_only_optional_real_axis_contract() -> None:
    schema = json.loads((ROOT / "schema/bs-pack-v1.schema.json").read_text())
    assert set(schema["required"]) == CANONICAL_PACK_REQUIRED
    assert "real_axis" not in schema["required"]
    assert schema["properties"]["source_format"]["enum"] == [
        "canonical-npy-v1",
        "banana-smasher-materialized-layer-v1",
    ]
    real_axis = schema["properties"]["real_axis"]
    assert real_axis["additionalProperties"] is False
    assert set(real_axis["required"]) == {
        "manifest_sha256",
        "layer_descriptor_sha256",
        "head_sha256",
    }
    assert real_axis["properties"]["layer_descriptor_sha256"]["minItems"] == 1


def test_tree_manifest_hash_is_relative_and_content_bound(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    for root in (first, second):
        (root / "nested").mkdir(parents=True)
        (root / "nested/member.bin").write_bytes(b"portable-member")
        (root / "manifest.json").write_text('{"schema":"portable"}\n')

    first_identity = tree_identity(first)
    second_identity = tree_identity(second)
    assert first_identity == second_identity
    assert first_identity["sha256"] == canonical_sha256(
        [
            {
                "path": "manifest.json",
                "bytes": (first / "manifest.json").stat().st_size,
                "sha256": __import__("hashlib").sha256(
                    (first / "manifest.json").read_bytes()
                ).hexdigest(),
            },
            {
                "path": "nested/member.bin",
                "bytes": (first / "nested/member.bin").stat().st_size,
                "sha256": __import__("hashlib").sha256(
                    (first / "nested/member.bin").read_bytes()
                ).hexdigest(),
            },
        ]
    )

    (second / "nested/member.bin").write_bytes(b"tampered-member")
    assert tree_identity(second)["sha256"] != first_identity["sha256"]
