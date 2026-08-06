import hashlib
import json

from banana_smasher import (
    BackpackFamilyActivation,
    BackpackFamilyProvider,
    BackpackPlan,
    BackpackWireArtifact,
    BackpackWirePrice,
    backpack_provider_from_declaration,
    builtin_backpack_family_providers,
    fixed_d4_backpack_provider,
    generate_backpack_candidate,
    materialize_backpack_assignment,
    native_mxfp4_backpack_provider,
    predict_backpack_candidate,
    qtip1_5_provider_declaration,
    qtip_ring_backpack_provider,
    resolve_backpack_family_provider,
    vector_vq_backpack_provider,
    verify_backpack_candidate,
)
from banana_smasher.knapsack import solve_class_balanced_options
from banana_smasher.cli import main


def test_builtin_provider_menu_and_declaration_only_qtip_extension():
    providers = builtin_backpack_family_providers()

    assert set(providers) == {
        "native-mxfp4",
        "qtip@2.00",
        "qtip@2.50",
        "qtip@3.00",
        "d4-k2048",
        "d4-k4096",
    }
    assert providers["native-mxfp4"].runtime_family == "native_mxfp4"
    assert providers["qtip@2.00"].runtime_family == "qtip2"
    assert providers["qtip@2.50"].runtime_family == "qtip2"
    assert providers["qtip@3.00"].runtime_family == "qtip3"
    assert providers["d4-k2048"].runtime_family == "truevq_d4"
    assert providers["d4-k4096"].runtime_family == "truevq_d4"
    assert all(isinstance(provider, BackpackFamilyProvider) for provider in providers.values())
    assert callable(generate_backpack_candidate)
    assert callable(materialize_backpack_assignment)
    assert callable(predict_backpack_candidate)
    assert callable(verify_backpack_candidate)
    assert native_mxfp4_backpack_provider() == providers["native-mxfp4"]
    assert qtip_ring_backpack_provider(2.5) == providers["qtip@2.50"]
    assert fixed_d4_backpack_provider(2048) == providers["d4-k2048"]
    assert resolve_backpack_family_provider(
        {"family": "vector_vq", "dimension": 8, "bits": 2}
    ) == vector_vq_backpack_provider(dimension=8, codebook_size=4)

    assert backpack_provider_from_declaration("qtip@2.00") == providers["qtip@2.00"]
    assert backpack_provider_from_declaration("qtip2") == providers["qtip@2.00"]
    assert backpack_provider_from_declaration("qtip2.5") == providers["qtip@2.50"]
    assert backpack_provider_from_declaration("qtip3") == providers["qtip@3.00"]

    extension = qtip1_5_provider_declaration()
    assert extension.provider_id == "qtip1_5"
    assert extension.tier == "qtip@1.50"
    assert [(row.geometry.K, row.geometry.V, row.quarters) for row in extension.components] == [
        (1, 1, 2),
        (2, 2, 2),
    ]

    price = providers["qtip@2.00"].price(
        {
            "physical_bytes": 7,
            "activation_artifacts": ({"id": "qtip-lut", "bytes": 11},),
        }
    )
    assert price.cell_payload_bytes == 7
    assert price.activation_bytes == 11
    assert price.full_wire_bytes == 18
    assert isinstance(price, BackpackWirePrice)
    assert isinstance(price.activations[0], BackpackFamilyActivation)
    assert isinstance(price.activations[0].artifacts[0], BackpackWireArtifact)


def test_cli_lists_the_same_builtin_provider_menu(capsys):
    assert main(["backpack", "providers"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "PASS"
    assert [row["id"] for row in output["providers"]] == list(
        builtin_backpack_family_providers()
    )


def test_shared_activation_artifact_is_charged_once():
    cells = ["cell-a", "cell-b"]
    tiers = ["qtip", "d4"]
    activation = {"id": "qtip2-lut", "bytes": 5}

    result = solve_class_balanced_options(
        cells=cells,
        tiers=tiers,
        bytes_by_option={
            ("cell-a", "qtip"): 2,
            ("cell-a", "d4"): 4,
            ("cell-b", "qtip"): 2,
            ("cell-b", "d4"): 4,
        },
        activation_artifacts_by_option={
            ("cell-a", "qtip"): (activation,),
            ("cell-b", "qtip"): (activation,),
            ("cell-a", "d4"): (),
            ("cell-b", "d4"): (),
        },
        class_costs_by_option={
            (cell, tier): {"chat": 0.0}
            for cell in cells
            for tier in tiers
        },
        envelope_bytes=9,
        class_caps={"chat": 1.0},
        exact_envelope=True,
    )

    assert [row["tier"] for row in result["assignments"]] == ["qtip", "qtip"]
    assert result["cell_payload_bytes"] == 4
    assert result["activation_bytes"] == 5
    assert result["assigned_bytes"] == 9
    assert result["activated_artifacts"] == [activation]


def test_wire_price_is_bound_to_physical_receipt_and_shared_artifact(tmp_path):
    activation = tmp_path / "qtip.tlut"
    activation.write_bytes(b"shared-tlut")
    activation_sha = hashlib.sha256(activation.read_bytes()).hexdigest()
    receipt = tmp_path / "candidate.json"
    receipt.write_text(
        json.dumps(
            {
                "cell_payload_bytes": 7,
                "activation_artifacts": [
                    {
                        "id": "qtip-tlut",
                        "path": str(activation),
                        "bytes": activation.stat().st_size,
                        "sha256": activation_sha,
                    }
                ],
            }
        )
        + "\n"
    )

    price = builtin_backpack_family_providers()["qtip@2.00"].price(receipt)

    assert price.receipt == str(receipt.resolve())
    assert price.receipt_sha256 == hashlib.sha256(receipt.read_bytes()).hexdigest()
    assert price.cell_payload_bytes == 7
    assert price.activation_bytes == activation.stat().st_size
    assert price.full_wire_bytes == 7 + activation.stat().st_size


def test_plan_accepts_receipt_bound_provider_activation_declaration(tmp_path):
    plan = BackpackPlan.from_mapping(
        {
            "schema": "banana-smasher-backpack-plan-v1",
            "model": {"root": str(tmp_path / "model"), "revision": "test"},
            "target": {"exact_bytes": 100},
            "tiers": [
                {
                    "id": "qtip-1.5",
                    "family": "qtip",
                    "bpw": 1.5,
                    "backend": "fixture_reference",
                    "activation_artifacts": [
                        {"id": "qtip-lut", "bytes": 11, "sha256": "a" * 64}
                    ],
                }
            ],
            "anchor": {"bank": str(tmp_path / "anchor.npz"), "teacher": "model"},
            "prediction": {"class_caps": {name: 1.0 for name in (
                "agentic", "chat", "code", "multilingual", "prose", "reasoning"
            )}},
            "repair": {"method": "none"},
            "output": {
                "pack": str(tmp_path / "pack"),
                "model_id": "test",
                "instance_id": "test-v1",
            },
        },
        base_dir=tmp_path,
    )

    assert plan.tiers[0]["activation_artifacts"] == [
        {"id": "qtip-lut", "bytes": 11, "sha256": "a" * 64}
    ]
