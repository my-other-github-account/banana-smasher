"""Regression: two-rank continuation checkpoints must reach the sealed scorer.

Observed live on 2026-08-30 (task t_9d869d9a, spark-1 rank0).  The Candidate-B
CVaR ``UPDATE_024.pt`` (sha256
ab99a45f0b882892ffb6c298fdae2836ca9117e5b106c524441de8b865deff00) was persisted
by a ``world_size=2`` continuation.  It passed ``adapt_checkpointed_envelope``'s
format admission and the sealed scorer's format check, materialised all 43
layers, then failed *inside* checkpoint admission at::

    File ".../joint_v7_runner.alias.py", line 465, in load_tensor_state
        raise RuntimeError("checkpoint dense state key/order drift")

``load_tensor_state`` compares the live roster against the saved surface
**positionally**::

    if list(live) != list(saved):
        raise RuntimeError("checkpoint dense state key/order drift")

Measured facts about that artifact (read-only ``torch.load``):

* norms: 235 names, ``set()`` equal to the admission roster, 0 missing, 0 extra
* outputs: 43 names, ``set()`` equal to the admission roster, 0 missing, 0 extra
* every per-name shape matches
* but 159 of 235 norm positions differ from the roster order, first divergence
  at index 75 (``model.layers.3.input_layernorm`` where the roster has
  ``model.layers.21.input_layernorm``)
* the saved order is exactly ``rank0 block (layers 0..20, 113 norms)`` followed
  by ``rank1 block (layers 21..42, 122 norms)`` — the pipeline layer_split
  ``{0: [0, 20], 1: [21, 42]}`` partition order
* the single-rank ``UPDATE_020.pt`` start checkpoint of the *same* lineage is
  already in roster order, and the admission roster order is plain ``sorted()``

So a two-rank continuation is scientifically identical to a single-rank one but
was rejected purely on mapping insertion order.  The adapter now normalises the
dense surfaces to canonical sorted order, carrying every value by identity.
"""
from __future__ import annotations

import unittest

import repair_api
from repair_api import ArtifactError

SCORER_REQUIRED_FORMAT = "banana-smasher-qtip2-v7-joint-checkpoint-v1"

# Layer split of the observed two-rank Candidate-B run.
RANK0_LAYERS = range(0, 21)
RANK1_LAYERS = range(21, 43)

# Measured on the real artifact.
OBSERVED_NORM_COUNT = 235
OBSERVED_OUTPUT_COUNT = 43
OBSERVED_RANK0_NORM_COUNT = 113
OBSERVED_RANK1_NORM_COUNT = 122
OBSERVED_ORDER_DIVERGENCE_COUNT = 159
OBSERVED_FIRST_DIVERGENCE = (
    75,
    "model.layers.3.input_layernorm",
    "model.layers.21.input_layernorm",
)
OBSERVED_OUTPUT_DIVERGENCE_COUNT = 29
OBSERVED_FIRST_OUTPUT_DIVERGENCE = (
    14,
    "model.layers.3.self_attn.o_b_proj.output_log_gain",
    "model.layers.21.self_attn.o_b_proj.output_log_gain",
)

# Measured per-layer RMSNorm topology of the real artifact: three classes.
#   layers 0,1                      -> 4 norms  (no compressor)
#   odd layers 3..41                -> 5 norms  (compressor.kv_norm)
#   even layers 2..42               -> 6 norms  (+ compressor.indexer.kv_norm)
# plus one trailing final ``model.norm``.  4*2 + 5*20 + 6*21 + 1 == 235.
_BASE_NORM_SUFFIXES = (
    "input_layernorm",
    "post_attention_layernorm",
    "self_attn.kv_norm",
    "self_attn.q_a_norm",
)
_COMPRESSOR_SUFFIX = "self_attn.compressor.kv_norm"
_INDEXER_SUFFIX = "self_attn.compressor.indexer.kv_norm"


def _norm_suffixes_for_layer(layer: int):
    if layer in (0, 1):
        return list(_BASE_NORM_SUFFIXES)
    if layer % 2 == 1:
        return list(_BASE_NORM_SUFFIXES) + [_COMPRESSOR_SUFFIX]
    return list(_BASE_NORM_SUFFIXES) + [_COMPRESSOR_SUFFIX, _INDEXER_SUFFIX]


def _norm_names_for(layers):
    """Every RMSNorm name the given layers contribute, canonical spelling."""
    names = []
    for layer in layers:
        names.extend(f"model.layers.{layer}.{suffix}" for suffix in _norm_suffixes_for_layer(layer))
    return names


def canonical_norm_roster():
    """The live roster order the scorer builds: plain sorted names."""
    return sorted(_norm_names_for(range(0, 43)) + ["model.norm"])


def canonical_output_roster():
    return sorted(
        f"model.layers.{layer}.self_attn.o_b_proj.output_log_gain" for layer in range(0, 43)
    )


def rank_partitioned(names):
    """Re-order a canonical roster into rank0-block-then-rank1-block order."""

    def layer_of(name):
        parts = name.split(".")
        return int(parts[2]) if len(parts) > 2 and parts[1] == "layers" else 10 ** 6

    rank0 = [n for n in names if layer_of(n) <= 20]
    rank1 = [n for n in names if layer_of(n) > 20]
    return rank0 + rank1


def sealed_scorer_load_tensor_state(live_names, saved):
    """Verbatim transcription of the sealed runner's positional order gate."""
    if list(live_names) != list(saved):
        raise RuntimeError("checkpoint dense state key/order drift")
    return True


def two_rank_continuation_checkpoint():
    """A payload shaped exactly like the observed world_size=2 UPDATE_024."""
    norm_names = rank_partitioned(canonical_norm_roster())
    output_names = rank_partitioned(canonical_output_roster())
    # Distinct sentinels so identity preservation is observable per name.
    norms = {name: object() for name in norm_names}
    outputs = {name: object() for name in output_names}
    luts = {f"layers.{layer}.qtip2_v7.layer_lut": object() for layer in range(43)}
    identity = {
        "schema": "resident-continuation-checkpoint-identity-v1",
        "basis_sha256": "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b",
        "checkpoint": "UPDATE_024",
        "next_update": 24,
        "parent_checkpoint_sha256": "f2b4688f47088f0c42a9d3b05493019727499326633e854943b38693fa6ebfaa",
        "parent_identity_sha256": "475ae87514235d61e2b3924e0b87a7aa524aac9eafda499eb5e1fb4de1fe1f8b",
        "state_sha256": "0ae7d86429b7e40732b5a5fa212571c1b54d723216604387e05a9b25f6b5c972",
        "optimizer_scheduler_lineage": "fresh-published-pre-adam-lambdalr",
        "checkpoint_loaded": True,
        "world_size": 2,
        "identity_sha256": "36a83528a9499c4fd5e13a3dd6016ce2cef6bc7d39dd8198a77f8d171141a597",
    }
    return {
        "schema": "resident-continuation-checkpoint-v1",
        "next_update": 24,
        "state": {"luts": luts, "norms": norms, "outputs": outputs},
        "optimizer_state": {"steps": 24},
        "scheduler_state": {"steps": 24},
        "optimizer_scheduler_delta": {"optimizer_steps": 24, "scheduler_steps": 24},
        "identity": identity,
        "controlled_arm_id": None,
    }


class TwoRankDenseOrderAdmissionTests(unittest.TestCase):
    def test_fixture_reproduces_the_measured_artifact_shape(self) -> None:
        payload = two_rank_continuation_checkpoint()
        norms = payload["state"]["norms"]
        outputs = payload["state"]["outputs"]
        self.assertEqual(len(norms), OBSERVED_NORM_COUNT)
        self.assertEqual(len(outputs), OBSERVED_OUTPUT_COUNT)
        self.assertEqual(set(norms), set(canonical_norm_roster()))
        self.assertEqual(set(outputs), set(canonical_output_roster()))
        self.assertEqual(payload["identity"]["world_size"], 2)
        self.assertEqual(len(_norm_names_for(RANK0_LAYERS)), OBSERVED_RANK0_NORM_COUNT)
        # rank1 block carries the trailing model.norm, hence 122 not 121.
        self.assertEqual(
            len(_norm_names_for(RANK1_LAYERS)) + 1, OBSERVED_RANK1_NORM_COUNT
        )

    def test_fixture_reproduces_the_measured_order_divergence(self) -> None:
        payload = two_rank_continuation_checkpoint()
        saved = list(payload["state"]["norms"])
        roster = canonical_norm_roster()
        self.assertNotEqual(saved, roster)
        divergences = [
            (i, a, b) for i, (a, b) in enumerate(zip(saved, roster)) if a != b
        ]
        self.assertEqual(len(divergences), OBSERVED_ORDER_DIVERGENCE_COUNT)
        self.assertEqual(divergences[0], OBSERVED_FIRST_DIVERGENCE)

        saved_out = list(payload["state"]["outputs"])
        roster_out = canonical_output_roster()
        out_divergences = [
            (i, a, b) for i, (a, b) in enumerate(zip(saved_out, roster_out)) if a != b
        ]
        self.assertEqual(len(out_divergences), OBSERVED_OUTPUT_DIVERGENCE_COUNT)
        self.assertEqual(out_divergences[0], OBSERVED_FIRST_OUTPUT_DIVERGENCE)

    def test_raw_two_rank_checkpoint_is_refused_by_the_sealed_order_gate(self) -> None:
        """The defect under repair, reproduced without the adapter."""
        payload = two_rank_continuation_checkpoint()
        with self.assertRaises(RuntimeError) as caught:
            sealed_scorer_load_tensor_state(
                canonical_norm_roster(), payload["state"]["norms"]
            )
        self.assertIn("dense state key/order drift", str(caught.exception))

    def test_adapter_admits_the_two_rank_checkpoint_through_the_order_gate(self) -> None:
        payload = two_rank_continuation_checkpoint()
        admitted = repair_api.adapt_checkpointed_envelope(payload)
        self.assertEqual(admitted["format"], SCORER_REQUIRED_FORMAT)
        self.assertTrue(
            sealed_scorer_load_tensor_state(
                canonical_norm_roster(), admitted["state"]["norms"]
            )
        )
        self.assertTrue(
            sealed_scorer_load_tensor_state(
                canonical_output_roster(), admitted["state"]["outputs"]
            )
        )

    def test_reordering_preserves_every_value_by_identity(self) -> None:
        payload = two_rank_continuation_checkpoint()
        before_norms = dict(payload["state"]["norms"])
        before_outputs = dict(payload["state"]["outputs"])
        before_luts = payload["state"]["luts"]
        admitted = repair_api.adapt_checkpointed_envelope(payload)
        for name, value in before_norms.items():
            self.assertIs(admitted["state"]["norms"][name], value)
        for name, value in before_outputs.items():
            self.assertIs(admitted["state"]["outputs"][name], value)
        # LUTs are keyed by the scorer's own name/order contract and are not a
        # dense surface; they must be passed straight through.
        self.assertIs(admitted["state"]["luts"], before_luts)
        self.assertEqual(set(admitted["state"]), {"luts", "norms", "outputs"})

    def test_adapter_does_not_mutate_the_input_payload(self) -> None:
        payload = two_rank_continuation_checkpoint()
        original_norm_order = list(payload["state"]["norms"])
        original_state = payload["state"]
        repair_api.adapt_checkpointed_envelope(payload)
        self.assertEqual(list(payload["state"]["norms"]), original_norm_order)
        self.assertIs(payload["state"], original_state)
        self.assertIsNone(payload.get("format"))

    def test_single_rank_sorted_checkpoint_is_unchanged_in_order(self) -> None:
        """The already-working world_size=1 path must be a no-op reorder."""
        payload = two_rank_continuation_checkpoint()
        payload["state"]["norms"] = {n: object() for n in canonical_norm_roster()}
        payload["state"]["outputs"] = {n: object() for n in canonical_output_roster()}
        payload["identity"]["world_size"] = 1
        admitted = repair_api.adapt_checkpointed_envelope(payload)
        self.assertEqual(list(admitted["state"]["norms"]), canonical_norm_roster())
        self.assertEqual(list(admitted["state"]["outputs"]), canonical_output_roster())

    def test_already_formatted_payload_is_still_passed_through(self) -> None:
        payload = two_rank_continuation_checkpoint()
        payload["format"] = SCORER_REQUIRED_FORMAT
        admitted = repair_api.adapt_checkpointed_envelope(payload)
        self.assertEqual(admitted["format"], SCORER_REQUIRED_FORMAT)

    def test_non_string_dense_keys_are_refused_rather_than_sorted(self) -> None:
        payload = two_rank_continuation_checkpoint()
        payload["state"]["norms"] = {0: object(), "model.norm": object()}
        with self.assertRaises(ArtifactError):
            repair_api.adapt_checkpointed_envelope(payload)

    def test_envelope_gates_are_still_enforced(self) -> None:
        """Normalisation must not weaken any pre-existing admission gate."""
        bad_schema = two_rank_continuation_checkpoint()
        bad_schema["schema"] = "some-other-schema"
        with self.assertRaises(ArtifactError):
            repair_api.adapt_checkpointed_envelope(bad_schema)

        bad_cursor = two_rank_continuation_checkpoint()
        bad_cursor["identity"]["next_update"] = 23
        with self.assertRaises(ArtifactError):
            repair_api.adapt_checkpointed_envelope(bad_cursor)

        not_loaded = two_rank_continuation_checkpoint()
        not_loaded["identity"]["checkpoint_loaded"] = False
        with self.assertRaises(ArtifactError):
            repair_api.adapt_checkpointed_envelope(not_loaded)

        bad_surfaces = two_rank_continuation_checkpoint()
        del bad_surfaces["state"]["outputs"]
        with self.assertRaises(ArtifactError):
            repair_api.adapt_checkpointed_envelope(bad_surfaces)


if __name__ == "__main__":
    unittest.main()
