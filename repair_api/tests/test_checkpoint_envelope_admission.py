"""Regression: canonical persisted checkpoints must reach the sealed scorer envelope.

The sealed candidate scorer
``/home/dnola/missions/QTIP2_V7_HALF_t_70421052_s8/code/fixed6_candidate_score_u20.py``
(sha256 1d49832ed2e1824b15dc80bdae3f09e8c3a7d7fdf588b9dd8c0319c27ac8827e) admits a
checkpoint only through this exact discriminator::

    if checkpoint.get("format") != "banana-smasher-qtip2-v7-joint-checkpoint-v1":
        raise RuntimeError(f"checkpoint format refused {checkpoint.get('format')}")

Checkpoints persisted by the public API
(:meth:`repair_api.api.ResidentRepairAPI._persist_continuation_checkpoint`) predate that
discriminator and therefore carry ``format`` = ``None``.  ``UPDATE_024.pt`` of mission
``QTIP2_V7_HALF_t_70421052_s8`` (sha256
650dbe534f2918f1fdbc0e59ef7dae03164f42d9d04f409f55cb6611840858ca) is exactly such an
otherwise canonical checkpoint; its observed top-level and identity key sets are pinned
below as :data:`PERSISTED_TOP_LEVEL_KEYS` / :data:`PERSISTED_IDENTITY_KEYS`, so the
fixture cannot drift away from the real artifact.  The checkpoint file itself is never
read, copied, or mutated by this test.

On the original code the public API exposes no adapter, the raw payload reaches the
scorer unchanged, and this regression fails at the format check with
``checkpoint format refused None``.  With ``repair_api.adapt_checkpointed_envelope`` the
same payload is admitted while every tensor, metadata, and resident-state object is
preserved by identity.
"""
from __future__ import annotations

import unittest

import repair_api

SCORER_REQUIRED_FORMAT = "banana-smasher-qtip2-v7-joint-checkpoint-v1"

# Observed on the real canonical artifact (torch.load of UPDATE_024.pt, read-only).
PERSISTED_TOP_LEVEL_KEYS = {
    "controlled_arm_id",
    "identity",
    "next_update",
    "optimizer_scheduler_delta",
    "optimizer_state",
    "scheduler_state",
    "schema",
    "state",
}
PERSISTED_IDENTITY_KEYS = {
    "basis_sha256",
    "checkpoint",
    "checkpoint_loaded",
    "identity_sha256",
    "next_update",
    "optimizer_scheduler_lineage",
    "parent_checkpoint_sha256",
    "parent_identity_sha256",
    "schema",
    "state_sha256",
    "world_size",
}
PERSISTED_STATE_SURFACES = {"luts", "norms", "outputs"}


def sealed_scorer_format_check(checkpoint):
    """Verbatim transcription of the sealed scorer's checkpoint admission gate."""
    if checkpoint.get("format") != SCORER_REQUIRED_FORMAT:
        raise RuntimeError(f"checkpoint format refused {checkpoint.get('format')}")
    return checkpoint


def public_api_envelope_adapter():
    """Resolve the public-API adapter without breaking import on the original code.

    Returning the identity function when the adapter is absent is what makes this
    regression fail at the sealed scorer's format check (the defect under repair)
    rather than at import time.
    """
    return getattr(repair_api, "adapt_checkpointed_envelope", lambda payload: payload)


def canonical_persisted_checkpoint():
    """A payload with the exact shape the public API persists for UPDATE_024."""
    # Stand-ins for the real tensors/state objects; identity preservation is what
    # this regression asserts, so opaque sentinels are sufficient and torch-free.
    luts, norms, outputs = object(), object(), object()
    identity = {
        "schema": "resident-continuation-checkpoint-identity-v1",
        "basis_sha256": "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b",
        "checkpoint": "UPDATE_024",
        "next_update": 24,
        "parent_checkpoint_sha256": "b" * 64,
        "parent_identity_sha256": "c" * 64,
        "state_sha256": "d" * 64,
        "optimizer_scheduler_lineage": "resident-adam-lambdalr",
        "checkpoint_loaded": True,
        "world_size": 1,
        "identity_sha256": "e" * 64,
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


class CanonicalCheckpointEnvelopeAdmissionTests(unittest.TestCase):
    def test_fixture_matches_the_real_persisted_checkpoint_shape(self) -> None:
        payload = canonical_persisted_checkpoint()
        self.assertEqual(set(payload), PERSISTED_TOP_LEVEL_KEYS)
        self.assertEqual(set(payload["identity"]), PERSISTED_IDENTITY_KEYS)
        self.assertEqual(set(payload["state"]), PERSISTED_STATE_SURFACES)
        # The defect being reproduced: raw canonical checkpoints have no format.
        self.assertIsNone(payload.get("format"))

    def test_raw_canonical_checkpoint_is_refused_by_the_sealed_scorer(self) -> None:
        payload = canonical_persisted_checkpoint()
        with self.assertRaises(RuntimeError) as caught:
            sealed_scorer_format_check(payload)
        self.assertEqual(str(caught.exception), "checkpoint format refused None")

    def test_public_api_supplies_the_required_envelope_for_the_sealed_scorer(self) -> None:
        payload = canonical_persisted_checkpoint()
        adapt = public_api_envelope_adapter()

        admitted = adapt(payload)

        # Fails on the original code here: format is still None at the scorer gate.
        sealed_scorer_format_check(admitted)
        self.assertEqual(admitted["format"], SCORER_REQUIRED_FORMAT)

    def test_admission_preserves_payload_metadata_and_resident_state_semantics(self) -> None:
        payload = canonical_persisted_checkpoint()
        adapt = public_api_envelope_adapter()

        admitted = adapt(payload)
        sealed_scorer_format_check(admitted)

        # The envelope adds the discriminator and nothing else.
        self.assertEqual(set(admitted) - set(payload), {"format"})
        self.assertEqual(set(payload) - set(admitted), set())

        # Model tensors and every resident-state object survive by identity.
        self.assertIs(admitted["state"], payload["state"])
        for surface in sorted(PERSISTED_STATE_SURFACES):
            self.assertIs(admitted["state"][surface], payload["state"][surface])
        for key in ("optimizer_state", "scheduler_state", "optimizer_scheduler_delta", "identity"):
            self.assertIs(admitted[key], payload[key])

        # Metadata and the resident cursor/lineage semantics are unchanged.
        self.assertEqual(admitted["schema"], "resident-continuation-checkpoint-v1")
        self.assertEqual(admitted["next_update"], payload["next_update"])
        self.assertEqual(admitted["controlled_arm_id"], payload["controlled_arm_id"])
        self.assertEqual(admitted["identity"]["identity_sha256"], payload["identity"]["identity_sha256"])
        self.assertEqual(admitted["identity"]["state_sha256"], payload["identity"]["state_sha256"])
        self.assertIs(admitted["identity"]["checkpoint_loaded"], True)

        # The persisted checkpoint object is never mutated in place.
        self.assertNotIn("format", payload)

    def test_admission_is_idempotent_for_an_already_stamped_envelope(self) -> None:
        adapt = public_api_envelope_adapter()
        payload = dict(canonical_persisted_checkpoint(), format=SCORER_REQUIRED_FORMAT)

        admitted = adapt(payload)

        sealed_scorer_format_check(admitted)
        self.assertEqual(admitted["next_update"], payload["next_update"])
        self.assertIs(admitted["state"], payload["state"])


class EnvelopeAdmissionRefusalTests(unittest.TestCase):
    """The adapter must stay narrow: only canonical persisted checkpoints pass."""

    def setUp(self) -> None:
        if not hasattr(repair_api, "adapt_checkpointed_envelope"):
            self.skipTest("public-API envelope adapter is not available")
        self.adapt = public_api_envelope_adapter()

    def assert_refused(self, payload) -> None:
        with self.assertRaises(repair_api.ArtifactError):
            self.adapt(payload)

    def test_refuses_non_mapping_payload(self) -> None:
        self.assert_refused(["not", "a", "checkpoint"])

    def test_refuses_foreign_schema(self) -> None:
        self.assert_refused(dict(canonical_persisted_checkpoint(), schema="some-other-schema-v1"))

    def test_refuses_incomplete_state_surfaces(self) -> None:
        payload = canonical_persisted_checkpoint()
        payload["state"] = {"luts": object(), "norms": object()}
        self.assert_refused(payload)

    def test_refuses_unloaded_checkpoint_identity(self) -> None:
        payload = canonical_persisted_checkpoint()
        payload["identity"] = dict(payload["identity"], checkpoint_loaded=False)
        self.assert_refused(payload)

    def test_refuses_identity_cursor_drift(self) -> None:
        payload = canonical_persisted_checkpoint()
        payload["identity"] = dict(payload["identity"], next_update=23)
        self.assert_refused(payload)

    def test_refuses_invalid_update_cursor(self) -> None:
        payload = canonical_persisted_checkpoint()
        payload["next_update"] = -1
        payload["identity"] = dict(payload["identity"], next_update=-1)
        self.assert_refused(payload)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    unittest.main()
