from repair_api.api import _admit_resident_validation_windows
from repair_api.balanced64 import ArtifactError

U64 = "4cab94648b62e15dbcc0be4e3701c5c2b38af71f8662011455ba0312448da624"
CANONICAL = (10, 12, 19, 24, 37, 45, 57, 60, 61, 63, 66, 69, 70, 75, 81, 89, 90, 109, 112, 113, 120, 128, 145, 146, 147, 148, 151, 159, 216, 226, 230, 244, 247, 267, 274, 276, 278, 279, 292, 303, 310, 322, 329, 333, 339, 346, 352, 378, 383, 386, 388, 397, 404, 405, 412, 419, 421, 435, 445, 450, 463, 487, 490, 506)
LEAKED = (28, 56, 68, 71, 76, 99, 107, 122, 124, 130, 141, 156, 160, 171, 180, 183, 185, 186, 196, 210, 212, 213, 218, 228, 232, 235, 249, 270, 272, 273, 283, 288, 290, 295, 297, 306, 307, 309, 311, 328, 331, 357, 362, 365, 368, 374, 376, 380, 384, 385, 391, 396, 413, 429, 430, 437, 442, 447, 454, 462, 464, 475, 489, 499)

def test_exact_u64_admits_only_w28_or_canonical_balanced64():
    assert _admit_resident_validation_windows((28,), artifact_windows=LEAKED, checkpoint_sha256=U64) == "W28"
    assert _admit_resident_validation_windows(CANONICAL, artifact_windows=LEAKED, checkpoint_sha256=U64) == "CANONICAL_BALANCED64"
    try:
        _admit_resident_validation_windows(LEAKED, artifact_windows=LEAKED, checkpoint_sha256=U64)
    except ArtifactError as exc:
        assert "canonical Balanced64" in str(exc)
    else:
        raise AssertionError("authenticated U64 admitted leaked fixture roster")

def test_non_u64_retains_manifest_compatibility():
    assert _admit_resident_validation_windows(LEAKED, artifact_windows=LEAKED, checkpoint_sha256="other") == "ARTIFACT_BALANCED64"
