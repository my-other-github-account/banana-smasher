import ast
import hashlib
from pathlib import Path


def test_provider_binds_the_exact_static_w28_trainer_bytes() -> None:
    root = Path(__file__).parents[1]
    provider_path = root / "modern_green_resident.py"
    trainer_path = root / "assets" / "static_w28_modern_green_clean_u0.py"

    tree = ast.parse(provider_path.read_text())
    declared = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "TRAINER_SHA256"
            for target in node.targets
        ):
            declared = ast.literal_eval(node.value)
            break

    assert declared == hashlib.sha256(trainer_path.read_bytes()).hexdigest()
