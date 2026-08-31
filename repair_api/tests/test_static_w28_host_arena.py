from pathlib import Path


def test_static_w28_streams_bounded_pageable_chunks_into_resident_cuda() -> None:
    source = (
        Path(__file__).parents[1]
        / "assets"
        / "static_w28_fast_v7_expert_base.py"
    ).read_text()

    class_start = source.index("class FullyResidentGroupedV7Experts")
    constructor = source[class_start:]
    stream = source[source.index("def _stream_projection_payloads("):class_start]
    assert "HOST_STREAM_EXPERTS = 16" in source
    assert "for start in range(0, len(paths), HOST_STREAM_EXPERTS):" in stream
    assert "packed_view = arena_cpu[:count].reshape" in stream
    assert "packed_cuda[start:end].copy_(" in stream
    assert "su_cuda[start:end].copy_(" in stream
    assert "sv_cuda[start:end].copy_(" in stream
    assert "torch.cuda.synchronize(device=device)" in stream
    assert "_stream_projection_payloads(" in constructor
    assert "self._packed_host_arena_owner" not in constructor
    assert "arena_shape = (len(PROJECTIONS), projection_elements)" not in constructor
    assert "arena_pointer, arena_owner" not in constructor


def test_streamed_projection_preserves_bytes_across_chunk_boundary(
    monkeypatch, tmp_path
) -> None:
    import ast
    from concurrent.futures import ThreadPoolExecutor
    import os

    import numpy as np
    import torch

    source_path = (
        Path(__file__).parents[1]
        / "assets"
        / "static_w28_fast_v7_expert_base.py"
    )
    tree = ast.parse(source_path.read_text())
    selected: list[ast.stmt] = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {
            "_shared_packed_host_arena",
            "_load_projection_payloads_into",
            "_stream_projection_payloads",
        }
    ]
    namespace = {
        "Path": Path,
        "ThreadPoolExecutor": ThreadPoolExecutor,
        "np": np,
        "os": os,
        "torch": torch,
        "PACKED_BYTES": 64,
        "HOST_STREAM_EXPERTS": 16,
        "_PACKED_HOST_ARENA": None,
    }
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(source_path), "exec"),
        namespace,
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    paths = []
    expected = []
    for expert in range(17):
        packed = (np.arange(32, dtype="<i2") + expert * 100).reshape(1, 1, 32)
        su = np.arange(16, dtype="<f2") + expert
        sv = np.arange(16, dtype="<f2") + expert * 2
        path = tmp_path / f"E{expert:03d}_w1.q2v7wire"
        path.write_bytes(
            packed.tobytes() + su.tobytes() + sv.tobytes() + expert.to_bytes(4, "little")
        )
        paths.append(path)
        expected.append(torch.from_numpy(packed.copy()))

    packed, su, sv, calls, read_bytes = namespace["_stream_projection_payloads"](
        paths, m=16, k=16, packed_bytes=64, device=torch.device("cpu")
    )

    assert torch.equal(packed, torch.stack(expected))
    assert torch.equal(su[:, 0], torch.arange(17, dtype=torch.float16))
    assert torch.equal(sv[:, 0], torch.arange(17, dtype=torch.float16) * 2)
    assert calls == 17
    assert read_bytes == 17 * 132
    assert namespace["_PACKED_HOST_ARENA"].shape == (16, 32)


def test_projection_reads_drop_each_authenticated_member_from_page_cache(
    monkeypatch, tmp_path
) -> None:
    import ast
    from concurrent.futures import ThreadPoolExecutor
    import os

    import numpy as np
    import torch

    source_path = (
        Path(__file__).parents[1]
        / "assets"
        / "static_w28_fast_v7_expert_base.py"
    )
    tree = ast.parse(source_path.read_text())
    selected = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_load_projection_payloads_into"
    )
    namespace = {
        "Path": Path,
        "ThreadPoolExecutor": ThreadPoolExecutor,
        "np": np,
        "os": os,
        "torch": torch,
        "PACKED_BYTES": 64,
    }
    exec(
        compile(ast.Module(body=[selected], type_ignores=[]), str(source_path), "exec"),
        namespace,
    )
    paths = []
    for expert in range(3):
        path = tmp_path / f"E{expert:03d}_w1.q2v7wire"
        path.write_bytes(bytes([expert]) * 64 + bytes(68))
        paths.append(path)
    advice_calls = []
    real_close = os.close

    def record_advice(fd, offset, length, advice):
        os.fstat(fd)
        advice_calls.append((offset, length, advice))

    monkeypatch.setattr(os, "POSIX_FADV_DONTNEED", 4, raising=False)
    monkeypatch.setattr(os, "posix_fadvise", record_advice, raising=False)
    packed = np.empty((3, 1, 1, 32), dtype="<i2")
    namespace["_load_projection_payloads_into"](
        paths, packed, m=16, k=16, packed_bytes=64, pin_memory=False
    )

    assert advice_calls == [(0, 0, 4)] * 3
    assert os.close is real_close
