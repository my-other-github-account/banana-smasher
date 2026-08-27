from types import SimpleNamespace

import launch_v21


def test_copy_l034_wires_targets_only_requested_host(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=b"sealed-tar", stderr=b"")

    monkeypatch.setattr(launch_v21.subprocess, "run", fake_run)

    launch_v21.copy_l034_wires(
        "spark-3", "/dev/shm/MODERN_GREEN_API_ONLY_t_1d62fc5e_s3_v21"
    )

    assert len(calls) == 2
    source_command = calls[0][0]
    destination_command = calls[1][0]
    assert source_command[0:6] == [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "spark-7",
    ]
    assert destination_command[0:6] == [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "spark-3",
    ]
    assert destination_command[-1].endswith(
        "tar -C /dev/shm/MODERN_GREEN_API_ONLY_t_1d62fc5e_s3_v21/l034 -xf -"
    )
    assert source_command[-1].endswith(
        "-cf - run/staged_wire run/complete_wire"
    )
    assert calls[1][1]["input"] == b"sealed-tar"


def test_l034_closure_passes_remote_script_over_stdin(monkeypatch):
    observed = {}

    def fake_ssh(host, command, **kwargs):
        observed.update(host=host, command=command, kwargs=kwargs)
        return '{"schema":"x","member_count":768,"rows":[]}\n'

    monkeypatch.setattr(launch_v21, "ssh", fake_ssh)
    result = launch_v21.l034_closure("spark-3", "/dev/shm/root-v21")

    assert result["member_count"] == 768
    assert observed["command"] == "python3 - /dev/shm/root-v21"
    script = observed["kwargs"]["input_bytes"].decode()
    assert "MISSING_L034_MEMBER {p}" in script
    assert "L034_MEMBER_DRIFT {p}" in script
