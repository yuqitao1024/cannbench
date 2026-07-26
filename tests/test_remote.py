import json
from types import SimpleNamespace

import pytest

from cannbench.core.execution import RemoteExecutionArtifacts, RemoteProfileArtifacts
from cannbench.core.prepared_input import build_prepared_operator_input, write_prepared_operator_input
from cannbench.core.profile import DeviceProfileSummary, ProfileKernelSelection
from cannbench.core.remote import (
    RemoteCollectionResult,
    RemoteEndpoint,
    collect_remote_artifacts,
    preinstall_remote_simt_op,
    read_remote_endpoint,
)


def _write_softmax_prepared(path):
    return write_prepared_operator_input(
        path,
        build_prepared_operator_input(
            op="softmax",
            dtype="float16",
            dataset="smoke",
            case_id="tiny_logits",
            seed=0,
        ),
    )


def test_read_remote_endpoint_config(tmp_path):
    path = tmp_path / "ascend.json"
    path.write_text(
        json.dumps(
            {
                "name": "ascend-a2",
                "backend": "ascend",
                "host": "user@ascend-host",
                "workdir": "/opt/cannbench",
                "python": "python3",
                "env": {"ASCEND_VISIBLE_DEVICES": "0"},
            }
        )
    )

    endpoint = read_remote_endpoint(path)

    assert endpoint == RemoteEndpoint(
        name="ascend-a2",
        backend="ascend",
        host="user@ascend-host",
        port=None,
        workdir="/opt/cannbench",
        python="python3",
        env={"ASCEND_VISIBLE_DEVICES": "0"},
    )


def test_read_remote_endpoint_parses_host_port_suffix(tmp_path):
    path = tmp_path / "ascend.json"
    path.write_text(
        json.dumps(
            {
                "name": "ascend-a2",
                "backend": "ascend",
                "host": "root@121.41.199.170:20002",
                "workdir": "/home/y00621698/cannbench",
            }
        )
    )

    endpoint = read_remote_endpoint(path)

    assert endpoint.host == "root@121.41.199.170"
    assert endpoint.port == 20002


def test_read_remote_endpoint_accepts_setup_command(tmp_path):
    path = tmp_path / "ascend.json"
    path.write_text(
        json.dumps(
            {
                "name": "ascend-a2",
                "backend": "ascend",
                "host": "root@121.41.199.170",
                "workdir": "/home/y00621698/cannbench",
                "setup": "source /usr/local/Ascend/cann/set_env.sh",
            }
        )
    )

    endpoint = read_remote_endpoint(path)

    assert endpoint.setup == "source /usr/local/Ascend/cann/set_env.sh"


def test_collect_remote_artifacts_runs_capture_and_downloads_output(tmp_path):
    commands: list[list[str]] = []

    def fake_runner(command):
        commands.append(command)

    endpoint = RemoteEndpoint(
        name="ascend-a2",
        backend="ascend",
        host="user@ascend-host",
        port=None,
        workdir="/opt/cannbench",
        python="python3",
        env={"ASCEND_VISIBLE_DEVICES": "0"},
    )
    prepared_input = tmp_path / "prepared.json"
    _write_softmax_prepared(prepared_input)

    result = collect_remote_artifacts(
        endpoint=endpoint,
        prepared_input=prepared_input,
        output_dir=tmp_path / "results",
        run_id="softmax-run",
        capture_output=True,
        implementation="simt",
        runner=fake_runner,
    )

    assert result.local_output_dir == tmp_path / "results"
    assert result.remote_run_dir == "/opt/cannbench/.cannbench-runs/softmax-run"
    assert commands == [
        [
            "ssh",
            "user@ascend-host",
            "mkdir -p /opt/cannbench/.cannbench-runs/softmax-run",
        ],
        [
            "scp",
            str(prepared_input),
            "user@ascend-host:/opt/cannbench/.cannbench-runs/softmax-run/prepared.json",
        ],
        [
            "ssh",
            "user@ascend-host",
            "cd /opt/cannbench && src/cannbench/operators/builtin/softmax/simt/v1/install.sh",
        ],
        [
            "ssh",
            "user@ascend-host",
            "cd /opt/cannbench && ASCEND_VISIBLE_DEVICES=0 CANNBENCH_SKIP_SIMT_INSTALL=1 python3 -m cannbench internal-run --backend ascend --prepared-input .cannbench-runs/softmax-run/prepared.json --output-dir .cannbench-runs/softmax-run/output --run-name captured-output --implementation simt",
        ],
        [
            "scp",
            "-r",
            "user@ascend-host:/opt/cannbench/.cannbench-runs/softmax-run/output",
            str(tmp_path / "results" / "output"),
        ],
    ]


def test_collect_remote_artifacts_runs_ascend_profile_and_downloads_profile(tmp_path):
    commands: list[list[str]] = []

    def fake_runner(command):
        commands.append(command)
        if command[:2] == ["scp", "-r"] and command[-1].endswith("/profile"):
            profile_dir = tmp_path / "results" / "profile"
            profile_dir.mkdir(parents=True)
            (profile_dir / "op_summary.csv").write_text(
                "Op Name,Task Duration(us)\nsoftmax,1000\n"
            )
        if command[:2] == ["scp", "-r"] and command[-1].endswith("/perf"):
            perf_dir = tmp_path / "results" / "perf"
            perf_dir.mkdir(parents=True)
            (perf_dir / "benchmark.json").write_text(
                json.dumps(
                    {
                        "backend": "ascend",
                        "device_name": "Ascend 910B",
                    }
                )
                + "\n"
            )

    endpoint = RemoteEndpoint(
        name="ascend-a2",
        backend="ascend",
        host="user@ascend-host",
        port=None,
        workdir="/opt/cannbench",
        python="python3",
        setup="source /usr/local/Ascend/cann/set_env.sh",
        env={"ASCEND_VISIBLE_DEVICES": "0"},
    )
    prepared_input = tmp_path / "prepared.json"
    _write_softmax_prepared(prepared_input)

    collect_remote_artifacts(
        endpoint=endpoint,
        prepared_input=prepared_input,
        output_dir=tmp_path / "results",
        run_id="softmax-run",
        capture_output=False,
        profile_device_time=True,
        runner=fake_runner,
    )

    assert commands == [
        [
            "ssh",
            "user@ascend-host",
            "mkdir -p /opt/cannbench/.cannbench-runs/softmax-run /opt/cannbench/.cannbench-runs/softmax-run/profile",
        ],
        [
            "scp",
            str(prepared_input),
            "user@ascend-host:/opt/cannbench/.cannbench-runs/softmax-run/prepared.json",
        ],
        [
            "ssh",
            "user@ascend-host",
            "cd /opt/cannbench && source /usr/local/Ascend/cann/set_env.sh && ASCEND_VISIBLE_DEVICES=0 msopprof --output=/opt/cannbench/.cannbench-runs/softmax-run/profile --launch-count=10 python3 -m cannbench internal-run --backend ascend --prepared-input .cannbench-runs/softmax-run/prepared.json --output-dir .cannbench-runs/softmax-run/perf --run-name benchmark",
        ],
        [
            "scp",
            "-r",
            "user@ascend-host:/opt/cannbench/.cannbench-runs/softmax-run/profile",
            str(tmp_path / "results" / "profile"),
        ],
        [
            "scp",
            "-r",
            "user@ascend-host:/opt/cannbench/.cannbench-runs/softmax-run/perf",
            str(tmp_path / "results" / "perf"),
        ],
    ]


def test_collect_remote_artifacts_passes_simt_op_version_to_internal_run(tmp_path):
    commands: list[list[str]] = []

    def fake_runner(command):
        commands.append(command)
        if command[:2] == ["scp", "-r"] and command[-1].endswith("/profile"):
            profile_dir = tmp_path / "results" / "profile"
            profile_dir.mkdir(parents=True)
            (profile_dir / "op_summary.csv").write_text(
                "Op Name,Task Duration(us)\nsoftmax,1000\n"
            )
        if command[:2] == ["scp", "-r"] and command[-1].endswith("/perf"):
            perf_dir = tmp_path / "results" / "perf"
            perf_dir.mkdir(parents=True)
            (perf_dir / "benchmark.json").write_text(
                json.dumps(
                    {
                        "backend": "ascend",
                        "device_name": "Ascend 910B",
                    }
                )
                + "\n"
            )

    endpoint = RemoteEndpoint(
        name="ascend-a2",
        backend="ascend",
        host="user@ascend-host",
        workdir="/opt/cannbench",
        python="python3",
        env={},
    )
    prepared_input = tmp_path / "prepared.json"
    _write_softmax_prepared(prepared_input)

    collect_remote_artifacts(
        endpoint=endpoint,
        prepared_input=prepared_input,
        output_dir=tmp_path / "results",
        run_id="softmax-run",
        capture_output=False,
        profile_device_time=True,
        implementation="simt",
        implementation_version="v2",
        runner=fake_runner,
    )

    assert commands[2] == [
        "ssh",
        "user@ascend-host",
        "cd /opt/cannbench && src/cannbench/operators/builtin/softmax/simt/v2/install.sh",
    ]
    assert "--implementation simt" in commands[3][2]
    assert "--implementation-version v2" in commands[3][2]


def test_collect_remote_artifacts_passes_external_implementation_to_internal_run(tmp_path):
    commands: list[list[str]] = []

    def fake_runner(command):
        commands.append(command)
        if command[:2] == ["scp", "-r"] and command[-1].endswith("/profile"):
            profile_dir = tmp_path / "results" / "profile"
            profile_dir.mkdir(parents=True)
            (profile_dir / "op_summary.csv").write_text(
                "Op Name,Task Duration(us)\nlightning_indexer,1000\n"
            )
        if command[:2] == ["scp", "-r"] and command[-1].endswith("/perf"):
            perf_dir = tmp_path / "results" / "perf"
            perf_dir.mkdir(parents=True)
            (perf_dir / "benchmark.json").write_text(
                json.dumps(
                    {
                        "backend": "ascend",
                        "device_name": "Ascend 910B",
                    }
                )
                + "\n"
            )

    endpoint = RemoteEndpoint(
        name="ascend-a2",
        backend="ascend",
        host="user@ascend-host",
        workdir="/opt/cannbench",
        python="python3",
        env={},
    )
    prepared_input = tmp_path / "prepared.json"
    write_prepared_operator_input(
        prepared_input,
        build_prepared_operator_input(
            op="lightning_indexer",
            dtype="float16",
            dataset="smoke",
            case_id="tiny_decode_top4",
            seed=0,
        ),
    )

    collect_remote_artifacts(
        endpoint=endpoint,
        prepared_input=prepared_input,
        output_dir=tmp_path / "results",
        run_id="dsa-run",
        capture_output=False,
        profile_device_time=True,
        implementation="vllm_ascend",
        runner=fake_runner,
    )

    assert "--implementation vllm_ascend" in commands[2][2]


def test_collect_remote_artifacts_can_use_predeployed_simt_op_without_deploying(tmp_path):
    commands: list[list[str]] = []

    def fake_runner(command):
        commands.append(command)
        if command[:2] == ["scp", "-r"] and command[-1].endswith("/profile"):
            profile_dir = tmp_path / "results" / "profile"
            profile_dir.mkdir(parents=True)
            (profile_dir / "op_summary.csv").write_text(
                "Op Name,Task Duration(us)\nsoftmax,1000\n"
            )
        if command[:2] == ["scp", "-r"] and command[-1].endswith("/perf"):
            perf_dir = tmp_path / "results" / "perf"
            perf_dir.mkdir(parents=True)
            (perf_dir / "benchmark.json").write_text(
                json.dumps(
                    {
                        "backend": "ascend",
                        "device_name": "Ascend 910B",
                    }
                )
                + "\n"
            )

    endpoint = RemoteEndpoint(
        name="ascend-a2",
        backend="ascend",
        host="user@ascend-host",
        workdir="/opt/cannbench",
        python="python3",
        env={},
    )
    prepared_input = tmp_path / "prepared.json"
    _write_softmax_prepared(prepared_input)

    collect_remote_artifacts(
        endpoint=endpoint,
        prepared_input=prepared_input,
        output_dir=tmp_path / "results",
        run_id="softmax-run",
        capture_output=False,
        profile_device_time=True,
        implementation="simt",
        implementation_version="v2",
        runner=fake_runner,
    )

    assert commands[2] == [
        "ssh",
        "user@ascend-host",
        "cd /opt/cannbench && src/cannbench/operators/builtin/softmax/simt/v2/install.sh",
    ]
    assert (
        "CANNBENCH_SKIP_SIMT_INSTALL=1 msopprof --output=/opt/cannbench/.cannbench-runs/softmax-run/profile --launch-count=10 "
        "python3 -m cannbench internal-run --backend ascend --prepared-input .cannbench-runs/softmax-run/prepared.json "
        "--output-dir .cannbench-runs/softmax-run/perf --run-name benchmark --implementation simt --implementation-version v2"
    ) in commands[3][2]


def test_preinstall_remote_simt_op_runs_install_once(tmp_path):
    commands: list[list[str]] = []

    def fake_runner(command):
        commands.append(command)

    endpoint = RemoteEndpoint(
        name="ascend-a2",
        backend="ascend",
        host="user@ascend-host",
        workdir="/opt/cannbench",
        python="python3",
        env={},
    )

    preinstall_remote_simt_op(
        endpoint=endpoint,
        op="softmax",
        implementation_version="v3",
        runner=fake_runner,
    )

    assert commands == [
        [
            "ssh",
            "user@ascend-host",
            "cd /opt/cannbench && src/cannbench/operators/builtin/softmax/simt/v3/install.sh",
        ],
    ]


def test_collect_remote_artifacts_skips_simt_install_when_preinstalled(tmp_path):
    commands: list[list[str]] = []

    def fake_runner(command):
        commands.append(command)
        if command[:2] == ["scp", "-r"] and command[-1].endswith("/profile"):
            profile_dir = tmp_path / "results" / "profile"
            profile_dir.mkdir(parents=True)
            (profile_dir / "op_summary.csv").write_text(
                "Op Name,Task Duration(us)\nsoftmax,1000\n"
            )
        if command[:2] == ["scp", "-r"] and command[-1].endswith("/perf"):
            perf_dir = tmp_path / "results" / "perf"
            perf_dir.mkdir(parents=True)
            (perf_dir / "benchmark.json").write_text(
                json.dumps(
                    {
                        "backend": "ascend",
                        "device_name": "Ascend 910B",
                    }
                )
                + "\n"
            )

    endpoint = RemoteEndpoint(
        name="ascend-a2",
        backend="ascend",
        host="user@ascend-host",
        workdir="/opt/cannbench",
        python="python3",
        env={"ASCEND_VISIBLE_DEVICES": "0"},
    )
    prepared_input = tmp_path / "prepared.json"
    _write_softmax_prepared(prepared_input)

    collect_remote_artifacts(
        endpoint=endpoint,
        prepared_input=prepared_input,
        output_dir=tmp_path / "results",
        run_id="softmax-run",
        capture_output=False,
        profile_device_time=True,
        implementation="simt",
        implementation_version="v3",
        preinstalled_simt=True,
        runner=fake_runner,
    )

    assert commands == [
        [
            "ssh",
            "user@ascend-host",
            "mkdir -p /opt/cannbench/.cannbench-runs/softmax-run /opt/cannbench/.cannbench-runs/softmax-run/profile",
        ],
        [
            "scp",
            str(prepared_input),
            "user@ascend-host:/opt/cannbench/.cannbench-runs/softmax-run/prepared.json",
        ],
        [
            "ssh",
            "user@ascend-host",
            "cd /opt/cannbench && ASCEND_VISIBLE_DEVICES=0 CANNBENCH_SKIP_SIMT_INSTALL=1 msopprof --output=/opt/cannbench/.cannbench-runs/softmax-run/profile --launch-count=10 python3 -m cannbench internal-run --backend ascend --prepared-input .cannbench-runs/softmax-run/prepared.json --output-dir .cannbench-runs/softmax-run/perf --run-name benchmark --implementation simt --implementation-version v3",
        ],
        [
            "scp",
            "-r",
            "user@ascend-host:/opt/cannbench/.cannbench-runs/softmax-run/profile",
            str(tmp_path / "results" / "profile"),
        ],
        [
            "scp",
            "-r",
            "user@ascend-host:/opt/cannbench/.cannbench-runs/softmax-run/perf",
            str(tmp_path / "results" / "perf"),
        ],
    ]


def test_collect_remote_artifacts_runs_nvidia_ncu_profile(tmp_path, monkeypatch):
    commands: list[list[str]] = []

    def fake_runner(command):
        commands.append(command)
        if command[:2] == ["scp", "-r"] and command[-1].endswith("/profile"):
            profile_dir = tmp_path / "results" / "profile"
            profile_dir.mkdir(parents=True)
            (profile_dir / "ncu.csv").write_text(
                '"ID","Kernel Name","gpu__time_duration.avg"\n'
                '"","","usecond"\n'
                '"1","softmax","1000"\n'
            )
        if command[:2] == ["scp", "-r"] and command[-1].endswith("/perf"):
            perf_dir = tmp_path / "results" / "perf"
            perf_dir.mkdir(parents=True)
            (perf_dir / "benchmark.json").write_text(
                json.dumps(
                    {
                        "backend": "nvidia",
                        "device_name": "NVIDIA H800",
                    }
                )
                + "\n"
            )

    endpoint = RemoteEndpoint(
        name="nvidia-h100",
        backend="nvidia",
        host="user@nvidia-host",
        port=None,
        workdir="/opt/cannbench",
        python="python3",
        env={"CUDA_VISIBLE_DEVICES": "0"},
    )
    prepared_input = tmp_path / "prepared.json"
    _write_softmax_prepared(prepared_input)
    monkeypatch.setattr(
        "cannbench.core.remote.get_operator_plugin",
        lambda op: SimpleNamespace(
            profile_kernel_selection=lambda **kwargs: ProfileKernelSelection(
                kernel_name_patterns=(op,),
                launch_count=7,
            )
        ),
    )

    collect_remote_artifacts(
        endpoint=endpoint,
        prepared_input=prepared_input,
        output_dir=tmp_path / "results",
        run_id="softmax-run",
        capture_output=False,
        profile_device_time=True,
        runner=fake_runner,
    )

    assert commands[2][0:2] == [
        "ssh",
        "user@nvidia-host",
    ]
    command = commands[2][2]
    assert "ncu --target-processes all --force-overwrite --launch-count 7 --csv" in command
    assert "python3 -m cannbench internal-run --backend nvidia" in command


def test_collect_remote_artifacts_can_summarize_downloaded_profile(tmp_path):
    def fake_runner(command):
        if command[:2] == ["scp", "-r"] and command[-1].endswith("/profile"):
            profile_dir = tmp_path / "results" / "profile"
            profile_dir.mkdir(parents=True)
            (profile_dir / "op_summary.csv").write_text(
                "Op Name,Task Duration(us)\nsoftmax,1000\n"
            )
        if command[:2] == ["scp", "-r"] and command[-1].endswith("/perf"):
            perf_dir = tmp_path / "results" / "perf"
            perf_dir.mkdir(parents=True)
            (perf_dir / "benchmark.json").write_text(
                json.dumps(
                    {
                        "backend": "ascend",
                        "device_name": "Ascend 910B",
                    }
                )
                + "\n"
            )

    endpoint = RemoteEndpoint(
        name="ascend-a2",
        backend="ascend",
        host="user@ascend-host",
        port=None,
        workdir="/opt/cannbench",
        python="python3",
        env={},
    )
    prepared_input = tmp_path / "prepared.json"
    _write_softmax_prepared(prepared_input)

    collect_remote_artifacts(
        endpoint=endpoint,
        prepared_input=prepared_input,
        output_dir=tmp_path / "results",
        run_id="softmax-run",
        capture_output=False,
        profile_device_time=True,
        summarize_profile=True,
        runner=fake_runner,
    )

    summary = tmp_path / "results" / "profile-summary.json"
    assert summary.is_file()
    assert '"latency_ms": 1.0' in summary.read_text()


def test_collect_remote_artifacts_returns_unified_artifacts(tmp_path):
    def fake_runner(command):
        if command[:2] == ["scp", "-r"] and command[-1].endswith("/profile"):
            profile_dir = tmp_path / "results" / "profile"
            profile_dir.mkdir(parents=True)
            (profile_dir / "op_summary.csv").write_text(
                "Op Name,Task Duration(us)\nsoftmax,1000\n"
            )
        if command[:2] == ["scp", "-r"] and command[-1].endswith("/perf"):
            perf_dir = tmp_path / "results" / "perf"
            perf_dir.mkdir(parents=True)
            (perf_dir / "benchmark.json").write_text(
                json.dumps(
                    {
                        "backend": "ascend",
                        "device_name": "Ascend 910B",
                    }
                )
                + "\n"
            )
        if command[:2] == ["scp", "-r"] and command[-1].endswith("/output"):
            output_dir = tmp_path / "results" / "output"
            output_dir.mkdir(parents=True)
            (output_dir / "tensor.json").write_text("{}")

    endpoint = RemoteEndpoint(
        name="ascend-a2",
        backend="ascend",
        host="user@ascend-host",
        port=None,
        workdir="/opt/cannbench",
        python="python3",
        env={},
    )
    prepared_input = tmp_path / "prepared.json"
    _write_softmax_prepared(prepared_input)

    result = collect_remote_artifacts(
        endpoint=endpoint,
        prepared_input=prepared_input,
        output_dir=tmp_path / "results",
        run_id="softmax-run",
        capture_output=True,
        profile_device_time=True,
        summarize_profile=True,
        runner=fake_runner,
    )

    assert result.artifacts.output_artifacts == (("tensor.json", b"{}"),)
    assert result.artifacts.profile is not None
    assert result.artifacts.profile.device_name == "Ascend 910B"
    assert result.artifacts.profile.profile_summary.backend == "ascend"
    assert result.artifacts.profile.profile_artifacts[0][0] == "op_summary.csv"
    assert result.artifacts.profile.perf_artifacts[0][0] == "benchmark.json"


def test_collect_remote_artifacts_rejects_unexpected_profile_kernel(tmp_path):
    def fake_runner(command):
        if command[:2] == ["scp", "-r"] and command[-1].endswith("/profile"):
            profile_dir = tmp_path / "results" / "profile"
            profile_dir.mkdir(parents=True)
            (profile_dir / "op_summary.csv").write_text(
                "Op Name,Task Duration(us)\nStatelessRandomNormalV3,1000\n"
            )
        if command[:2] == ["scp", "-r"] and command[-1].endswith("/perf"):
            perf_dir = tmp_path / "results" / "perf"
            perf_dir.mkdir(parents=True)
            (perf_dir / "benchmark.json").write_text(
                json.dumps(
                    {
                        "backend": "ascend",
                        "device_name": "Ascend 910B",
                    }
                )
                + "\n"
            )

    endpoint = RemoteEndpoint(
        name="ascend-a2",
        backend="ascend",
        host="user@ascend-host",
        port=None,
        workdir="/opt/cannbench",
        python="python3",
        env={},
    )
    prepared_input = tmp_path / "prepared.json"
    _write_softmax_prepared(prepared_input)

    with pytest.raises(ValueError, match="expected profiler kernel"):
        collect_remote_artifacts(
            endpoint=endpoint,
            prepared_input=prepared_input,
            output_dir=tmp_path / "results",
            run_id="softmax-run",
            capture_output=False,
            profile_device_time=True,
            runner=fake_runner,
        )


def test_collect_remote_artifacts_uses_ssh_and_scp_port(tmp_path):
    commands: list[list[str]] = []

    def fake_runner(command):
        commands.append(command)

    endpoint = RemoteEndpoint(
        name="ascend-a2",
        backend="ascend",
        host="root@121.41.199.170",
        port=20002,
        workdir="/home/y00621698/cannbench",
        python="python3",
        env={},
    )
    prepared_input = tmp_path / "prepared.json"
    _write_softmax_prepared(prepared_input)

    collect_remote_artifacts(
        endpoint=endpoint,
        prepared_input=prepared_input,
        output_dir=tmp_path / "results",
        run_id="softmax-run",
        capture_output=True,
        runner=fake_runner,
    )

    assert commands[0][:4] == ["ssh", "-p", "20002", "root@121.41.199.170"]
    assert commands[1][:3] == ["scp", "-P", "20002"]
    assert commands[2][:4] == ["ssh", "-p", "20002", "root@121.41.199.170"]
    assert commands[3][:4] == ["scp", "-P", "20002", "-r"]
