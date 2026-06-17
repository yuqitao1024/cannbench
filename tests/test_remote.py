import json

from cannbench.core.remote import (
    RemoteEndpoint,
    collect_remote_artifacts,
    read_remote_endpoint,
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
        workdir="/opt/cannbench",
        python="python3",
        env={"ASCEND_VISIBLE_DEVICES": "0"},
    )


def test_collect_remote_artifacts_runs_capture_and_downloads_output(tmp_path):
    commands: list[list[str]] = []

    def fake_runner(command):
        commands.append(command)

    endpoint = RemoteEndpoint(
        name="ascend-a2",
        backend="ascend",
        host="user@ascend-host",
        workdir="/opt/cannbench",
        python="python3",
        env={"ASCEND_VISIBLE_DEVICES": "0"},
    )
    prepared_input = tmp_path / "prepared.json"
    prepared_input.write_text("{}")

    result = collect_remote_artifacts(
        endpoint=endpoint,
        prepared_input=prepared_input,
        output_dir=tmp_path / "results",
        run_id="softmax-run",
        capture_output=True,
        deploy_custom_op=True,
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
            "cd /opt/cannbench && ASCEND_VISIBLE_DEVICES=0 python3 -m cannbench capture-output --backend ascend --prepared-input .cannbench-runs/softmax-run/prepared.json --output .cannbench-runs/softmax-run/output --deploy-custom-op",
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

    endpoint = RemoteEndpoint(
        name="ascend-a2",
        backend="ascend",
        host="user@ascend-host",
        workdir="/opt/cannbench",
        python="python3",
        env={"ASCEND_VISIBLE_DEVICES": "0"},
    )
    prepared_input = tmp_path / "prepared.json"
    prepared_input.write_text("{}")

    collect_remote_artifacts(
        endpoint=endpoint,
        prepared_input=prepared_input,
        output_dir=tmp_path / "results",
        run_id="softmax-run",
        capture_output=False,
        profile_device_time=True,
        warmup=3,
        iterations=5,
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
            "cd /opt/cannbench && ASCEND_VISIBLE_DEVICES=0 msprof op --output=/opt/cannbench/.cannbench-runs/softmax-run/profile python3 -m cannbench operator --backend ascend --prepared-input .cannbench-runs/softmax-run/prepared.json --warmup 3 --iterations 5 --output-dir .cannbench-runs/softmax-run/perf --run-name benchmark",
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


def test_collect_remote_artifacts_runs_nvidia_ncu_profile(tmp_path):
    commands: list[list[str]] = []

    def fake_runner(command):
        commands.append(command)

    endpoint = RemoteEndpoint(
        name="nvidia-h100",
        backend="nvidia",
        host="user@nvidia-host",
        workdir="/opt/cannbench",
        python="python3",
        env={"CUDA_VISIBLE_DEVICES": "0"},
    )
    prepared_input = tmp_path / "prepared.json"
    prepared_input.write_text("{}")

    collect_remote_artifacts(
        endpoint=endpoint,
        prepared_input=prepared_input,
        output_dir=tmp_path / "results",
        run_id="softmax-run",
        capture_output=False,
        profile_device_time=True,
        warmup=3,
        iterations=5,
        runner=fake_runner,
    )

    assert commands[2] == [
        "ssh",
        "user@nvidia-host",
        "cd /opt/cannbench && CUDA_VISIBLE_DEVICES=0 ncu --target-processes all --force-overwrite --csv --log-file /opt/cannbench/.cannbench-runs/softmax-run/profile/ncu.csv --export /opt/cannbench/.cannbench-runs/softmax-run/profile/ncu-report python3 -m cannbench operator --backend nvidia --prepared-input .cannbench-runs/softmax-run/prepared.json --warmup 3 --iterations 5 --output-dir .cannbench-runs/softmax-run/perf --run-name benchmark",
    ]


def test_collect_remote_artifacts_can_summarize_downloaded_profile(tmp_path):
    def fake_runner(command):
        if command[:2] == ["scp", "-r"] and command[-1].endswith("/profile"):
            profile_dir = tmp_path / "results" / "profile"
            profile_dir.mkdir(parents=True)
            (profile_dir / "op_summary.csv").write_text(
                "Op Name,Task Duration(us)\nsoftmax,1000\n"
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
    prepared_input.write_text("{}")

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
    assert '"latency_ms_avg": 1.0' in summary.read_text()
