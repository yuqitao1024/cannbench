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
