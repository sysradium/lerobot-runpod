from __future__ import annotations

import json
from datetime import datetime

import pytest
from conftest import make_dataset

from lerobot_runpod import cli, policies, runner


class ScriptedFollower:
    def __init__(self, system, container, started=True):
        self.system = system
        self.container = container
        self.started = started

    def wait_for_container(self, on_line):
        for line in self.system:
            on_line(line)
        return self.started

    def follow_container(self, on_line):
        for line in self.container:
            if on_line(line):
                return "finished"
        return "pod_gone"


def make_run(lerobot_home):
    p = runner.build_plan(
        dataset="local/ds",
        hub_dataset="me/ds",
        policy=policies.get("act"),
        gpu_id="NVIDIA GeForce RTX 3090",
        cloud="COMMUNITY",
        hf_user="me",
        steps=200,
        lerobot_ref="abc",
        now=datetime(2026, 10, 18, 21, 41, 54),
    )
    events = []
    run = runner.TrainingRun(p, "hf_x", events.append)
    run.pod_id = "pod1"
    return run, events


def test_follow_to_done_tracks_state_and_writes_lelab_log(lerobot_home):
    run, events = make_run(lerobot_home)
    follower = ScriptedFollower(
        ["start container for runpod/pytorch: begin"],
        [
            "@@STAGE train@@",
            "\rTraining: 100%|█| 200/200 [03:01<00:00,  1.10step/s]"
            "INFO x step:200 loss:3.150 grdn:55.364 lr:1.0e-05",
            "@@DONE@@",
            "never seen",
        ],
    )
    assert run.follow(follower) == "done"
    assert run.state.stage == "done"
    assert run.state.progress.step == 200
    assert run.state.metrics.loss == 3.15
    assert any(isinstance(e, runner.SystemLine) for e in events)
    lines = [json.loads(line)["message"] for line in run.log_path.read_text().splitlines()]
    assert lines[0] == "@@STAGE train@@"
    assert lines[-1] == "@@DONE@@"
    assert "never seen" not in lines

    info = run.job_info()
    assert info.job_id == "act_runpod_ds_2026-10-18_21-41-54"
    assert (info.loss, info.lr, info.grad_norm) == (3.15, 1e-05, 55.364)
    assert info.name == "ACT · me/ds · RunPod NVIDIA GeForce RTX 3090"


def test_follow_reports_failure_and_keeps_pod(lerobot_home):
    run, _ = make_run(lerobot_home)
    result = run.follow(ScriptedFollower([], ["@@STAGE install@@", "@@FAILED install@@"]))
    assert result == "failed: install"
    assert run.pod_id == "pod1"
    assert "runpodctl pod delete pod1" in run.kept_message()


def test_follow_pod_gone(lerobot_home):
    run, _ = make_run(lerobot_home)
    assert run.follow(ScriptedFollower([], [], started=False)) == "pod disappeared"


def test_check_cameras_uses_local_info_when_not_on_hub(lerobot_home, monkeypatch):
    make_dataset(lerobot_home, "local/ds", cams=("front",))
    status = runner.DatasetStatus("local/ds", "me/ds", False, lerobot_home / "local/ds")
    model = {"input_features": {"observation.images.camera1": {"type": "VISUAL"}}}
    monkeypatch.setattr(runner.hf, "download_json", lambda *a, **k: model)
    with pytest.raises(runner.cameras.CameraMismatch):
        runner.check_cameras(policies.get("smolvla"), status)
    assert runner.check_cameras(policies.get("act"), status) is None


def test_teardown_defaults_to_delete(lerobot_home, monkeypatch):
    run, _ = make_run(lerobot_home)
    deleted = []
    monkeypatch.setattr(runner.runpod, "delete_pod", lambda pod: deleted.append(pod) or True)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    cli.teardown(run, assume_yes=False)
    assert deleted == ["pod1"]
    assert run.pod_id is None
