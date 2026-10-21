from __future__ import annotations

import json
import os
import threading
from datetime import datetime

import pytest

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


def test_check_cameras_uses_local_info_when_not_on_hub(make_dataset, lerobot_home, monkeypatch):
    make_dataset("local/ds", cams=("front",))
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


def test_follow_mirrors_running_then_failed_into_job_json(lerobot_home):
    run, _ = make_run(lerobot_home)
    states = []
    real_mark = run.mark

    def spy(state, error=None):
        real_mark(state, error)
        states.append(json.loads((run.plan.job_dir / "job.json").read_text()))

    run.mark = spy
    run.follow(ScriptedFollower([], ["@@STAGE train@@"]))
    running, final = states[0], states[-1]
    assert running["state"] == "running" and running["runner"] == "local"
    assert running["process_pid"] == os.getpid() and running["ended_at"] is None
    assert final["state"] == "failed" and final["process_pid"] is None
    assert final["error_message"] == "pod disappeared"


def legacy_pod(**env):
    base = {
        "DATASET": "sysradium/second_try_cam1",
        "REPO": "sysradium/act_second_try_cam1_20261021_2239",
        "STEPS": "30000",
        "BATCH": "32",
        "MAX_HOURS": "4",
        "DEADLINE": "1792629546",
        "LOG_FREQ": "25",
        "LEROBOT_REF": "30da8e68",
        "POLICY_ARGS": "--policy.type=act",
    }
    base.update(env)
    return {
        "id": "q2x44ipzuu0xo4",
        "gpuId": "NVIDIA GeForce RTX 3090",
        "costPerHr": 0.22,
        "createdAt": "2026-10-21 20:39:18.98 +0000 UTC",
        "env": base,
    }


def test_plan_from_pod_uses_stored_job_id_policy_and_cloud(lerobot_home):
    pod = legacy_pod(JOB_ID="act_runpod_second_try_cam1_2026-10-21_22-39-17", POLICY="act", CLOUD="SECURE")
    p = runner.plan_from_pod(pod)
    assert p.job_id == "act_runpod_second_try_cam1_2026-10-21_22-39-17"
    assert (p.policy.name, p.cloud, p.steps, p.batch, p.log_freq) == ("act", "SECURE", 30000, 32, 25)
    assert p.deadline == 1792629546 and p.repo.endswith("_2239")


def test_plan_from_legacy_pod_infers_policy_cloud_and_job_dir(lerobot_home):
    created = runner._pod_created(legacy_pod())
    stamp = created.replace(second=0) if created.second >= 10 else created
    existing = lerobot_home / "outputs/train" / f"act_runpod_second_try_cam1_{stamp:%Y-%m-%d_%H-%M-%S}"
    existing.mkdir(parents=True)
    offers = [runner.runpod.GpuOffer("NVIDIA GeForce RTX 3090", 24, True, 0.46, 0.22, "Low")]
    p = runner.plan_from_pod(legacy_pod(), offers)
    assert p.policy.name == "act"
    assert p.cloud == "COMMUNITY"
    assert p.job_id == existing.name


def test_plan_from_pod_maps_base_model_back_to_policy(lerobot_home):
    pod = legacy_pod(POLICY_ARGS="--policy.path=lerobot/smolvla_base --policy.scheduler_decay_steps=10000")
    assert runner.plan_from_pod(pod).policy.name == "smolvla"


def test_plan_from_foreign_pod_is_rejected():
    with pytest.raises(ValueError, match="not created by lerobot-runpod"):
        runner.plan_from_pod({"id": "x", "env": {"FOO": "1"}})


def test_active_follower_ignores_self_and_dead_pids(lerobot_home):
    run, _ = make_run(lerobot_home)
    run.mark("running")
    assert runner.active_follower(run.plan.job_dir) is None  # our own pid
    record = json.loads((run.plan.job_dir / "job.json").read_text())
    record["process_pid"] = os.getppid()
    (run.plan.job_dir / "job.json").write_text(json.dumps(record))
    assert runner.active_follower(run.plan.job_dir) == os.getppid()
    record["process_pid"] = 2**22 + 12345
    (run.plan.job_dir / "job.json").write_text(json.dumps(record))
    assert runner.active_follower(run.plan.job_dir) is None


class TerminatedRun:
    def __init__(self, train_plan):
        self.plan = train_plan
        self.pod_id = "pod1"
        self.stop = threading.Event()
        self.log_path = "/tmp/log.jsonl"
        self.deleted = False
        self.marks = []

    def follow(self):
        raise cli.Terminated()

    def mark(self, state, error=None):
        self.marks.append((state, error))

    def delete_pod(self):
        self.deleted = True
        self.pod_id = None

    def kept_message(self):
        return "kept"


def test_sigterm_from_lelab_deletes_the_pod_without_asking(lerobot_home, monkeypatch):
    run, _ = make_run(lerobot_home)
    fake = TerminatedRun(run.plan)

    def no_prompt(question, *, default=False, assume_yes=False):
        assert assume_yes, "must not prompt on SIGTERM"
        return True

    monkeypatch.setattr(cli, "confirm", no_prompt)
    assert cli.follow_and_deliver(fake, assume_yes=False) == 1
    assert fake.deleted
    assert fake.marks == [("interrupted", "stopped from LeLab")]


def test_attach_refuses_when_another_process_follows(lerobot_home, monkeypatch):
    run, _ = make_run(lerobot_home)
    monkeypatch.setattr(cli.runpod, "get_pod", lambda pod_id: {"id": pod_id})
    monkeypatch.setattr(cli.runpod, "list_gpus", lambda: [])
    monkeypatch.setattr(cli.runner, "plan_from_pod", lambda pod, offers: run.plan)
    monkeypatch.setattr(cli.runner, "active_follower", lambda job_dir: 4242)
    with pytest.raises(cli.Abort, match="pid 4242 is already following"):
        cli.cmd_attach(cli.build_parser().parse_args(["attach", "pod1"]))
