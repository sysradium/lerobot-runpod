from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

from lerobot_runpod import deliver

LELAB_PYTHON = Path.home() / ".local/share/uv/tools/lelab/bin/python"


def make_info(**overrides):
    kwargs = dict(
        job_id="act_runpod_second_try_2026-10-18_21-41-54",
        name="ACT · sysradium/second_try · RunPod NVIDIA GeForce RTX 3090",
        policy="act",
        dataset="sysradium/second_try",
        repo="sysradium/act_second_try_20261018_2141",
        steps=200,
        batch=32,
        save_freq=40,
        started_at=1792352517,
        ended_at=1792352724,
        loss=3.15,
        lr=1e-05,
        grad_norm=55.364,
    )
    kwargs.update(overrides)
    return deliver.JobInfo(**kwargs)


def test_job_id_format():
    assert (
        deliver.job_id("smolvla", "ds", datetime(2026, 10, 18, 22, 57, 30))
        == "smolvla_runpod_ds_2026-10-18_22-57-30"
    )


def test_job_record_matches_lelab_expectations(tmp_path):
    record = deliver.job_record(make_info(), tmp_path / "job")
    assert record["state"] == "done"
    assert record["runner"] == "local"
    assert record["process_pid"] is None
    assert record["exit_code"] == 0
    assert record["hf_repo_id"] is None
    assert record["output_dir"] == str(tmp_path / "job" / "run")
    assert record["metrics"] == {
        "current_step": 200,
        "total_steps": 200,
        "current_loss": 3.15,
        "current_lr": 1e-05,
        "grad_norm": 55.364,
        "eta_seconds": 0,
    }
    assert record["config"]["policy_repo_id"] == "sysradium/act_second_try_20261018_2141"
    assert record["config"]["policy_base_model_path"] is None
    assert record["config"]["log_freq"] == 100
    assert deliver.job_record(make_info(log_freq=25), tmp_path / "job")["config"]["log_freq"] == 25


def fake_download(files):
    def download(repo_id, local_dir):
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / ".cache").mkdir()
        for name, content in files.items():
            (local_dir / name).write_text(json.dumps(content))

    return download


def test_deliver_lays_out_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(deliver.hf, "download_repo", fake_download({"config.json": {"type": "act"}}))
    job_dir = tmp_path / "job"
    ckpt = deliver.deliver(make_info(), job_dir)
    assert ckpt == job_dir / "run/checkpoints/000200/pretrained_model"
    assert (ckpt / "config.json").is_file()
    assert not (ckpt / ".cache").exists()
    last = job_dir / "run/checkpoints/last"
    assert last.is_symlink() and str(last.readlink()) == "000200"
    assert json.loads((job_dir / "job.json").read_text())["id"] == make_info().job_id


def test_deliver_fails_without_config(tmp_path, monkeypatch):
    monkeypatch.setattr(deliver.hf, "download_repo", fake_download({}))
    with pytest.raises(RuntimeError, match="download incomplete"):
        deliver.deliver(make_info(), tmp_path / "job")


def test_fetch_reads_train_config(lerobot_home, monkeypatch):
    files = {
        "config.json": {"type": "smolvla"},
        "train_config.json": {
            "steps": 10000,
            "batch_size": 32,
            "save_freq": 2000,
            "dataset": {"repo_id": "sysradium/second_try_cam1"},
            "policy": {"type": "smolvla", "pretrained_path": "lerobot/smolvla_base"},
        },
    }
    monkeypatch.setattr(deliver.hf, "download_repo", fake_download(files))
    job_dir = deliver.fetch("sysradium/smolvla_x")
    assert job_dir.name.startswith("smolvla_runpod_second_try_cam1_")
    record = json.loads((job_dir / "job.json").read_text())
    assert record["config"]["steps"] == 10000
    assert record["config"]["policy_base_model_path"] == "lerobot/smolvla_base"
    assert (job_dir / "run/checkpoints/010000/pretrained_model/config.json").is_file()
    assert not list((lerobot_home / "outputs/train").glob(".fetch_*"))


def test_append_log(tmp_path):
    path = tmp_path / "log.jsonl"
    deliver.append_log(path, "hello", 1.5)
    deliver.append_log(path, "world")
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines[0] == {"timestamp": 1.5, "message": "hello"}
    assert lines[1]["message"] == "world"


@pytest.mark.skipif(not LELAB_PYTHON.exists(), reason="LeLab is not installed as a uv tool")
def test_job_record_validates_against_lelab(tmp_path):
    record = deliver.job_record(make_info(base_model=None), tmp_path / "job")
    code = "import sys; from lelab.jobs import JobRecord; JobRecord.model_validate_json(sys.stdin.read())"
    proc = subprocess.run(
        [str(LELAB_PYTHON), "-c", code], input=json.dumps(record), capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr


def test_running_record_points_lelab_at_the_follower(tmp_path):
    record = deliver.job_record(
        make_info(current_step=1200), tmp_path / "job", state="running", process_pid=4242
    )
    assert (record["state"], record["runner"], record["process_pid"]) == ("running", "local", 4242)
    assert record["ended_at"] is None and record["exit_code"] is None
    assert record["metrics"]["current_step"] == 1200 and record["metrics"]["total_steps"] == 200


def test_failed_record_keeps_the_reason(tmp_path):
    path = deliver.write_job(
        make_info(), tmp_path / "job", state="failed", process_pid=4242, error="pod disappeared"
    )
    record = json.loads(path.read_text())
    assert (record["state"], record["exit_code"], record["process_pid"]) == ("failed", 1, None)
    assert record["error_message"] == "pod disappeared"


@pytest.mark.skipif(not LELAB_PYTHON.exists(), reason="LeLab not installed")
@pytest.mark.parametrize("state", ["running", "failed", "interrupted"])
def test_live_records_validate_against_lelab(tmp_path, state):
    record = deliver.job_record(make_info(current_step=5), tmp_path / "job", state=state, process_pid=4242)
    code = "import sys; from lelab.jobs import JobRecord; JobRecord.model_validate_json(sys.stdin.read())"
    proc = subprocess.run(
        [str(LELAB_PYTHON), "-c", code], input=json.dumps(record), capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
