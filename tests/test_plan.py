from __future__ import annotations

import base64
import json
import subprocess
from datetime import datetime

import pytest

from lerobot_runpod import config, plan, policies

TOKEN = "hf_FAKEtoken1234567890"


def make_plan(**overrides):
    kwargs = dict(
        dataset="local/second_try",
        hub_dataset="sysradium/second_try",
        policy=policies.get("smolvla"),
        steps=10000,
        batch=32,
        max_hours=6,
        gpu_id="NVIDIA GeForce RTX 3090",
        cloud="community",
        repo="sysradium/smolvla_second_try_20261018_2257",
        lerobot_ref="b9cb121c0000",
        created_at=datetime(2026, 10, 18, 22, 57, 30),
    )
    kwargs.update(overrides)
    return plan.TrainPlan(**kwargs)


def test_derived_names(lerobot_home):
    p = make_plan()
    assert p.cloud == "COMMUNITY"
    assert p.save_freq == 2000
    assert p.job_id == "smolvla_runpod_second_try_2026-10-18_22-57-30"
    assert p.job_dir == lerobot_home / "outputs/train" / p.job_id
    assert p.pod_name == "lerobot-smolvla-20261018_2257"
    assert p.deadline == int(datetime(2026, 10, 18, 22, 57, 30).timestamp()) + 6 * 3600


def test_save_freq_never_zero():
    assert make_plan(steps=3).save_freq == 3


@pytest.mark.parametrize("field", ["steps", "batch", "max_hours"])
def test_rejects_non_positive(field):
    with pytest.raises(ValueError):
        make_plan(**{field: 0})


def test_pod_env_carries_bootstrap_and_params():
    env = make_plan().pod_env(TOKEN)
    assert env["HF_TOKEN"] == TOKEN
    assert env["DATASET"] == "sysradium/second_try"
    assert env["POLICY_ARGS"] == (
        "--policy.path=lerobot/smolvla_base --policy.scheduler_decay_steps=10000 "
        "--policy.freeze_vision_encoder=false --policy.train_expert_only=false"
    )
    assert env["EXTRAS"] == "training,smolvla"
    assert env["SAVE_FREQ"] == "2000"
    assert env["LOG_FREQ"] == str(plan.DEFAULT_LOG_FREQ)
    assert (env["JOB_ID"], env["POLICY"], env["CLOUD"]) == (make_plan().job_id, "smolvla", "COMMUNITY")
    assert env["FAIL_GRACE_S"] == "1800"
    assert all(isinstance(v, str) for v in env.values())
    boot = base64.b64decode(env["BOOT"]).decode()
    assert boot == plan.bootstrap_script()
    assert "@@STAGE train@@" in boot and "podTerminate" in boot


def test_log_freq_is_passed_to_lerobot_train(lerobot_home):
    assert make_plan(log_freq=10).pod_env(TOKEN)["LOG_FREQ"] == "10"
    assert '--log_freq="${LOG_FREQ:-100}"' in plan.bootstrap_script()


def test_log_freq_must_be_positive():
    with pytest.raises(ValueError, match="log freq"):
        make_plan(log_freq=0)


def test_bootstrap_is_valid_bash():
    subprocess.run(["bash", "-n"], input=plan.bootstrap_script(), text=True, check=True)


def test_pod_command_redaction():
    p = make_plan()
    cmd = p.pod_command(TOKEN)
    env = json.loads(cmd[cmd.index("--env") + 1])
    assert env["HF_TOKEN"] == TOKEN
    shown = plan.format_command(cmd, TOKEN)
    assert TOKEN not in shown
    assert plan.TOKEN_PLACEHOLDER in shown
    assert "--gpu-id 'NVIDIA GeForce RTX 3090'" in shown


def test_with_fresh_deadline_moves_forward():
    p = make_plan(created_at=datetime(2020, 1, 1))
    assert p.with_fresh_deadline().deadline > p.deadline


def test_default_repo():
    assert plan.default_repo("me", "act", "ds", datetime(2026, 10, 18, 21, 41)) == "me/act_ds_20261018_2141"


def test_lerobot_ref_without_checkout():
    ref = plan.resolve_lerobot_ref(None)
    assert ref.ref == "main"
    assert ref.warning


def _fake_lelab(tmp_path, dist_name, direct_url=None):
    dist = tmp_path / "lelab/lib/python3.13/site-packages" / dist_name
    dist.mkdir(parents=True)
    if direct_url is not None:
        (dist / "direct_url.json").write_text(json.dumps(direct_url))
    return tmp_path / "lelab"


def test_lelab_lerobot_ref_reads_git_commit(tmp_path):
    tool = _fake_lelab(
        tmp_path,
        "lerobot-0.6.0.dist-info",
        {"url": "https://github.com/huggingface/lerobot.git", "vcs_info": {"commit_id": "30da8e68abc"}},
    )
    assert config.lelab_lerobot_ref(tool) == "30da8e68abc"


def test_lelab_lerobot_ref_falls_back_to_release_tag(tmp_path):
    assert config.lelab_lerobot_ref(_fake_lelab(tmp_path, "lerobot-0.6.1.dist-info")) == "v0.6.1"


def test_lelab_lerobot_ref_without_lelab(tmp_path):
    assert config.lelab_lerobot_ref(tmp_path / "missing") is None


def test_default_ref_prefers_lelab(tmp_path, monkeypatch):
    tool = _fake_lelab(tmp_path, "lerobot-0.6.0.dist-info", {"vcs_info": {"commit_id": "30da8e68abc"}})
    monkeypatch.setenv("LELAB_TOOL_DIR", str(tool))
    ref = plan.default_lerobot_ref(None)
    assert ref.ref == "30da8e68abc" and "LeLab" in (ref.note or "")


def test_default_ref_without_lelab_uses_local_resolution(tmp_path, monkeypatch):
    monkeypatch.setenv("LELAB_TOOL_DIR", str(tmp_path / "missing"))
    ref = plan.default_lerobot_ref(None)
    assert ref.ref == config.FALLBACK_LEROBOT_REF and ref.warning


def test_bootstrap_only_requests_bf16_when_supported():
    boot = plan.bootstrap_script()
    assert "--accelerator.mixed_precision=bf16 \\" not in boot
    assert 'grep -q -- "--accelerator.mixed_precision"' in boot
    assert '"${precision[@]}"' in boot
    assert "export ACCELERATE_MIXED_PRECISION=bf16" in boot
    assert "is_bf16_supported" in boot
