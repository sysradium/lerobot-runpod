from __future__ import annotations

import base64
import json
import subprocess
from datetime import datetime

import pytest

from lerobot_runpod import plan, policies

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
    assert env["FAIL_GRACE_S"] == "1800"
    assert all(isinstance(v, str) for v in env.values())
    boot = base64.b64decode(env["BOOT"]).decode()
    assert boot == plan.bootstrap_script()
    assert "@@STAGE train@@" in boot and "podTerminate" in boot


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
    assert (
        plan.default_repo("me", "act", "ds", datetime(2026, 10, 18, 21, 41)) == "me/act_ds_20261018_2141"
    )


def test_lerobot_ref_without_checkout():
    ref = plan.resolve_lerobot_ref(None)
    assert ref.ref == "main"
    assert ref.warning
