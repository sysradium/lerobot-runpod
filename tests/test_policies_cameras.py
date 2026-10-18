from __future__ import annotations

import pytest

from lerobot_runpod import cameras, policies


def test_scratch_policy_args():
    assert policies.get("act").train_args(200) == ["--policy.type=act"]
    assert policies.get("diffusion").extras_spec == "training,diffusion"


def test_base_model_policy_args():
    assert policies.get("smolvla").train_args(10000) == [
        "--policy.path=lerobot/smolvla_base",
        "--policy.scheduler_decay_steps=10000",
        "--policy.freeze_vision_encoder=false",
        "--policy.train_expert_only=false",
    ]
    pi0 = policies.get("pi0")
    assert pi0.train_args(5) == ["--policy.path=lerobot/pi0_base", "--policy.scheduler_decay_steps=5"]
    assert (pi0.min_vram_gb, pi0.extras_spec) == (80, "training,pi")


def test_policy_table_defaults():
    table = {p.name: (p.steps, p.batch, p.min_vram_gb, p.max_hours) for p in policies.POLICIES.values()}
    assert table == {
        "act": (50000, 32, 24, 4),
        "smolvla": (20000, 32, 24, 6),
        "diffusion": (100000, 64, 24, 8),
        "pi0": (30000, 32, 80, 8),
    }


def test_unknown_policy():
    with pytest.raises(ValueError, match="unsupported policy"):
        policies.get("gr00t")


MODEL = {
    "input_features": {
        "observation.state": {"type": "STATE", "shape": [6]},
        "observation.images.camera1": {"type": "VISUAL", "shape": [3, 256, 256]},
        "observation.images.camera2": {"type": "VISUAL", "shape": [3, 256, 256]},
        "observation.images.camera3": {"type": "VISUAL", "shape": [3, 256, 256]},
    }
}


def info(*cams, dtype="video"):
    features = {"action": {"dtype": "float32"}}
    features.update({f"observation.images.{c}": {"dtype": dtype} for c in cams})
    return {"features": features}


@pytest.mark.parametrize(
    ("cams", "ok"),
    [
        (("camera1",), True),
        (("camera1", "camera2", "camera3"), True),
        (("camera1", "camera2", "camera3", "camera4"), True),
        (("front",), False),
        (("camera1", "wrist"), False),
    ],
)
def test_camera_compare(cams, ok):
    assert cameras.compare(MODEL, info(*cams)).ok is ok


def test_camera_compare_accepts_image_features():
    check = cameras.compare(MODEL, info("camera2", dtype="image"))
    assert check.dataset == ["observation.images.camera2"]
    assert check.ok


def test_preflight_raises_with_both_sets(monkeypatch):
    def fake_download(repo_id, filename, repo_type="model"):
        return MODEL if repo_type == "model" else info("front")

    monkeypatch.setattr(cameras.hf, "download_json", fake_download)
    with pytest.raises(cameras.CameraMismatch, match="observation.images.front"):
        cameras.preflight("lerobot/smolvla_base", "me/ds")


def test_rename_command_uses_packaged_script(tmp_path):
    cmd = cameras.rename_command(tmp_path / "a", tmp_path / "b", "front", "camera1")
    assert cmd[-5].endswith("rename_camera.py")
    assert cmd[-2:] == ["front", "camera1"]
