from __future__ import annotations

from lerobot_runpod import datasets, hf


def test_discover_sorts_newest_first_and_skips_reserved_dirs(make_dataset, lerobot_home):
    make_dataset("sysradium/old", mtime=1_000)
    make_dataset("local/new", episodes=50, frames=26903, fps=30, mtime=2_000)
    for reserved in ("outputs", "calibration", "ports", "robots", "inference_logs"):
        make_dataset(f"{reserved}/thing", mtime=3_000)
    (lerobot_home / "broken/ds/meta").mkdir(parents=True)
    (lerobot_home / "broken/ds/meta/info.json").write_text("{not json")

    found = datasets.discover()

    assert [d.repo_id for d in found] == ["local/new", "sysradium/old"]
    assert (found[0].episodes, found[0].frames, found[0].fps) == (50, 26903, 30)
    assert found[0].name == "new"
    assert found[0].path == lerobot_home / "local/new"


def test_discover_empty_home(tmp_path):
    assert datasets.discover(tmp_path / "nope") == []


def test_hub_id_maps_local_owner_to_hf_user():
    assert datasets.hub_id("local/pick_cube", "sysradium") == "sysradium/pick_cube"
    assert datasets.hub_id("someone/pick_cube", "sysradium") == "someone/pick_cube"


def test_camera_keys_reads_video_and_image_features(make_dataset, lerobot_home):
    make_dataset("a/b", cams=("wrist", "front"))
    (ds,) = datasets.discover()
    assert datasets.camera_keys(ds.info) == ["observation.images.front", "observation.images.wrist"]


def test_push_command_prefers_local_checkout(tmp_path):
    cmd = hf.push_dataset_command("me/ds", tmp_path / "ds", tmp_path / "lerobot")
    assert cmd[:4] == ["uv", "run", "--project", str(tmp_path / "lerobot")]
    assert cmd[-2:] == ["me/ds", str(tmp_path / "ds")]
    cmd = hf.push_dataset_command("me/ds", tmp_path / "ds", None)
    assert cmd[:4] == ["uv", "run", "--with", "lerobot"]
