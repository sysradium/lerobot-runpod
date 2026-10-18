from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


def make_dataset(home: Path, repo_id: str, *, episodes=2, frames=100, fps=30, cams=("front",), mtime=None):
    meta = home / repo_id / "meta"
    meta.mkdir(parents=True)
    features = {
        "action": {"dtype": "float32", "shape": [6]},
        "observation.state": {"dtype": "float32", "shape": [6]},
    }
    for cam in cams:
        features[f"observation.images.{cam}"] = {"dtype": "video", "shape": [480, 640, 3]}
    info = {"total_episodes": episodes, "total_frames": frames, "fps": fps, "features": features}
    path = meta / "info.json"
    path.write_text(json.dumps(info))
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


@pytest.fixture
def lerobot_home(tmp_path, monkeypatch):
    home = tmp_path / "lerobot"
    home.mkdir()
    monkeypatch.setenv("HF_LEROBOT_HOME", str(home))
    return home
