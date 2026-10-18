from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest


def _write_dataset(
    home: Path,
    repo_id: str,
    *,
    episodes: int = 2,
    frames: int = 100,
    fps: int = 30,
    cams: tuple[str, ...] = ("front",),
    mtime: float | None = None,
) -> Path:
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


@pytest.fixture
def make_dataset(lerobot_home: Path) -> Callable[..., Path]:
    """Write a minimal LeLab dataset (meta/info.json) under the temp lerobot home."""

    def make(repo_id: str, **kwargs: Any) -> Path:
        return _write_dataset(lerobot_home, repo_id, **kwargs)

    return make
