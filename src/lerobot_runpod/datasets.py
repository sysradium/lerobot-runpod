"""Discover datasets recorded by LeLab in the local LeRobot cache."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .config import SKIP_DATASET_OWNERS, lerobot_home


@dataclass(frozen=True)
class LocalDataset:
    repo_id: str
    path: Path
    episodes: int
    frames: int
    fps: int
    mtime: float
    info: dict

    @property
    def owner(self) -> str:
        return self.repo_id.split("/", 1)[0]

    @property
    def name(self) -> str:
        return self.repo_id.split("/", 1)[1]

    @property
    def modified(self) -> str:
        return datetime.fromtimestamp(self.mtime).strftime("%Y-%m-%d %H:%M")


def discover(home: Path | None = None) -> list[LocalDataset]:
    home = home or lerobot_home()
    found = []
    for info_path in home.glob("*/*/meta/info.json"):
        ds_dir = info_path.parent.parent
        owner, name = ds_dir.parent.name, ds_dir.name
        if owner in SKIP_DATASET_OWNERS:
            continue
        try:
            info = json.loads(info_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        found.append(
            LocalDataset(
                repo_id=f"{owner}/{name}",
                path=ds_dir,
                episodes=int(info.get("total_episodes", 0)),
                frames=int(info.get("total_frames", 0)),
                fps=int(info.get("fps", 0)),
                mtime=info_path.stat().st_mtime,
                info=info,
            )
        )
    found.sort(key=lambda d: d.mtime, reverse=True)
    return found


def split_id(dataset_id: str) -> tuple[str, str]:
    if "/" not in dataset_id:
        raise ValueError(f"dataset id must look like owner/name, got {dataset_id!r}")
    owner, name = dataset_id.split("/", 1)
    return owner, name


def hub_id(dataset_id: str, hf_user: str) -> str:
    """The pod can only fetch from the Hub, so `local/<name>` maps to `<hf_user>/<name>`."""
    owner, name = split_id(dataset_id)
    return f"{hf_user}/{name}" if owner == "local" else dataset_id


def local_path(dataset_id: str, home: Path | None = None) -> Path:
    return (home or lerobot_home()) / dataset_id


def camera_keys(info: dict) -> list[str]:
    return sorted(k for k, v in info.get("features", {}).items() if v.get("dtype") in ("video", "image"))
