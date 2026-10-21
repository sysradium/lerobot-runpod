from __future__ import annotations

import json
import os
from pathlib import Path

# RunPod's own image is pre-cached on their hosts (starts in seconds) and runs as
# root; huggingface/lerobot-gpu (~6 GB) pulled for ~20 min and, as non-root,
# hit "CUDA unknown error" on some hosts.
IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
MIN_CUDA_VERSION = "12.8"
CONTAINER_DISK_GB = 50
FAIL_GRACE_S = 1800
FALLBACK_LEROBOT_REF = "main"
POD_NAME_PREFIX = "lerobot-"

SKIP_DATASET_OWNERS = frozenset({"calibration", "outputs", "ports", "robots", "inference_logs"})


def lerobot_home() -> Path:
    env = os.environ.get("HF_LEROBOT_HOME")
    return Path(env).expanduser() if env else Path.home() / ".cache/huggingface/lerobot"


def outputs_dir() -> Path:
    return lerobot_home() / "outputs" / "train"


def lerobot_src_candidates(cwd: Path | None = None) -> list[Path]:
    env = os.environ.get("LEROBOT_SRC")
    if env:
        return [Path(env).expanduser()]
    cwd = cwd or Path.cwd()
    return [cwd / "lerobot", cwd.parent / "lerobot"]


def find_lerobot_src(cwd: Path | None = None) -> Path | None:
    for path in lerobot_src_candidates(cwd):
        if (path / ".git").exists():
            return path
    return None


def lelab_tool_dir() -> Path:
    env = os.environ.get("LELAB_TOOL_DIR")
    return Path(env).expanduser() if env else Path.home() / ".local/share/uv/tools/lelab"


def lelab_lerobot_ref(tool_dir: Path | None = None) -> str | None:
    """The lerobot commit (or release tag) bundled with the installed LeLab.

    Models saved by a newer lerobot can carry config keys and processor layouts
    that LeLab's lerobot rejects at inference, so training should match it.
    """
    for dist in sorted((tool_dir or lelab_tool_dir()).glob("lib/python*/site-packages/lerobot-*.dist-info")):
        direct_url = dist / "direct_url.json"
        if direct_url.exists():
            try:
                commit = json.loads(direct_url.read_text()).get("vcs_info", {}).get("commit_id")
            except (OSError, json.JSONDecodeError):
                commit = None
            if commit:
                return str(commit)
        version = dist.name.removeprefix("lerobot-").removesuffix(".dist-info")
        if version:
            return f"v{version}"
    return None
