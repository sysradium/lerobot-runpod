"""Thin wrappers around the `hf` CLI, which is assumed to be installed and logged in."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from .shell import CommandError, run


class HfError(RuntimeError):
    pass


def whoami() -> str:
    # The plain-text output differs between a TTY and a pipe, so only trust --json.
    try:
        out = run(["hf", "auth", "whoami", "--json"]).stdout
        user = json.loads(out).get("user")
    except (CommandError, json.JSONDecodeError) as exc:
        raise HfError("not logged in to Hugging Face (run: hf auth login)") from exc
    if not user:
        raise HfError("not logged in to Hugging Face (run: hf auth login)")
    return user


def token() -> str:
    try:
        value = run(["hf", "auth", "token"], check=False).stdout.strip()
    except OSError:
        value = ""
    if not value.startswith("hf_"):
        path = Path.home() / ".cache/huggingface/token"
        value = path.read_text().strip() if path.exists() else ""
    if not value:
        raise HfError("could not read the HF token (run: hf auth login)")
    return value


def dataset_exists(repo_id: str) -> bool:
    return run(["hf", "datasets", "info", repo_id], check=False).returncode == 0


def download_json(repo_id: str, filename: str, repo_type: str = "model") -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        cmd = ["hf", "download", repo_id, filename, "--local-dir", tmp, "--quiet"]
        if repo_type != "model":
            cmd[2:2] = ["--repo-type", repo_type]
        try:
            run(cmd, timeout=300)
        except CommandError as exc:
            raise HfError(f"could not fetch {repo_id}/{filename}") from exc
        return json.loads((Path(tmp) / filename).read_text())


def download_repo(repo_id: str, local_dir: Path) -> None:
    try:
        run(["hf", "download", repo_id, "--local-dir", str(local_dir), "--quiet"], timeout=3600)
    except CommandError as exc:
        raise HfError(f"could not download {repo_id}: {exc}") from exc


PUSH_SNIPPET = """
import sys
from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo_id, root = sys.argv[1], sys.argv[2]
LeRobotDataset(repo_id, root=root).push_to_hub(private=True, tags=["so101", "lerobot"])
"""


def push_dataset_command(repo_id: str, root: Path, lerobot_src: Path | None) -> list[str]:
    """Pushing needs lerobot itself; use the local checkout if there is one, else a throwaway env."""
    if lerobot_src is not None:
        return ["uv", "run", "--project", str(lerobot_src), "python", "-c", PUSH_SNIPPET, repo_id, str(root)]
    return ["uv", "run", "--with", "lerobot", "python", "-c", PUSH_SNIPPET, repo_id, str(root)]


def push_dataset(repo_id: str, root: Path, lerobot_src: Path | None) -> None:
    try:
        run(push_dataset_command(repo_id, root, lerobot_src), timeout=3600)
    except CommandError as exc:
        raise HfError(f"pushing {root} to {repo_id} failed: {exc}") from exc
