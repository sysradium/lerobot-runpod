"""Camera-name preflight for policies fine-tuned from a base model."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from . import hf
from .datasets import camera_keys
from .shell import CommandError, run


class CameraMismatch(RuntimeError):
    pass


@dataclass(frozen=True)
class CameraCheck:
    model: list[str]
    dataset: list[str]

    @property
    def ok(self) -> bool:
        # lerobot accepts camera sets that are a subset or superset of the base
        # model's; anything else fails at train time.
        m, d = set(self.model), set(self.dataset)
        return d <= m or m <= d

    def message(self, base_model: str) -> str:
        return (
            f"camera names don't match {base_model}\n"
            f"  dataset: {self.dataset}\n"
            f"  model:   {self.model}\n"
            "rename the dataset's cameras (and the LeLab robot camera) to the model's names first, "
            "e.g. with `lerobot-runpod rename-camera`"
        )


def model_camera_keys(config: dict) -> list[str]:
    return sorted(k for k, v in config.get("input_features", {}).items() if v.get("type") == "VISUAL")


def compare(model_config: dict, dataset_info: dict) -> CameraCheck:
    return CameraCheck(model=model_camera_keys(model_config), dataset=camera_keys(dataset_info))


def preflight(base_model: str, hub_dataset: str) -> CameraCheck:
    """LeLab can't remap camera names at inference, so catch mismatches before paying for a GPU."""
    check = compare(
        hf.download_json(base_model, "config.json"),
        hf.download_json(hub_dataset, "meta/info.json", repo_type="dataset"),
    )
    if not check.ok:
        raise CameraMismatch(check.message(base_model))
    return check


def rename_command(src: Path, dst: Path, old: str, new: str) -> list[str]:
    script = resources.files("lerobot_runpod").joinpath("rename_camera.py")
    return [
        "uv", "run", "--no-project", "--with", "pandas", "--with", "pyarrow",
        "python", str(script), str(src), str(dst), old, new,
    ]  # fmt: skip


def rename(src: Path, dst: Path, old: str, new: str) -> str:
    try:
        return run(rename_command(src, dst, old, new), timeout=1800).stdout.strip()
    except CommandError as exc:
        raise RuntimeError(f"camera rename failed: {exc.output.strip()}") from exc
