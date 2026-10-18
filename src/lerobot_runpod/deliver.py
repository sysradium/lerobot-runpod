"""Register a trained model as a finished run in LeLab's outputs/train directory."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import hf
from .config import outputs_dir


@dataclass
class JobInfo:
    job_id: str
    name: str
    policy: str
    dataset: str
    repo: str
    steps: int
    batch: int
    save_freq: int
    started_at: float
    ended_at: float
    base_model: str | None = None
    loss: float | None = None
    lr: float | None = None
    grad_norm: float | None = None


def job_id(policy: str, dataset_name: str, when: datetime, runner: str = "runpod") -> str:
    return f"{policy}_{runner}_{dataset_name}_{when:%Y-%m-%d_%H-%M-%S}"


def job_record(info: JobInfo, job_dir: Path) -> dict:
    """The shape LeLab's pydantic JobRecord accepts.

    runner="imported" with hf_repo_id=None makes LeLab read checkpoints from
    output_dir/checkpoints/<step>/pretrained_model instead of the Hub.
    """
    return {
        "id": info.job_id,
        "name": info.name,
        "state": "done",
        "config": {
            "dataset_repo_id": info.dataset,
            "policy_type": info.policy,
            "steps": info.steps,
            "batch_size": info.batch,
            "log_freq": 100,
            "save_freq": info.save_freq,
            "policy_device": "cuda",
            "policy_use_bf16": True,
            "policy_push_to_hub": True,
            "policy_repo_id": info.repo,
            "policy_base_model_path": info.base_model,
        },
        "output_dir": str(job_dir / "run"),
        "started_at": info.started_at,
        "ended_at": info.ended_at,
        "exit_code": 0,
        "error_message": None,
        "metrics": {
            "current_step": info.steps,
            "total_steps": info.steps,
            "current_loss": info.loss,
            "current_lr": info.lr,
            "grad_norm": info.grad_norm,
            "eta_seconds": 0,
        },
        "runner": "imported",
        "hf_repo_id": None,
        "checkpoint_count": 0,
    }


def checkpoint_dir(job_dir: Path, steps: int) -> Path:
    return job_dir / "run" / "checkpoints" / f"{steps:06d}" / "pretrained_model"


def install_checkpoint(job_dir: Path, steps: int, source: Path | None = None) -> Path:
    """Lay out <job>/run/checkpoints/<steps:06d>/pretrained_model plus a `last` symlink,
    the way lerobot-train writes it locally."""
    ckpt = checkpoint_dir(job_dir, steps)
    if source is not None and source != ckpt:
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), ckpt)
    shutil.rmtree(ckpt / ".cache", ignore_errors=True)
    if not (ckpt / "config.json").is_file():
        raise RuntimeError(f"download incomplete: {ckpt / 'config.json'} missing")
    last = ckpt.parent.parent / "last"
    if last.is_symlink() or last.exists():
        last.unlink()
    last.symlink_to(f"{steps:06d}")
    return ckpt


def write_job(info: JobInfo, job_dir: Path) -> Path:
    job_dir.mkdir(parents=True, exist_ok=True)
    path = job_dir / "job.json"
    path.write_text(json.dumps(job_record(info, job_dir), indent=2) + "\n")
    return path


def deliver(info: JobInfo, job_dir: Path) -> Path:
    ckpt = checkpoint_dir(job_dir, info.steps)
    hf.download_repo(info.repo, ckpt)
    install_checkpoint(job_dir, info.steps)
    write_job(info, job_dir)
    return ckpt


def append_log(path: Path, message: str, timestamp: float | None = None) -> None:
    """LeLab-style {timestamp, message} lines, so the run's log shows up in LeLab."""
    with path.open("a") as fh:
        fh.write(json.dumps({"timestamp": timestamp or time.time(), "message": message}) + "\n")


def fetch(
    repo: str, *, steps: int | None = None, policy: str | None = None, dataset: str | None = None
) -> Path:
    """Deliver an already-trained model from the Hub, e.g. after the laptop slept through a run."""
    now = datetime.now()
    staging = outputs_dir() / f".fetch_{now:%Y%m%d_%H%M%S}"
    hf.download_repo(repo, staging)
    try:
        train_cfg = _read_json(staging / "train_config.json")
        model_cfg = _read_json(staging / "config.json")
        steps = steps or int(train_cfg.get("steps") or 0)
        if not steps:
            raise RuntimeError(f"{repo} has no train_config.json with steps; pass --steps")
        policy = policy or model_cfg.get("type") or train_cfg.get("policy", {}).get("type") or "act"
        dataset = dataset or train_cfg.get("dataset", {}).get("repo_id") or "unknown/unknown"
        dataset_name = dataset.split("/", 1)[-1]
        jid = job_id(policy, dataset_name, now)
        job_dir = outputs_dir() / jid
        install_checkpoint(job_dir, steps, staging)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    stamp = now.timestamp()
    info = JobInfo(
        job_id=jid,
        name=f"{policy.upper()} · {dataset} · imported from {repo}",
        policy=policy,
        dataset=dataset,
        repo=repo,
        steps=steps,
        batch=int(train_cfg.get("batch_size") or 0) or 8,
        save_freq=int(train_cfg.get("save_freq") or 0) or steps,
        started_at=stamp,
        ended_at=stamp,
        base_model=train_cfg.get("policy", {}).get("pretrained_path"),
    )
    write_job(info, job_dir)
    return job_dir


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
