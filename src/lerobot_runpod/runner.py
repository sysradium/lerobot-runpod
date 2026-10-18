"""Pre-flight checks and the pod lifecycle, shared by the CLI and the TUI."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from . import cameras, datasets, deliver, hf, logs, plan, policies, runpod
from .config import find_lerobot_src


@dataclass(frozen=True)
class Info:
    text: str
    level: str = "info"


@dataclass(frozen=True)
class SystemLine:
    text: str


@dataclass(frozen=True)
class PodCreated:
    pod_id: str
    deadline: int


RunEvent = logs.Event | Info | SystemLine | PodCreated
Emit = Callable[[RunEvent], None]


@dataclass(frozen=True)
class DatasetStatus:
    dataset: str
    hub_dataset: str
    on_hub: bool
    local_path: Path

    @property
    def can_push(self) -> bool:
        return (self.local_path / "meta/info.json").is_file()


def dataset_status(dataset: str, hf_user: str) -> DatasetStatus:
    hub_id = datasets.hub_id(dataset, hf_user)
    return DatasetStatus(dataset, hub_id, hf.dataset_exists(hub_id), datasets.local_path(dataset))


def check_cameras(policy: policies.Policy, status: DatasetStatus) -> cameras.CameraCheck | None:
    """Uses the local info.json when the dataset isn't on the Hub yet."""
    if policy.base_model is None:
        return None
    if status.on_hub:
        return cameras.preflight(policy.base_model, status.hub_dataset)
    info = json.loads((status.local_path / "meta/info.json").read_text())
    check = cameras.compare(hf.download_json(policy.base_model, "config.json"), info)
    if not check.ok:
        raise cameras.CameraMismatch(check.message(policy.base_model))
    return check


def build_plan(
    *,
    dataset: str,
    hub_dataset: str,
    policy: policies.Policy,
    gpu_id: str,
    cloud: str,
    hf_user: str,
    steps: int | None = None,
    batch: int | None = None,
    max_hours: int | None = None,
    repo: str | None = None,
    lerobot_ref: str | None = None,
    now: datetime | None = None,
) -> plan.TrainPlan:
    now = now or datetime.now()
    name = datasets.split_id(dataset)[1]
    return plan.TrainPlan(
        dataset=dataset,
        hub_dataset=hub_dataset,
        policy=policy,
        steps=steps or policy.steps,
        batch=batch or policy.batch,
        max_hours=max_hours or policy.max_hours,
        gpu_id=gpu_id,
        cloud=cloud,
        repo=repo or plan.default_repo(hf_user, policy.name, name, now),
        lerobot_ref=lerobot_ref or plan.resolve_lerobot_ref(find_lerobot_src()).ref,
        created_at=now,
    )


@dataclass
class RunState:
    stage: str = "pending"
    progress: logs.Progress | None = None
    metrics: logs.Metrics | None = None
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    result: str | None = None


class TrainingRun:
    """Creates the pod, follows it to the end, and delivers the model.

    Never deletes a pod on failure by itself: the caller decides (and asks the user).
    """

    def __init__(self, train_plan: plan.TrainPlan, hf_token: str, emit: Emit):
        self.plan = train_plan
        self.hf_token = hf_token
        self.emit = emit
        self.stop = threading.Event()
        self.pod_id: str | None = None
        self.state = RunState()

    @property
    def log_path(self) -> Path:
        return self.plan.job_dir / "log.jsonl"

    def create_pod(self) -> str:
        self.plan = self.plan.with_fresh_deadline()
        self.emit(Info(f"creating pod on {self.plan.gpu_id} ({self.plan.cloud})"))
        self.pod_id = runpod.create_pod(self.plan.pod_command(self.hf_token))
        self.state.started_at = time.time()
        self.emit(PodCreated(self.pod_id, self.plan.deadline))
        self.emit(Info(f"pod {self.pod_id} created: {runpod.console_url(self.pod_id)}"))
        return self.pod_id

    def follow(self, follower: logs.LogFollower | None = None) -> str:
        assert self.pod_id, "create_pod() first"
        self.plan.job_dir.joinpath("run", "checkpoints").mkdir(parents=True, exist_ok=True)
        follower = follower or logs.LogFollower(self.pod_id, stop=self.stop)
        self.state.stage = "starting"
        self.emit(Info("waiting for the container (image pull can take a while on a fresh host)"))
        if not follower.wait_for_container(lambda line: self.emit(SystemLine(line))):
            return self._finish("stopped" if self.stop.is_set() else "pod disappeared")
        outcome = follower.follow_container(self._on_container_line)
        if outcome == "pod_gone":
            self.pod_id = None
            return self._finish("pod disappeared")
        if outcome == "stopped":
            return self._finish("stopped")
        return self._finish(self.state.result or "log stream ended unexpectedly")

    def _finish(self, result: str) -> str:
        self.state.result = result
        self.state.ended_at = time.time()
        return result

    def _on_container_line(self, line: str) -> bool:
        deliver.append_log(self.log_path, line.split("\r")[-1] or line)
        for event in logs.parse_line(line):
            match event:
                case logs.Stage(name):
                    self.state.stage = name
                case logs.Progress():
                    self.state.progress = event
                case logs.Metrics():
                    self.state.metrics = event
                case logs.Done():
                    self.state.stage = "done"
                    self.state.result = "done"
                case logs.Failed(reason):
                    self.state.stage = "failed"
                    self.state.result = f"failed: {reason}"
            self.emit(event)
            if self.state.result:
                return True
        return False

    def delete_pod(self) -> bool:
        if not self.pod_id:
            return False
        pod_id, self.pod_id = self.pod_id, None
        deleted = runpod.delete_pod(pod_id)
        self.emit(Info(f"pod {pod_id} deleted" if deleted else f"pod {pod_id} already gone"))
        return deleted

    def kept_message(self) -> str:
        deadline = datetime.fromtimestamp(self.plan.deadline).strftime("%H:%M")
        return (
            f"pod {self.pod_id} kept. It deletes itself by {deadline} at the latest, "
            f"or run: runpodctl pod delete {self.pod_id}"
        )

    def job_info(self) -> deliver.JobInfo:
        metrics = self.state.metrics
        p = self.plan
        return deliver.JobInfo(
            job_id=p.job_id,
            name=f"{p.policy.name.upper()} · {p.hub_dataset} · RunPod {p.gpu_id}",
            policy=p.policy.name,
            dataset=p.hub_dataset,
            repo=p.repo,
            steps=p.steps,
            batch=p.batch,
            save_freq=p.save_freq,
            started_at=int(self.state.started_at),
            ended_at=int(self.state.ended_at or time.time()),
            base_model=p.policy.base_model,
            loss=metrics.loss if metrics else None,
            lr=metrics.lr if metrics else None,
            grad_norm=metrics.grad_norm if metrics else None,
        )

    def deliver(self) -> Path:
        self.emit(Info(f"downloading {self.plan.repo} into LeLab"))
        ckpt = deliver.deliver(self.job_info(), self.plan.job_dir)
        self.emit(Info(f"model ready in LeLab: {ckpt}"))
        return ckpt
