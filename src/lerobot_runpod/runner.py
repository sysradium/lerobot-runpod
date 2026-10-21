"""Pre-flight checks and the pod lifecycle, shared by the CLI and the TUI."""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import cameras, datasets, deliver, hf, logs, plan, policies, runpod
from .config import find_lerobot_src, outputs_dir


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
    log_freq: int | None = None,
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
        log_freq=log_freq or plan.DEFAULT_LOG_FREQ,
        gpu_id=gpu_id,
        cloud=cloud,
        repo=repo or plan.default_repo(hf_user, policy.name, name, now),
        lerobot_ref=lerobot_ref or plan.resolve_lerobot_ref(find_lerobot_src()).ref,
        created_at=now,
    )


def _policy_from_args(policy_args: str) -> policies.Policy:
    for arg in policy_args.split():
        key, _, value = arg.partition("=")
        if key == "--policy.type":
            return policies.get(value)
        if key == "--policy.path":
            for p in policies.POLICIES.values():
                if p.base_model == value:
                    return p
    raise ValueError(f"cannot tell the policy from POLICY_ARGS={policy_args!r}")


def _pod_created(pod: dict[str, Any]) -> datetime:
    raw = str(pod.get("createdAt") or "")
    try:
        utc = datetime.strptime(raw.split(".")[0].replace(" +0000 UTC", ""), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return datetime.now()
    return utc.replace(tzinfo=UTC).astimezone().replace(tzinfo=None)


def guess_cloud(gpu_id: str, cost_per_hr: float | None, offers: list[runpod.GpuOffer]) -> str:
    """Pods created before CLOUD was stored in their env: match the hourly price."""
    if cost_per_hr is None:
        return "SECURE"
    for o in offers:
        if (
            o.gpu_id == gpu_id
            and o.community_price is not None
            and abs(o.community_price - cost_per_hr) < 0.005
        ):
            return "COMMUNITY"
    return "SECURE"


def find_job_id(policy: str, dataset_name: str, created: datetime, root: Path | None = None) -> str | None:
    """An existing job dir started for this pod: same policy/dataset, stamped shortly before it."""
    best: tuple[float, str] | None = None
    for d in (root or outputs_dir()).glob(f"{policy}_runpod_{dataset_name}_*"):
        _, day, clock = d.name.rsplit("_", 2)
        try:
            stamp = datetime.strptime(f"{day}_{clock}", "%Y-%m-%d_%H-%M-%S")
        except ValueError:
            continue
        gap = (created - stamp).total_seconds()
        if -120 <= gap <= 600 and (best is None or abs(gap) < best[0]):
            best = (abs(gap), d.name)
    return best[1] if best else None


def plan_from_pod(pod: dict[str, Any], offers: list[runpod.GpuOffer] | None = None) -> plan.TrainPlan:
    """Rebuild the plan a pod was created from, out of the env vars it was given."""
    env = pod.get("env") or {}
    if not isinstance(env, dict) or "DATASET" not in env or "REPO" not in env:
        raise ValueError(f"pod {pod.get('id')} was not created by lerobot-runpod (no DATASET/REPO env)")
    policy = (
        policies.get(env["POLICY"]) if env.get("POLICY") else _policy_from_args(env.get("POLICY_ARGS", ""))
    )
    created = _pod_created(pod)
    gpu_id = str(pod.get("gpuId") or "?")
    cloud = env.get("CLOUD") or guess_cloud(gpu_id, pod.get("costPerHr"), offers or [])
    name = datasets.split_id(env["DATASET"])[1]
    job_id = env.get("JOB_ID") or find_job_id(policy.name, name, created)
    return plan.TrainPlan(
        dataset=env["DATASET"],
        hub_dataset=env["DATASET"],
        policy=policy,
        steps=int(env["STEPS"]),
        batch=int(env["BATCH"]),
        max_hours=int(env.get("MAX_HOURS") or policy.max_hours),
        gpu_id=gpu_id,
        cloud=cloud,
        repo=env["REPO"],
        lerobot_ref=env.get("LEROBOT_REF") or "?",
        log_freq=int(env.get("LOG_FREQ") or 100),
        fixed_job_id=job_id,
        created_at=created,
        deadline=int(env.get("DEADLINE") or 0),
    )


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def active_follower(job_dir: Path) -> int | None:
    """PID of another live process already following this job (per its job.json)."""
    record = deliver.read_job(job_dir)
    pid = record.get("process_pid")
    if record.get("state") == "running" and isinstance(pid, int) and pid != os.getpid() and _pid_alive(pid):
        return pid
    return None


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

    def attach(self, pod_id: str) -> None:
        """Follow a pod that already exists (created by an earlier run)."""
        self.pod_id = pod_id
        self.state.started_at = self.plan.created_at.timestamp()
        self.emit(Info(f"attached to pod {pod_id}: {runpod.console_url(pod_id)}"))

    def mark(self, state: str, error: str | None = None) -> None:
        """Mirror the run into LeLab's job.json (running jobs show live progress there)."""
        deliver.write_job(
            self.job_info(), self.plan.job_dir, state=state, process_pid=os.getpid(), error=error
        )

    def follow(self, follower: logs.LogFollower | None = None) -> str:
        assert self.pod_id, "create_pod() first"
        self.plan.job_dir.joinpath("run", "checkpoints").mkdir(parents=True, exist_ok=True)
        self.mark("running")
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
        if result != "done":
            self.mark("interrupted" if result == "stopped" else "failed", error=result)
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
            log_freq=p.log_freq,
            started_at=int(self.state.started_at),
            ended_at=int(self.state.ended_at or time.time()),
            base_model=p.policy.base_model,
            loss=metrics.loss if metrics else None,
            lr=metrics.lr if metrics else None,
            grad_norm=metrics.grad_norm if metrics else None,
            current_step=self.state.progress.step if self.state.progress else None,
        )

    def deliver(self) -> Path:
        self.emit(Info(f"downloading {self.plan.repo} into LeLab"))
        try:
            ckpt = deliver.deliver(self.job_info(), self.plan.job_dir)
        except Exception as exc:
            self.mark("failed", error=f"training finished but the download failed: {exc}")
            raise
        self.emit(Info(f"model ready in LeLab: {ckpt}"))
        return ckpt
