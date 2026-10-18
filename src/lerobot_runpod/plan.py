"""Everything decided before a pod is created, and the pod command derived from it."""

from __future__ import annotations

import base64
import shlex
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from importlib import resources
from pathlib import Path

from . import runpod
from .config import FAIL_GRACE_S, FALLBACK_LEROBOT_REF, POD_NAME_PREFIX, outputs_dir
from .datasets import split_id
from .policies import Policy
from .shell import run

TOKEN_PLACEHOLDER = "<HF_TOKEN>"


def bootstrap_script() -> str:
    return resources.files("lerobot_runpod").joinpath("bootstrap.sh").read_text()


@dataclass(frozen=True)
class LerobotRef:
    ref: str
    warning: str | None = None


def resolve_lerobot_ref(src: Path | None) -> LerobotRef:
    """The pod installs lerobot from GitHub at the local checkout's commit so it trains with the
    same code; fall back to origin/main if HEAD isn't pushed."""
    if src is None:
        return LerobotRef(
            FALLBACK_LEROBOT_REF, "no local lerobot checkout found; the pod will use lerobot@main"
        )
    git = ["git", "-C", str(src)]
    run([*git, "fetch", "-q", "origin", "main"], check=False, timeout=60)
    if run([*git, "merge-base", "--is-ancestor", "HEAD", "origin/main"], check=False).returncode == 0:
        head = run([*git, "rev-parse", "HEAD"], check=False).stdout.strip()
        if head:
            return LerobotRef(head)
    main = run([*git, "rev-parse", "origin/main"], check=False).stdout.strip()
    if main:
        return LerobotRef(
            main, f"local lerobot HEAD isn't on origin/main; the pod will use origin/main {main[:8]}"
        )
    return LerobotRef(
        FALLBACK_LEROBOT_REF, f"could not read git state of {src}; the pod will use lerobot@main"
    )


@dataclass(frozen=True)
class TrainPlan:
    dataset: str
    hub_dataset: str
    policy: Policy
    steps: int
    batch: int
    max_hours: int
    gpu_id: str
    cloud: str
    repo: str
    lerobot_ref: str
    created_at: datetime = field(default_factory=datetime.now)
    deadline: int = 0

    def __post_init__(self) -> None:
        if self.steps <= 0 or self.batch <= 0:
            raise ValueError("steps and batch must be positive integers")
        if self.max_hours <= 0:
            raise ValueError("max hours must be a positive integer")
        object.__setattr__(self, "cloud", runpod.normalize_cloud(self.cloud))
        if not self.deadline:
            object.__setattr__(self, "deadline", int(self.created_at.timestamp()) + self.max_hours * 3600)

    def with_fresh_deadline(self) -> TrainPlan:
        """The deadline is absolute, so recompute it right before the pod is created."""
        return replace(self, deadline=int(time.time()) + self.max_hours * 3600)

    @property
    def dataset_name(self) -> str:
        return split_id(self.dataset)[1]

    @property
    def save_freq(self) -> int:
        return self.steps // 5 or self.steps

    @property
    def train_args(self) -> list[str]:
        return self.policy.train_args(self.steps)

    @property
    def job_id(self) -> str:
        return f"{self.policy.name}_runpod_{self.dataset_name}_{self.created_at:%Y-%m-%d_%H-%M-%S}"

    @property
    def job_dir(self) -> Path:
        return outputs_dir() / self.job_id

    @property
    def pod_name(self) -> str:
        return f"{POD_NAME_PREFIX}{self.policy.name}-{self.created_at:%Y%m%d_%H%M}"

    @property
    def deadline_str(self) -> str:
        return datetime.fromtimestamp(self.deadline).strftime("%H:%M")

    def pod_env(self, hf_token: str) -> dict[str, str]:
        boot = base64.b64encode(bootstrap_script().encode()).decode()
        return {
            "HF_TOKEN": hf_token,
            "BOOT": boot,
            "DATASET": self.hub_dataset,
            "POLICY_ARGS": " ".join(self.train_args),
            "REPO": self.repo,
            "STEPS": str(self.steps),
            "BATCH": str(self.batch),
            "SAVE_FREQ": str(self.save_freq),
            "LEROBOT_REF": self.lerobot_ref,
            "EXTRAS": self.policy.extras_spec,
            "MAX_HOURS": str(self.max_hours),
            "DEADLINE": str(self.deadline),
            "FAIL_GRACE_S": str(FAIL_GRACE_S),
        }

    def pod_command(self, hf_token: str) -> list[str]:
        return runpod.create_command(
            name=self.pod_name, gpu_id=self.gpu_id, cloud=self.cloud, env=self.pod_env(hf_token)
        )

    def summary(self) -> list[tuple[str, str]]:
        return [
            ("dataset", self.hub_dataset),
            ("policy", self.policy.name),
            ("steps", str(self.steps)),
            ("batch", str(self.batch)),
            ("lerobot", f"{self.lerobot_ref[:8]} [{self.policy.extras_spec}]"),
            ("gpu", f"{self.gpu_id} ({self.cloud})"),
            ("max time", f"{self.max_hours}h (until {self.deadline_str})"),
            ("model", f"https://huggingface.co/{self.repo} (private)"),
            ("local", str(self.job_dir)),
        ]


def default_repo(hf_user: str, policy: str, dataset_name: str, now: datetime) -> str:
    return f"{hf_user}/{policy}_{dataset_name}_{now:%Y%m%d_%H%M}"


def redact(text: str, secret: str) -> str:
    return text.replace(secret, TOKEN_PLACEHOLDER) if secret else text


def format_command(cmd: list[str], secret: str) -> str:
    return redact(shlex.join(cmd), secret)
