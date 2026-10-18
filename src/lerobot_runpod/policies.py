"""Supported policies and their training defaults. Add a row to POLICIES to support a new one."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Policy:
    name: str
    description: str
    steps: int
    batch: int
    min_vram_gb: int
    max_hours: int
    extras: tuple[str, ...] = ("training",)
    # set => fine-tune from this checkpoint instead of training from scratch
    base_model: str | None = None
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    def train_args(self, steps: int) -> list[str]:
        if self.base_model is None:
            args = [f"--policy.type={self.name}"]
        else:
            args = [f"--policy.path={self.base_model}", f"--policy.scheduler_decay_steps={steps}"]
        return args + list(self.extra_args)

    @property
    def extras_spec(self) -> str:
        return ",".join(self.extras)


POLICIES: dict[str, Policy] = {
    p.name: p
    for p in (
        Policy("act", "fast, reliable baseline for SO-101 (recommended)", 50000, 32, 24, 4),
        Policy(
            "smolvla",
            "VLA fine-tuned from lerobot/smolvla_base, runs on a Mac",
            20000,
            32,
            24,
            6,
            extras=("training", "smolvla"),
            base_model="lerobot/smolvla_base",
            extra_args=("--policy.freeze_vision_encoder=false", "--policy.train_expert_only=false"),
        ),
        Policy(
            "diffusion",
            "diffusion policy, slower to train",
            100000,
            64,
            24,
            8,
            extras=("training", "diffusion"),
        ),
        Policy(
            "pi0",
            "large VLA fine-tuned from lerobot/pi0_base, needs >=80GB",
            30000,
            32,
            80,
            8,
            extras=("training", "pi"),
            base_model="lerobot/pi0_base",
        ),
    )
}


def get(name: str) -> Policy:
    try:
        return POLICIES[name]
    except KeyError:
        raise ValueError(f"unsupported policy {name!r}; choose from {', '.join(POLICIES)}") from None
