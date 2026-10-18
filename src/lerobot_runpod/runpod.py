"""Thin wrappers around the `runpodctl` CLI, which is assumed to be installed and configured."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .config import CONTAINER_DISK_GB, IMAGE, MIN_CUDA_VERSION, POD_NAME_PREFIX
from .shell import CommandError, run

CLOUDS = ("SECURE", "COMMUNITY")
DOCKER_ARGS = "bash -c 'echo $BOOT | base64 -d > /tmp/boot.sh && bash /tmp/boot.sh'"


class RunPodError(RuntimeError):
    pass


class CapacityError(RunPodError):
    """The GPU listing said "available", but RunPod had no machine when we asked for one."""


@dataclass(frozen=True)
class GpuOffer:
    gpu_id: str
    memory_gb: int
    available: bool
    secure_price: float | None
    community_price: float | None
    stock: str | None

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> GpuOffer:
        return cls(
            gpu_id=row["gpuId"],
            memory_gb=int(row.get("memoryInGb") or 0),
            available=bool(row.get("available")),
            secure_price=row.get("securePricePerHr"),
            community_price=row.get("communityPricePerHr"),
            stock=row.get("stockStatus"),
        )

    def price(self, cloud: str) -> float | None:
        return self.secure_price if normalize_cloud(cloud) == "SECURE" else self.community_price


def normalize_cloud(cloud: str) -> str:
    value = cloud.upper()
    if value not in CLOUDS:
        raise ValueError(f"cloud must be SECURE or COMMUNITY, got {cloud!r}")
    return value


def parse_gpus(raw: str) -> list[GpuOffer]:
    return [GpuOffer.from_json(row) for row in json.loads(raw) if row.get("gpuId")]


def list_gpus() -> list[GpuOffer]:
    return parse_gpus(run(["runpodctl", "gpu", "list"]).stdout)


def filter_offers(offers: list[GpuOffer], min_vram_gb: int, cloud: str) -> list[GpuOffer]:
    usable = [o for o in offers if o.available and o.memory_gb >= min_vram_gb and o.price(cloud) is not None]
    return sorted(usable, key=lambda o: (o.price(cloud), -o.memory_gb))


def create_command(
    *, name: str, gpu_id: str, cloud: str, env: dict[str, str], image: str = IMAGE
) -> list[str]:
    return [
        "runpodctl", "pod", "create",
        "--image", image,
        "--name", name,
        "--gpu-id", gpu_id,
        "--cloud-type", normalize_cloud(cloud),
        "--min-cuda-version", MIN_CUDA_VERSION,
        "--container-disk-in-gb", str(CONTAINER_DISK_GB),
        "--env", json.dumps(env, separators=(",", ":")),
        "--docker-args", DOCKER_ARGS,
    ]  # fmt: skip


def is_capacity_error(text: str) -> bool:
    lowered = text.lower()
    return "no longer any instances available" in lowered or "no instances currently available" in lowered


def create_pod(cmd: list[str]) -> str:
    try:
        out = run(cmd, timeout=300).stdout
    except CommandError as exc:
        if is_capacity_error(exc.output):
            raise CapacityError(
                "RunPod has no capacity for this GPU right now (the listing can say 'available' anyway); "
                "pick another GPU or cloud"
            ) from exc
        raise RunPodError(f"pod creation failed: {exc.output.strip()}") from exc
    try:
        pod_id = json.loads(out).get("id")
    except json.JSONDecodeError:
        pod_id = None
    if not pod_id:
        if is_capacity_error(out):
            raise CapacityError("RunPod has no capacity for this GPU right now; pick another GPU or cloud")
        raise RunPodError(f"pod creation returned no id: {out.strip()[:500]}")
    return pod_id


def delete_pod(pod_id: str) -> bool:
    """Returns False when the pod was already gone."""
    return run(["runpodctl", "pod", "delete", pod_id], check=False).returncode == 0


def list_pods() -> list[dict[str, Any]]:
    pods = json.loads(run(["runpodctl", "pod", "list"]).stdout or "[]")
    # an error comes back as a JSON object; treating it as "no pods" would make
    # a live pod look gone to the log follower
    if not isinstance(pods, list):
        raise RunPodError(f"unexpected pod list output: {str(pods)[:200]}")
    return [p for p in pods if isinstance(p, dict)]


def own_pods(pods: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [p for p in pods if str(p.get("name", "")).startswith(POD_NAME_PREFIX)]


def pod_exists(pod_id: str) -> bool:
    try:
        return any(p.get("id") == pod_id for p in list_pods())
    except (CommandError, json.JSONDecodeError):
        # can't tell; assume it's still there rather than abandoning a billing pod
        return True


def logs_command(pod_id: str, source: str, *, follow: bool = True, tail: int | None = None) -> list[str]:
    cmd = ["runpodctl", "pod", "logs", pod_id, "--source", source]
    if follow:
        cmd.append("--follow")
    if tail is not None:
        cmd += ["--tail", str(tail)]
    return cmd


def container_started(pod_id: str) -> bool:
    out = run(logs_command(pod_id, "container", follow=False, tail=1), check=False, timeout=60).stdout
    return bool(out.strip())


def console_url(pod_id: str) -> str:
    return f"https://console.runpod.io/pods?id={pod_id}"
