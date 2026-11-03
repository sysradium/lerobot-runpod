"""Thin wrappers around the `runpodctl` CLI, which is assumed to be installed and configured,
plus RunPod's v2 catalog API for per-cloud GPU availability (`runpodctl gpu list` ignores the cloud
and CUDA filters a pod is created with, so it lists GPUs that can't actually be placed)."""

from __future__ import annotations

import json
import os
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import __version__
from .config import CONTAINER_DISK_GB, IMAGE, MIN_CUDA_VERSION, POD_NAME_PREFIX
from .shell import CommandError, run

API_URL = "https://api.runpod.io/v2"
CLOUDS = ("SECURE", "COMMUNITY")
DOCKER_ARGS = "bash -c 'echo $BOOT | base64 -d > /tmp/boot.sh && bash /tmp/boot.sh'"
STOCK_RANK = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


class RunPodError(RuntimeError):
    pass


class CapacityError(RunPodError):
    """Nothing could be placed right now; trying again later may work."""


@dataclass(frozen=True)
class GpuOffer:
    gpu_id: str
    memory_gb: int
    available: bool
    secure_price: float | None
    community_price: float | None
    stock: str | None
    # in-stock data centers for the cloud the catalog was asked about, most stock first
    data_centers: tuple[str, ...] = ()

    @classmethod
    def from_catalog(cls, row: dict[str, Any]) -> GpuOffer:
        price = row.get("price") or {}
        stocked = [dc for dc in row.get("dataCenters") or [] if dc.get("availability") in STOCK_RANK]
        stocked.sort(key=lambda dc: STOCK_RANK[dc["availability"]])
        return cls(
            gpu_id=row["id"],
            memory_gb=int(row.get("memory") or 0),
            available=row.get("availability") in STOCK_RANK,
            secure_price=(price.get("secure") or None) if row.get("secure", True) else None,
            community_price=(price.get("community") or None) if row.get("community", True) else None,
            stock=row.get("availability"),
            data_centers=tuple(dc["id"] for dc in stocked),
        )

    def price(self, cloud: str) -> float | None:
        return self.secure_price if normalize_cloud(cloud) == "SECURE" else self.community_price


def normalize_cloud(cloud: str) -> str:
    value = cloud.upper()
    if value not in CLOUDS:
        raise ValueError(f"cloud must be SECURE or COMMUNITY, got {cloud!r}")
    return value


def api_key(config_path: Path | None = None) -> str:
    """RUNPOD_API_KEY, else the key `runpodctl config` stored."""
    if key := os.environ.get("RUNPOD_API_KEY"):
        return key
    path = config_path or Path.home() / ".runpod/config.toml"
    try:
        key = tomllib.loads(path.read_text()).get("apikey")
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise RunPodError(f"no RunPod API key: set RUNPOD_API_KEY or run `runpodctl config` ({exc})") from exc
    if not key:
        raise RunPodError(f"no apikey in {path}: set RUNPOD_API_KEY or run `runpodctl config`")
    return str(key)


def _problem_detail(exc: urllib.error.HTTPError) -> str:
    try:
        body = json.loads(exc.read() or b"{}")
    except (OSError, json.JSONDecodeError):
        return str(exc.reason)
    return str(body.get("detail") or body.get("title") or exc.reason)


def api_get(path: str, params: dict[str, str]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{API_URL}{path}?{urllib.parse.urlencode(params)}",
        headers={"Authorization": f"Bearer {api_key()}", "User-Agent": f"lerobot-runpod/{__version__}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = _problem_detail(exc)
        # rate limits and outages pass; anything else (bad key, bad query) won't fix itself
        if exc.code == 429 or exc.code >= 500:
            raise CapacityError(f"RunPod API unavailable ({exc.code}): {detail}") from exc
        raise RunPodError(f"RunPod API error {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise CapacityError(f"cannot reach the RunPod API: {exc}") from exc


def parse_catalog(data: dict[str, Any]) -> list[GpuOffer]:
    return [GpuOffer.from_catalog(row) for row in data.get("gpus") or [] if row.get("id")]


def list_gpus(cloud: str) -> list[GpuOffer]:
    """Availability as a pod created with `create_command` on this cloud would see it."""
    params = {
        "include": "AVAILABILITY",
        "product": "POD",
        "cloud": normalize_cloud(cloud),
        "minCudaVersion": MIN_CUDA_VERSION,
    }
    return parse_catalog(api_get("/catalog/gpus", params))


def filter_offers(offers: list[GpuOffer], min_vram_gb: int, cloud: str) -> list[GpuOffer]:
    """GPUs that fit and are sold on this cloud: in stock first, then cheapest."""
    usable = [o for o in offers if o.memory_gb >= min_vram_gb and o.price(cloud) is not None]
    return sorted(usable, key=lambda o: (not o.available, o.price(cloud), -o.memory_gb))


def placement_candidates(
    offers: list[GpuOffer], gpu_id: str, cloud: str, min_vram_gb: int, max_price: float | None
) -> list[tuple[str, str | None]]:
    """(gpu, data center) pairs to try in order, as RunPod's docs suggest, since a create call
    places one GPU type and never falls back: the chosen GPU wherever it is in stock, then the
    other fitting GPUs up to max_price, cheapest first.

    The catalog is a hint, not a reservation, so a chosen GPU it shows no stock for still gets
    one try with the data center left to RunPod. So does a GPU in stock without per-data-center
    stock, which is how community cloud reports.
    """
    chosen = next((o for o in offers if o.gpu_id == gpu_id), None)
    pairs = [(gpu_id, dc) for dc in chosen.data_centers] if chosen else []
    pairs = pairs or [(gpu_id, None)]
    if max_price is not None:
        for o in filter_offers(offers, min_vram_gb, cloud):
            price = o.price(cloud)
            if o.gpu_id != gpu_id and o.available and price is not None and price <= max_price:
                pairs += [(o.gpu_id, dc) for dc in o.data_centers] or [(o.gpu_id, None)]
    return pairs


def create_command(
    *,
    name: str,
    gpu_id: str,
    cloud: str,
    env: dict[str, str],
    image: str = IMAGE,
    data_center_id: str | None = None,
) -> list[str]:
    cmd = [
        "runpodctl", "pod", "create",
        "--image", image,
        "--name", name,
        "--gpu-id", gpu_id,
        "--cloud-type", normalize_cloud(cloud),
        "--min-cuda-version", MIN_CUDA_VERSION,
        "--container-disk-in-gb", str(CONTAINER_DISK_GB),
        "--env", json.dumps(env, separators=(",", ":")),
    ]  # fmt: skip
    if data_center_id:
        cmd += ["--data-center-ids", data_center_id]
    return [*cmd, "--docker-args", DOCKER_ARGS]


def is_capacity_error(text: str) -> bool:
    lowered = text.lower()
    return "no longer any instances available" in lowered or "no instances currently available" in lowered


def create_pod(cmd: list[str]) -> str:
    try:
        out = run(cmd, timeout=300).stdout
    except CommandError as exc:
        if is_capacity_error(exc.output):
            raise CapacityError("RunPod has no capacity for this GPU right now") from exc
        raise RunPodError(f"pod creation failed: {exc.output.strip()}") from exc
    try:
        pod_id = json.loads(out).get("id")
    except json.JSONDecodeError:
        pod_id = None
    if not pod_id:
        if is_capacity_error(out):
            raise CapacityError("RunPod has no capacity for this GPU right now")
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


def get_pod(pod_id: str) -> dict[str, Any]:
    pod = json.loads(run(["runpodctl", "pod", "get", pod_id]).stdout or "{}")
    if not isinstance(pod, dict) or "error" in pod or not pod.get("id"):
        raise RunPodError(f"pod {pod_id} not found: {str(pod)[:200]}")
    return pod


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
