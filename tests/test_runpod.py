from __future__ import annotations

import json

import pytest

from lerobot_runpod import runpod
from lerobot_runpod.shell import CommandError, Result

GPU_LIST = json.dumps(
    [
        {
            "available": True,
            "communityPricePerHr": 1.19,
            "gpuId": "NVIDIA A100 80GB PCIe",
            "memoryInGb": 80,
            "securePricePerHr": 1.79,
            "stockStatus": "Low",
        },
        {
            "available": True,
            "communityPricePerHr": 1,
            "gpuId": "NVIDIA A100-SXM4-40GB",
            "memoryInGb": 40,
            "securePricePerHr": None,
            "stockStatus": "Low",
        },
        {
            "available": True,
            "communityPricePerHr": 0.22,
            "gpuId": "NVIDIA GeForce RTX 3090",
            "memoryInGb": 24,
            "securePricePerHr": 0.46,
            "stockStatus": "High",
        },
        {
            "available": False,
            "communityPricePerHr": 0.1,
            "gpuId": "NVIDIA GeForce RTX 3080",
            "memoryInGb": 10,
            "securePricePerHr": 0.2,
        },
    ]
)


def test_filter_offers_by_vram_cloud_and_price():
    offers = runpod.parse_gpus(GPU_LIST)
    community = runpod.filter_offers(offers, 24, "community")
    assert [o.gpu_id for o in community] == [
        "NVIDIA GeForce RTX 3090",
        "NVIDIA A100-SXM4-40GB",
        "NVIDIA A100 80GB PCIe",
    ]
    secure = runpod.filter_offers(offers, 24, "SECURE")
    assert [o.gpu_id for o in secure] == ["NVIDIA GeForce RTX 3090", "NVIDIA A100 80GB PCIe"]
    assert [o.gpu_id for o in runpod.filter_offers(offers, 80, "SECURE")] == ["NVIDIA A100 80GB PCIe"]


def test_normalize_cloud():
    assert runpod.normalize_cloud("community") == "COMMUNITY"
    with pytest.raises(ValueError):
        runpod.normalize_cloud("spot")


def test_create_command_shape():
    cmd = runpod.create_command(name="lerobot-act-1", gpu_id="NVIDIA GeForce RTX 3090", cloud="community", env={"A": "1"})
    assert cmd[:3] == ["runpodctl", "pod", "create"]
    assert cmd[cmd.index("--cloud-type") + 1] == "COMMUNITY"
    assert cmd[cmd.index("--env") + 1] == '{"A":"1"}'
    assert cmd[cmd.index("--min-cuda-version") + 1] == "12.8"
    assert cmd[cmd.index("--container-disk-in-gb") + 1] == "50"
    assert cmd[-1] == "bash -c 'echo $BOOT | base64 -d > /tmp/boot.sh && bash /tmp/boot.sh'"


def test_create_pod_capacity_error(monkeypatch):
    def fake_run(cmd, **kwargs):
        raise CommandError(cmd, 1, "", "Error: There are no longer any instances available with the requested specifications.")

    monkeypatch.setattr(runpod, "run", fake_run)
    with pytest.raises(runpod.CapacityError):
        runpod.create_pod(["runpodctl", "pod", "create"])


def test_create_pod_returns_id(monkeypatch):
    monkeypatch.setattr(runpod, "run", lambda cmd, **kw: Result(0, '{"id": "abc123", "name": "x"}', ""))
    assert runpod.create_pod(["runpodctl"]) == "abc123"


def test_own_pods_filters_by_prefix():
    pods = [{"id": "1", "name": "lerobot-act-x"}, {"id": "2", "name": "comfy"}]
    assert runpod.own_pods(pods) == [pods[0]]
