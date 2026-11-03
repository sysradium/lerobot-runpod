from __future__ import annotations

import pytest

from lerobot_runpod import runpod
from lerobot_runpod.shell import CommandError, Result

CATALOG = {
    "gpus": [
        {
            "id": "NVIDIA A100 80GB PCIe",
            "memory": 80,
            "secure": True,
            "community": True,
            "price": {"secure": 1.79, "community": 1.19},
            "availability": "LOW",
            "dataCenters": [{"id": "CA-MTL-3", "availability": "LOW"}],
        },
        {
            "id": "NVIDIA A100-SXM4-40GB",
            "memory": 40,
            "secure": False,
            "community": True,
            "price": {"secure": 0, "community": 1},
            "availability": "NONE",
            "dataCenters": [],
        },
        {
            "id": "NVIDIA GeForce RTX 3090",
            "memory": 24,
            "secure": True,
            "community": True,
            "price": {"secure": 0.46, "community": 0.22},
            "availability": "HIGH",
            "dataCenters": [
                {"id": "EU-RO-1", "availability": "LOW"},
                {"id": "US-TX-3", "availability": "NONE"},
                {"id": "EU-SE-1", "availability": "HIGH"},
            ],
        },
        {
            "id": "NVIDIA GeForce RTX 3080",
            "memory": 10,
            "price": {"secure": 0.2, "community": 0.1},
            "availability": "HIGH",
            "dataCenters": [{"id": "US-TX-3", "availability": "HIGH"}],
        },
    ]
}


def test_parse_catalog_keeps_stocked_data_centers_best_first():
    offers = {o.gpu_id: o for o in runpod.parse_catalog(CATALOG)}
    assert offers["NVIDIA GeForce RTX 3090"].data_centers == ("EU-SE-1", "EU-RO-1")
    assert offers["NVIDIA GeForce RTX 3090"].available
    sxm = offers["NVIDIA A100-SXM4-40GB"]
    assert (sxm.available, sxm.secure_price, sxm.community_price, sxm.data_centers) == (False, None, 1, ())


def test_filter_offers_puts_stock_first_then_price():
    offers = runpod.parse_catalog(CATALOG)
    community = runpod.filter_offers(offers, 24, "community")
    assert [o.gpu_id for o in community] == [
        "NVIDIA GeForce RTX 3090",
        "NVIDIA A100 80GB PCIe",
        "NVIDIA A100-SXM4-40GB",
    ]
    secure = runpod.filter_offers(offers, 24, "SECURE")
    assert [o.gpu_id for o in secure] == ["NVIDIA GeForce RTX 3090", "NVIDIA A100 80GB PCIe"]
    assert [o.gpu_id for o in runpod.filter_offers(offers, 80, "SECURE")] == ["NVIDIA A100 80GB PCIe"]


def test_placement_candidates_order_and_price_cap():
    offers = runpod.parse_catalog(CATALOG)
    pinned = runpod.placement_candidates(offers, "NVIDIA GeForce RTX 3090", "SECURE", 24, None)
    assert pinned == [("NVIDIA GeForce RTX 3090", "EU-SE-1"), ("NVIDIA GeForce RTX 3090", "EU-RO-1")]
    capped = runpod.placement_candidates(offers, "NVIDIA GeForce RTX 3090", "SECURE", 24, 1.0)
    assert capped == pinned
    fallback = runpod.placement_candidates(offers, "NVIDIA GeForce RTX 3090", "SECURE", 24, 2.0)
    assert fallback == [*pinned, ("NVIDIA A100 80GB PCIe", "CA-MTL-3")]
    out_of_stock = runpod.placement_candidates(offers, "NVIDIA A100-SXM4-40GB", "COMMUNITY", 24, 0.5)
    assert out_of_stock == [
        ("NVIDIA A100-SXM4-40GB", None),
        ("NVIDIA GeForce RTX 3090", "EU-SE-1"),
        ("NVIDIA GeForce RTX 3090", "EU-RO-1"),
    ]


def test_placement_candidates_without_data_centers_let_runpod_choose():
    offers = [
        runpod.GpuOffer("NVIDIA GeForce RTX 3090", 24, False, 0.46, 0.22, "NONE"),
        runpod.GpuOffer("NVIDIA A100-SXM4-80GB", 80, True, 1.79, 1.39, "LOW"),
    ]
    assert runpod.placement_candidates(offers, "NVIDIA GeForce RTX 3090", "COMMUNITY", 24, 2.0) == [
        ("NVIDIA GeForce RTX 3090", None),
        ("NVIDIA A100-SXM4-80GB", None),
    ]


def test_list_gpus_asks_the_catalog_for_this_cloud_and_cuda(monkeypatch):
    seen = {}

    def fake_get(path, params):
        seen.update(path=path, **params)
        return CATALOG

    monkeypatch.setattr(runpod, "api_get", fake_get)
    assert len(runpod.list_gpus("community")) == 4
    assert seen == {
        "path": "/catalog/gpus",
        "include": "AVAILABILITY",
        "product": "POD",
        "cloud": "COMMUNITY",
        "minCudaVersion": "12.8",
    }


def test_api_key_from_env_or_runpodctl_config(tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text('apikey = "rpa_123"\napiurl = "https://api.runpod.io/graphql"\n')
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    assert runpod.api_key(config) == "rpa_123"
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_env")
    assert runpod.api_key(config) == "rpa_env"
    monkeypatch.delenv("RUNPOD_API_KEY")
    with pytest.raises(runpod.RunPodError, match="RUNPOD_API_KEY"):
        runpod.api_key(tmp_path / "missing.toml")


def test_normalize_cloud():
    assert runpod.normalize_cloud("community") == "COMMUNITY"
    with pytest.raises(ValueError):
        runpod.normalize_cloud("spot")


def test_create_command_shape():
    cmd = runpod.create_command(
        name="lerobot-act-1", gpu_id="NVIDIA GeForce RTX 3090", cloud="community", env={"A": "1"}
    )
    assert cmd[:3] == ["runpodctl", "pod", "create"]
    assert cmd[cmd.index("--cloud-type") + 1] == "COMMUNITY"
    assert cmd[cmd.index("--env") + 1] == '{"A":"1"}'
    assert cmd[cmd.index("--min-cuda-version") + 1] == "12.8"
    assert cmd[cmd.index("--container-disk-in-gb") + 1] == "50"
    assert cmd[-1] == "bash -c 'echo $BOOT | base64 -d > /tmp/boot.sh && bash /tmp/boot.sh'"
    assert "--data-center-ids" not in cmd
    placed = runpod.create_command(
        name="x", gpu_id="NVIDIA GeForce RTX 3090", cloud="SECURE", env={}, data_center_id="EU-RO-1"
    )
    assert placed[placed.index("--data-center-ids") + 1] == "EU-RO-1"


def test_create_pod_capacity_error(monkeypatch):
    def fake_run(cmd, **kwargs):
        raise CommandError(
            cmd,
            1,
            "",
            "Error: There are no longer any instances available with the requested specifications.",
        )

    monkeypatch.setattr(runpod, "run", fake_run)
    with pytest.raises(runpod.CapacityError):
        runpod.create_pod(["runpodctl", "pod", "create"])


def test_create_pod_returns_id(monkeypatch):
    monkeypatch.setattr(runpod, "run", lambda cmd, **kw: Result(0, '{"id": "abc123", "name": "x"}', ""))
    assert runpod.create_pod(["runpodctl"]) == "abc123"


def test_own_pods_filters_by_prefix():
    pods = [{"id": "1", "name": "lerobot-act-x"}, {"id": "2", "name": "comfy"}]
    assert runpod.own_pods(pods) == [pods[0]]
