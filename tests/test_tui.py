from __future__ import annotations

import asyncio
import threading
import time

from textual.widgets import DataTable, Input, ProgressBar, Select, Static

from lerobot_runpod import datasets, logs, plan, runner, runpod, tui

OFFERS = [
    runpod.GpuOffer("NVIDIA GeForce RTX 3090", 24, True, 0.46, 0.22, "High"),
    runpod.GpuOffer("NVIDIA RTX A6000", 48, True, 0.59, 0.33, "Low"),
    runpod.GpuOffer("NVIDIA A100 80GB PCIe", 80, True, 1.79, 1.19, "Low"),
    runpod.GpuOffer("NVIDIA GeForce RTX 3080", 10, True, 0.2, 0.1, "Low"),
]


def local_ds(repo_id, mtime, tmp_path):
    return datasets.LocalDataset(repo_id, tmp_path / repo_id, 50, 26903, 30, mtime, {})


class FakeRun:
    def __init__(self, train_plan, emit):
        self.plan = train_plan
        self.emit = emit
        self.pod_id = None
        self.deleted = []
        self.delivered = False
        self.log_path = "/tmp/log.jsonl"
        self.stop = threading.Event()

    def attach(self, pod_id):
        self.pod_id = pod_id
        self.attached = pod_id

    def create_pod(self):
        self.pod_id = "pod1"
        self.emit(runner.PodCreated("pod1", int(time.time()) + 3600))
        return "pod1"

    def follow(self):
        self.emit(runner.SystemLine("start container for runpod/pytorch: begin"))
        for line in [
            "@@STAGE install@@",
            "@@STAGE train@@",
            "\rTraining: 100%|█| 200/200 [03:01<00:00,  1.10step/s]"
            "INFO x step:200 loss:3.150 grdn:55.364 lr:1.0e-05",
            "@@DONE@@",
        ]:
            for event in logs.parse_line(line):
                self.emit(event)
        return "done"

    def delete_pod(self):
        self.deleted.append(self.pod_id)
        self.pod_id = None
        return True

    def deliver(self):
        self.delivered = True
        return "/tmp/ckpt"

    def kept_message(self):
        return "kept"


def make_services(tmp_path, runs):
    def make_run(train_plan, token, emit):
        run = FakeRun(train_plan, emit)
        runs.append(run)
        return run

    services = tui.Services(
        discover=lambda: [
            local_ds("sysradium/second_try_cam1", 2000, tmp_path),
            local_ds("local/first", 1000, tmp_path),
        ],
        list_gpus=lambda: OFFERS,
        whoami=lambda: "sysradium",
        token=lambda: "hf_fake",
        dataset_status=lambda ds, user: runner.DatasetStatus(ds, ds, True, tmp_path / ds),
        check_cameras=lambda policy, status: None,
        lerobot_ref=lambda: plan.LerobotRef("abc123"),
        list_pods=lambda: [],
        active_follower=lambda job_dir: None,
    )
    services.make_run = make_run
    return services


async def wait_for(pilot, condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        await pilot.pause(0.05)


def test_setup_screen_renders_and_follows_policy(tmp_path):
    async def scenario():
        app = tui.LerobotRunpodApp(make_services(tmp_path, []))
        async with app.run_test(size=(140, 45)) as pilot:
            screen = app.screen
            assert isinstance(screen, tui.SetupScreen)
            ds_table = screen.query_one("#datasets", DataTable)
            gpus = screen.query_one("#gpus", DataTable)
            await wait_for(pilot, lambda: ds_table.row_count == 2 and gpus.row_count == 3)
            assert screen.query_one("#steps", Input).value == "50000"
            assert gpus.get_row_at(0)[3] == "NVIDIA GeForce RTX 3090"

            screen.query_one("#policy", Select).value = "pi0"
            await pilot.pause()
            assert screen.query_one("#steps", Input).value == "30000"
            assert screen.query_one("#max_hours", Input).value == "8"
            assert [gpus.get_row_at(i)[3] for i in range(gpus.row_count)] == ["NVIDIA A100 80GB PCIe"]

            screen.query_one("#cloud", Select).value = "SECURE"
            await pilot.pause()
            assert gpus.get_row_at(0)[0] == "$1.79/h"

    asyncio.run(scenario())


def test_start_confirm_and_run_to_done(tmp_path, lerobot_home):
    runs = []

    async def scenario():
        app = tui.LerobotRunpodApp(make_services(tmp_path, runs))
        async with app.run_test(size=(140, 45)) as pilot:
            setup = app.screen
            await wait_for(pilot, lambda: setup.query_one("#gpus", DataTable).row_count == 3)
            setup.query_one("#steps", Input).value = "200"
            await pilot.click("#start")
            await wait_for(pilot, lambda: isinstance(app.screen, tui.ConfirmScreen))
            assert "Training plan" in app.screen.message
            assert "sysradium/second_try_cam1" in app.screen.message
            await pilot.click("#yes")

            await wait_for(pilot, lambda: isinstance(app.screen, tui.RunScreen))
            run_screen = app.screen
            await wait_for(pilot, lambda: runs and runs[0].delivered)
            await pilot.pause(0.1)
            assert run_screen.stage == "done"
            bar = run_screen.query_one("#progress", ProgressBar)
            assert (bar.progress, bar.total) == (200, 200)
            assert "loss 3.15" in str(run_screen.query_one("#loss", Static).render())
            assert runs[0].deleted == ["pod1"]
            assert runs[0].plan.steps == 200

    asyncio.run(scenario())


class BlockingRun(FakeRun):
    def follow(self):
        self.emit(logs.Stage("train"))
        self.stop.wait(10)
        return "stopped"


def test_quit_with_running_pod_asks_and_deletes_by_default(tmp_path, lerobot_home):
    runs = []

    async def scenario():
        services = make_services(tmp_path, runs)
        services.make_run = lambda p, token, emit: runs.append(BlockingRun(p, emit)) or runs[-1]
        app = tui.LerobotRunpodApp(services)
        async with app.run_test(size=(140, 45)) as pilot:
            await wait_for(pilot, lambda: app.screen.query_one("#gpus", DataTable).row_count == 3)
            await pilot.click("#start")
            await wait_for(pilot, lambda: isinstance(app.screen, tui.ConfirmScreen))
            await pilot.click("#yes")
            await wait_for(pilot, lambda: runs and runs[0].pod_id == "pod1")
            await wait_for(pilot, lambda: app.screen.stage == "train")
            await pilot.press("q")
            await wait_for(pilot, lambda: isinstance(app.screen, tui.ConfirmScreen))
            assert "still running" in app.screen.message
            await pilot.press("enter")
            await wait_for(pilot, lambda: runs[0].deleted == ["pod1"])
            assert runs[0].stop.is_set()

    asyncio.run(scenario())


POD = {
    "id": "q2x44ipzuu0xo4",
    "name": "lerobot-act-20261021_2239",
    "gpuId": "NVIDIA GeForce RTX 3090",
    "costPerHr": 0.22,
    "desiredStatus": "RUNNING",
}


def attachable_services(tmp_path, runs, follower=None):
    services = make_services(tmp_path, runs)
    services.list_pods = lambda: [POD]
    services.active_follower = lambda job_dir: follower
    services.attach_plan = lambda pod_id: runner.build_plan(
        dataset="sysradium/second_try_cam1",
        hub_dataset="sysradium/second_try_cam1",
        policy=tui.policies.get("act"),
        gpu_id="NVIDIA GeForce RTX 3090",
        cloud="COMMUNITY",
        hf_user="sysradium",
        steps=200,
        lerobot_ref="30da8e68",
    )
    return services


def test_running_pods_are_listed_and_attachable(tmp_path, lerobot_home):
    runs = []

    async def scenario():
        app = tui.LerobotRunpodApp(attachable_services(tmp_path, runs))
        async with app.run_test(size=(140, 50)) as pilot:
            setup = app.screen
            pods = setup.query_one("#pods", DataTable)
            await wait_for(pilot, lambda: pods.row_count == 1)
            assert pods.get_row_at(0)[0] == "q2x44ipzuu0xo4"
            assert pods.get_row_at(0)[3] == "$0.22/h"
            await pilot.click("#attach")
            await wait_for(pilot, lambda: isinstance(app.screen, tui.RunScreen))
            await wait_for(pilot, lambda: runs and runs[0].delivered)
            assert runs[0].attached == "q2x44ipzuu0xo4"
            assert runs[0].deleted == ["q2x44ipzuu0xo4"]

    asyncio.run(scenario())


def test_attach_refuses_a_pod_someone_else_follows(tmp_path, lerobot_home):
    runs = []

    async def scenario():
        app = tui.LerobotRunpodApp(attachable_services(tmp_path, runs, follower=4242))
        async with app.run_test(size=(140, 50)) as pilot:
            pods = app.screen.query_one("#pods", DataTable)
            await wait_for(pilot, lambda: pods.row_count == 1)
            await pilot.click("#attach")
            await pilot.pause(0.3)
            assert isinstance(app.screen, tui.SetupScreen)
            assert "4242" in str(app.screen.query_one("#status", Static).render())
            assert not runs

    asyncio.run(scenario())
