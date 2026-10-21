"""A small Textual front end: one setup screen, one run screen."""

from __future__ import annotations

import asyncio
import contextlib
import signal
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    ProgressBar,
    RichLog,
    Select,
    Static,
)

from . import datasets, hf, logs, plan, policies, runner, runpod
from .config import find_lerobot_src

STAGES = ("pod", "starting", "install", "env", "download", "train", "done")


@dataclass
class Services:
    """Everything the TUI does to the outside world, swappable in tests."""

    discover: Callable[[], list[datasets.LocalDataset]] = datasets.discover
    list_gpus: Callable[[], list[runpod.GpuOffer]] = runpod.list_gpus
    whoami: Callable[[], str] = hf.whoami
    token: Callable[[], str] = hf.token
    dataset_status: Callable[[str, str], runner.DatasetStatus] = runner.dataset_status
    check_cameras: Callable[..., object] = runner.check_cameras
    lerobot_ref: Callable[[], plan.LerobotRef] = lambda: plan.default_lerobot_ref(find_lerobot_src())
    list_pods: Callable[[], list[dict[str, Any]]] = lambda: runpod.own_pods(runpod.list_pods())
    active_follower: Callable[..., int | None] = runner.active_follower

    def attach_plan(self, pod_id: str) -> plan.TrainPlan:
        offers: list[runpod.GpuOffer] = []
        for _ in range(2):
            try:
                offers = self.list_gpus()
                break
            except Exception:
                continue
        return runner.plan_from_pod(runpod.get_pod(pod_id), offers)

    def push_dataset(self, status: runner.DatasetStatus) -> None:
        hf.push_dataset(status.hub_dataset, status.local_path, find_lerobot_src())

    def make_run(self, train_plan: plan.TrainPlan, token: str, emit: runner.Emit) -> runner.TrainingRun:
        return runner.TrainingRun(train_plan, token, emit)


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class ConfirmScreen(ModalScreen[bool]):
    BINDINGS = [Binding("escape", "answer(False)", "Cancel"), Binding("y", "answer(True)", "Yes")]
    DEFAULT_CSS = """
    ConfirmScreen { align: center middle; }
    ConfirmScreen > Vertical {
        width: 110; max-width: 95%; height: auto; border: thick $accent; padding: 1 2; background: $surface;
    }
    ConfirmScreen Horizontal { height: auto; margin-top: 1; align-horizontal: right; }
    ConfirmScreen Button { margin-left: 2; }
    """

    def __init__(self, message: str, *, yes: str = "Yes", no: str = "Cancel", default_yes: bool = False):
        super().__init__()
        self.message = message
        self.yes = yes
        self.no = no
        self.default_yes = default_yes

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(self.message, markup=False)
            with Horizontal():
                yield Button(self.no, id="no")
                yield Button(self.yes, id="yes", variant="primary")

    def on_mount(self) -> None:
        self.query_one("#yes" if self.default_yes else "#no", Button).focus()

    @on(Button.Pressed)
    def pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")

    def action_answer(self, value: bool) -> None:
        self.dismiss(value)


class SetupScreen(Screen):
    BINDINGS = [Binding("r", "refresh", "Refresh"), Binding("q", "app.quit", "Quit")]
    DEFAULT_CSS = """
    SetupScreen { layout: vertical; padding: 0 1; }
    SetupScreen .title { margin-top: 1; text-style: bold; }
    #pods { height: auto; max-height: 5; }
    #datasets { height: 1fr; min-height: 4; }
    #gpus { height: 1fr; min-height: 4; }
    #params { height: auto; margin-top: 1; }
    #params > Vertical { width: 1fr; height: auto; margin-right: 1; }
    #params Input, #params Select { width: 100%; }
    #policy-hint { color: $text-muted; }
    #bottom { height: auto; margin: 1 0; }
    #status { width: 1fr; padding-top: 1; }
    """

    def __init__(self, services: Services):
        super().__init__()
        self.services = services
        self.offers: list[runpod.GpuOffer] = []
        self.dataset_rows: list[datasets.LocalDataset] = []
        self.gpu_rows: list[runpod.GpuOffer] = []
        self.pod_rows: list[dict[str, Any]] = []

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("Running pods", id="pods-title", classes="title")
        yield DataTable(id="pods", cursor_type="row", zebra_stripes=True)
        yield Label("LeLab datasets (newest first)", classes="title")
        yield DataTable(id="datasets", cursor_type="row", zebra_stripes=True)
        with Horizontal(id="params"):
            with Vertical():
                yield Label("Policy")
                yield Select(
                    [(p.name, p.name) for p in policies.POLICIES.values()],
                    value="act",
                    allow_blank=False,
                    id="policy",
                )
            with Vertical():
                yield Label("Steps")
                yield Input(id="steps", type="integer")
            with Vertical():
                yield Label("Batch")
                yield Input(id="batch", type="integer")
            with Vertical():
                yield Label("Max hours")
                yield Input(id="max_hours", type="integer")
            with Vertical():
                yield Label("Cloud")
                yield Select(
                    [("COMMUNITY", "COMMUNITY"), ("SECURE", "SECURE")],
                    value="COMMUNITY",
                    allow_blank=False,
                    id="cloud",
                )
        yield Static("", id="policy-hint")
        yield Label("GPU", id="gpu-title", classes="title")
        yield DataTable(id="gpus", cursor_type="row", zebra_stripes=True)
        with Horizontal(id="bottom"):
            yield Static("", id="status")
            yield Button("Attach", id="attach")
            yield Button("Start", id="start", variant="success")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#datasets", DataTable).add_columns("modified", "dataset", "episodes", "frames", "fps")
        self.query_one("#gpus", DataTable).add_columns("price", "vram", "stock", "gpu")
        self.query_one("#pods", DataTable).add_columns("pod", "name", "gpu", "price", "status")
        self._apply_policy_defaults()
        self.load_datasets()
        self.load_gpus()
        self.load_pods()
        self.set_interval(30, self.load_pods)

    @property
    def policy(self) -> policies.Policy:
        return policies.get(str(self.query_one("#policy", Select).value))

    @property
    def cloud(self) -> str:
        return str(self.query_one("#cloud", Select).value)

    def set_status(self, text: str) -> None:
        self.query_one("#status", Static).update(text)

    def _apply_policy_defaults(self) -> None:
        p = self.policy
        self.query_one("#steps", Input).value = str(p.steps)
        self.query_one("#batch", Input).value = str(p.batch)
        self.query_one("#max_hours", Input).value = str(p.max_hours)
        self.query_one("#policy-hint", Static).update(
            f"{p.name}: {p.description}. COMMUNITY cloud is cheaper but less reliable."
        )

    def action_refresh(self) -> None:
        self.load_datasets()
        self.load_gpus()

    @work(exclusive=True, group="datasets")
    async def load_datasets(self) -> None:
        found = await asyncio.to_thread(self.services.discover)
        table = self.query_one("#datasets", DataTable)
        table.clear()
        self.dataset_rows = found
        for d in found:
            table.add_row(d.modified, d.repo_id, str(d.episodes), str(d.frames), str(d.fps))
        if not found:
            self.set_status("no LeLab datasets found; record one in LeLab first")

    @work(exclusive=True, group="pods")
    async def load_pods(self) -> None:
        try:
            pods = await asyncio.to_thread(self.services.list_pods)
        except Exception as exc:
            self.query_one("#pods-title", Label).update(f"Running pods (could not list: {exc})")
            return
        self.pod_rows = pods
        table = self.query_one("#pods", DataTable)
        table.clear()
        for p in pods:
            price = p.get("costPerHr")
            table.add_row(
                str(p.get("id", "")),
                str(p.get("name", "")),
                str(p.get("gpuId") or "-"),
                f"${price:.2f}/h" if isinstance(price, int | float) else "-",
                str(p.get("desiredStatus") or "-"),
            )
        self.query_one("#pods-title", Label).update(
            f"Running pods ({len(pods)}) — select one and press Attach to follow it"
            if pods
            else "Running pods (none)"
        )

    @on(Button.Pressed, "#attach")
    def attach_pressed(self) -> None:
        if not self.pod_rows:
            self.notify("no running lerobot-* pods", severity="warning")
            return
        pod = self.pod_rows[self.query_one("#pods", DataTable).cursor_row]
        self.attach(str(pod.get("id")))

    @work(exclusive=True, group="start")
    async def attach(self, pod_id: str) -> None:
        self.set_status(f"reading pod {pod_id}...")
        try:
            train_plan = await asyncio.to_thread(self.services.attach_plan, pod_id)
        except Exception as exc:
            self.set_status(f"[red]{exc}[/red]")
            self.notify(str(exc), severity="error", timeout=15)
            return
        other = await asyncio.to_thread(self.services.active_follower, train_plan.job_dir, pod_id)
        if other:
            self.set_status(f"[red]pid {other} is already following {pod_id}[/red]")
            self.notify(f"pid {other} is already following {pod_id}", severity="error", timeout=15)
            return
        self.set_status("")
        self.app.push_screen(RunScreen(self.services, train_plan, "", attach_pod=pod_id))

    @work(exclusive=True, group="gpus")
    async def load_gpus(self) -> None:
        self.set_status("loading GPU prices...")
        try:
            self.offers = await asyncio.to_thread(self.services.list_gpus)
        except Exception as exc:
            self.set_status(f"could not list GPUs: {exc}")
            return
        self.set_status("")
        self._fill_gpus()

    def _fill_gpus(self) -> None:
        p, cloud = self.policy, self.cloud
        self.gpu_rows = runpod.filter_offers(self.offers, p.min_vram_gb, cloud)
        table = self.query_one("#gpus", DataTable)
        table.clear()
        for o in self.gpu_rows:
            table.add_row(f"${o.price(cloud):.2f}/h", f"{o.memory_gb}GB", o.stock or "-", o.gpu_id)
        self.query_one("#gpu-title", Label).update(f"GPU (>= {p.min_vram_gb}GB, {cloud}, cheapest first)")

    @on(Select.Changed, "#policy")
    def policy_changed(self) -> None:
        self._apply_policy_defaults()
        self._fill_gpus()

    @on(Select.Changed, "#cloud")
    def cloud_changed(self) -> None:
        self._fill_gpus()

    def _int(self, widget_id: str) -> int:
        raw = self.query_one(f"#{widget_id}", Input).value.strip()
        if not raw.isdigit() or int(raw) <= 0:
            raise ValueError(f"{widget_id.replace('_', ' ')} must be a positive integer")
        return int(raw)

    @on(Button.Pressed, "#start")
    def start_pressed(self) -> None:
        ds_table, gpu_table = self.query_one("#datasets", DataTable), self.query_one("#gpus", DataTable)
        if not self.dataset_rows:
            self.notify("no dataset selected", severity="error")
            return
        if not self.gpu_rows:
            self.notify("no GPU available for this policy/cloud", severity="error")
            return
        try:
            steps, batch, max_hours = self._int("steps"), self._int("batch"), self._int("max_hours")
        except ValueError as exc:
            self.notify(str(exc), severity="error")
            return
        dataset = self.dataset_rows[ds_table.cursor_row]
        gpu = self.gpu_rows[gpu_table.cursor_row]
        self.prepare_and_start(dataset.repo_id, self.policy, steps, batch, max_hours, gpu.gpu_id)

    @work(exclusive=True, group="start")
    async def prepare_and_start(
        self, dataset: str, policy: policies.Policy, steps: int, batch: int, max_hours: int, gpu_id: str
    ) -> None:
        svc = self.services
        button = self.query_one("#start", Button)
        button.disabled = True
        try:
            self.set_status("checking Hugging Face login...")
            user = await asyncio.to_thread(svc.whoami)
            token = await asyncio.to_thread(svc.token)
            self.set_status(f"checking whether {dataset} is on the Hub...")
            status = await asyncio.to_thread(svc.dataset_status, dataset, user)
            if not status.on_hub:
                if not status.can_push:
                    raise RuntimeError(f"{status.hub_dataset} is not on the Hub and has no local copy")
                push = await self.app.push_screen_wait(
                    ConfirmScreen(
                        f"{status.hub_dataset} is not on the Hub yet, and the pod can only download from "
                        f"the Hub.\n\nPush {status.local_path} there as a PRIVATE dataset?",
                        yes="Push privately",
                    )
                )
                if not push:
                    self.set_status("cancelled: the dataset must be on the Hub")
                    return
                self.set_status(f"pushing {status.hub_dataset} (private)... this can take a while")
                await asyncio.to_thread(svc.push_dataset, status)
                status = replace(status, on_hub=True)
            if policy.base_model:
                self.set_status(f"checking camera names against {policy.base_model}...")
                await asyncio.to_thread(svc.check_cameras, policy, status)
            self.set_status("resolving lerobot version...")
            ref = await asyncio.to_thread(svc.lerobot_ref)
            train_plan = runner.build_plan(
                dataset=dataset,
                hub_dataset=status.hub_dataset,
                policy=policy,
                gpu_id=gpu_id,
                cloud=self.cloud,
                hf_user=user,
                steps=steps,
                batch=batch,
                max_hours=max_hours,
                lerobot_ref=ref.ref,
            )
            price = next((o.price(self.cloud) for o in self.gpu_rows if o.gpu_id == gpu_id), None)
            lines = [f"{k:<10} {v}" for k, v in train_plan.summary()]
            if price is not None:
                lines.append(f"{'cost':<10} ${price:.2f}/h, at most ~${price * max_hours:.2f}")
            if ref.warning:
                lines.append(f"\nnote: {ref.warning}")
            self.set_status("")
            ok = await self.app.push_screen_wait(
                ConfirmScreen("Training plan\n\n" + "\n".join(lines), yes="Create pod and train")
            )
            if ok:
                self.app.push_screen(RunScreen(svc, train_plan, token))
        except Exception as exc:
            self.set_status(f"[red]{exc}[/red]")
            self.notify(str(exc), severity="error", timeout=15)
        finally:
            button.disabled = False


class RunScreen(Screen):
    BINDINGS = [
        Binding("x", "stop", "Stop & delete pod"),
        Binding("q", "leave", "Quit"),
    ]
    DEFAULT_CSS = """
    RunScreen { layout: vertical; padding: 0 1; }
    #plan { margin-top: 1; color: $text-muted; }
    #stages { margin-top: 1; }
    #progress-row { margin-top: 1; height: 1; }
    #progress { width: auto; }
    #progress Bar { width: 50; }
    #stats { width: auto; margin-left: 2; }
    #loss { margin-top: 1; }
    #log { height: 1fr; margin-top: 1; border: round $panel; }
    """

    def __init__(
        self, services: Services, train_plan: plan.TrainPlan, token: str, attach_pod: str | None = None
    ):
        super().__init__()
        self.services = services
        self.train_plan = train_plan
        self.training = services.make_run(train_plan, token, self._emit)
        self.attach_pod = attach_pod
        self.stage = "pod"
        self.finished = False

    def compose(self) -> ComposeResult:
        yield Header()
        p = self.train_plan
        yield Static(
            f"{p.policy.name} on {p.hub_dataset} · {p.steps} steps · {p.gpu_id} ({p.cloud}) · "
            f"max {p.max_hours}h",
            id="plan",
            markup=False,
        )
        yield Static("", id="stages")
        with Horizontal(id="progress-row"):
            yield ProgressBar(total=p.steps, show_eta=False, id="progress")
            yield Static("", id="stats")
        yield Static("loss: -", id="loss")
        yield RichLog(id="log", max_lines=5000, markup=False, wrap=False)
        yield Footer()

    def on_mount(self) -> None:
        self._render_stages()
        self.drive()

    def _emit(self, event: runner.RunEvent) -> None:
        self.app.call_from_thread(self.handle_event, event)

    def log_line(self, text: str | Text) -> None:
        self.query_one("#log", RichLog).write(text)

    def _render_stages(self) -> None:
        parts = []
        current = STAGES.index(self.stage) if self.stage in STAGES else -1
        for i, name in enumerate(STAGES):
            if self.stage == "failed" and i == max(current, 0):
                parts.append(f"[b red]{name}[/]")
            elif i < current:
                parts.append(f"[green]{name}[/]")
            elif i == current:
                parts.append(f"[b reverse] {name} [/]")
            else:
                parts.append(f"[dim]{name}[/]")
        if self.stage == "failed":
            parts.append("[b red]failed[/]")
        self.query_one("#stages", Static).update(" > ".join(parts))

    def set_stage(self, stage: str) -> None:
        if stage == "failed" or stage in STAGES:
            self.stage = stage
            self._render_stages()

    def handle_event(self, event: runner.RunEvent) -> None:
        match event:
            case logs.Progress():
                self.query_one("#progress", ProgressBar).update(total=event.total, progress=event.step)
                rate = f"{event.steps_per_s:.2f} it/s" if event.steps_per_s else "? it/s"
                self.query_one("#stats", Static).update(
                    f"{event.step}/{event.total} · {fmt_duration(event.elapsed_s)} elapsed · "
                    f"ETA {fmt_duration(event.eta_s)} · {rate}"
                )
            case logs.Metrics():
                parts = [f"step {event.step}" if event.step is not None else ""]
                parts += [f"loss {event.loss}", f"grad norm {event.grad_norm}", f"lr {event.lr}"]
                self.query_one("#loss", Static).update(" · ".join(p for p in parts if p))
                self.log_line(event.text)
            case logs.Stage(name=name):
                self.set_stage(name)
                self.log_line(Text(f"── {name} ──", style="bold"))
            case logs.Done():
                self.set_stage("done")
                self.log_line(Text("training finished", style="bold green"))
            case logs.Failed(reason=reason):
                self.set_stage("failed")
                self.log_line(Text(f"FAILED: {reason}", style="bold red"))
            case logs.Warn(text=text):
                self.log_line(Text(f"warning: {text}", style="yellow"))
            case logs.Log(text=text):
                self.log_line(text)
            case runner.SystemLine(text=text):
                self.log_line(Text(f"[pod] {text}", style="dim"))
            case runner.PodCreated():
                self.set_stage("starting")
            case runner.Info(text=text, level=level):
                self.log_line(Text(f"==> {text}", style="yellow" if level == "warn" else "cyan"))

    @work(thread=True, exclusive=True, group="run")
    def drive(self) -> None:
        run = self.training
        try:
            if self.attach_pod:
                run.attach(self.attach_pod)
                self._emit(runner.PodCreated(self.attach_pod, run.plan.deadline))
            else:
                run.create_pod()
        except runpod.CapacityError as exc:
            self.app.call_from_thread(self._back_to_setup, str(exc))
            return
        except Exception as exc:
            self.app.call_from_thread(self._back_to_setup, f"pod creation failed: {exc}")
            return
        result = run.follow()
        if result == "stopped":
            return
        if result != "done":
            self.app.call_from_thread(self.handle_failure, result)
            return
        run.delete_pod()
        try:
            ckpt = run.deliver()
        except Exception as exc:
            self._emit(runner.Info(f"{exc}; retry with: lerobot-runpod fetch {run.plan.repo}", "warn"))
            return
        self.finished = True
        self._emit(runner.Info(f"done. Restart or refresh LeLab to see the run. ({ckpt})"))

    def _back_to_setup(self, message: str) -> None:
        self.notify(message, severity="error", timeout=20)
        self.app.pop_screen()

    @work(exclusive=True, group="ask")
    async def handle_failure(self, result: str) -> None:
        self.log_line(Text(f"{result}  (full log: {self.training.log_path})", style="bold red"))
        if not self.training.pod_id:
            return
        delete = await self.app.push_screen_wait(
            ConfirmScreen(
                f"{result}\n\nPod {self.training.pod_id} is still running (billing!). Delete it now?\n"
                "Keeping it lets you inspect it; it deletes itself after a grace period anyway.",
                yes="Delete pod",
                no="Keep pod",
                default_yes=True,
            )
        )
        await self._teardown(delete)

    async def _teardown(self, delete: bool) -> None:
        if not self.training.pod_id:
            return
        if delete:
            await asyncio.to_thread(self.training.delete_pod)
        else:
            self.log_line(Text(self.training.kept_message(), style="yellow"))

    @work(exclusive=True, group="ask")
    async def action_stop(self) -> None:
        if not self.training.pod_id:
            self.notify("no pod running")
            return
        if await self.app.push_screen_wait(
            ConfirmScreen(f"Stop training and delete pod {self.training.pod_id}?", yes="Stop & delete")
        ):
            self.training.stop.set()
            await self._teardown(True)
            self.set_stage("failed")

    @work(exclusive=True, group="ask")
    async def action_leave(self) -> None:
        message = None
        if self.training.pod_id:
            delete = await self.app.push_screen_wait(
                ConfirmScreen(
                    f"Pod {self.training.pod_id} is still running (billing!). Delete it before quitting?",
                    yes="Delete pod",
                    no="Keep pod",
                    default_yes=True,
                )
            )
            self.training.stop.set()
            if delete:
                await asyncio.to_thread(self.training.delete_pod)
            else:
                message = self.training.kept_message()
        self.training.stop.set()
        self.app.exit(message=message)


class LerobotRunpodApp(App):
    TITLE = "lerobot-runpod"
    SUB_TITLE = "train LeLab datasets on RunPod"

    def __init__(self, services: Services | None = None):
        super().__init__()
        self.services = services or Services()

    def on_mount(self) -> None:
        self.push_screen(SetupScreen(self.services))
        # no SIGTERM handlers on Windows event loops
        with contextlib.suppress(NotImplementedError, RuntimeError):
            asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, self._on_sigterm)

    def _on_sigterm(self) -> None:
        self.run_worker(self._terminate(), exclusive=True, group="terminate")

    async def _terminate(self) -> None:
        """LeLab's Stop button SIGTERMs the follower: stop training and delete the pod."""
        for screen in self.screen_stack:
            if isinstance(screen, RunScreen) and screen.training.pod_id:
                screen.training.stop.set()
                await asyncio.to_thread(screen.training.mark, "interrupted", "stopped from LeLab")
                await asyncio.to_thread(screen.training.delete_pod)
        self.exit(message="stopped from LeLab; pod deleted")

    async def action_quit(self) -> None:
        # never leave a billing pod behind silently
        run_screens = [s for s in self.screen_stack if isinstance(s, RunScreen)]
        if run_screens and run_screens[-1].training.pod_id:
            while self.screen is not run_screens[-1]:
                self.pop_screen()
            run_screens[-1].action_leave()
        else:
            self.exit()


def run_tui() -> None:
    app = LerobotRunpodApp()
    app.run()
