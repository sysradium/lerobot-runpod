"""A small Textual front end: one setup screen, one run screen."""

from __future__ import annotations

import asyncio
import contextlib
import signal
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from rich.text import Text
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen, Screen
from textual.validation import Integer, ValidationResult, Validator
from textual.widgets import (
    Button,
    Checkbox,
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
from .config import CAPACITY_POLL_S, FALLBACK_PRICE_FACTOR, find_lerobot_src

STAGES = ("pod", "starting", "install", "env", "download", "train", "done")
# shown under the Training panel while that field has focus
FIELD_HELP = {
    "steps": "optimizer steps: more steps train longer and cost more",
    "batch": "samples per step: lower it if the GPU runs out of memory",
    "max_hours": "hard limit: the pod deletes itself after this, even if this machine is gone",
    "log_freq": "print a loss line every N steps",
    "cloud": "COMMUNITY is roughly half the price of SECURE, but its hosts are less reliable",
    "streaming": "read the dataset from the Hub while training instead of downloading it to the pod "
    "first: no download wait or disk use, frames are shuffled within a buffer",
    "image_transforms": "random brightness/contrast/saturation/hue/sharpness/affine on camera frames, "
    "for robustness to lighting and camera placement",
    "train_args": "extra lerobot-train settings, space separated, e.g. policy.chunk_size=50 or "
    "dataset.image_transforms.max_num_transforms=2",
}
PANELS = {"1": "#pods", "2": "#datasets", "3": "#policy", "4": "#gpus"}


class VimTable(DataTable):
    BINDINGS = [
        Binding("j", "cursor_down", show=False),
        Binding("k", "cursor_up", show=False),
        Binding("g", "scroll_top", show=False),
        Binding("G", "scroll_bottom", show=False),
        Binding("ctrl+d", "page_down", show=False),
        Binding("ctrl+u", "page_up", show=False),
    ]


class VimLog(RichLog):
    BINDINGS = [
        Binding("j", "scroll_down", show=False),
        Binding("k", "scroll_up", show=False),
        Binding("g", "scroll_home", show=False),
        Binding("G", "scroll_end", show=False),
        Binding("ctrl+d", "page_down", show=False),
        Binding("ctrl+u", "page_up", show=False),
    ]


class TrainArgs(Validator):
    def validate(self, value: str) -> ValidationResult:
        try:
            plan.parse_train_args(value.split())
        except ValueError as exc:
            return self.failure(str(exc))
        return self.success()


@dataclass(frozen=True)
class SetupForm:
    dataset: datasets.LocalDataset
    gpu: runpod.GpuOffer
    policy: policies.Policy
    cloud: str
    steps: int
    batch: int
    max_hours: int
    log_freq: int
    streaming: bool
    image_transforms: bool
    extra_train_args: tuple[str, ...]

    @property
    def price(self) -> float | None:
        return self.gpu.price(self.cloud)


@dataclass
class Services:
    """Everything the TUI does to the outside world, swappable in tests."""

    discover: Callable[[], list[datasets.LocalDataset]] = datasets.discover
    list_gpus: Callable[[str], list[runpod.GpuOffer]] = runpod.list_gpus
    whoami: Callable[[], str] = hf.whoami
    token: Callable[[], str] = hf.token
    dataset_status: Callable[[str, str], runner.DatasetStatus] = runner.dataset_status
    tag_dataset: Callable[[str, str], None] = hf.tag_dataset
    check_cameras: Callable[..., object] = runner.check_cameras
    lerobot_ref: Callable[[], plan.LerobotRef] = lambda: plan.default_lerobot_ref(find_lerobot_src())
    list_pods: Callable[[], list[dict[str, Any]]] = lambda: runpod.own_pods(runpod.list_pods())
    active_follower: Callable[..., int | None] = runner.active_follower

    def attach_plan(self, pod_id: str) -> plan.TrainPlan:
        offers: list[runpod.GpuOffer] = []
        for _ in range(2):
            try:
                # only the prices are used, and the catalog returns both clouds' prices either way
                offers = self.list_gpus("SECURE")
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
    BINDINGS = [
        Binding("s", "start", "Start"),
        Binding("a", "attach", "Attach"),
        Binding("r", "refresh", "Refresh"),
        Binding("q", "app.quit", "Quit"),
        Binding("escape", "leave_field", show=False),
        *(Binding(key, f"focus_panel('{key}')", show=False) for key in PANELS),
    ]
    DEFAULT_CSS = """
    SetupScreen { layout: vertical; padding: 0 1; }
    SetupScreen VimTable, #training {
        border: round $panel; border-title-color: $text-muted; background: $surface;
    }
    SetupScreen VimTable:focus, #training:focus-within {
        border: round $accent; border-title-color: $accent; border-title-style: bold;
    }
    #pods { height: auto; max-height: 7; }
    #datasets { height: 1fr; min-height: 6; }
    #gpus { height: 1fr; min-height: 6; }
    #training { height: auto; padding: 0 1; }
    #training Horizontal { height: auto; }
    #training Horizontal > Vertical { width: 1fr; height: auto; margin-right: 1; }
    #training Input, #training Select { width: 100%; }
    #training Label { color: $text-muted; }
    #data-field { width: 2fr; }
    #train-args-field { width: 3fr; }
    #hint { color: $text-muted; padding: 0 1; }
    #bottom { height: auto; margin-bottom: 1; }
    #bottom > Vertical { width: 1fr; height: auto; }
    #summary { text-style: bold; }
    #status { color: $text-muted; }
    #bottom Button { margin-left: 1; }
    """

    def __init__(self, services: Services):
        super().__init__()
        self.services = services
        self.offers: list[runpod.GpuOffer] = []
        self.dataset_rows: list[datasets.LocalDataset] = []
        self.gpu_rows: list[runpod.GpuOffer] = []
        self.pod_rows: list[dict[str, Any]] = []

    def compose(self) -> ComposeResult:
        positive = [Integer(minimum=1, failure_description="must be a positive integer")]
        yield Header()
        yield VimTable(id="pods", cursor_type="row", zebra_stripes=True)
        yield VimTable(id="datasets", cursor_type="row", zebra_stripes=True)
        with Vertical(id="training"):
            with Horizontal():
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
                    yield Input(id="steps", type="integer", validators=positive)
                with Vertical():
                    yield Label("Batch")
                    yield Input(id="batch", type="integer", validators=positive)
                with Vertical():
                    yield Label("Max hours")
                    yield Input(id="max_hours", type="integer", validators=positive)
                with Vertical():
                    yield Label("Log every")
                    yield Input(
                        str(plan.DEFAULT_LOG_FREQ), id="log_freq", type="integer", validators=positive
                    )
                with Vertical():
                    yield Label("Cloud")
                    yield Select(
                        [("COMMUNITY", "COMMUNITY"), ("SECURE", "SECURE")],
                        value="COMMUNITY",
                        allow_blank=False,
                        id="cloud",
                    )
            with Horizontal():
                with Vertical(id="data-field"):
                    yield Label("Data")
                    with Horizontal():
                        yield Checkbox("Stream from Hub", id="streaming")
                        yield Checkbox("Image transforms", id="image_transforms")
                with Vertical(id="train-args-field"):
                    yield Label("Extra lerobot-train args")
                    yield Input(
                        placeholder="policy.chunk_size=50 policy.optimizer_lr=3e-5",
                        id="train_args",
                        validators=[TrainArgs()],
                    )
            yield Static("", id="hint")
        yield VimTable(id="gpus", cursor_type="row", zebra_stripes=True)
        with Horizontal(id="bottom"):
            with Vertical():
                yield Static("", id="summary")
                yield Static("", id="status")
            yield Button("Attach", id="attach")
            yield Button("Start", id="start", variant="success")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#datasets", DataTable).add_columns("modified", "dataset", "episodes", "frames", "fps")
        self.query_one("#gpus", DataTable).add_columns("price", "vram", "stock", "gpu", "in stock at")
        self.query_one("#pods", DataTable).add_columns("pod", "name", "gpu", "price", "status")
        self.query_one("#training").border_title = "[3] Training"
        self.query_one("#datasets").border_title = "[2] LeLab datasets · newest first · Enter to configure"
        self._apply_policy_defaults()
        self.load_datasets()
        self.load_gpus()
        self.load_pods()
        self.set_interval(30, self.load_pods)
        self.query_one("#datasets").focus()

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
        self._update_hint()

    def _update_hint(self) -> None:
        focused = self.focused.id if self.focused else None
        if focused in FIELD_HELP:
            text = FIELD_HELP[focused]
        else:
            p = self.policy
            text = (
                f"{p.name}: {p.description}  ·  Tab next field · 1-4 panels · Esc leave field · "
                "j/k/g/G in tables"
            )
        self.query_one("#hint", Static).update(text)

    def on_descendant_focus(self, event: events.DescendantFocus) -> None:
        self._update_hint()

    def _read_form(self) -> SetupForm:
        """Everything Start needs, or a ValueError naming the first problem."""
        if not self.dataset_rows:
            raise ValueError("no dataset selected")
        if not self.gpu_rows:
            raise ValueError("no GPU for this policy/cloud" if self.offers else "loading GPUs...")
        values = {}
        for field in ("steps", "batch", "max_hours", "log_freq"):
            raw = self.query_one(f"#{field}", Input).value.strip()
            if not raw.isdigit() or int(raw) <= 0:
                raise ValueError(f"{field.replace('_', ' ')} must be a positive integer")
            values[field] = int(raw)
        extra = plan.parse_train_args(self.query_one("#train_args", Input).value.split())
        return SetupForm(
            dataset=self.dataset_rows[self.query_one("#datasets", DataTable).cursor_row],
            gpu=self.gpu_rows[self.query_one("#gpus", DataTable).cursor_row],
            policy=self.policy,
            cloud=self.cloud,
            streaming=self.query_one("#streaming", Checkbox).value,
            image_transforms=self.query_one("#image_transforms", Checkbox).value,
            extra_train_args=extra,
            **values,
        )

    def _update_summary(self) -> None:
        summary = self.query_one("#summary", Static)
        try:
            form = self._read_form()
        except ValueError as exc:
            summary.update(Text(str(exc), style="yellow"))
            return
        parts = [form.dataset.repo_id, form.policy.name, f"{form.steps} steps", form.gpu.gpu_id]
        if form.price is not None:
            parts.append(
                f"${form.price:.2f}/h × {form.max_hours}h = up to ${form.price * form.max_hours:.2f}"
            )
        if form.streaming:
            parts.append("streaming")
        if form.image_transforms:
            parts.append("image transforms")
        summary.update(" · ".join(parts))

    @on(DataTable.RowHighlighted, "#datasets")
    @on(DataTable.RowHighlighted, "#gpus")
    @on(Input.Changed)
    @on(Checkbox.Changed)
    def form_changed(self) -> None:
        self._update_summary()

    def action_refresh(self) -> None:
        self.load_datasets()
        self.load_gpus()

    def action_focus_panel(self, key: str) -> None:
        self.query_one(PANELS[key]).focus()

    def action_leave_field(self) -> None:
        self.query_one("#gpus").focus()

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
        self._update_summary()

    @work(exclusive=True, group="pods")
    async def load_pods(self) -> None:
        table = self.query_one("#pods", DataTable)
        try:
            pods = await asyncio.to_thread(self.services.list_pods)
        except Exception as exc:
            table.border_title = f"[1] Running pods · could not list: {exc}"
            return
        self.pod_rows = pods
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
        table.border_title = (
            f"[1] Running pods ({len(pods)}) · Enter or a to attach and follow one"
            if pods
            else "[1] Running pods (none)"
        )

    @on(DataTable.RowSelected, "#pods")
    def action_attach(self) -> None:
        if not self.pod_rows:
            self.notify("no running lerobot-* pods", severity="warning")
            return
        pod = self.pod_rows[self.query_one("#pods", DataTable).cursor_row]
        self.attach(str(pod.get("id")))

    @on(Button.Pressed, "#attach")
    def attach_pressed(self) -> None:
        self.action_attach()

    @on(DataTable.RowSelected, "#datasets")
    def dataset_selected(self) -> None:
        self.query_one("#policy").focus()

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
        cloud = self.cloud
        self.offers = []
        self._fill_gpus()
        self.set_status(f"loading {cloud} GPU availability...")
        try:
            self.offers = await asyncio.to_thread(self.services.list_gpus, cloud)
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
            stock = (o.stock or "-").lower()
            table.add_row(
                f"${o.price(cloud):.2f}/h",
                f"{o.memory_gb}GB",
                stock if o.available else Text(stock, style="dim"),
                o.gpu_id if o.available else Text(o.gpu_id, style="dim"),
                ", ".join(o.data_centers),
            )
        table.border_title = (
            f"[4] GPU · >= {p.min_vram_gb}GB, {cloud}, in stock first, then cheapest · Enter or s to start"
        )
        self._update_summary()

    @on(Select.Changed, "#policy")
    def policy_changed(self) -> None:
        self._apply_policy_defaults()
        self._fill_gpus()

    @on(Select.Changed, "#cloud")
    def cloud_changed(self) -> None:
        self.load_gpus()

    @on(DataTable.RowSelected, "#gpus")
    @on(Button.Pressed, "#start")
    def action_start(self) -> None:
        if self.query_one("#start", Button).disabled:
            return
        try:
            form = self._read_form()
        except ValueError as exc:
            self.notify(str(exc), severity="error")
            return
        self.prepare_and_start(form)

    @work(exclusive=True, group="start")
    async def prepare_and_start(self, form: SetupForm) -> None:
        svc = self.services
        dataset, policy = form.dataset.repo_id, form.policy
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
            if tag := status.missing_tag:
                add = await self.app.push_screen_wait(
                    ConfirmScreen(
                        f"{status.hub_dataset} has no {tag} tag. lerobot loads Hub datasets at that tag "
                        "(a plain folder upload doesn't create it), so training would crash on the pod."
                        f"\n\nAdd the {tag} tag to {status.hub_dataset}?",
                        yes="Add tag",
                        default_yes=True,
                    )
                )
                if not add:
                    self.set_status(f"cancelled: training needs the {tag} tag")
                    return
                self.set_status(f"tagging {status.hub_dataset} with {tag}...")
                await asyncio.to_thread(svc.tag_dataset, status.hub_dataset, tag)
            if policy.base_model:
                self.set_status(f"checking camera names against {policy.base_model}...")
                await asyncio.to_thread(svc.check_cameras, policy, status)
            self.set_status("resolving lerobot version...")
            ref = await asyncio.to_thread(svc.lerobot_ref)
            price = form.price
            max_price = price * FALLBACK_PRICE_FACTOR if price is not None else None
            train_plan = runner.build_plan(
                dataset=dataset,
                hub_dataset=status.hub_dataset,
                policy=policy,
                gpu_id=form.gpu.gpu_id,
                cloud=form.cloud,
                hf_user=user,
                steps=form.steps,
                batch=form.batch,
                max_hours=form.max_hours,
                log_freq=form.log_freq,
                extra_train_args=form.extra_train_args,
                streaming=form.streaming,
                image_transforms=form.image_transforms,
                lerobot_ref=ref.ref,
                max_price=max_price,
            )
            lines = [f"{k:<10} {v}" for k, v in train_plan.summary()]
            if price is not None and max_price is not None:
                lines.append(
                    f"{'cost':<10} ${price:.2f}/h, at most ~${price * form.max_hours:.2f} "
                    f"(~${max_price * form.max_hours:.2f} on a fallback GPU)"
                )
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
    #log:focus { border: round $accent; }
    """

    def __init__(
        self, services: Services, train_plan: plan.TrainPlan, token: str, attach_pod: str | None = None
    ):
        super().__init__()
        self.services = services
        self.train_plan = train_plan
        self.training = services.make_run(train_plan, token, self._emit)
        self.attach_pod = attach_pod
        self.wait_for_capacity = False
        self.stage = "pod"
        self.finished = False

    def compose(self) -> ComposeResult:
        yield Header()
        p = self.train_plan
        yield Static(self._plan_text(), id="plan", markup=False)
        yield Static("", id="stages")
        with Horizontal(id="progress-row"):
            yield ProgressBar(total=p.steps, show_eta=False, id="progress")
            yield Static("", id="stats")
        yield Static("loss: -", id="loss")
        yield VimLog(id="log", max_lines=5000, markup=False, wrap=False)
        yield Footer()

    def _plan_text(self) -> str:
        p = self.training.plan
        parts = [
            f"{p.policy.name} on {p.hub_dataset}",
            f"{p.steps} steps",
            f"{p.gpu_id} ({p.cloud})",
            f"max {p.max_hours}h",
        ]
        if p.streaming:
            parts.append("streaming")
        if p.image_transforms:
            parts.append("image transforms")
        return " · ".join(parts)

    def on_mount(self) -> None:
        self._render_stages()
        self.query_one("#log").focus()
        self.drive()

    def _emit(self, event: runner.RunEvent) -> None:
        self.app.call_from_thread(self.handle_event, event)

    def log_line(self, text: str | Text) -> None:
        log = self.query_one("#log", RichLog)
        # don't yank the view back down while someone scrolled up to read
        log.write(text, scroll_end=log.is_vertical_scroll_end)

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
                # a fallback GPU may have been placed instead of the chosen one
                self.query_one("#plan", Static).update(self._plan_text())
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
                run.create_pod(wait=self.wait_for_capacity)
        except runpod.CapacityError as exc:
            if run.stop.is_set() or self.wait_for_capacity:
                self.app.call_from_thread(self._back_to_setup, str(exc))
            else:
                self.app.call_from_thread(self.ask_to_wait, str(exc))
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

    @work(exclusive=True, group="ask")
    async def ask_to_wait(self, message: str) -> None:
        wait = await self.app.push_screen_wait(
            ConfirmScreen(
                f"{message}\n\nKeep checking every {CAPACITY_POLL_S}s and create the pod as soon as "
                "one is free? Press x to stop waiting.",
                yes="Wait for a GPU",
                no="Back",
                default_yes=True,
            )
        )
        if not wait:
            self.app.pop_screen()
            return
        self.wait_for_capacity = True
        self.log_line(Text(f"==> waiting for capacity, checking every {CAPACITY_POLL_S}s", style="yellow"))
        self.drive()

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
            # cancels waiting for capacity, if that's what's going on
            self.training.stop.set()
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
            if not isinstance(screen, RunScreen):
                continue
            screen.training.stop.set()
            if screen.training.pod_id:
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


def run_tui(lerobot_ref: str | None = None) -> None:
    services = Services(lerobot_ref=lambda: plan.LerobotRef(lerobot_ref)) if lerobot_ref else None
    app = LerobotRunpodApp(services)
    app.run()
