"""Parse lerobot-train output from the pod and follow `runpodctl pod logs`."""

from __future__ import annotations

import contextlib
import json
import queue
import re
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass

from . import runpod


@dataclass(frozen=True)
class Stage:
    name: str


@dataclass(frozen=True)
class Done:
    pass


@dataclass(frozen=True)
class Failed:
    reason: str


@dataclass(frozen=True)
class Warn:
    text: str


@dataclass(frozen=True)
class Progress:
    step: int
    total: int
    elapsed_s: int
    eta_s: int | None
    steps_per_s: float | None
    text: str

    @property
    def fraction(self) -> float:
        return self.step / self.total if self.total else 0.0


@dataclass(frozen=True)
class Metrics:
    step: int | None
    loss: float | None
    grad_norm: float | None
    lr: float | None
    text: str


@dataclass(frozen=True)
class Log:
    text: str


Event = Stage | Done | Failed | Warn | Progress | Metrics | Log

_STAGE = re.compile(r"@@STAGE (\S+?)@@")
_FAILED = re.compile(r"@@FAILED (.*?)@@")
_WARN = re.compile(r"@@WARN (.*?)@@")
_TQDM = re.compile(r"(\d+)/(\d+) \[([\d:]+)<([\d:?]+),\s*(\?|[\d.]+)\s*(s/step|step/s|s/it|it/s)")
_STEP = re.compile(r"\bstep:([\d.]+)([KMBT]?)")
# leading space: otherwise it matches kld_loss / losses_after_forward
_LOSS = re.compile(r" loss:([\d.]+)")
_GRDN = re.compile(r"\bgrdn:([\d.]+)")
_LR = re.compile(r"\blr:([\d.]+(?:e[+-]?\d+)?)")
_SUFFIX = {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000, "T": 1_000_000_000_000}


def _clock(value: str) -> int | None:
    if "?" in value:
        return None
    seconds = 0
    for part in value.split(":"):
        seconds = seconds * 60 + int(part)
    return seconds


def _float(pattern: re.Pattern[str], text: str) -> float | None:
    m = pattern.search(text)
    return float(m.group(1)) if m else None


def parse_progress(frame: str) -> Progress | None:
    m = _TQDM.search(frame)
    if not m:
        return None
    step, total, elapsed, eta, rate, unit = m.groups()
    steps_per_s: float | None = None
    if rate != "?":
        value = float(rate)
        steps_per_s = value if unit in ("step/s", "it/s") else (1 / value if value else None)
    text = frame.split("Training:", 1)[-1].strip()
    return Progress(int(step), int(total), _clock(elapsed) or 0, _clock(eta), steps_per_s, text)


def parse_metrics(text: str) -> Metrics | None:
    if "step:" not in text or " loss:" not in text:
        return None
    step = None
    if m := _STEP.search(text):
        # lerobot abbreviates big numbers (step:2K), so this is approximate past 1000
        step = int(float(m.group(1)) * _SUFFIX[m.group(2)])
    return Metrics(step, _float(_LOSS, text), _float(_GRDN, text), _float(_LR, text), text)


def _split_glued(segment: str) -> tuple[str | None, str]:
    """tqdm doesn't end its frame with a newline, so a log line can arrive glued after it."""
    if "Training:" not in segment:
        return None, segment
    start = segment.index("Training:")
    cut = segment.find("INFO", start)
    if cut == -1:
        return segment, ""
    return segment[:cut], segment[cut:]


def _parse_text(text: str) -> list[Event]:
    if not text.strip():
        return []
    if "@@DONE@@" in text:
        return [Done()]
    if m := _FAILED.search(text):
        return [Failed(m.group(1))]
    if m := _STAGE.search(text):
        return [Stage(m.group(1))]
    if m := _WARN.search(text):
        return [Warn(m.group(1))]
    if metrics := parse_metrics(text):
        return [metrics]
    return [Log(text.rstrip())]


def parse_line(line: str) -> list[Event]:
    """One container log line -> events. tqdm redraws with \\r, so only the last frame is kept
    as progress, but log lines glued to earlier frames are not dropped."""
    segments = line.split("\r")
    events: list[Event] = []
    progress = None
    for segment in segments:
        frame, rest = _split_glued(segment)
        if frame is not None:
            progress = parse_progress(frame) or progress
        events.extend(_parse_text(rest))
    if progress is not None:
        events.insert(0, progress)
    return events


@dataclass(frozen=True)
class Record:
    source: str
    line: str
    ts: str


def parse_record(raw: str) -> Record:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return Record("", raw.rstrip("\n"), "")
    if not isinstance(obj, dict):
        return Record("", raw.rstrip("\n"), "")
    return Record(str(obj.get("source", "")), str(obj.get("line", "")), str(obj.get("ts", "")))


def ts_key(ts: str) -> str:
    """RFC3339Nano trims trailing zeros, which breaks plain string comparison; pad the fraction."""
    if not ts:
        return ""
    base, _, frac = ts.rstrip("Z").partition(".")
    return f"{base}.{frac.ljust(9, '0')}"


_TIMEOUT = object()
_EOF = object()


class _Stream:
    def __init__(self, cmd: list[str], popen: Callable[..., subprocess.Popen[str]]):
        self._proc = popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._queue: queue.Queue[object] = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            self._queue.put(line)
        self._queue.put(_EOF)

    def get(self, timeout: float) -> object:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return _TIMEOUT

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            self._proc.wait(timeout=5)


class LogFollower:
    """Follows a pod's logs in two phases.

    A `runpodctl pod logs --follow` stream opened before the container exists only ever
    delivers platform (source=system) lines, so phase 1 follows system lines until the
    container starts, and phase 2 opens a fresh container stream, reconnecting on EOF or
    when it goes silent for `stall_timeout` (a --follow stream can hang without closing).
    """

    def __init__(
        self,
        pod_id: str,
        *,
        stop: threading.Event | None = None,
        popen: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
        pod_exists: Callable[[str], bool] = runpod.pod_exists,
        container_started: Callable[[str], bool] = runpod.container_started,
        quiet_timeout: float = 30,
        stall_timeout: float = 120,
        retry_delay: float = 5,
        poll: float = 1,
    ):
        self.pod_id = pod_id
        self.stop = stop or threading.Event()
        self._popen = popen
        self._pod_exists = pod_exists
        self._container_started = container_started
        self._quiet_timeout = quiet_timeout
        self._stall_timeout = stall_timeout
        self._retry_delay = retry_delay
        self._poll = poll

    def wait_for_container(self, on_line: Callable[[str], None]) -> bool:
        """Returns False if the pod disappeared or we were stopped."""
        while not self.stop.is_set():
            stream = _Stream(runpod.logs_command(self.pod_id, "system"), self._popen)
            try:
                quiet = 0.0
                while not self.stop.is_set():
                    item = stream.get(self._poll)
                    if item is _TIMEOUT:
                        quiet += self._poll
                        if quiet < self._quiet_timeout:
                            continue
                    if item is _TIMEOUT or item is _EOF:
                        # quiet stream or EOF: the container may already be up
                        if self._container_started(self.pod_id):
                            return True
                        if not self._pod_exists(self.pod_id):
                            return False
                        if item is _EOF:
                            break
                        quiet = 0.0
                        continue
                    quiet = 0.0
                    line = parse_record(str(item)).line
                    if line:
                        on_line(line)
                    if "start container" in line:
                        return True
            finally:
                stream.close()
            self.stop.wait(self._retry_delay)
        return False

    def follow_container(self, on_line: Callable[[str], bool]) -> str:
        """Feeds container lines to `on_line` until it returns True.

        Returns "finished", "pod_gone" or "stopped".
        """
        last = ""
        while not self.stop.is_set():
            stream = _Stream(runpod.logs_command(self.pod_id, "container"), self._popen)
            try:
                quiet = 0.0
                while not self.stop.is_set():
                    item = stream.get(self._poll)
                    if item is _TIMEOUT:
                        quiet += self._poll
                        if quiet < self._stall_timeout:
                            continue
                    if item is _TIMEOUT or item is _EOF:
                        break
                    quiet = 0.0
                    record = parse_record(str(item))
                    key = ts_key(record.ts)
                    # reconnects replay recent history
                    if key and last and key <= last:
                        continue
                    if key:
                        last = key
                    if record.line and on_line(record.line):
                        return "finished"
            finally:
                stream.close()
            if self.stop.is_set():
                break
            if not self._pod_exists(self.pod_id):
                return "pod_gone"
            self.stop.wait(self._retry_delay)
        return "stopped"
