from __future__ import annotations

import json
import threading

import pytest

from lerobot_runpod import logs
from lerobot_runpod.logs import Done, Failed, Log, Metrics, Progress, Stage, Warn

GLUED_DONE = (
    "\rTraining: 100%|██████████| 200/200 [03:01<00:00,  1.10step/s]"
    "INFO 2026-10-18 21:45:02 ot_train.py:970 step:200 smpl:6K ep:12 epch:0.24 loss:3.150 "
    "grdn:55.364 lr:1.0e-05 data_s:0.010 updt_s:0.300"
)
SMOLVLA_STEP100 = (
    "\rTraining:   1%|          | 99/10000 [01:18<1:54:40,  1.44step/s]"
    "\rTraining:   1%|          | 100/10000 [01:19<1:54:36,  1.44step/s]"
    "INFO 2026-10-18 21:01:53 ot_train.py:970 step:100 smpl:3K ep:6 epch:0.12 loss:0.546 grdn:10.161 "
    "lr:5.1e-06 data_s:0.083 prep_s:0.003 updt_s:0.703 step_s:0.792 smp/s:40 mem_gb:18.41 "
    "losses_after_forward:0.536 losses_after_in_ep_bound:0.522 losses_after_rm_padding:0.522"
)
SLOW_FRAMES = (
    "\rTraining:   0%|          | 1/10000 [00:10<29:25:36, 10.59s/step]"
    "\rTraining:   0%|          | 2/10000 [00:11<13:14:01,  4.77s/step]"
    "\rTraining:   0%|          | 3/10000 [00:11<8:03:45,  2.90s/step] "
)
START = (
    "\rTraining:   0%|          | 0/10000 [00:00<?, ?step/s]INFO 2026-10-18 21:00:33 ot_train.py:918 "
    "Start offline training on a fixed dataset, with effective batch size: 32"
)


def test_glued_info_line_after_final_frame():
    progress, metrics = logs.parse_line(GLUED_DONE)
    assert isinstance(progress, Progress)
    assert (progress.step, progress.total, progress.elapsed_s, progress.eta_s) == (200, 200, 181, 0)
    assert progress.steps_per_s == pytest.approx(1.10)
    assert isinstance(metrics, Metrics)
    assert (metrics.step, metrics.loss, metrics.grad_norm, metrics.lr) == (200, 3.150, 55.364, 1.0e-05)
    assert metrics.text.startswith("INFO 2026-10-18")


def test_loss_does_not_match_prefixed_loss_keys():
    progress, metrics = logs.parse_line(SMOLVLA_STEP100)
    assert progress.step == 100
    assert progress.eta_s == 1 * 3600 + 54 * 60 + 36
    assert metrics.loss == 0.546
    assert metrics.lr == pytest.approx(5.1e-06)
    assert metrics.grad_norm == 10.161


def test_kld_loss_alone_is_not_the_loss():
    (metrics,) = logs.parse_line("INFO x step:1K loss:1.250 kld_loss:9.999 grdn:2.0 lr:1.0e-04")
    assert metrics.loss == 1.25
    assert metrics.step == 1000


def test_only_last_tqdm_frame_is_kept_and_seconds_per_step_inverted():
    (progress,) = logs.parse_line(SLOW_FRAMES)
    assert progress.step == 3
    assert progress.eta_s == 8 * 3600 + 3 * 60 + 45
    assert progress.steps_per_s == pytest.approx(1 / 2.90)
    assert progress.text.startswith("0%|")


def test_unknown_rate_and_glued_plain_info():
    progress, log = logs.parse_line(START)
    assert (progress.step, progress.eta_s, progress.steps_per_s) == (0, None, None)
    assert isinstance(log, Log) and log.text.startswith("INFO") and "Start offline training" in log.text


def test_info_glued_to_an_earlier_frame_is_not_dropped():
    line = (
        "\rTraining:  1%| | 100/10000 [01:19<1:54:36,  1.44step/s]INFO x step:100 loss:0.5 grdn:1 lr:1e-05"
        "\rTraining:  1%| | 101/10000 [01:20<1:54:30,  1.44step/s]"
    )
    events = logs.parse_line(line)
    assert events[0].step == 101
    assert isinstance(events[1], Metrics) and events[1].loss == 0.5


@pytest.mark.parametrize(
    ("line", "event"),
    [
        ("@@STAGE install@@", Stage("install")),
        ("@@DONE@@", Done()),
        ("@@FAILED train rc=1@@", Failed("train rc=1")),
        ("@@FAILED timeout: hit the 6h limit@@", Failed("timeout: hit the 6h limit")),
        (
            "@@WARN no RUNPOD_API_KEY: this pod cannot delete itself@@",
            Warn("no RUNPOD_API_KEY: this pod cannot delete itself"),
        ),
        (
            "INFO 2026-10-18 21:00:33 ot_train.py:800 cfg.steps=10000 (10K)",
            Log("INFO 2026-10-18 21:00:33 ot_train.py:800 cfg.steps=10000 (10K)"),
        ),
    ],
)
def test_markers(line, event):
    assert logs.parse_line(line) == [event]


def test_blank_lines_are_ignored():
    assert logs.parse_line("") == []
    assert logs.parse_line("\r   ") == []


def test_parse_record():
    raw = json.dumps({"source": "container", "line": "@@STAGE env@@", "ts": "2026-10-18T21:00:03.122211573Z"})
    assert logs.parse_record(raw) == logs.Record(
        "container", "@@STAGE env@@", "2026-10-18T21:00:03.122211573Z"
    )
    assert logs.parse_record("not json\n").line == "not json"


def test_ts_key_orders_trimmed_fractions():
    assert logs.ts_key("2026-10-18T21:00:17.2257833Z") < logs.ts_key("2026-10-18T21:00:17.225783329Z")
    assert logs.ts_key("2026-10-18T21:00:17Z") < logs.ts_key("2026-10-18T21:00:17.000000001Z")


class FakeProc:
    def __init__(self, lines):
        self.stdout = iter(lines)
        self.killed = False

    def poll(self):
        return 0

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return 0


class FakePopen:
    def __init__(self, streams):
        self.streams = list(streams)
        self.cmds = []

    def __call__(self, cmd, **kwargs):
        self.cmds.append(cmd)
        return FakeProc(self.streams.pop(0) if self.streams else [])


def rec(line, ts, source="container"):
    return json.dumps({"source": source, "line": line, "ts": ts}) + "\n"


def follower(popen, **kw):
    kw.setdefault("pod_exists", lambda pod: True)
    kw.setdefault("container_started", lambda pod: False)
    return logs.LogFollower("pod1", popen=popen, retry_delay=0, poll=0.01, quiet_timeout=0.05, **kw)


def test_phase1_stops_at_start_container():
    popen = FakePopen(
        [
            [
                rec("create container runpod/pytorch", "2026-10-18T20:57:58Z", "system"),
                rec("start container for runpod/pytorch:1.0.2: begin", "2026-10-18T20:57:59Z", "system"),
                rec("never read", "2026-10-18T20:58:00Z", "system"),
            ]
        ]
    )
    seen = []
    assert follower(popen).wait_for_container(seen.append) is True
    assert seen[-1].startswith("start container")
    assert popen.cmds[0][-2:] == ["system", "--follow"]


def test_phase1_falls_back_to_container_probe_on_eof():
    assert follower(FakePopen([[]]), container_started=lambda pod: True).wait_for_container(print) is True


def test_phase1_gives_up_when_pod_is_gone():
    assert follower(FakePopen([[]]), pod_exists=lambda pod: False).wait_for_container(print) is False


def test_phase2_reconnects_and_dedupes_by_ts():
    first = [rec("@@STAGE install@@", "2026-10-18T20:57:59.6Z"), rec("installing", "2026-10-18T20:58:00.1Z")]
    second = [
        rec("@@STAGE install@@", "2026-10-18T20:57:59.6Z"),
        rec("installing", "2026-10-18T20:58:00.1Z"),
        rec("@@STAGE env@@", "2026-10-18T21:00:03.122211573Z"),
        rec("@@DONE@@", "2026-10-18T21:10:00.5Z"),
    ]
    popen = FakePopen([first, second])
    seen = []

    def on_line(line):
        seen.append(line)
        return line == "@@DONE@@"

    assert follower(popen).follow_container(on_line) == "finished"
    assert seen == ["@@STAGE install@@", "installing", "@@STAGE env@@", "@@DONE@@"]
    assert len(popen.cmds) == 2


class HungProc(FakeProc):
    """Delivers its lines, then blocks like a --follow stream that stopped sending without closing."""

    def __init__(self, lines):
        super().__init__(lines)
        self._release = threading.Event()
        self.stdout = self._lines(lines)

    def _lines(self, lines):
        yield from lines
        self._release.wait(5)

    def poll(self):
        return None

    def kill(self):
        super().kill()
        self._release.set()


def test_phase2_restarts_a_stalled_stream():
    hung = HungProc(
        [rec("@@STAGE train@@", "2026-10-21T20:51:50Z"), rec("step:3K loss:0.375", "2026-10-21T20:51:54Z")]
    )
    fresh = [
        rec("step:3K loss:0.375", "2026-10-21T20:51:54Z"),
        rec("step:7K loss:0.183", "2026-10-21T21:05:29Z"),
        rec("@@DONE@@", "2026-10-21T22:30:00Z"),
    ]
    procs = [hung, FakeProc(fresh)]
    cmds = []

    def popen(cmd, **kwargs):
        cmds.append(cmd)
        return procs.pop(0)

    seen = []

    def on_line(line):
        seen.append(line)
        return line == "@@DONE@@"

    result = follower(popen, stall_timeout=0.1).follow_container(on_line)
    assert result == "finished"
    assert hung.killed
    assert seen == ["@@STAGE train@@", "step:3K loss:0.375", "step:7K loss:0.183", "@@DONE@@"]
    assert len(cmds) == 2


def test_phase2_reports_pod_gone():
    assert (
        follower(FakePopen([[]]), pod_exists=lambda pod: False).follow_container(lambda line: False)
        == "pod_gone"
    )
