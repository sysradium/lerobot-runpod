from __future__ import annotations

import argparse
import sys

from . import __version__, cameras, datasets, deliver, hf, logs, plan, policies, runner, runpod
from .config import find_lerobot_src, lerobot_home
from .shell import missing_binaries

TTY = sys.stdout.isatty()


def _style(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if TTY else text


def info(msg: str) -> None:
    print(f"{_style('36', '==>')} {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"{_style('33', '!!')} {msg}", file=sys.stderr, flush=True)


def bold(msg: str) -> None:
    print(_style("1", msg), flush=True)


class Abort(Exception):
    pass


def confirm(question: str, *, default: bool = False, assume_yes: bool = False) -> bool:
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        return default
    hint = "[Y/n]" if default else "[y/N]"
    try:
        reply = input(f"{question} {hint}: ").strip().lower()
    except EOFError:
        return default
    return default if not reply else reply.startswith("y")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lerobot-runpod",
        description="Train LeRobot policies on RunPod GPUs from LeLab datasets. "
        "Run without a command for the TUI.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")

    train = sub.add_parser("train", help="train non-interactively")
    train.add_argument(
        "--dataset", required=True, help="HF dataset id, or a LeLab cache id like local/<name>"
    )
    train.add_argument("--policy", required=True, choices=list(policies.POLICIES))
    train.add_argument("--steps", type=int, help="training steps (default: per policy)")
    train.add_argument("--batch", type=int, help="batch size (default: per policy)")
    train.add_argument(
        "--max-hours", type=int, help="hard limit on pod lifetime; the pod deletes itself after"
    )
    train.add_argument("--cloud", default="SECURE", type=str.upper, choices=runpod.CLOUDS)
    train.add_argument("--gpu", help='RunPod gpu id, e.g. "NVIDIA GeForce RTX 4090" (default: cheapest fit)')
    train.add_argument("--repo", help="model repo to push to (default <hf_user>/<policy>_<dataset>_<ts>)")
    train.add_argument(
        "--push-dataset", action="store_true", help="push a local-only dataset to the Hub (private)"
    )
    train.add_argument("--yes", "-y", action="store_true", help="skip the final confirmation")
    train.add_argument(
        "--dry-run", action="store_true", help="print the plan and pod command, create nothing"
    )

    sub.add_parser("datasets", help="list LeLab datasets in the local cache")

    gpus = sub.add_parser("gpus", help="list RunPod GPUs with prices")
    gpus.add_argument("--min-vram", type=int, default=24)
    gpus.add_argument("--cloud", default="SECURE", type=str.upper, choices=runpod.CLOUDS)

    pods = sub.add_parser("pods", help="list (and delete) lerobot-* pods")
    pods.add_argument("--delete", metavar="POD_ID", action="append", default=[], help="delete this pod")
    pods.add_argument("--delete-all", action="store_true", help="delete every lerobot-* pod")
    pods.add_argument("--yes", "-y", action="store_true")

    fetch = sub.add_parser("fetch", help="deliver an already-trained HF model into LeLab")
    fetch.add_argument("repo")
    fetch.add_argument("--steps", type=int, help="override the step count read from train_config.json")
    fetch.add_argument("--policy")
    fetch.add_argument("--dataset")

    rename = sub.add_parser("rename-camera", help="copy a local dataset with one camera renamed")
    rename.add_argument("dataset", help="local dataset id, e.g. local/pick_cube")
    rename.add_argument("old", help='camera name, e.g. "front" for observation.images.front')
    rename.add_argument("new", help="new camera name, e.g. camera1")
    rename.add_argument("--to", help="new dataset id (default <owner>/<name>_<new>)")
    return parser


def cmd_datasets(_: argparse.Namespace) -> int:
    found = datasets.discover()
    if not found:
        warn(f"no datasets found under {lerobot_home()}")
        return 1
    for d in found:
        print(f"{d.modified}  {d.repo_id:<50} {d.episodes:>4} eps {d.frames:>8} frames {d.fps:>3} fps")
    return 0


def cmd_gpus(args: argparse.Namespace) -> int:
    for o in runpod.filter_offers(runpod.list_gpus(), args.min_vram, args.cloud):
        print(f"${o.price(args.cloud):<6.2f}/h  {o.memory_gb:>3}GB  {o.stock or '-':<7} {o.gpu_id}")
    return 0


def cmd_pods(args: argparse.Namespace) -> int:
    pods = runpod.own_pods(runpod.list_pods())
    for p in pods:
        print(
            f"{p.get('id')}  {p.get('name', ''):<32} {p.get('gpuId', ''):<28} "
            f"${p.get('costPerHr', 0)}/h  {p.get('desiredStatus', '')}  "
            f"up {int(p.get('uptimeSeconds') or 0) // 60} min"
        )
    if not pods:
        info("no lerobot-* pods")
    targets = [p["id"] for p in pods] if args.delete_all else args.delete
    for pod_id in targets:
        if confirm(f"Delete pod {pod_id}?", assume_yes=args.yes):
            info(f"pod {pod_id} deleted" if runpod.delete_pod(pod_id) else f"pod {pod_id} already gone")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    info(f"downloading {args.repo}")
    job_dir = deliver.fetch(args.repo, steps=args.steps, policy=args.policy, dataset=args.dataset)
    info(f"registered in LeLab: {job_dir}")
    print("  Restart LeLab (or refresh) to see the run in its training list.")
    return 0


def cmd_rename(args: argparse.Namespace) -> int:
    owner, name = datasets.split_id(args.dataset)
    target = args.to or f"{owner}/{name}_{args.new}"
    print(cameras.rename(datasets.local_path(args.dataset), datasets.local_path(target), args.old, args.new))
    info(f"new dataset: {target}. Rename the camera in LeLab's robot setup to {args.new!r} too.")
    return 0


def pick_gpu(policy: policies.Policy, cloud: str) -> str:
    offers = runpod.filter_offers(runpod.list_gpus(), policy.min_vram_gb, cloud)
    if not offers:
        raise Abort(f"no {cloud} GPU with >= {policy.min_vram_gb}GB available")
    cheapest = offers[0]
    info(f"cheapest fit: {cheapest.gpu_id} ({cheapest.memory_gb}GB) at ${cheapest.price(cloud):.2f}/h")
    return cheapest.gpu_id


def prepare(args: argparse.Namespace) -> tuple[plan.TrainPlan, str]:
    policy = policies.get(args.policy)
    hf_user = hf.whoami()
    token = hf.token()

    status = runner.dataset_status(args.dataset, hf_user)
    if status.on_hub:
        info(f"dataset on Hub: {status.hub_dataset}")
    else:
        if not status.can_push:
            raise Abort(f"{status.hub_dataset} is not on the Hub and {status.local_path} does not exist")
        warn(f"{status.hub_dataset} is not on the Hub yet")
        if args.dry_run:
            info(f"dry run: would push {status.local_path} to {status.hub_dataset} (private)")
        elif args.push_dataset or confirm(f"Push {status.local_path} to {status.hub_dataset} (private)?"):
            info(f"pushing {status.hub_dataset} (private)")
            hf.push_dataset(status.hub_dataset, status.local_path, find_lerobot_src())
            info(f"pushed {status.hub_dataset}")
            status = runner.DatasetStatus(status.dataset, status.hub_dataset, True, status.local_path)
        else:
            raise Abort("the dataset must be on the Hub for the pod to use it (pass --push-dataset)")

    if policy.base_model:
        check = runner.check_cameras(policy, status)
        info(f"cameras match {policy.base_model}: {', '.join(check.dataset) if check else '-'}")

    ref = plan.resolve_lerobot_ref(find_lerobot_src())
    if ref.warning:
        warn(ref.warning)

    gpu = args.gpu or pick_gpu(policy, args.cloud)
    train_plan = runner.build_plan(
        dataset=args.dataset,
        hub_dataset=status.hub_dataset,
        policy=policy,
        gpu_id=gpu,
        cloud=args.cloud,
        hf_user=hf_user,
        steps=args.steps,
        batch=args.batch,
        max_hours=args.max_hours,
        repo=args.repo,
        lerobot_ref=ref.ref,
    )
    return train_plan, token


def print_plan(train_plan: plan.TrainPlan) -> None:
    print()
    bold("Training plan")
    for key, value in train_plan.summary():
        print(f"  {key:<10} {value}")
    print()


class ConsolePrinter:
    def __init__(self) -> None:
        self.tqdm_open = False

    def _end_tqdm(self) -> None:
        if self.tqdm_open:
            print()
            self.tqdm_open = False

    def __call__(self, event: runner.RunEvent) -> None:
        if isinstance(event, logs.Progress):
            if TTY:
                print(f"\r\033[K{event.text}", end="", flush=True)
                self.tqdm_open = True
            return
        self._end_tqdm()
        match event:
            case runner.SystemLine(text=text):
                print(_style("2", f"[pod] {text}"), flush=True)
            case logs.Stage(name=name):
                bold(f"── {name} ──")
            case logs.Metrics(text=text) | logs.Log(text=text):
                print(text, flush=True)
            case logs.Warn(text=text):
                warn(text)
            case logs.Failed(reason=reason):
                warn(f"failed: {reason}")
            case logs.Done():
                info("training finished")
            case runner.Info(text=text, level=level):
                (warn if level == "warn" else info)(text)


def cmd_train(args: argparse.Namespace) -> int:
    train_plan, token = prepare(args)
    print_plan(train_plan)
    if args.dry_run:
        bold("Dry run, would execute:")
        print("  " + plan.format_command(train_plan.pod_command(token), token))
        return 0
    if not confirm("Create the pod and start training?", assume_yes=args.yes):
        raise Abort("aborted")

    run = runner.TrainingRun(train_plan, token, ConsolePrinter())
    try:
        run.create_pod()
    except runpod.CapacityError as exc:
        raise Abort(f"{exc}\n   see: lerobot-runpod gpus --cloud {train_plan.cloud}") from exc

    try:
        result = run.follow()
    except KeyboardInterrupt:
        run.stop.set()
        print()
        result = "interrupted"

    if result != "done":
        warn(result)
        warn(f"full log: {run.log_path}")
        teardown(run, args.yes)
        return 1

    run.delete_pod()
    try:
        ckpt = run.deliver()
    except RuntimeError as exc:
        raise Abort(
            f"{exc}\n   the model is on the Hub; retry with: lerobot-runpod fetch {train_plan.repo}"
        ) from exc
    minutes = int(((run.state.ended_at or 0) - run.state.started_at) / 60)
    print()
    bold("Done")
    print(f"  model    https://huggingface.co/{train_plan.repo}\n  local    {ckpt}\n  time     {minutes} min")
    print("  Restart LeLab (or refresh) to see the run in its training list.")
    return 0


def teardown(run: runner.TrainingRun, assume_yes: bool) -> None:
    if not run.pod_id:
        return
    warn(f"pod {run.pod_id} is still running (billing!)")
    try:
        delete = confirm(f"Delete pod {run.pod_id}?", default=True, assume_yes=assume_yes)
    except KeyboardInterrupt:
        delete = True
    if delete:
        run.delete_pod()
    else:
        warn(run.kept_message())


COMMANDS = {
    "train": cmd_train,
    "datasets": cmd_datasets,
    "gpus": cmd_gpus,
    "pods": cmd_pods,
    "fetch": cmd_fetch,
    "rename-camera": cmd_rename,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if missing := missing_binaries():
        print(f"missing dependency: {', '.join(missing)}", file=sys.stderr)
        return 1
    if args.command is None:
        from .tui import run_tui

        run_tui()
        return 0
    try:
        return COMMANDS[args.command](args)
    except (Abort, RuntimeError, ValueError) as exc:
        print(f"{_style('31', 'xx')} {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
