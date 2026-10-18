from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass


class CommandError(RuntimeError):
    def __init__(self, cmd: Sequence[str], returncode: int, stdout: str, stderr: str):
        self.cmd = list(cmd)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        detail = (stderr or stdout).strip().splitlines()
        tail = detail[-1] if detail else f"exit code {returncode}"
        super().__init__(f"{cmd[0]} {cmd[1] if len(cmd) > 1 else ''} failed: {tail}")

    @property
    def output(self) -> str:
        return f"{self.stdout}\n{self.stderr}"


@dataclass
class Result:
    returncode: int
    stdout: str
    stderr: str


def run(
    cmd: Sequence[str], *, check: bool = True, timeout: float | None = 120, cwd: str | None = None
) -> Result:
    proc = subprocess.run(list(cmd), capture_output=True, text=True, timeout=timeout, cwd=cwd)
    if check and proc.returncode != 0:
        raise CommandError(cmd, proc.returncode, proc.stdout, proc.stderr)
    return Result(proc.returncode, proc.stdout, proc.stderr)


def missing_binaries(names: Sequence[str] = ("runpodctl", "hf", "uv")) -> list[str]:
    return [name for name in names if shutil.which(name) is None]
