"""Where a result came from (runs spec 8): the commit, the machine, the versions, the
command and the time. Checkpoints, run segments and offline results all record it."""

import datetime
import hashlib
import pathlib
import platform
import socket
import subprocess
import sys

import torch

REPOSITORY = pathlib.Path(__file__).resolve().parents[3]


def _git(*arguments: str) -> str | None:
    try:
        done = subprocess.run(("git", *arguments), cwd=REPOSITORY, capture_output=True,
                              text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def commit() -> tuple[str | None, bool]:
    """(the checked-out commit, whether the tree has uncommitted changes); (None, False)
    outside a git checkout."""
    head = _git("rev-parse", "HEAD")
    if head is None:
        return None, False
    return head, bool(_git("status", "--porcelain", "--untracked-files=no"))


def now() -> str:
    """The time, as an ISO 8601 string with the UTC offset."""
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def provenance() -> dict:
    """The commit and whether the tree was dirty, the host and the GPU (when this process
    has used one: provenance never initialises CUDA), the Python, torch and CUDA versions,
    the command line and the time."""
    head, dirty = commit()
    gpu = torch.cuda.get_device_name() if torch.cuda.is_initialized() else None
    return dict(commit=head, dirty=dirty, host=socket.gethostname(), gpu=gpu,
                python=platform.python_version(), torch=torch.__version__,
                cuda=torch.version.cuda, command=list(sys.argv), time=now())


def file_hash(path: pathlib.Path) -> str:
    """A file's SHA-256, hex."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
