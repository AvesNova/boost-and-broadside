"""Where a result came from (runs spec 8): the commit, the machine, the versions, the
command and the time. Checkpoints, run segments and offline results all record it."""

import datetime
import hashlib
import json
import pathlib
import platform
import socket
import subprocess
import sys

import torch

REPOSITORY = pathlib.Path(__file__).resolve().parents[3]


def _git(*arguments: str) -> str | None:
    try:
        done = subprocess.run(
            ("git", *arguments), cwd=REPOSITORY, capture_output=True, text=True, timeout=10
        )
    except OSError, subprocess.TimeoutExpired:
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
    return dict(
        commit=head,
        dirty=dirty,
        host=socket.gethostname(),
        gpu=gpu,
        python=platform.python_version(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        command=list(sys.argv),
        time=now(),
    )


def file_hash(path: pathlib.Path) -> str:
    """A file's SHA-256, hex."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def policy_hash(directory: pathlib.Path) -> str:
    """A policy file's identity: the SHA-256 of its checkpoint.json, which holds the hash of
    every other file."""
    return file_hash(pathlib.Path(directory) / "checkpoint.json")


def write_result(
    run, kind: str, payload: dict, inputs: dict[str, str | pathlib.Path]
) -> pathlib.Path:
    """Write an offline result (ratings, head-to-head, clips) of `run` (a `records.Run` or
    its directory) as `artifacts/<kind>/<id>/result.json`: the provenance, the payload and
    the inputs, each policy file by path and hash. Returns the result's directory."""
    run_directory = pathlib.Path(getattr(run, "directory", run))
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d-%H%M%S")
    parent = run_directory / "artifacts" / kind
    parent.mkdir(parents=True, exist_ok=True)
    count = sum(1 for path in parent.iterdir() if path.name.startswith(stamp))
    directory = parent / (stamp if count == 0 else f"{stamp}-{count + 1}")
    directory.mkdir()
    result = dict(
        provenance=provenance(),
        kind=kind,
        payload=payload,
        inputs={
            name: dict(path=str(pathlib.Path(path).resolve()), sha256=policy_hash(path))
            for name, path in inputs.items()
        },
    )
    temporary = directory / "result.json.tmp"
    temporary.write_text(json.dumps(result, indent=1) + "\n")
    temporary.replace(directory / "result.json")
    return directory
