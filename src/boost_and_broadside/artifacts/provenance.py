"""Execution, code, and machine provenance recorded in every durable artifact.

An artifact has to answer "what produced this, from which code, on what" long
after the shell that made it is gone. Everything recorded here is an allowlist:
the environment description names a fixed set of variables that change how a
measurement runs and nothing else, because dumping the process environment into
a file that may be committed is how credentials escape.

The repository this describes is the one the *package* was imported from, not
the working directory, so a mode that runs inside a temporary root still records
the code that actually executed.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

import boost_and_broadside

# Variables that change what a measurement does. Values are recorded because
# each is a device/backend selection, not a credential; anything outside this
# list is not recorded at all, not even its name.
_ENVIRONMENT_ALLOWLIST = (
    "CUDA_VISIBLE_DEVICES",
    "CUBLAS_WORKSPACE_CONFIG",
    "PYTORCH_CUDA_ALLOC_CONF",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "MPLBACKEND",
)

PRODUCER_NAME = "boost-and-broadside"


def package_repository() -> Path:
    """The checkout the imported package lives in (``src`` layout), if any."""

    return Path(boost_and_broadside.__file__).resolve().parents[2]


def _git(repository: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def code_provenance(repository: Path | None = None) -> dict[str, Any]:
    """Git commit, cleanliness, and locked-dependency identity of the code that ran.

    An installed wheel has no checkout to interrogate; that is recorded as an
    absent commit rather than guessed at. Publication requires a clean commit
    (see the publication manifest), so ``dirty`` is a decision input, not a note.
    """

    root = package_repository() if repository is None else Path(repository)
    commit = _git(root, "rev-parse", "HEAD")
    status = None if commit is None else _git(root, "status", "--porcelain=v1")
    lock = root / "uv.lock"
    return {
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
        "repository_available": commit is not None,
        "uv_lock_sha256": (
            hashlib.sha256(lock.read_bytes()).hexdigest() if lock.is_file() else None
        ),
    }


def _device_provenance(device: str | None) -> dict[str, Any]:
    """Identity of the accelerator a measurement ran on, where one exists."""

    record: dict[str, Any] = {"requested": device}
    if device is None or not device.startswith("cuda") or not torch.cuda.is_available():
        return record
    index = torch.device(device).index or 0
    properties = torch.cuda.get_device_properties(index)
    uuid = getattr(properties, "uuid", None)
    record.update(
        {
            "name": properties.name,
            "index": index,
            "total_memory_bytes": int(properties.total_memory),
            "capability": f"{properties.major}.{properties.minor}",
            "uuid": None if uuid is None else str(uuid),
        }
    )
    return record


def runtime_provenance(device: str | None = None) -> dict[str, Any]:
    """Interpreter, framework, accelerator, and determinism settings."""

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "device": _device_provenance(device),
    }


def environment_provenance() -> dict[str, str]:
    """The allowlisted environment variables that are actually set."""

    return {name: os.environ[name] for name in _ENVIRONMENT_ALLOWLIST if name in os.environ}


def normalized_command(argv: Sequence[str]) -> str:
    """The canonical ``uv run ...`` spelling of an invocation.

    The original argv is recorded verbatim beside this; the normalized form is
    what a reader can paste to reproduce the measurement regardless of whether
    it was launched through ``uv run``, an activated virtual environment, or an
    absolute path to the installed executable.

    Twelve of the thirteen producers are ``bnb`` subcommands, whose ``argv[0]``
    is the installed executable and carries no information. Ingestion is a
    script, and dropping *its* ``argv[0]`` produced a normalized command with no
    subcommand in it — something that would fail at argparse, in the one field a
    reader is invited to paste. A script keeps its path.
    """

    if not argv:
        return "uv run bnb"
    if argv[0].endswith(".py"):
        return " ".join(["uv", "run", *argv])
    return " ".join(["uv", "run", "bnb", *argv[1:]])


def producer_provenance() -> dict[str, Any]:
    """Which producer wrote the artifact, and at which package version."""

    return {"name": PRODUCER_NAME, "version": _package_version()}


def _package_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("boost-and-broadside")
    except PackageNotFoundError:  # pragma: no cover - editable checkout without metadata
        return "unknown"


__all__ = [
    "code_provenance",
    "environment_provenance",
    "normalized_command",
    "package_repository",
    "producer_provenance",
    "runtime_provenance",
]
