"""Periodic backup of run folders and evaluation data to a private Hugging Face repository.

    uv run --no-sync python box/hf_backup.py --profile box        # on the rented box
    uv run --no-sync python box/hf_backup.py --profile laptop     # on the laptop
    ... --dry-run                                                 # the plan, no token needed

Each source folder is uploaded with `HfApi.upload_folder(path_in_repo=...)`, which in
huggingface_hub 2.x (with hf_xet) streams the files in several commits, skips files the repo
already holds and resumes when the same call is repeated. A run folder is one source, so a
finished run costs one listing per backup. Compile caches, skipped-step dumps, bytecode, secrets,
pid files, virtual environments and .git are never uploaded. A lock keeps two backups from
overlapping. The token comes from HF_TOKEN, else ~/.config/bnb/hf.key, else HF_TOKEN in
/workspace/secrets.env; it is never printed or logged.

One log line per source: bytes uploaded (new or changed), bytes skipped (already in the repo, by
size), files, seconds. Exit status 1 if any source failed (the others still run).
"""

import argparse
import contextlib
import dataclasses
import fcntl
import os
import pathlib
import re
import sys
import tempfile
import time

from huggingface_hub.utils import filter_repo_objects

REPOSITORY = "julianzia/boost-and-broadside-runs"

# Names whose folders are never uploaded, at any depth, and file patterns likewise.
IGNORED_FOLDERS = (
    "__pycache__",
    ".git",
    "venv",
    ".venv",
    "inductor",
    "torchinductor*",
    "triton",
    ".triton",
    "dumps",  # the skipped-step batch dumps (~10 GB each at full scale)
)
IGNORED_FILES = ("*.dump", "*.pid", "*.key", "*secrets.env", "*.pyc", "*.tmp", "*.lock")


def ignore_patterns() -> list[str]:
    """The fnmatch patterns (on the path relative to a source; `*` crosses `/`) skipped."""
    patterns = list(IGNORED_FILES)
    for name in IGNORED_FOLDERS:
        patterns += [f"{name}/*", f"*/{name}/*"]
    return patterns


@dataclasses.dataclass
class Source:
    folder: pathlib.Path
    prefix: str  # the path in the repo
    allow: list[str] | None = None  # only these files (a folder's loose top-level files)


def expand(root: pathlib.Path, prefix: str, skip: tuple[str, ...] = ()) -> list[Source]:
    """One source per child folder of `root`, and one for its loose top-level files."""
    sources, loose = [], []
    if not root.is_dir():
        return sources
    for child in sorted(root.iterdir()):
        if child.name in skip or child.name in IGNORED_FOLDERS:
            continue
        if child.is_dir():
            sources.append(Source(child, f"{prefix}/{child.name}"))
        elif child.is_file():
            loose.append(child.name)
    loose = list(filter_repo_objects(loose, ignore_patterns=ignore_patterns()))
    if loose:
        sources.append(Source(root, prefix, allow=loose))
    return sources


def profile_sources(name: str) -> list[Source]:
    home = pathlib.Path.home() / "avesnova"
    if name == "box":
        volume = pathlib.Path(os.environ.get("VOLUME", "/workspace"))
        return expand(volume / "runs", "box/runs")
    if name == "laptop":
        sources = expand(home / "ueval-runs", "laptop/ueval-runs", skip=("cache",))
        sources += expand(home / "watch-runs", "laptop/watch-runs")
        sources.append(Source(home / "boost-and-broadside-v2" / "review", "laptop/review"))
        return [s for s in sources if s.folder.exists()]
    raise ValueError(f"unknown profile {name!r}")


def plan(source: Source) -> list[tuple[str, int]]:
    """(path relative to the source, size) of the files the source would upload."""
    names = []
    for directory, folders, files in os.walk(source.folder):
        folders[:] = [f for f in folders if f not in IGNORED_FOLDERS]
        if source.allow is not None and directory != str(source.folder):
            folders[:] = []
            continue
        relative = pathlib.Path(directory).relative_to(source.folder)
        names += [str(relative / f) if str(relative) != "." else f for f in files]
    if source.allow is not None:
        names = [n for n in names if n in set(source.allow)]
    kept = filter_repo_objects(names, ignore_patterns=ignore_patterns())
    sizes = []
    for name in kept:
        with contextlib.suppress(OSError):  # a file removed mid-scan
            sizes.append((name, (source.folder / name).stat().st_size))
    return sizes


def gigabytes(size: int) -> str:
    return f"{size / 1e9:.2f} GB" if size >= 1e8 else f"{size / 1e6:.1f} MB"


def read_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token.strip()
    key = pathlib.Path.home() / ".config" / "bnb" / "hf.key"
    if key.is_file():
        return key.read_text().strip() or None
    secrets = pathlib.Path("/workspace/secrets.env")
    if secrets.is_file():
        match = re.search(
            r"^\s*(?:export\s+)?HF_TOKEN=['\"]?([^'\"\s]+)", secrets.read_text(), re.M
        )
        if match:
            return match.group(1)
    return None


def remote_sizes(api, repository: str, source: Source, token: str) -> dict[str, int]:
    from huggingface_hub.errors import EntryNotFoundError

    sizes = {}
    try:
        for entry in api.list_repo_tree(
            repository, path_in_repo=source.prefix, recursive=True, token=token
        ):
            if hasattr(entry, "size") and not entry.path.endswith("/"):
                sizes[entry.path] = entry.size
    except EntryNotFoundError:
        pass
    return sizes


def upload(api, repository: str, source: Source, token: str) -> tuple[int, int, int]:
    """Upload one source; returns (bytes uploaded, bytes skipped, files)."""
    files = plan(source)
    if not files:
        return 0, 0, 0
    remote = remote_sizes(api, repository, source, token)
    skipped = sum(s for n, s in files if remote.get(f"{source.prefix}/{n}") == s)
    total = sum(s for _, s in files)
    if skipped < total:
        api.upload_folder(
            repo_id=repository,
            repo_type="model",
            folder_path=source.folder,
            path_in_repo=source.prefix,
            allow_patterns=source.allow,
            ignore_patterns=ignore_patterns(),
            commit_message=f"backup {source.prefix}",
            token=token,
        )
    return total - skipped, skipped, len(files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--profile", choices=("box", "laptop"), required=True)
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        metavar="FOLDER=PREFIX",
        help="an additional folder and its path in the repo (repeatable)",
    )
    parser.add_argument("--repository", default=REPOSITORY)
    parser.add_argument("--dry-run", action="store_true", help="list the plan with sizes; no token")
    parser.add_argument(
        "--lock",
        type=pathlib.Path,
        default=pathlib.Path(tempfile.gettempdir()) / "bnb-hf-backup.lock",
    )
    args = parser.parse_args()

    sources = profile_sources(args.profile)
    for extra in args.extra:
        folder, prefix = extra.split("=", 1)
        sources.append(Source(pathlib.Path(folder), prefix))

    if args.dry_run:
        grand = 0
        for source in sources:
            files = plan(source)
            size = sum(s for _, s in files)
            grand += size
            print(f"{source.prefix:60s} {len(files):6d} files {gigabytes(size):>10s}", flush=True)
        print(f"{'total':60s} {'':6s}       {gigabytes(grand):>10s}")
        return 0

    lock = open(args.lock, "w")  # noqa: SIM115 - held for the process lifetime
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("hf_backup: another backup holds the lock; nothing done", flush=True)
        return 0
    token = read_token()
    if token is None:
        print(
            "hf_backup: no token (HF_TOKEN, ~/.config/bnb/hf.key or /workspace/secrets.env)",
            file=sys.stderr,
        )
        return 2
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repository, repo_type="model", private=True, exist_ok=True, token=token)
    failed = 0
    for source in sources:
        started = time.monotonic()
        try:
            uploaded, skipped, count = upload(api, args.repository, source, token)
            status = "ok"
        except Exception as error:  # report and go on with the other sources
            uploaded = skipped = count = 0
            failed += 1
            status = f"FAILED {type(error).__name__}"
        print(
            f"hf_backup: {source.prefix} {status} uploaded={gigabytes(uploaded)} "
            f"skipped={gigabytes(skipped)} files={count} seconds={time.monotonic() - started:.0f}",
            flush=True,
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
