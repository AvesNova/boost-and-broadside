"""The on-disk cache of things that are slow to make and fixed by the code that made them
(the play shell's built tick, the scripted agent's ship characteristics).

`ROOT` is under the user's cache directory (`BNB_CACHE` overrides it). `fingerprint` hashes
the source of some of bnb's packages and torch's version, so an entry made by other code is
never read: a change to the game makes its entries again.
"""

import functools
import hashlib
import os
import pathlib

import platformdirs
import torch

ROOT = pathlib.Path(
    os.environ.get("BNB_CACHE") or platformdirs.user_cache_dir("boost-and-broadside")
)


@functools.cache
def fingerprint(*packages: str) -> str:
    """A hash of these bnb packages' (or modules', as "runtime/realtime.py") source and
    torch's version."""
    root = pathlib.Path(__file__).parent
    digest = hashlib.sha256(torch.__version__.encode())
    for package in packages:
        target = root / package
        paths = [target] if target.is_file() else sorted(target.rglob("*.py"))
        for path in paths:
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def path(kind: str, *parts, packages: tuple[str, ...], suffix: str) -> pathlib.Path:
    """Where the entry of this kind for these parts (their reprs) and code is kept."""
    key = hashlib.sha256(
        "|".join([fingerprint(*packages), *(repr(part) for part in parts)]).encode()
    ).hexdigest()[:32]
    return ROOT / kind / f"{key}{suffix}"
