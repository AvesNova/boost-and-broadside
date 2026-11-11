"""fetch_champion.py RUNS_DIR: download the champion's (and the null's) policies the
tournament rates from the public hub repo julianzia/boost-and-broadside-runs into
RUNS_DIR/<run>/policies/step_<n>/<variant>/, with no token (the repo is public). Every LFS
file's SHA-256 is checked against the hub's listing; a file already present and matching is
skipped. Standard library only (runs with any python)."""

import hashlib
import json
import pathlib
import sys
import urllib.request

REPO = "julianzia/boost-and-broadside-runs"
CHAMPION_BC = "2026-11-09-bc-deadreckon-split-hiddenphysics"
CHAMPION_RL = "2026-11-09-rl-deadreckon-split-hiddenphysics"
NULL_RL = "2026-11-09-rl-absolute"
WANTED = [(CHAMPION_BC, 120586240, "uniform_average")]
WANTED += [
    (CHAMPION_RL, step, variant)
    for step in (60817408, 121634816, 180355072)
    for variant in ("uniform_average", "live")
]
WANTED += [(NULL_RL, 180355072, "uniform_average")]


def listing(path: str) -> list[dict]:
    url = f"https://huggingface.co/api/models/{REPO}/tree/main/{path}"
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    runs = pathlib.Path(sys.argv[1])
    for run, step, variant in WANTED:
        remote = f"runs/{run}/policies/step_{step}/{variant}"
        for entry in listing(remote):
            if entry["type"] != "file":
                continue
            local = runs / entry["path"].removeprefix("runs/")
            want = entry.get("lfs", {}).get("oid")
            if local.exists() and (want is None or sha256(local) == want):
                continue
            local.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://huggingface.co/{REPO}/resolve/main/{entry['path']}"
            partial = local.with_suffix(local.suffix + ".part")
            urllib.request.urlretrieve(url, partial)
            if want is not None and sha256(partial) != want:
                raise SystemExit(f"{entry['path']}: SHA-256 differs from the hub's listing")
            partial.rename(local)
            print(f"fetched {entry['path']}", flush=True)
        print(f"ok {remote}", flush=True)


if __name__ == "__main__":
    main()
