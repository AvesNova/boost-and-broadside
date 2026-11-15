"""hub_verify.py: before the box is given back (2026-11-10), every local file the hub should hold,
checked byte for byte against the hub's copy: runs/2026-*/** -> runs/<run>/**, runs/ratings/** ->
ratings/**, and runs/hub/runs/** (laptop runs fetched from the hub) -> runs/**. A file stored in LFS
is compared by SHA-256, any other by its git blob id. Missing or different files are uploaded
(`--repair`) and checked again. Writes the hub's whole file list (path, size, SHA-256 or blob id)
and the hub revision to OUT (a manifest for the git repo).
Usage: hub_verify.py OUT.jsonl [--repair]"""

import hashlib
import json
import os
import pathlib
import sys

from huggingface_hub import HfApi

REPO = "julianzia/boost-and-broadside-runs"
ROOT = pathlib.Path("/workspace/boost-and-broadside/runs")
api = HfApi(token=os.environ["HF_TOKEN"])


def hub_files() -> tuple[dict, str]:
    info = api.model_info(REPO, files_metadata=True)
    return {s.rfilename: s for s in info.siblings}, info.sha


def local_files() -> dict[str, pathlib.Path]:
    out = {}
    for run in sorted(ROOT.glob("2026-*")):
        if run.is_dir():
            for f in run.rglob("*"):
                if f.is_file() and ".tmp" not in str(f.relative_to(run)):
                    out[f"runs/{run.name}/{f.relative_to(run)}"] = f
    for f in (ROOT / "ratings").rglob("*"):
        if f.is_file():
            out[f"ratings/{f.relative_to(ROOT / 'ratings')}"] = f
    for f in (ROOT / "hub" / "runs").rglob("*"):
        if f.is_file():
            out[f"runs/{f.relative_to(ROOT / 'hub' / 'runs')}"] = f
    return out


def matches(path: pathlib.Path, sibling) -> bool:
    data = path.read_bytes()
    if sibling.lfs is not None:
        return hashlib.sha256(data).hexdigest() == sibling.lfs.sha256
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() == sibling.blob_id


def check(local: dict, hub: dict) -> tuple[list, list]:
    missing = [k for k in local if k not in hub]
    different = [k for k in local if k in hub and not matches(local[k], hub[k])]
    return missing, different


hub, revision = hub_files()
local = local_files()
missing, different = check(local, hub)
print(f"{len(local)} local files, {len(hub)} hub files: {len(missing)} missing on the hub, "
      f"{len(different)} different")
for k in (missing + different)[:40]:
    print("   ", "missing" if k in missing else "different", k)
if "--repair" in sys.argv and (missing or different):
    for k in missing + different:
        api.upload_file(path_or_fileobj=str(local[k]), path_in_repo=k, repo_id=REPO,
                        commit_message=f"final sync {k}")
    hub, revision = hub_files()
    missing, different = check(local, hub)
    print(f"after the repair: {len(missing)} missing, {len(different)} different")
with open(sys.argv[1], "w") as f:
    f.write(json.dumps(dict(repo=f"https://huggingface.co/{REPO}", revision=revision,
                            files=len(hub))) + "\n")
    for name, s in sorted(hub.items()):
        f.write(json.dumps(dict(path=name, size=s.size,
                                sha256=s.lfs.sha256 if s.lfs else None,
                                git_blob=None if s.lfs else s.blob_id)) + "\n")
print(f"manifest: {sys.argv[1]} (hub revision {revision})")
