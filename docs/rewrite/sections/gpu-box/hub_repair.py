"""hub_repair.py RUNS_DIR: check every local policy and checkpoint file against its own
checkpoint.json record (SHA-256) and against the hub's copy; where the hub's copy differs from
the record and the local file matches it, upload the local file again, then re-check.

Why: on 2026-11-09/10 the hub held a wrong copy (same size, different bytes) of 1 checkpoint and
3 policy files although the local files matched their records: uploads can pair a path with
the wrong content, so an upload is not trusted until its hash is checked."""

import hashlib
import json
import os
import pathlib
import sys

from huggingface_hub import HfApi

REPO = "julianzia/boost-and-broadside-runs"
runs = pathlib.Path(sys.argv[1])
api = HfApi(token=os.environ["HF_TOKEN"])


def hub_hashes() -> dict:
    info = api.model_info(REPO, files_metadata=True)
    return {s.rfilename: (s.lfs.sha256 if s.lfs else None) for s in info.siblings}


hub = hub_hashes()
repaired, local_bad = [], []
for record in sorted(runs.glob("2026-*/**/checkpoint.json")):
    if ".tmp" in str(record):
        continue
    for name, expected in json.load(open(record)).get("files", {}).items():
        path = record.parent / name
        key = "runs/" + str(path.relative_to(runs))
        if hub.get(key) == expected:
            continue
        if hub.get(key) is None:
            continue  # not uploaded yet: the next sync uploads it
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            local_bad.append(key)
            continue
        api.upload_file(path_or_fileobj=str(path), path_in_repo=key, repo_id=REPO,
                        commit_message=f"repair {key}")
        repaired.append(key)
if repaired:
    hub = hub_hashes()
still = [k for k in repaired if hub.get(k) is None]
print(f"hub repair: {len(repaired)} re-uploaded, {len(local_bad)} local files not matching "
      f"their own record, {len(still)} still wrong after the repair")
for k in repaired + local_bad:
    print("  ", k)
