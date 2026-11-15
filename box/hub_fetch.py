"""hub_fetch.py RUN: exit 1 unless RUN's 30M uniform average is on the hub; else download its
policy files and records into runs/hub/runs/RUN (the image's /venv/main has huggingface_hub)."""
import os
import sys

from huggingface_hub import HfApi, snapshot_download

run = sys.argv[1]
repo = "julianzia/boost-and-broadside-runs"
token = os.environ["HF_TOKEN"]
want = f"runs/{run}/policies/step_30408704/uniform_average/checkpoint.json"
if want not in HfApi(token=token).list_repo_files(repo):
    sys.exit(1)
snapshot_download(
    repo,
    local_dir="runs/hub",
    allow_patterns=[f"runs/{run}/policies/**", f"runs/{run}/run.json",
                    f"runs/{run}/config.json", f"runs/{run}/metrics.jsonl"],
    token=token,
)
