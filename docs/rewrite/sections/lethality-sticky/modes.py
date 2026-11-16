"""Frontline and elimination 5v5 still work: `bench.mode_dynamics` (drawn lineups, fields
on, 256 games each) and the freeze check's measures (`freeze_report.measures`), into
data/modes-LABEL.jsonl. Prints each mode's measures.

    PYTHONPATH=src:. uv run --no-sync python \
        docs/rewrite/sections/lethality-sticky/modes.py LABEL OUT_DIR
"""

import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "freeze-final"))
import freeze_report  # noqa: E402

label, out_dir = sys.argv[1], pathlib.Path(sys.argv[2])
path = out_dir / f"modes-{label}.jsonl"
for mode in ("frontline", "elimination"):
    tag = f"{label}-{mode}"
    if not freeze_report.load(path, tag)[0]:
        command = [
            sys.executable, "-m", "bench.mode_dynamics", mode, "--games", "256", "--batch",
            "256", "--ships", "5", "--fields", "--seed", "23", "--drawn-lineups", "--label",
            tag, "--out", str(path),
        ]  # fmt: skip
        subprocess.run(command, check=True)
    games, entries = freeze_report.load(path, tag)
    print(tag, json.dumps(freeze_report.measures(mode, games, entries)), flush=True)
