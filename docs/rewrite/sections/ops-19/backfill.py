"""Log a lean.train train.log to W&B after the fact, as lean.train would have (lean.wandb_log).

    uv run --no-sync python docs/rewrite/sections/ops-19/backfill.py RUN_DIR/train.log
        [--run-dir DIR] [--mode online|offline] [--wandb-dir DIR] [--no-keep-id]
        [--checkpoint PATH | --no-checkpoint]

Reads the update lines (`<stage> update N steps S key value ... k=v ... advantage_shares [..]`),
the gradient decomposition (`<stage> update N gradients_json {...}`) and the warnings
(`WARNING value support: ...`, `WARNING approximate KL above ... (epochs: ...)`), and logs one
record per update to the W&B run lean.train names for the run directory (default: the log's
directory) and stage: `<dir>-<stage>` in boost-and-broadside-v2, the id kept in DIR/wandb.json
(`--no-keep-id`: a fresh id, nothing written to DIR), so a later `--resume DIR` continues the
same W&B run. The names are lean.train's: the list fields by position (league_slots/<kind>,
league_win_rates/<i>), the advantage shares by level (advantage_share_<level>, the
LearnConfig's levels), steps as environment_steps. An update printed twice (a resume from an
earlier checkpoint) keeps its last line. Lines a live run is still writing are skipped.

The config: the stage, the commit from the log's "== ... start commit X" line, and the learner's
and model's configs from `--checkpoint` (default: DIR's latest checkpoint; `--no-checkpoint`
skips them). Run it once per log: running it again appends the same updates to the same run.
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from lean import wandb_log  # noqa: E402

UPDATE = re.compile(r"^(bc|rl) update (\d+) (.*)$")
COMMIT = re.compile(r"^== .* start commit (\S+)")
RANGE_KINDS = ("narrow", "clamped")


def tokens(text: str) -> list[str]:
    """Space-separated tokens, a bracketed list as one token."""
    found, depth, current = [], 0, ""
    for character in text:
        if character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
        if character == " " and depth == 0:
            if current:
                found.append(current)
            current = ""
        else:
            current += character
    if current:
        found.append(current)
    return found


def number(text: str):
    value = float(text)
    return int(value) if value.is_integer() and "." not in text and "e" not in text else value


def update_line(rest: str, levels: tuple[str, ...]) -> dict:
    """An update line's fields (after `<stage> update N `) under lean.train's W&B names."""
    record, items, position = {}, tokens(rest), 0
    while position < len(items):
        item = items[position]
        if "=" in item and not item.startswith("["):
            name, value = item.split("=", 1)
            record[name] = number(value)
            position += 1
            continue
        value = items[position + 1]
        position += 2
        if value.startswith("["):
            entries = value[1:-1].split()
            if item == "league_slots":
                for entry in entries:
                    kind, count = entry.split(":")
                    record[f"league_slots/{kind}"] = int(count)
            elif item == "advantage_shares":
                for level, entry in zip(levels, entries, strict=True):
                    record[f"advantage_share_{level}"] = float(entry)
            else:
                for index, entry in enumerate(entries):
                    record[f"{item}/{index}"] = float(entry)
        else:
            record[wandb_log.STEP if item == "steps" else item] = number(value)
    return record


def parse(lines, levels: tuple[str, ...]) -> tuple[str | None, str | None, dict, int]:
    """(stage, commit, {update: record}, lines skipped) from a train.log's lines."""
    stage, commit, updates, skipped = None, None, {}, 0
    for line in lines:
        line = line.rstrip("\n")
        match = COMMIT.match(line)
        if match:
            commit = match.group(1)
            continue
        match = UPDATE.match(line)
        if not match:
            continue
        line_stage, index, rest = match.group(1), int(match.group(2)), match.group(3)
        if stage is None:
            stage = line_stage
        elif line_stage != stage:
            raise ValueError(f"two stages in one log ({stage}, {line_stage}): one log per stage")
        try:
            if rest.startswith("steps "):
                record = update_line(rest, levels)
                record["update"] = index
                record["kl_warning"] = 0
                for kind in RANGE_KINDS:
                    for level in levels:
                        record[f"range_warning_{kind}_{level}"] = 0
                updates[index] = record  # a later line for the same update replaces it
            elif rest.startswith("gradients_json "):
                updates[index].update(json.loads(rest.removeprefix("gradients_json ")))
            elif rest.startswith("WARNING value support: "):
                for name in rest.removeprefix("WARNING value support: ").split():
                    updates[index][f"range_warning_{name}"] = 1
            elif rest.startswith("WARNING approximate KL above "):
                updates[index]["kl_warning"] = 1
                epochs = re.search(r"\(epochs: ([^)]*)\)", rest)
                if epochs:
                    for epoch, value in enumerate(epochs.group(1).split()):
                        updates[index][f"approximate_kl_epoch_{epoch}"] = float(value)
        except ValueError, KeyError, IndexError:
            skipped += 1  # a line still being written, or one for an update not seen
    return stage, commit, updates, skipped


def latest(directory: Path) -> Path | None:
    found = sorted((directory / "checkpoints").glob("*.pt"))
    return found[-1] if found else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("log", type=Path)
    parser.add_argument("--run-dir", type=Path, default=None, help="default: the log's directory")
    parser.add_argument("--mode", choices=("online", "offline"), default="online")
    parser.add_argument("--wandb-dir", type=Path, default=None, help="default: the run directory")
    parser.add_argument(
        "--no-keep-id",
        action="store_true",
        help="a fresh W&B run id; nothing written to the run directory",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--no-checkpoint", action="store_true")
    arguments = parser.parse_args()
    directory = arguments.run_dir or arguments.log.resolve().parent

    config: dict = {"backfilled_from": arguments.log.name, "run_directory": directory.name}
    levels = None
    checkpoint_path = None if arguments.no_checkpoint else arguments.checkpoint or latest(directory)
    if checkpoint_path is not None:
        from lean.learn import LearnConfig, load_checkpoint

        checkpoint = load_checkpoint(checkpoint_path, "cpu")
        learn = LearnConfig(**checkpoint["learn_config"])
        levels = learn.level_names
        plan = dict(checkpoint.get("plan") or {})
        for key in ("out", "start"):  # directory names, not paths
            if plan.get(key) is not None:
                plan[key] = Path(plan[key]).name
        config.update(learn=learn, model=checkpoint["model_config"], plan=plan)
        config["seed"] = plan.get("seed")
        config["checkpoint"] = checkpoint_path.name
        del checkpoint
    if levels is None:
        from lean.learn import LearnConfig

        levels = LearnConfig().level_names

    with open(arguments.log) as handle:
        stage, commit, updates, skipped = parse(handle, levels)
    if stage is None:
        raise SystemExit(f"no update lines in {arguments.log}")
    config.update(stage=stage, commit=commit)
    print(
        f"{arguments.log}: {stage}, {len(updates)} updates "
        f"({sum('gradient/actor_share' in record for record in updates.values())} with the "
        f"gradient decomposition), {skipped} lines skipped, commit {commit}",
        flush=True,
    )
    log = wandb_log.WandbLog(
        arguments.mode,
        directory,
        stage,
        wandb_log.as_config(config),
        wandb_directory=arguments.wandb_dir,
        keep_id=not arguments.no_keep_id,
    )
    if log.session is None:
        raise SystemExit("W&B did not start: nothing logged")
    for index in sorted(updates):
        log.log(updates[index])
    log.finish()
    print(f"logged {len(updates)} updates to {wandb_log.run_name(directory, stage)}", flush=True)


if __name__ == "__main__":
    main()
