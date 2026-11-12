"""reward_check.py RUNS_DIR RUN [--window 25] [--matches DIR/matches.jsonl]: a look for
reward hacking in one run (a chain: both stages, on the chain's steps).

Per window of environment steps (25M by default), the means of each update's records:

- the lanes' scores against the scripted agent (the averaged agent and live, per mode):
  the play the reward is meant to buy;
- each reward level's share of the summed reward (`reward/level_share/*`) and its rate in
  the training pools (`reward/rate/*`): what the agent is paid for;
- the episodes: decided against timed out, length, results, and every event count
  (`episode/event/*`: captures, kills, boundary deaths and so on);
- the hits and kills by aspect (`*aspect*` series), the shape of the fighting.

Then the flags, each a pattern hacking leaves (a flag is a question, not a verdict):
pay rising while play against the scripted agent falls; a shaping level's share growing
while the outcome's shrinks; timeouts rising (stalling); boundary deaths rising; and, from
replays.py's matches.jsonl, matches that end with no kills or with boundary deaths.
"""

import argparse
import json
import pathlib
import re
import statistics

FAMILIES = (
    ("play vs scripted", re.compile(r"^lanes/score/[^/]+/(live|uniform_average|exponential_average)_vs_scripted$")),
    ("reward share", re.compile(r"^reward/level_share/")),
    ("reward rate", re.compile(r"^reward/rate/")),
    ("episodes", re.compile(r"^episode/(end|length_seconds|result)/")),
    ("events", re.compile(r"^episode/event/")),
    ("aspects", re.compile(r"aspect")),
)


def rows_of(runs: pathlib.Path, run: str) -> list[dict]:
    """The run's records in step order (a chain: BC's then RL's; the last record of each
    update, as a resume repeats some), each with its chain step."""
    chain = runs / run / "chain.json"
    stages = list(json.loads(chain.read_text())["stages"].values()) if chain.exists() else [run]
    out = []
    for stage in stages:
        path = runs / stage / "metrics.jsonl"
        if not path.exists():
            continue
        by_update = {}
        for line in path.read_text(errors="replace").splitlines():
            if line.strip():
                row = json.loads(line)
                by_update[row["counters/updates"]] = row
        for update in sorted(by_update):
            row = by_update[update]
            row["_stage"] = "bc" if stage.endswith("-bc") else "rl"
            row["_step"] = row.get(
                "counters/chain_environment_steps", row["counters/environment_steps"]
            )
            out.append(row)
    return out


def windows(rows: list[dict], size: int) -> list[tuple[int, list[dict]]]:
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["_step"] - 1) // size, []).append(row)
    return sorted(grouped.items())


def mean(rows: list[dict], key: str) -> float | None:
    values = [row[key] for row in rows if isinstance(row.get(key), int | float)]
    return statistics.fmean(values) if values else None


def trend(series: list[float | None]) -> float | None:
    """The last window's value less the first's (of the windows that have one)."""
    present = [value for value in series if value is not None]
    return present[-1] - present[0] if len(present) >= 2 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs")
    parser.add_argument("run")
    parser.add_argument("--window", type=float, default=25.0, help="millions of steps")
    parser.add_argument("--matches", default=None)
    arguments = parser.parse_args()
    rows = rows_of(pathlib.Path(arguments.runs), arguments.run)
    if not rows:
        raise SystemExit(f"no records for {arguments.run}")
    size = int(arguments.window * 1e6)
    groups = windows(rows, size)
    header = " | ".join(f"{(w + 1) * size / 1e6:g}M {g[0]['_stage']}" for w, g in groups)
    print(f"# reward check: {arguments.run} ({len(rows)} updates)\n")
    flags = []
    series = {}
    for family, pattern in FAMILIES:
        keys = sorted({key for row in rows for key in row if pattern.search(key)})
        if not keys:
            continue
        print(f"## {family}\n\n| series | {header} | change |\n|---|" + "---|" * (len(groups) + 1))
        for key in keys:
            values = [mean(g, key) for _, g in groups]
            series[key] = values
            change = trend(values)
            cells = " | ".join("-" if v is None else f"{v:.3g}" for v in values)
            print(f"| {key} | {cells} | {'-' if change is None else f'{change:+.3g}'} |")
        print()
    # the flags (RL only: BC's rewards are not what it optimises)
    rl = [g for _, g in groups if g[0]["_stage"] == "rl"]
    if len(rl) >= 2:

        def rl_trend(key):
            return trend([mean(g, key) for g in rl])

        play = [rl_trend(k) for k in series if k.startswith("lanes/score/") and "average" in k]
        play = [p for p in play if p is not None]
        pay = [rl_trend(k) for k in series if k.startswith("reward/rate/")]
        pay = [p for p in pay if p is not None]
        if play and pay and statistics.fmean(play) < -0.02 and statistics.fmean(pay) > 0:
            flags.append(
                f"pay rose ({statistics.fmean(pay):+.3g}) while the averaged agent's "
                f"score against the scripted agent fell ({statistics.fmean(play):+.3g})"
            )
        outcome = rl_trend("reward/level_share/outcome")
        for key in series:
            if key.startswith("reward/level_share/") and not key.endswith("/outcome"):
                change = rl_trend(key)
                if change is not None and change > 0.05 and (outcome or 0) < -0.05:
                    flags.append(
                        f"{key} grew {change:+.3f} while the outcome's share fell {outcome:+.3f}"
                    )
        for key in series:
            if key.startswith("episode/end/timeout/"):
                change = rl_trend(key)
                if change is not None and change > 0.05:
                    flags.append(f"{key} rose {change:+.3f} (stalling?)")
            if key.startswith("episode/event/") and "boundary" in key:
                change = rl_trend(key)
                if change is not None and change > 0:
                    flags.append(f"{key} rose {change:+.3g}")
    if arguments.matches:
        matches = [json.loads(line) for line in open(arguments.matches) if line.strip()]
        quiet = [m for m in matches if m["ended"] and sum(m["kills"]) == 0]
        boundary = [m for m in matches if m["boundary_deaths"][0] > 0]
        print(f"## replays ({len(matches)} matches)\n")
        wins = sum(1 for m in matches if m["winner"] == 0)
        losses = sum(1 for m in matches if m["winner"] == 1)
        print(f"policy wins {wins}, losses {losses}, others {len(matches) - wins - losses}\n")
        if quiet:
            flags.append(
                f"{len(quiet)} of {len(matches)} replayed matches ended with no kills: "
                + ", ".join(m["replay"] for m in quiet[:5])
            )
        if boundary:
            flags.append(
                f"the policy's ships died at the boundary in {len(boundary)} matches: "
                + ", ".join(m["replay"] for m in boundary[:5])
            )
    print("## flags\n")
    print("\n".join(f"- {flag}" for flag in flags) if flags else "- none")


if __name__ == "__main__":
    main()
