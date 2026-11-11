"""The temperature sweep: does a trained RL policy play best at the entropy it was trained to?

    uv run --no-sync python -m bench.temperature_sweep play --out DIR POLICY [POLICY ...]
    uv run --no-sync python -m bench.temperature_sweep report --out DIR

`play`: one field of every POLICY at each `--temperatures` (`POLICY@T=<temperature>`,
`bnb.league.matches.temperature_of`; T = 1 is the policy as it is) and the scripted agent
(the 1000 anchor), rated in each of `--presets` on the league's match engine with the
A-optimal allocation (`bnb.league.tournament.rate`) to `--target-stderr`: every batch is
appended to `DIR/<preset>.batches.jsonl` with each block's first-game statistics (the
networks' summed entropy among them), and a record found is resumed. `--hours` bounds the
whole run.

`report`: from the records, per preset: each player's Elo and standard error (the fit the
last batch recorded) and its mean entropy, nats per living ship-decision of the joint
action distribution it sampled from (`STATISTICS`' entropy over its ship-ticks), as a
Markdown table of policy by temperature; `sweep.json` in DIR.

A temperature divides the logits of the whole joint action, so it moves every factor at
once; it rates a policy's sampling, not a policy trained to that entropy.
"""

import argparse
import datetime
import json
import pathlib
import time

import torch

from bnb.league import tournament
from bnb.league.matches import STATISTICS, ArenaConfig, player, temperature_of

ANCHOR = "scripted"


def _clock(seconds: float) -> str:
    later = datetime.datetime.now().astimezone() + datetime.timedelta(seconds=seconds)
    return later.strftime("%H:%M")


def field(policies: list[str], temperatures: list[float]) -> list[str]:
    """The field's names: each policy at each temperature (T = 1 by its own name), then the
    anchor."""
    names = [
        policy if temperature == 1.0 else f"{policy}@T={temperature:g}"
        for policy in policies
        for temperature in temperatures
    ]
    return names + [ANCHOR]


def play(args) -> None:
    """The tournaments the module describes, appended to the records batch by batch."""
    names = field(args.policies, args.temperatures)
    players = [player(name) for name in names]
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = tournament.TournamentConfig(
        ArenaConfig(blocks=args.blocks, block_environments=args.block_games),
        target_stderr=args.target_stderr,
        budget=args.budget,
    )
    deadline = time.time() + 3600.0 * args.hours

    def log(line: str) -> None:
        print(f"{datetime.datetime.now():%H:%M:%S} {line}", flush=True)
        if time.time() > deadline:
            raise SystemExit(f"past the {args.hours} h bound: resume from {out}")

    log(
        f"{len(names)} players, presets {args.presets}, target standard error "
        f"{args.target_stderr:g}, budget {args.budget} games a preset in batches of "
        f"{config.arena.games}; bound {_clock(deadline - time.time())}"
    )
    for preset in args.presets:
        rows = tournament.rate(
            players,
            preset,
            torch.device("cuda"),
            config,
            log,
            record=out / f"{preset}.batches.jsonl",
            statistics=True,
        )
        for row in sorted(rows, key=lambda row: -row["calibrated_elo"]):
            log(f"{preset} {row['player']} {row['calibrated_elo']:.0f} +- {row['stderr']:.0f}")


def summary(record: pathlib.Path) -> list[dict]:
    """From a preset's record: each player's name, policy, temperature, Elo and standard
    error (the last batch's fit, the anchor at 1000), games, and mean entropy (nats per
    living ship-decision; 0 for the scripted agent)."""
    lines = [json.loads(line) for line in record.read_text().splitlines() if line.strip()]
    last = lines[-1]
    names = [row["player"] for row in last["ratings"]]
    sums = {name: torch.zeros(len(STATISTICS), dtype=torch.float64) for name in names}
    games = dict.fromkeys(names, 0)
    for line in lines:
        for (first, second), result, (mine, theirs) in zip(
            line["matchups"], line["results"], line["statistics"], strict=True
        ):
            sums[first] += torch.tensor(mine, dtype=torch.float64)
            sums[second] += torch.tensor(theirs, dtype=torch.float64)
            games[first] += sum(result)
            games[second] += sum(result)
    anchor = next(row["rating"] for row in last["ratings"] if row["player"] == ANCHOR)
    entropy, ticks = STATISTICS.index("entropy"), STATISTICS.index("ship_ticks")
    rows = []
    for row in last["ratings"]:
        policy, temperature = temperature_of(row["player"])
        rows.append(
            dict(
                player=row["player"],
                policy=policy,
                temperature=1.0 if temperature is None else temperature,
                elo=1000.0 + row["rating"] - anchor,
                stderr=row["stderr"],
                games=games[row["player"]],
                entropy=float(
                    sums[row["player"]][entropy] / sums[row["player"]][ticks].clamp_min(1)
                ),
            )
        )
    return rows


def report(args) -> None:
    """The tables the module describes, from the records."""
    out = pathlib.Path(args.out)
    result = {}
    for record in sorted(out.glob("*.batches.jsonl")):
        preset = record.name.removesuffix(".batches.jsonl")
        rows = summary(record)
        result[preset] = rows
        temperatures = sorted({row["temperature"] for row in rows if row["policy"] != ANCHOR})
        policies = list(dict.fromkeys(row["policy"] for row in rows if row["policy"] != ANCHOR))
        at = {(row["policy"], row["temperature"]): row for row in rows}
        print(f"\n{preset} (Elo +- standard error; mean entropy, nats)\n")
        print("| policy | " + " | ".join(f"T = {t:g}" for t in temperatures) + " |")
        print("|---|" + "---|" * len(temperatures))
        for policy in policies:
            cells = [
                f"{at[policy, t]['elo']:.0f} +- {at[policy, t]['stderr']:.0f} "
                f"({at[policy, t]['entropy']:.2f})"
                for t in temperatures
            ]
            print(f"| {policy} | " + " | ".join(cells) + " |")
    (out / "sweep.json").write_text(json.dumps(result, indent=1) + "\n")


def main() -> None:
    """The command line: `play` or `report`."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    playing = commands.add_parser("play")
    playing.add_argument("policies", nargs="+")
    playing.add_argument("--out", required=True)
    playing.add_argument("--temperatures", type=float, nargs="+", default=[0.7, 1.0, 1.4])
    playing.add_argument("--presets", nargs="+", default=["frontline", "deathmatch"])
    playing.add_argument("--target-stderr", type=float, default=12.0)
    playing.add_argument("--budget", type=int, default=60_000)
    playing.add_argument("--blocks", type=int, default=8)
    playing.add_argument("--block-games", type=int, default=256)
    playing.add_argument("--hours", type=float, default=6.0)
    reporting = commands.add_parser("report")
    reporting.add_argument("--out", required=True)
    args = parser.parse_args()
    play(args) if args.command == "play" else report(args)


if __name__ == "__main__":
    main()
