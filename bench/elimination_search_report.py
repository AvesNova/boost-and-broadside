"""Tables for `bench/elimination_search_run.sh`: per label, game length, how games end, and
the losing team's deaths in order.

    python bench/elimination_search_report.py FILE.jsonl [FILE.jsonl ...]

Definitions (each environment's first game; `bench.mode_dynamics` records every ship's first
death: its tick, whether the boundary killed it (no shooter) and its distance from the
centre):
- length: the game's end, in s (a game still on at the clock counts as the clock's length);
- wiped: the share of games that ended with one team dead (elimination), or by any rule;
- loser: the team with more deaths (none in a tie: those games are left out of the death
  columns); its k-th death's time, median and 90th percentile over the games where it had a
  k-th death, and n, how many did;
- boundary: the share of each ship's first death with no shooter, and of the loser's k-th
  deaths;
- first hit: the game's first enemy hit, median (s).
Deathmatch rows add the deaths per team per minute over the game. gap: the median over games
of the mean distance from a living ship to its nearest living ally (m).
"""

import json
import pathlib
import statistics
import sys


def percentile(values: list[float], share: float) -> float:
    """The nearest-rank percentile (nan for none)."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(share * len(ordered)))]


def summary(entries: list[dict]) -> dict:
    """The label's numbers over its entries' games."""
    first = entries[0]
    tick = first["tick_length"]
    episode = first["episode_ticks"]
    games = [game for entry in entries for game in entry["games"]]
    ships = first["ships"]
    lengths, wiped, hits, gaps = [], 0, [], []
    deaths = boundary = first_deaths = 0
    by_k = [[] for _ in range(ships)]
    boundary_k = [0] * ships
    decided = 0
    for game in games:
        lengths.append(min(game["end_tick"], episode) * tick)
        wiped += bool(game["by_rule"])
        if game.get("ally_gap") is not None:
            gaps.append(game["ally_gap"])
        if game["first_hit"] >= 0:
            hits.append(game["first_hit"] * tick)
        ticks, by_boundary = game["death_tick"], game["death_boundary"]
        deaths += game["deaths"]  # every death (deathmatch's ships respawn)
        first_deaths += sum(t >= 0 for t in ticks)
        boundary += sum(b for t, b in zip(ticks, by_boundary) if t >= 0)
        per_team = [
            sorted(
                (t, b)
                for t, b in zip(
                    ticks[team * ships : (team + 1) * ships], by_boundary[team * ships :]
                )
                if t >= 0
            )
            for team in (0, 1)
        ]
        if len(per_team[0]) == len(per_team[1]):
            continue
        decided += 1
        loser = per_team[0] if len(per_team[0]) > len(per_team[1]) else per_team[1]
        for k, (t, b) in enumerate(loser):
            by_k[k].append(t * tick)
            boundary_k[k] += b
    return dict(
        games=len(games),
        wiped=wiped / len(games),
        median=statistics.median(lengths),
        mean=statistics.fmean(lengths),
        p80=percentile(lengths, 0.8),
        p90=percentile(lengths, 0.9),
        first_hit=statistics.median(hits) if hits else float("nan"),
        boundary=boundary / max(first_deaths, 1),
        decided=decided,
        ally_gap=statistics.median(gaps) if gaps else float("nan"),
        deaths_per_team_minute=deaths / 2 / (sum(lengths) / 60.0),
        k=[
            (
                len(times),
                percentile(times, 0.5),
                percentile(times, 0.9),
                boundary_k[k] / max(len(times), 1),
            )
            for k, times in enumerate(by_k)
        ],
    )


def main() -> None:
    entries: dict[str, list[dict]] = {}
    for name in sys.argv[1:]:
        for line in pathlib.Path(name).read_text().splitlines():
            entry = json.loads(line)
            entries.setdefault(entry["label"], []).append(entry)
    print(
        f"{'label':<16} {'games':>5} {'ended':>5} {'mean':>5} {'med':>5} {'p80':>5} {'p90':>5}"
        f" {'hit1':>5} {'bdry':>5} {'kills/min':>9} {'gap m':>6}"
    )
    rows = {label: summary(group) for label, group in entries.items()}
    for label, row in rows.items():
        print(
            f"{label:<16} {row['games']:>5} {row['wiped']:>5.2f} {row['mean']:>5.0f}"
            f" {row['median']:>5.0f} {row['p80']:>5.0f} {row['p90']:>5.0f}"
            f" {row['first_hit']:>5.0f} {row['boundary']:>5.2f}"
            f" {row['deaths_per_team_minute']:>9.2f} {row['ally_gap']:>6.0f}"
        )
    print()
    print("loser's k-th death: median/p90 s [boundary share] (n games with it)")
    for label, row in rows.items():
        cells = "  ".join(
            f"{median:>4.0f}/{p90:<4.0f}[{share:.2f}]({n})" for n, median, p90, share in row["k"]
        )
        print(f"{label:<16} {cells}")


if __name__ == "__main__":
    main()
