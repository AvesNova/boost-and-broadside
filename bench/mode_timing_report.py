"""Report of the mode-timing measurements (`bench/mode_timing_run.sh`, files of
`bench.mode_dynamics`: the scripted agent on both seats, real-scale presets, fog on, the
preset's fields on, mirrored lineups alternating Yongma,Jujak / Jujak,Yongma over environments,
each environment's first game).

    PYTHONPATH=src python -m bench.mode_timing_report DIR

Definitions.
- Deathmatch kill rate: a point is one death (whatever its cause), scored for the other
  team. Rate per team = points scored AGAINST that team (its deaths) per minute of the game's
  clock (the tickets are off, so every game plays the whole clock). Total = both teams.
  Mean and standard deviation are over games; the interval is a Student t 95% interval of
  the mean over games. Time profile: deaths per minute (both teams) in 1-minute bins.
- Tickets: a game ends when a team's kills reach T (kills_per_ship x ships per team; the
  tickets of the rule are "kills to win"). Ended-by-tickets share at time limit L and T:
  the share of games in which the T-th point (of either team) came before L. Read directly
  from the long games, so no model of the rate is assumed; a game whose points never reach
  T within the measured clock (900 s) counts as not ended by tickets before any L <= 900.
  The table gives, per L in (10, 12, 15) minutes, the T (integer, per team) whose share is
  closest to, and at least, 70, 80 and 90%; the share is read at L, and the 15-minute
  limit is the measured clock.
- Elimination: the tick at which one team is fully dead (the game ends by the rule, before
  the clock); percentiles of that time over games that ended; never ended = games to the
  clock. Share of deaths by the boundary = deaths with no shooter over all deaths.
- Frontline: as `bench.front_reversal` (moves, P(reversal) with Wilson and game-clustered
  bootstrap 95% intervals, ended by capture or the clock).
"""

import json
import math
import pathlib
import random
import statistics
import sys

from bench.front_reversal import counts, wilson

TICK = 1.0 / 30.0


def load(path: pathlib.Path) -> tuple[list[dict], dict]:
    """(games, the last batch's header) of a measurement file."""
    games, header = [], {}
    for line in path.read_text().splitlines():
        entry = json.loads(line)
        games += entry["games"]
        header = {key: value for key, value in entry.items() if key not in ("games", "histograms")}
    return games, header


def t_interval(values: list[float]) -> tuple[float, float]:
    """The mean's 95% interval (normal-t table approximated by 1.96 above 60 games)."""
    n = len(values)
    mean = statistics.fmean(values)
    t = {5: 2.78, 10: 2.23, 20: 2.09, 30: 2.05, 40: 2.02, 60: 2.0}
    factor = next((v for k, v in sorted(t.items()) if n <= k), 1.96)
    half = factor * statistics.stdev(values) / math.sqrt(n)
    return mean - half, mean + half


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile q in [0, 100]."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low = int(math.floor(position))
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def deathmatch(directory: pathlib.Path) -> None:
    """Measurement 1."""
    games, header = load(directory / "deathmatch.jsonl")
    clock = header["episode_ticks"] * TICK
    minutes = clock / 60
    print(
        f"## deathmatch: {len(games)} games, clock {clock:.0f} s, "
        f"{header['ships']}v{header['ships']}, arena {header['radius']} m, "
        f"fields {header['fields']}, respawn {header['respawn_seconds']} s"
    )
    # deaths of a team = points scored by the other team
    team = [[sum(1 for _, t in g["events"] if t != side) for side in (0, 1)] for g in games]
    for name, values in (
        ("per team (deaths per minute)", [x / minutes for pair in team for x in pair]),
        ("per game, both teams", [sum(pair) / minutes for pair in team]),
        ("per game, team-mean", [sum(pair) / 2 / minutes for pair in team]),
    ):
        low, high = t_interval(values)
        print(
            f"{name}: mean {statistics.fmean(values):.3f}, sd {statistics.stdev(values):.3f}, "
            f"95% [{low:.3f}, {high:.3f}], n {len(values)}"
        )
    print("time profile (both teams, deaths per game-minute, mean over games [sd]):")
    bins = int(minutes)
    for b in range(bins):
        per_game = [
            sum(1 for at, _ in g["events"] if b * 1800 <= at < (b + 1) * 1800) for g in games
        ]
        print(
            f"  minute {b + 1:2d}: {statistics.fmean(per_game):.2f} "
            f"[{statistics.stdev(per_game):.2f}]"
        )
    first = [min((at for at, _ in g["events"]), default=None) for g in games]
    first = [f * TICK for f in first if f is not None]
    print(
        f"first death at {statistics.fmean(first):.1f} s mean "
        f"(median {statistics.median(first):.1f})"
    )
    # tickets: time the T-th point of either team arrives
    print("tickets T (kills per team to end the game) -> share of games ended by tickets before L:")
    limits = (10, 12, 15)
    times = {}
    for T in range(1, 80):
        ends = []
        for g in games:
            when = []
            for side in (0, 1):
                ticks = sorted(at for at, t in g["events"] if t == side)
                when.append(ticks[T - 1] * TICK if len(ticks) >= T else math.inf)
            ends.append(min(when))
        times[T] = ends
    for L in limits:
        shares = {T: sum(e < L * 60 for e in ends) / len(ends) for T, ends in times.items()}
        row = []
        for target in (0.7, 0.8, 0.9):
            # the largest T whose share is still at least the target
            ok = [T for T, s in shares.items() if s >= target]
            T = max(ok) if ok else None
            row.append(
                f"{int(target * 100)}%: T={T} (share {shares[T]:.2f})"
                if T
                else f"{int(target * 100)}%: none"
            )
        print(f"  L={L} min: " + "; ".join(row))
        if L == 15 and False:
            pass
    print("  share ended by tickets at each (L, T):")
    for L in limits:
        print(
            f"   L={L}: "
            + " ".join(
                f"T{T}={sum(e < L * 60 for e in times[T]) / len(games):.2f}"
                for T in range(10, 41, 2)
            )
        )
    print()


def elimination(directory: pathlib.Path) -> None:
    """Measurement 2."""
    for name in ("elimination_noshrink", "elimination_shrink"):
        path = directory / f"{name}.jsonl"
        if not path.exists():
            continue
        games, header = load(path)
        episode = header["episode_ticks"] * TICK
        # The shrink's hold is clamped to one tick before its end (`layout`: hold <= end - 1),
        # so with the shrink "off" by a long hold the boundary still collapses in the last
        # tick before clock minus the 8 s lead, killing whoever is left. A game ending then
        # did not end by a wipe: it is counted as never ended (its deaths are left out).
        collapse = episode - 8.0 - 1.0 if name == "elimination_noshrink" else math.inf
        games = [g for g in games if not (g["end_tick"] * TICK >= collapse)] + [
            dict(g, by_rule=False, deaths=0, boundary_deaths=0)
            for g in games
            if g["end_tick"] * TICK >= collapse
        ]
        ended = [g["end_tick"] * TICK for g in games if g["by_rule"]]
        print(
            f"## {name}: {len(games)} games, clock {episode:.0f} s, ended {len(ended)}, "
            f"never ended {len(games) - len(ended)}"
        )
        if ended:
            print(
                "  wipe time (s) percentiles of ended games: "
                + ", ".join(f"p{q} {percentile(ended, q):.0f}" for q in (5, 10, 25, 50, 75, 80, 90))
            )
            print(f"  mean {statistics.fmean(ended):.0f} s, sd {statistics.stdev(ended):.0f} s")
        # percentiles of all games with never-ended as infinity
        allt = [g["end_tick"] * TICK if g["by_rule"] else math.inf for g in games]
        print(
            "  percentiles over all games (never ended = inf): "
            + ", ".join(f"p{q} {percentile(allt, q):.0f}" for q in (5, 10, 25, 50, 75, 80, 90))
        )
        deaths = sum(g["deaths"] for g in games)
        boundary = sum(g["boundary_deaths"] for g in games)
        with_boundary = sum(1 for g in games if g["boundary_deaths"])
        print(
            f"  deaths {deaths}, by the boundary {boundary} ({boundary / deaths:.1%}); "
            f"games with a boundary death {with_boundary} ({with_boundary / len(games):.1%})"
        )
        print()


def frontline(directory: pathlib.Path) -> None:
    """Measurement 3."""
    for name in ("frontline_2v2", "frontline_3v3"):
        path = directory / f"{name}.jsonl"
        if not path.exists():
            continue
        games, header = load(path)
        per_game = [counts(g) for g in games]
        reversals, pairs = sum(r for r, _ in per_game), sum(n for _, n in per_game)
        low, high = wilson(reversals, pairs)
        rng = random.Random(0)
        boots = []
        for _ in range(4000):
            sample = [per_game[rng.randrange(len(per_game))] for _ in per_game]
            total = sum(n for _, n in sample)
            if total:
                boots.append(sum(r for r, _ in sample) / total)
        boots.sort()
        moves = [len(g["events"]) for g in games]
        by_rule = sum(g["by_rule"] for g in games)
        lengths = [g["end_tick"] * TICK for g in games]
        print(
            f"## {name}: {len(games)} games, {header['ships']}v{header['ships']}, arena "
            f"{header['radius']} m, clock {header['episode_ticks'] * TICK:.0f} s, "
            f"fields {header['fields']}"
        )
        low_m, high_m = t_interval(moves)
        print(
            f"  moves per game {statistics.fmean(moves):.2f} [95% {low_m:.2f}, {high_m:.2f}], "
            f"sd {statistics.stdev(moves):.2f}"
        )
        low_l, high_l = t_interval(lengths)
        print(
            f"  game length {statistics.fmean(lengths):.0f} s [95% {low_l:.0f}, {high_l:.0f}], "
            "median "
            f"{statistics.median(lengths):.0f}"
        )
        low_c, high_c = wilson(by_rule, len(games))
        print(
            f"  ended by capture {by_rule} ({by_rule / len(games):.1%}, "
            f"Wilson [{low_c:.3f}, {high_c:.3f}]); on the clock {len(games) - by_rule}"
        )
        print(
            f"  consecutive pairs {pairs}; reversals {reversals}; "
            f"P(reversal) {reversals / max(pairs, 1):.3f} Wilson [{low:.3f}, {high:.3f}] "
            f"bootstrap [{boots[int(0.025 * len(boots))]:.3f}, "
            f"{boots[int(0.975 * len(boots))]:.3f}]"
        )
        histogram = {}
        for count in moves:
            histogram[count] = histogram.get(count, 0) + 1
        print("  games by moves:", dict(sorted(histogram.items())))
        print()


def main() -> None:
    """Print every measurement found in the directory."""
    directory = pathlib.Path(sys.argv[1])
    if (directory / "deathmatch.jsonl").exists():
        deathmatch(directory)
    elimination(directory)
    frontline(directory)


if __name__ == "__main__":
    main()
