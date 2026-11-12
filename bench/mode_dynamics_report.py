"""Reports of `bench.mode_dynamics` records.

    PYTHONPATH=src python -m bench.mode_dynamics_report FILE [--label LABEL]

deathmatch: kills per minute per game (mean, median, quartiles, 5-95%), and for tickets
(points a team needs) by clock length, the share of games a team's tickets end, replayed
from every point's time; the tickets that end 80% of games at each clock.

frontline: the share of games the front ends (else the clock), the games' lengths, the
captures per game and per minute, and the capture dynamics: P(the next capture is by the
same team | a capture), with its standard error over the pairs (games weighted by their
pairs), and P(reversal) = 1 minus it.
"""

import argparse
import json
import math
import statistics


def quantiles(values: list[float]) -> str:
    ordered = sorted(values)

    def at(q):
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    return (
        f"mean {statistics.fmean(values):.2f}, median {statistics.median(values):.2f}, "
        f"quartiles {at(0.25):.2f}-{at(0.75):.2f}, 5-95% {at(0.05):.2f}-{at(0.95):.2f}"
    )


def deathmatch(entries: list[dict]) -> dict:
    tick = entries[0]["tick_length"]
    episode = entries[0]["episode_ticks"]
    games = [game for entry in entries for game in entry["games"]]
    minutes = episode * tick / 60.0
    rates = [len(game["events"]) / minutes for game in games]
    print(f"{len(games)} games of {minutes:.1f} min, {entries[0]['ships']}v{entries[0]['ships']}")
    print(f"kills per minute (both teams): {quantiles(rates)}")
    per_team = [
        max(sum(1 for _, team in game["events"] if team == t) for t in (0, 1)) for game in games
    ]
    print(f"the leading team's points at the clock: {quantiles([float(v) for v in per_team])}")

    def ended(tickets: int, clock_ticks: int) -> float:
        count = 0
        for game in games:
            points = [0, 0]
            for at, team in game["events"]:
                if at >= clock_ticks:
                    break
                points[team] += 1
                if points[team] >= tickets:
                    count += 1
                    break
        return count / len(games)

    table = {}
    clocks = sorted({int(episode * f) for f in (0.5, 2.0 / 3.0, 5.0 / 6.0, 1.0)})
    print("| clock (s) | " + " | ".join(f"T={t}" for t in range(5, 61, 5)) + " | T for 80% |")
    print("|---" * 14 + "|")
    for clock_ticks in clocks:
        shares = {t: ended(t, clock_ticks) for t in range(1, 121)}
        eighty = max((t for t, share in shares.items() if share >= 0.8), default=None)
        table[clock_ticks * tick] = dict(shares=shares, tickets_for_80=eighty)
        print(
            f"| {clock_ticks * tick:.0f} | "
            + " | ".join(f"{shares[t]:.2f}" for t in range(5, 61, 5))
            + f" | {eighty} |"
        )
    return dict(games=len(games), kills_per_minute=rates, table=table)


def frontline(entries: list[dict]) -> dict:
    tick = entries[0]["tick_length"]
    episode = entries[0]["episode_ticks"]
    games = [game for entry in entries for game in entry["games"]]
    by_front = sum(game["by_rule"] for game in games) / len(games)
    error = math.sqrt(by_front * (1 - by_front) / len(games))
    lengths = [(game["end_tick"] + 1) * tick for game in games]
    captures = [len(game["events"]) for game in games]
    same = pairs = 0
    for game in games:
        teams = [team for _, team in game["events"]]
        for first, second in zip(teams, teams[1:]):
            pairs += 1
            same += first == second
    p_same = same / pairs if pairs else math.nan
    p_error = math.sqrt(p_same * (1 - p_same) / pairs) if pairs else math.nan
    first_capture = [game["events"][0][0] * tick for game in games if game["events"]]
    print(
        f"{len(games)} games, {entries[0]['ships']}v{entries[0]['ships']}, R "
        f"{entries[0]['radius']:.0f} m, zone radius {entries[0]['zone_radius']:.0f} m (scale 1), "
        f"capture {entries[0]['capture_seconds']:.0f} s, clock {episode * tick:.0f} s"
    )
    print(f"ended by the front: {by_front:.3f} +- {error:.3f} (else the clock)")
    print(f"length (s): {quantiles(lengths)}")
    print(f"captures per game: {quantiles([float(c) for c in captures])}")
    if first_capture:
        print(f"first capture (s): {quantiles(first_capture)}")
    print(
        f"P(next capture by the same team | a capture): {p_same:.3f} +- {p_error:.3f} "
        f"over {pairs} pairs; P(reversal) {1 - p_same:.3f}"
    )
    return dict(
        games=len(games),
        ended_by_front=by_front,
        ended_by_front_error=error,
        p_same=p_same,
        p_same_error=p_error,
        pairs=pairs,
        mean_captures=statistics.fmean(captures),
        mean_length=statistics.fmean(lengths),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--label", default=None)
    arguments = parser.parse_args()
    entries = [json.loads(line) for line in open(arguments.file)]
    labels = sorted({entry["label"] for entry in entries})
    for label in labels if arguments.label is None else [arguments.label]:
        mine = [entry for entry in entries if entry["label"] == label]
        print(f"## {label or '(no label)'}")
        if mine[0]["kind"] == "deathmatch":
            deathmatch(mine)
        else:
            frontline(mine)


if __name__ == "__main__":
    main()
