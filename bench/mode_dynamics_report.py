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


def histogram_summary(entries: list[dict], name: str) -> str | None:
    """Mean, median and quartiles (s) of a probe histogram (1 s bins, summed over batches;
    a bin's middle stands for its times)."""
    counts = None
    for entry in entries:
        values = entry.get("histograms", {}).get(name)
        if values is None:
            continue
        counts = values if counts is None else [a + b for a, b in zip(counts, values)]
    if not counts or sum(counts) == 0:
        return None
    total = sum(counts)
    mean = sum((index + 0.5) * count for index, count in enumerate(counts)) / total

    def at(q):
        running = 0
        for index, count in enumerate(counts):
            running += count
            if running >= q * total:
                return index + 0.5
        return len(counts) - 0.5

    return (
        f"mean {mean:.1f}, median {at(0.5):.1f}, quartiles {at(0.25):.1f}-{at(0.75):.1f}, "
        f"5-95% {at(0.05):.1f}-{at(0.95):.1f} (n {total})"
    )


def probes(entries: list[dict], names: tuple[str, ...]) -> None:
    labels = dict(
        engage="spawn to first enemy hit dealt or taken (s)",
        life="spawn to death (s)",
        objective="spawn to first zone other than the spawn zone (s)",
        zone_to_zone="leaving a zone to entering another (s)",
        contest="all meters at zero to a capture (s)",
    )
    for name in names:
        summary = histogram_summary(entries, name)
        if summary:
            print(f"{labels[name]}: {summary}")


def deathmatch_timing(entries: list[dict]) -> None:
    """Game starts' first hit and first kill, time between deaths, trade shares."""
    tick = entries[0]["tick_length"]
    games = [game for entry in entries for game in entry["games"]]
    hits = [game["first_hit"] * tick for game in games if game.get("first_hit", -1) >= 0]
    if hits:
        print(f"match start to first enemy hit (s): {quantiles(hits)}")
    kills = [game["events"][0][0] * tick for game in games if game["events"]]
    print(f"match start to first death (s): {quantiles(kills)}")
    gaps = []
    for game in games:
        ticks = sorted(at for at, _ in game["events"])
        gaps += [(b - a) * tick for a, b in zip(ticks, ticks[1:])]
    print(f"time between deaths (s): {quantiles(gaps)}")
    for window in (0.5, 1.0, 2.0):
        reach = round(window / tick)
        trades = total = 0
        for game in games:
            events = game["events"]
            for at, team in events:
                total += 1
                trades += any(abs(other - at) <= reach and t != team for other, t in events)
        print(f"trade share (an enemy death within {window} s): {trades / total:.3f}")


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
    deathmatch_timing(entries)
    probes(entries, ("engage", "life"))

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
    mean_rate = statistics.fmean(rates)
    ratios = [
        row["tickets_for_80"] / (mean_rate * seconds / 60.0)
        for seconds, row in table.items()
        if row["tickets_for_80"]
    ]
    if ratios:
        print(
            f"T for 80% = {statistics.fmean(ratios):.3f} x kills per minute (both teams) x "
            f"clock minutes (ratios {', '.join(f'{r:.3f}' for r in ratios)})"
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
        f"capture {entries[0]['capture_seconds']:.0f} s, threshold "
        f"{entries[0].get('front_threshold', 3)}, clock {episode * tick:.0f} s"
    )
    print(f"ended by the front: {by_front:.3f} +- {error:.3f} (else the clock)")
    print(f"length (s): {quantiles(lengths)}")
    print(f"captures per game: {quantiles([float(c) for c in captures])}")
    if first_capture:
        print(f"first capture (s): {quantiles(first_capture)}")
    probes(entries, ("engage", "life", "objective", "zone_to_zone", "contest"))
    if (entries[0].get("experiments") or {}).get("b_parameters"):
        scores = []
        for entry in entries:
            for environment, game in enumerate(entry["games"]):
                result = game["result"] if environment % 2 == 0 else -game["result"]
                scores.append((result + 1) / 2)  # side A: win 1, draw 0.5
        mean = statistics.fmean(scores)
        error = statistics.pstdev(scores) / math.sqrt(len(scores))
        print(
            f"side A (default set) against B ({entries[0]['experiments']['b_parameters']}): "
            f"{mean:.3f} +- {error:.3f}"
        )
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
