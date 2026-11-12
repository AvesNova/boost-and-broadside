"""Report of the respawn-wave runs of `bench.frontline_design` (frontline and deathmatch).

    PYTHONPATH=src python -m bench.respawn_waves_report FILE.jsonl [FILE.jsonl ...]
        [--thresholds 2,3] [--clocks 900,1200]

Every variant's respawn waits (mean and 90th percentile, s, over every respawn of the first
games, episode resets aside). Frontline, per (threshold T, clock C) replayed from the traced
front (`bench.frontline_design_report.replay`): P(next capture by the same team), the
share ending by the front, the median length, captures per game, the first capture's median
and the median zone-to-zone leg. Deathmatch (the whole clock, no tickets): kills a minute
(both teams), the median life, the trade share (an enemy death within 1 s), the median time
between deaths, and the match's first enemy hit to its first kill (median).
"""

import argparse
import json
import math
import pathlib
import statistics

from bench.frontline_design_report import leg_median, replay


def waits(entries: list[dict]) -> tuple[float, float]:
    """The respawn waits' mean and 90th percentile (s, the percentile at bin resolution)."""
    total, histogram, width = 0.0, None, entries[0]["respawn_wait"]["bin_seconds"]
    for entry in entries:
        counts = entry["respawn_wait"]["histogram"]
        histogram = counts if histogram is None else [a + b for a, b in zip(histogram, counts)]
        total += entry["respawn_wait"]["total_seconds"]
    count = sum(histogram)
    if not count:
        return math.nan, math.nan
    running = 0
    for index, value in enumerate(histogram):
        running += value
        if running >= 0.9 * count:
            return total / count, (index + 1) * width
    return total / count, math.nan


def histogram_median(entries: list[dict], name: str) -> float:
    """The median of a probe histogram (1 s bins)."""
    total = None
    for entry in entries:
        counts = entry["histograms"][name]
        total = counts if total is None else [a + b for a, b in zip(total, counts)]
    half, running = sum(total) / 2, 0
    for second, count in enumerate(total):
        running += count
        if running >= half:
            return second + 0.5
    return math.nan


def frontline(by_label: dict, thresholds: list[int], clocks: list[float]) -> None:
    """The frontline table."""
    print(
        "| n | variant | T | clock (s) | P(same next) | by the front | median length (s) "
        "| captures / game | first capture (s) | zone-to-zone (s) | wait mean (s) "
        "| wait p90 (s) | games |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (ships, label), entries in sorted(by_label.items()):
        tick = entries[0]["tick_length"]
        games = [game for entry in entries for game in entry["games"]]
        leg = leg_median(entries)
        mean, p90 = waits(entries)
        for threshold in thresholds:
            for clock in clocks:
                r = replay(games, tick, threshold, clock)
                print(
                    f"| {ships} | {label} | {threshold} | {clock:.0f} | "
                    f"{r['p_same']:.3f} ± {r['p_same_error']:.3f} | "
                    f"{r['by_front']:.2f} ± {r['by_front_error']:.2f} | "
                    f"{r['median_length']:.0f} | {r['captures']:.1f} | "
                    f"{r['first_capture']:.0f} | {leg:.1f} | {mean:.1f} | {p90:.1f} | "
                    f"{r['games']} |"
                )


def deathmatch(by_label: dict) -> None:
    """The deathmatch table."""
    print(
        "| n | variant | kills / min | life median (s) | trade share (1 s) "
        "| between deaths median (s) | first hit to first kill (s) | wait mean (s) "
        "| wait p90 (s) | games |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for (ships, label), entries in sorted(by_label.items()):
        tick = entries[0]["tick_length"]
        minutes = entries[0]["episode_ticks"] * tick / 60.0
        games = [game for entry in entries for game in entry["games"]]
        rate = statistics.fmean(len(game["events"]) / minutes for game in games)
        reach = round(1.0 / tick)
        trades = total = 0
        gaps, contact = [], []
        for game in games:
            events = game["events"]
            for at, team in events:
                total += 1
                trades += any(abs(other - at) <= reach and t != team for other, t in events)
            ticks = sorted(at for at, _ in events)
            gaps += [(b - a) * tick for a, b in zip(ticks, ticks[1:])]
            if events and game.get("first_hit", -1) >= 0:
                contact.append((ticks[0] - game["first_hit"]) * tick)
        mean, p90 = waits(entries)
        print(
            f"| {ships} | {label} | {rate:.2f} | {histogram_median(entries, 'life'):.1f} | "
            f"{trades / max(total, 1):.3f} | {statistics.median(gaps):.1f} | "
            f"{statistics.median(contact):.1f} | {mean:.1f} | {p90:.1f} | {len(games)} |"
        )


def main() -> None:
    """The report the module describes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=pathlib.Path, nargs="+")
    parser.add_argument("--thresholds", default="2,3")
    parser.add_argument("--clocks", default="900,1200")
    arguments = parser.parse_args()
    kinds: dict[str, dict] = {}
    for path in arguments.files:
        for line in path.read_text().splitlines():
            entry = json.loads(line)
            key = (entry["ships"], entry["label"])
            kinds.setdefault(entry["kind"], {}).setdefault(key, []).append(entry)
    if "frontline" in kinds:
        frontline(
            kinds["frontline"],
            [int(value) for value in arguments.thresholds.split(",")],
            [float(value) for value in arguments.clocks.split(",")],
        )
    if "deathmatch" in kinds:
        deathmatch(kinds["deathmatch"])


if __name__ == "__main__":
    main()
