"""Elimination shrink sweep (`bench/elimination_shrink_run.sh`): per shrink start (hold) and
duration, game length (mean, median, 80th and 90th percentiles) and how much the boundary
bears: the share of games ended before the shrink starts, the shares ending with the
boundary below 75% and 50% of the arena radius (linear law, end fraction taken as 0), the
boundary's share of deaths, the share of games with a boundary death, and clock ends.

Usage: python bench/elimination_shrink_report.py DATA.jsonl
"""

import collections
import json
import statistics
import sys

TICKS_PER_SECOND = 30


def percentile(values: list[float], share: float) -> float:
    """The nearest-rank percentile of the values (share in [0, 1])."""
    ordered = sorted(values)
    return ordered[min(int(share * len(ordered)), len(ordered) - 1)]


def main(path: str) -> None:
    """Print the tables for the files on the command line."""
    games = collections.defaultdict(list)
    with open(path) as lines:
        for line in lines:
            record = json.loads(line)
            games[record["label"]] += record["games"]
    print(
        "start duration clock |  mean median  p80  p90 | before start | <75% | <50% |"
        " boundary deaths | games with one | clock ends"
    )
    for label, played in games.items():
        hold, duration = (int(part[1:]) for part in label.split("_"))
        clock = hold + duration + 8
        seconds = [game["end_tick"] / TICKS_PER_SECOND for game in played]
        fractions = [1 - min(max((second - hold) / duration, 0), 1) for second in seconds]
        deaths = sum(game["deaths"] for game in played)
        boundary = sum(game["boundary_deaths"] for game in played)
        count = len(played)
        print(
            f"{hold:5d} {duration:8d} {clock:5d} |"
            f" {statistics.mean(seconds):5.0f} {statistics.median(seconds):6.0f}"
            f" {percentile(seconds, 0.8):4.0f} {percentile(seconds, 0.9):4.0f} |"
            f" {sum(second < hold for second in seconds) / count:12.2f} |"
            f" {sum(fraction < 0.75 for fraction in fractions) / count:4.2f} |"
            f" {sum(fraction < 0.5 for fraction in fractions) / count:4.2f} |"
            f" {boundary / deaths:15.3f} |"
            f" {sum(game['boundary_deaths'] > 0 for game in played) / count:14.2f} |"
            f" {sum(not game['by_rule'] for game in played) / count:10.2f}"
        )


if __name__ == "__main__":
    main(sys.argv[1])
