"""Elimination at fixed map sizes (`bench/elimination_static_run.sh`): per radius, game length
(mean, median, 80th and 90th percentiles; a game the clock ends counts at the boundary's
collapse, 8 s before it), deaths per minute of both teams and the boundary's share of deaths.

Usage: python bench/elimination_static_report.py DATA.jsonl
"""

import collections
import json
import statistics
import sys

TICKS_PER_SECOND = 30
COLLAPSE_SECONDS = 892  # the 900 s clock less the 8 s end lead


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
    print("radius |  mean median  p80  p90 | never ended | deaths/min | boundary share")
    for label, played in sorted(games.items(), key=lambda item: -int(item[0][1:])):
        seconds = [
            game["end_tick"] / TICKS_PER_SECOND if game["by_rule"] else COLLAPSE_SECONDS
            for game in played
        ]
        ended = [game for game in played if game["by_rule"]]
        deaths = sum(game["deaths"] for game in ended)
        boundary = sum(game["boundary_deaths"] for game in ended)
        minutes = sum(game["end_tick"] / (60 * TICKS_PER_SECOND) for game in ended)
        print(
            f"{label[1:]:>6} | {statistics.mean(seconds):5.0f} {statistics.median(seconds):6.0f}"
            f" {percentile(seconds, 0.8):4.0f} {percentile(seconds, 0.9):4.0f} |"
            f" {1 - len(ended) / len(played):11.2f} | {deaths / max(minutes, 1e-9):10.1f} |"
            f" {boundary / max(deaths, 1):14.3f}"
        )


if __name__ == "__main__":
    main(sys.argv[1])
