"""Report of `bench.frontline_design` records, replayed at any threshold and clock.

    PYTHONPATH=src python -m bench.frontline_design_report FILE.jsonl [--thresholds 2,3]
        [--clocks 600,900,1200] [--markdown]

The runs play with the threshold off and a long clock; the scripted agent reads neither,
so a game under threshold T and clock C is the traced game up to the first capture that
puts the front at +-T, or to C. Per label and (T, C): the share ending by the front, the
median length, captures per game, P(next capture by the same team | a capture) with a
standard error clustered by game, the first capture's median, and the median zone-to-zone
leg (from the probe, the whole traced game).
"""

import argparse
import json
import math
import pathlib
import statistics


def replay(games: list[dict], tick: float, threshold: int, clock: float) -> dict:
    """The (T, C) game of each traced game, summarised."""
    limit = clock / tick
    ended = lengths = 0
    lengths, captures, firsts, same_by_game, pairs_by_game = [], [], [], [], []
    for game in games:
        front, kept, end = 0, [], min(limit, game["end_tick"])
        for at, team in game["events"]:
            if at >= limit:
                break
            front += 1 if team == 0 else -1
            kept.append(team)
            if abs(front) >= threshold:
                end = at
                ended += 1
                break
        lengths.append(end * tick)
        captures.append(len(kept))
        if game["events"] and game["events"][0][0] < limit:
            firsts.append(game["events"][0][0] * tick)
        same_by_game.append(sum(a == b for a, b in zip(kept, kept[1:])))
        pairs_by_game.append(max(0, len(kept) - 1))
    pairs = sum(pairs_by_game)
    p = sum(same_by_game) / pairs if pairs else math.nan
    if pairs:  # ratio estimator, clustered by game
        n = len(games)
        mean_pairs = pairs / n
        residual = [s - p * m for s, m in zip(same_by_game, pairs_by_game)]
        error = math.sqrt(sum(r * r for r in residual) / (n * (n - 1))) / mean_pairs
    else:
        error = math.nan
    share = ended / len(games)
    return dict(
        games=len(games),
        by_front=share,
        by_front_error=math.sqrt(share * (1 - share) / len(games)),
        median_length=statistics.median(lengths),
        captures=statistics.fmean(captures),
        p_same=p,
        p_same_error=error,
        first_capture=statistics.median(firsts) if firsts else math.nan,
    )


def leg_median(entries: list[dict]) -> float:
    """Median zone-to-zone leg (s) over the probe's 1 s bins."""
    total = None
    for entry in entries:
        counts = entry["histograms"]["zone_to_zone"]
        total = counts if total is None else [a + b for a, b in zip(total, counts)]
    if not total or not sum(total):
        return math.nan
    half, running = sum(total) / 2, 0
    for second, count in enumerate(total):
        running += count
        if running >= half:
            return second + 0.5
    return math.nan


def main() -> None:
    """The report the module describes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=pathlib.Path, nargs="+")
    parser.add_argument("--thresholds", default="2,3")
    parser.add_argument("--clocks", default="600,900,1200")
    arguments = parser.parse_args()
    by_label: dict[tuple, list[dict]] = {}
    for path in arguments.files:
        for line in path.read_text().splitlines():
            entry = json.loads(line)
            by_label.setdefault((entry["ships"], entry["label"]), []).append(entry)
    print(
        "| n | variant | T | clock (s) | P(same next) | by the front | median length (s) "
        "| captures / game | first capture (s) | zone-to-zone leg (s) | games |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for (ships, label), entries in sorted(by_label.items()):
        tick = entries[0]["tick_length"]
        games = [game for entry in entries for game in entry["games"]]
        leg = leg_median(entries)
        for threshold in map(int, arguments.thresholds.split(",")):
            for clock in map(float, arguments.clocks.split(",")):
                if clock > entries[0]["episode_ticks"] * tick + 1e-6:
                    continue
                r = replay(games, tick, threshold, clock)
                print(
                    f"| {ships} | {label} | {threshold} | {clock:.0f} | "
                    f"{r['p_same']:.3f} ± {r['p_same_error']:.3f} | "
                    f"{r['by_front']:.3f} ± {r['by_front_error']:.3f} | "
                    f"{r['median_length']:.0f} | {r['captures']:.1f} | "
                    f"{r['first_capture']:.0f} | {leg:.1f} | {r['games']} |"
                )


if __name__ == "__main__":
    main()
