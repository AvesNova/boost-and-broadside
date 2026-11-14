"""Is the frontline's front a random walk? P(reversal) of the scripted agent's captures.

Games come from `bench.mode_dynamics frontline` (the scripted agent on both seats, real-scale
frontline preset, 5v5, fog on, each environment's first game, one JSON line per batch):

    PYTHONPATH=src python -m bench.mode_dynamics frontline --out FILE --games N --label L \
        [--lineup yongma,jujak] [--device cpu --eager]
    PYTHONPATH=src python -m bench.front_reversal FILE [FILE ...]

Definitions. The front is the game's signed capture count (`ModeState.front`: +1 for a team 0
capture, -1 for a team 1 capture). A move is one capture, so each move is +1 or -1 (the
winning capture included). A reversal is a move whose direction is opposite to the previous
move's, in the same game; the first move of a game has no previous move and is not counted.
P(reversal) = reversals / (moves that have a previous move), pooled over games, with a
Wilson 95% interval and a game-clustered bootstrap 95% interval (moves within a game are not
independent). A random walk has P(reversal) = 0.5; below 0.5 is persistence, above it anti-persistence. Also reported: moves per
game, and how games ended: by capture (the front reached the threshold) or on the clock.
"""

import json
import math
import pathlib
import random
import sys


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """The Wilson score interval."""
    if total == 0:
        return (0.0, 1.0)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (centre - half, centre + half)


def counts(game: dict) -> tuple[int, int]:
    """(reversals, pairs of consecutive moves) of one game."""
    teams = [team for _, team in game["events"]]
    pairs = list(zip(teams, teams[1:]))
    return sum(a != b for a, b in pairs), len(pairs)


def main() -> None:
    """Report the files named on the command line."""
    games, headers = [], set()
    for name in sys.argv[1:]:
        for line in pathlib.Path(name).read_text().splitlines():
            entry = json.loads(line)
            games += entry["games"]
            headers.add(
                (entry["preset"], entry["ships"], entry["fog"], entry["lineup"], entry["tick_length"])
            )
    per_game = [counts(game) for game in games]
    reversals = sum(r for r, _ in per_game)
    pairs = sum(n for _, n in per_game)
    low, high = wilson(reversals, pairs)
    rng = random.Random(0)
    boots = []
    for _ in range(4000):
        sample = [per_game[rng.randrange(len(per_game))] for _ in per_game]
        total = sum(n for _, n in sample)
        if total:
            boots.append(sum(r for r, _ in sample) / total)
    boots.sort()
    moves = [len(game["events"]) for game in games]
    by_rule = sum(game["by_rule"] for game in games)
    print(f"settings: {sorted(headers, key=str)}")
    print(f"games {len(games)}; moves {sum(moves)} ({sum(moves) / len(games):.2f} per game)")
    print(f"consecutive move pairs {pairs}; reversals {reversals}")
    print(f"P(reversal) = {reversals / pairs:.3f}  Wilson 95% [{low:.3f}, {high:.3f}]")
    print(
        f"  game-clustered bootstrap 95% [{boots[int(0.025 * len(boots))]:.3f}, "
        f"{boots[int(0.975 * len(boots))]:.3f}]"
    )
    print(f"ended by capture {by_rule} ({by_rule / len(games):.1%}); on the clock "
          f"{len(games) - by_rule} ({1 - by_rule / len(games):.1%})")
    histogram = {}
    for count in moves:
        histogram[count] = histogram.get(count, 0) + 1
    print("games by number of moves:", dict(sorted(histogram.items())))


if __name__ == "__main__":
    main()
