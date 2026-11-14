"""Is the frontline's front a random walk? P(reversal) of the scripted agent's captures.

Games come from `bench.mode_dynamics frontline` (the scripted agent on both seats, real-scale
frontline preset, 5v5, fog on, each environment's first game, one JSON line per batch):

    PYTHONPATH=src python -m bench.mode_dynamics frontline --out FILE --games N --label L \
        [--lineup "yongma,jujak;jujak,yongma"] [--fields] [--device cpu --eager]
    PYTHONPATH=src python -m bench.front_reversal FILE [FILE ...] [--label=L ...]

Definitions. The front is the game's signed capture count (`ModeState.front`: +1 for a team 0
capture, -1 for a team 1 capture). A move is one capture, so each move is +1 or -1 (the
winning capture included). A reversal is a move whose direction is opposite to the previous
move's, in the same game; the first move of a game has no previous move and is not counted.
P(reversal) = reversals / (moves that have a previous move), pooled over games, with a
Wilson 95% interval and a game-clustered bootstrap 95% interval (moves within a game are not
independent). A random walk has P(reversal) = 0.5; below 0.5 is persistence, above it
anti-persistence. The same number is P(next capture by the team that just captured) =
1 - P(reversal); also by the front's size before the move. Also reported: moves per game,
how games ended (by capture, the front reaching the threshold, or on the clock), the mean
game length, the time from a capture to the next by whether it reverses, and (when the
batches carry them) the capture lock's occupancy tallies.
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
    """Report the files named on the command line (`--label=L`: only the batches labelled L)."""
    games, headers, tallies, tick_lengths, lengths = [], set(), {}, set(), []
    names = [name for name in sys.argv[1:] if not name.startswith("--label=")]
    labels = {name.split("=", 1)[1] for name in sys.argv[1:] if name.startswith("--label=")}
    for name in names:
        for line in pathlib.Path(name).read_text().splitlines():
            entry = json.loads(line)
            if labels and entry["label"] not in labels:
                continue
            games += entry["games"]
            tick_lengths.add(entry["tick_length"])
            lengths += [game["end_tick"] * entry["tick_length"] for game in entry["games"]]
            for key, value in entry.get("lock_tallies", {}).items():
                tallies[key] = tallies.get(key, 0) + value
            headers.add(
                (
                    entry["preset"],
                    entry["ships"],
                    entry["fog"],
                    entry["lineup"],
                    entry["tick_length"],
                )
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
    print(
        f"ended by capture {by_rule} ({by_rule / len(games):.1%}); on the clock "
        f"{len(games) - by_rule} ({1 - by_rule / len(games):.1%})"
    )
    # By the front's size after the previous move (the state a move is made from): |front| 0, 1, 2.
    by_state: dict[int, list[int]] = {}
    for game in games:
        front = 0
        previous = None
        for _, team in game["events"]:
            step = 1 if team == 0 else -1
            if previous is not None:
                cell = by_state.setdefault(abs(front), [0, 0])
                cell[0] += step != previous
                cell[1] += 1
            front += step
            previous = step
    for size, (r, n) in sorted(by_state.items()):
        lo, hi = wilson(r, n)
        print(
            f"  front |{size}| before the move: P(reversal) {r}/{n} = {r / n:.3f} "
            f"[{lo:.3f}, {hi:.3f}]"
        )
    print(f"P(next capture by the team that just captured) = {1 - reversals / pairs:.3f}")
    histogram = {}
    for count in moves:
        histogram[count] = histogram.get(count, 0) + 1
    print("games by number of moves:", dict(sorted(histogram.items())))
    print(f"mean game length {sum(lengths) / len(lengths):.0f} s")
    # The time from a capture to the next, by whether the next one reverses it.
    (tick,) = tick_lengths
    gaps: dict[bool, list[float]] = {True: [], False: []}
    for game in games:
        for (t0, a), (t1, b) in zip(game["events"], game["events"][1:]):
            gaps[a != b].append((t1 - t0) * tick)
    for reversal, values in gaps.items():
        values.sort()
        if values:
            q = lambda f: values[min(len(values) - 1, int(f * len(values)))]  # noqa: E731
            print(
                f"  time to the next capture, {'reversal' if reversal else 'same team'}: "
                f"n {len(values)}, quartiles {q(0.25):.0f} / {q(0.5):.0f} / {q(0.75):.0f} s"
            )
    if tallies.get("locked_alive"):
        alive = tallies["locked_alive"]
        print(
            f"while a team is locked out: its living ships' time in the locked zone "
            f"{tallies['locked_in_locked_zone'] / alive:.1%}, in its own defense zone "
            f"{tallies['locked_in_own_defense'] / alive:.1%}; the capturer's in its attack "
            f"zone {tallies['capturer_in_attack_zone'] / max(1, tallies['capturer_alive']):.1%}"
        )


if __name__ == "__main__":
    main()
