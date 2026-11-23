"""v1's crossover walk, played through the evaluator (lean.eval.evaluator)."""

from lean.eval.store import ScriptedAgent


def games_for(total_ships: int) -> int:
    """v1's games per matchup: max(48, min(256, 4,000,000 // total_ships ** 2))."""
    return max(48, min(256, 4_000_000 // max(1, total_ships) ** 2))


def walk(start: int, matches: dict, limit: int) -> tuple[int | None, tuple[int, int] | None]:
    """v1's walk for one network size over the matches played so far ({scripted_size:
    {"win_rate", ...}}), from the scripted size `start`: upward while the network wins at
    least half its games, or downward to the first count it does (a tie is a non-win).

    Returns (beats_up_to, None) when the walk is complete, else (None, (scripted_size,
    direction)): the next scripted size it needs and the way it is walking (+1 up, -1
    down, 0 at the start).
    """
    count = min(max(1, start), limit)
    if count not in matches:
        return None, (count, 0)
    if matches[count]["win_rate"] >= 0.5:
        while count < limit:
            if count + 1 not in matches:
                return None, (count + 1, 1)
            if matches[count + 1]["win_rate"] < 0.5:
                break
            count += 1
        return count, None
    while count > 1:
        if count - 1 not in matches:
            return None, (count - 1, -1)
        if matches[count - 1]["win_rate"] >= 0.5:
            break
        count -= 1
    return count - 1, None


def evaluator_round(evaluator, network, opponent=None):
    """The crossover's `play_round` on an evaluator: `network` (a NetworkAgent) against
    `opponent` (default the yardstick), every wanted match in one job; stored games count,
    so a match may hold more games than asked."""
    opponent = opponent or ScriptedAgent.of()

    def play_round(wanted):
        counts = evaluator.ensure([(network, opponent, (size, scripted), games) for size, scripted, games in wanted])
        return [(entry["wins"], entry["games"]) for entry in counts]

    return play_round


def crossover(
    play_round, sizes, games, *, maximum_total_ships: int = 320, start_ratio: float | dict = 1.0, spread: int = 2,
) -> dict:
    """v1's size-curve protocol, every network size at once.

    For each network team size in `sizes` the scripted team's size is walked one ship at a
    time (`walk`) from round(network_size * start_ratio) (parity by default; `start_ratio`
    may be a dict by network size, such as the previous evaluation's ratios): upward while
    the network wins at least half its games, or downward to the first count it does. A tie
    is a non-win. The walk assumes the win rate falls as scripted ships are added, and stops
    at `maximum_total_ships`.

    The walks run in rounds. Each round plays, in one call of `play_round`, the next
    scripted size every unfinished walk needs with its neighbours: `spread` sizes either
    side at the start, then 2 * `spread` further sizes in the walk's direction. A walk
    whose beats_up_to lies in start - spread .. start + spread - 1 finishes in one round.
    Under the walk's assumption the start changes which matches are played, not the
    answer.

    `games` is an int or a function of the total ship count (see `games_for`).
    `play_round([(network_size, scripted_size, games), ...])` returns each match's
    (wins, games) (evaluator_round, or a stand-in in tests).

    Returns {"rounds", "sizes": {network_size: {"beats_up_to", "beats_up_to_win_rate",
    "crossover", "crossover_win_rate" (None if capped), "ratio", "matches":
    {scripted_size: {"wins", "games", "win_rate"}}}}}; ratio = beats_up_to / network_size,
    the break-even ratio; beats_up_to is 0 when even one scripted ship is not beaten.
    "matches" holds every match played, the neighbours included.
    """
    def count_for(network_size, scripted_size):
        return games(network_size + scripted_size) if callable(games) else games

    played = {network_size: {} for network_size in sizes}
    beaten = {}
    rounds = 0
    while True:
        wanted = []
        for network_size in sorted(played):
            limit = maximum_total_ships - network_size
            ratio = start_ratio.get(network_size, 1.0) if isinstance(start_ratio, dict) else start_ratio
            beats_up_to, need = walk(round(network_size * ratio), played[network_size], limit)
            if need is None:
                beaten[network_size] = beats_up_to
                continue
            scripted_size, direction = need
            low, high = {0: (-spread, spread), 1: (0, 2 * spread), -1: (-2 * spread, 0)}[direction]
            for size in range(max(1, scripted_size + low), min(limit, scripted_size + high) + 1):
                if size not in played[network_size]:
                    wanted.append((network_size, size, count_for(network_size, size)))
        if not wanted:
            break
        rounds += 1
        for (network_size, size, _), (won, count) in zip(wanted, play_round(wanted), strict=True):
            played[network_size][size] = {"wins": won, "games": count, "win_rate": won / count}

    report = {}
    for network_size in sorted(played):
        matches = played[network_size]
        beats_up_to = beaten[network_size]
        first_lost = beats_up_to + 1 if beats_up_to + 1 in matches else None
        report[network_size] = {
            "beats_up_to": beats_up_to,
            "beats_up_to_win_rate": matches[beats_up_to]["win_rate"] if beats_up_to in matches else None,
            "crossover": first_lost,
            "crossover_win_rate": matches[first_lost]["win_rate"] if first_lost is not None else None,
            "ratio": beats_up_to / network_size,
            "matches": dict(sorted(matches.items())),
        }
    return {"rounds": rounds, "sizes": report}
