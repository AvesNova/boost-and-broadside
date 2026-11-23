"""Evaluation: batched matches, head-to-head counts and v1's crossover walk."""

import dataclasses
import importlib
import math

import torch
from torch import Tensor

from lean.learn.collect import (
    generator_device,
    reset_carry,
    sample_joint,
    scripted_actions,
    split_joint,
)


def wilson_interval(wins: float, games: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval (low, high) for a proportion wins / games; (0, 1) for no games."""
    if games <= 0:
        return 0.0, 1.0
    rate = wins / games
    denominator = 1.0 + z * z / games
    centre = (rate + z * z / (2 * games)) / denominator
    half = z * math.sqrt(rate * (1.0 - rate) / games + z * z / (4 * games * games)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def games_for(total_ships: int) -> int:
    """v1's games per matchup: max(48, min(256, 4,000,000 // total_ships ** 2))."""
    return max(48, min(256, 4_000_000 // max(1, total_ships) ** 2))


@torch.no_grad()
def play_results(
    game_config, policy, opponent, team_sizes: Tensor, generator,
    *, maximum_ticks: int = 8192, check_every: int = 32, game=None, scripted=None,
) -> Tensor:
    """Play one game per environment at once, `policy` as team 0, and return each
    environment's first result, (E,) long: 0 `policy` won, 1 it lost, 2 a draw, -1 still
    running after `maximum_ticks`.

    `team_sizes` (E, 2) gives each environment's ships per team; environments may differ
    (the game pads every one to the largest total). `opponent` plays team 1: anything with
    a `step` method is used as a policy, anything else is taken to be a ScriptedConfig
    (the yardstick, spread 0, its memory carried from tick to tick and renewed for each new
    game, as in collect). Actions are sampled. Finished environments are re-dealt at
    their own sizes and ignored. Games deal the standard spawn (GameConfig.training_spawn
    is turned off).
    """
    game = game or importlib.import_module("lean.game")
    if getattr(game_config, "training_spawn", False):  # evaluation always deals the standard spawn
        game_config = dataclasses.replace(game_config, training_spawn=False)
    device = generator_device(generator)
    sizes = team_sizes.to(device=device, dtype=torch.long).contiguous()
    games = sizes.shape[0]
    state = game.reset(game_config, games, sizes, generator, device)
    slots = state.exists.shape[1]
    carry = policy.initial_carry(games, slots, device)
    opponent_is_policy = hasattr(opponent, "step")
    if opponent_is_policy:
        opponent_carry = opponent.initial_carry(games, slots, device)
    else:
        scripted = scripted or importlib.import_module("lean.scripted")
        persona = scripted.personas(opponent, games, slots, generator, device, 0.0)
        memory = scripted.fresh_memory(games, slots, generator, device)
    result = torch.full((games,), -1, dtype=torch.long, device=device)

    for tick in range(maximum_ticks):
        logits, _, carry = policy.step(game.observe(game_config, state, 0), carry)
        action = split_joint(sample_joint(logits, generator))
        if opponent_is_policy:
            opponent_logits, _, opponent_carry = opponent.step(game.observe(game_config, state, 1), opponent_carry)
            opponent_action = split_joint(sample_joint(opponent_logits, generator))
        else:
            opponent_action, memory = scripted_actions(scripted, opponent, state, persona, memory, generator)
        action = torch.where((state.team == 0)[..., None], action, opponent_action)
        state, step_out = game.step(game_config, state, action, game.draw_noise(game_config, games, slots, generator, device))
        result = torch.where(step_out.done & (result < 0), step_out.result, result)
        state = game.reset_done(game_config, state, step_out.done, sizes, generator)
        carry = reset_carry(carry, step_out.done)
        if opponent_is_policy:
            opponent_carry = reset_carry(opponent_carry, step_out.done)
        else:
            memory = scripted.renew_memory(memory, step_out.done, generator)
        if (tick + 1) % check_every == 0 and bool((result >= 0).all()):
            break
    return result


def play(game_config, policy, opponent, team_sizes: tuple[int, int], games: int, generator, **options) -> dict[str, int]:
    """Play `games` games at one size, `policy` as team 0 with team_sizes[0] ships against
    `opponent` with team_sizes[1] (see `play_results`). Returns {"wins", "draws",
    "losses"} for `policy`; games still running at the tick limit count as draws."""
    sizes = torch.tensor(team_sizes, dtype=torch.long).expand(games, 2)
    result = play_results(game_config, policy, opponent, sizes, generator, **options)
    wins = int((result == 0).sum())
    losses = int((result == 1).sum())
    return {"wins": wins, "draws": games - wins - losses, "losses": losses}


def play_matches(
    game_config, policy, opponent, matches: list[tuple[int, int, int]], generator,
    *, maximum_games: int | None = None, **options,
) -> list[int]:
    """Wins of `policy` in each match (team 0 size, team 1 size, games), every match in
    one batch of environments (in consecutive batches of at most `maximum_games`
    environments when given; a match is never split)."""
    batches, batch, batch_games = [], [], 0
    for match in matches:
        if batch and maximum_games is not None and batch_games + match[2] > maximum_games:
            batches.append(batch)
            batch, batch_games = [], 0
        batch.append(match)
        batch_games += match[2]
    if batch:
        batches.append(batch)
    wins = []
    for batch in batches:
        counts = torch.tensor([games for _, _, games in batch])
        sizes = torch.tensor([(first, second) for first, second, _ in batch]).repeat_interleave(counts, dim=0)
        won = (play_results(game_config, policy, opponent, sizes, generator, **options) == 0).long().cpu()
        group = torch.arange(len(batch)).repeat_interleave(counts)
        wins += torch.zeros(len(batch), dtype=torch.long).index_add_(0, group, won).tolist()
    return wins


def head_to_head(game_config, policy_a, policy_b_or_scripted, team_sizes: tuple[int, int], games: int, generator=None, **options) -> dict:
    """Win, draw and loss counts of `policy_a` (team 0) with a 95% Wilson interval on its
    win rate (draws count as non-wins) and its score (wins plus half the draws)."""
    generator = generator or torch.Generator().manual_seed(0)
    counts = play(game_config, policy_a, policy_b_or_scripted, team_sizes, games, generator, **options)
    low, high = wilson_interval(counts["wins"], games)
    return {
        **counts, "games": games, "win_rate": counts["wins"] / games,
        "win_rate_low": low, "win_rate_high": high,
        "score": (counts["wins"] + 0.5 * counts["draws"]) / games,
    }


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


def crossover(
    game_config, policy, scripted_config, sizes, games, generator,
    *, maximum_total_ships: int = 320, start_ratio: float | dict = 1.0, spread: int = 2,
    maximum_games: int | None = None, match=None, **options,
) -> dict:
    """v1's size-curve protocol, every network size at once.

    The network is always team 0 and the scripted agent team 1. For each network team
    size in `sizes` the scripted team's size is walked one ship at a time (`walk`) from
    round(network_size * start_ratio) (parity by default; `start_ratio` may be a dict by
    network size, such as the previous evaluation's ratios): upward while the network wins at least half its games, or downward to the
    first count it does. A tie is a non-win. The walk assumes the win rate falls as
    scripted ships are added, and stops at `maximum_total_ships`.

    The walks run in rounds. Each round plays, in one batch of environments, the next
    scripted size every unfinished walk needs with its neighbours: `spread` sizes either
    side at the start, then 2 * `spread` further sizes in the walk's direction. A walk
    whose beats_up_to lies in start - spread .. start + spread - 1 finishes in one round.
    Under the walk's assumption the start changes which matches are played, not the
    answer.
    `maximum_games` caps the environments of one batch (a round then plays in several).

    `games` is an int or a function of the total ship count (see `games_for`).
    `match(network_size, scripted_size, games) -> wins` replaces the batched games when
    given (used by tests).

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
        if match is None:
            wins = play_matches(game_config, policy, scripted_config, wanted, generator, maximum_games=maximum_games, **options)
        else:
            wins = [match(network_size, size, count) for network_size, size, count in wanted]
        for (network_size, size, count), won in zip(wanted, wins, strict=True):
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
