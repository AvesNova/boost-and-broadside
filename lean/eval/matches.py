"""Evaluation: batched matches, head-to-head counts and v1's crossover walk."""

import importlib
import math

import torch
from torch import Tensor

from lean.learn.collect import generator_device, sample_joint, scripted_actions, select_environments, split_joint


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
def play(
    game_config, policy, opponent, team_sizes: tuple[int, int], games: int, generator,
    *, maximum_ticks: int = 8192, check_every: int = 32, game=None, scripted=None,
) -> dict[str, int]:
    """Play `games` games at once, `policy` as team 0 with team_sizes[0] ships.

    `opponent` plays team 1 with team_sizes[1] ships: anything with a `step` method is
    used as a policy, anything else is taken to be a ScriptedConfig (fixed personas,
    spread 0). Actions are sampled. Only the first result of each environment counts;
    finished environments are re-dealt and ignored. Games still running after
    `maximum_ticks` count as draws. Returns {"wins", "draws", "losses"} for `policy`.
    """
    game = game or importlib.import_module("lean.game")
    device = generator_device(generator)
    sizes = torch.tensor(team_sizes, dtype=torch.long, device=device).expand(games, 2).contiguous()
    state = game.reset(game_config, games, sizes, generator, device)
    slots = state.exists.shape[1]
    carry = policy.initial_carry(games, slots, device)
    opponent_is_policy = hasattr(opponent, "step")
    if opponent_is_policy:
        opponent_carry = opponent.initial_carry(games, slots, device)
    else:
        scripted = scripted or importlib.import_module("lean.scripted")
        persona = scripted.personas(opponent, games, slots, generator, device, 0.0)
    result = torch.full((games,), -1, dtype=torch.long, device=device)

    for tick in range(maximum_ticks):
        logits, _, carry = policy.step(game.observe(game_config, state, 0), carry)
        action = split_joint(sample_joint(logits, generator))
        if opponent_is_policy:
            opponent_logits, _, opponent_carry = opponent.step(game.observe(game_config, state, 1), opponent_carry)
            opponent_action = split_joint(sample_joint(opponent_logits, generator))
        else:
            opponent_action = scripted_actions(scripted, opponent, state, persona, generator)
        action = torch.where((state.team == 0)[..., None], action, opponent_action)
        state, step_out = game.step(game_config, state, action, game.draw_noise(game_config, games, slots, generator, device))
        result = torch.where(step_out.done & (result < 0), step_out.result, result)
        state = game.reset_done(game_config, state, step_out.done, sizes, generator)
        carry = select_environments(step_out.done, torch.zeros_like(carry), carry)
        if opponent_is_policy:
            opponent_carry = select_environments(step_out.done, torch.zeros_like(opponent_carry), opponent_carry)
        if (tick + 1) % check_every == 0 and bool((result >= 0).all()):
            break
    wins = int((result == 0).sum())
    losses = int((result == 1).sum())
    return {"wins": wins, "draws": games - wins - losses, "losses": losses}


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


def crossover(
    game_config, policy, scripted_config, sizes, games, generator,
    *, maximum_total_ships: int = 320, match=None, **options,
) -> dict:
    """v1's size-curve protocol.

    The network is always team 0 and the scripted agent team 1. For each network team
    size in `sizes` (ascending) the scripted team's size is walked one ship at a time,
    starting from the previous size's crossover (from parity for the first size): upward
    while the network wins at least half its games, or downward to the first count it
    does. A tie is a non-win. The walk assumes the win rate falls as scripted ships are
    added, and stops at `maximum_total_ships`.

    `games` is an int or a function of the total ship count (see `games_for`).
    `match(network_size, scripted_size, games) -> wins` replaces the batched games when
    given (used by tests).

    Returns {"sizes": {network_size: {"beats_up_to", "beats_up_to_win_rate",
    "crossover", "crossover_win_rate" (None if capped), "ratio", "matches":
    {scripted_size: {"wins", "games", "win_rate"}}}}}; ratio = beats_up_to / network_size,
    the break-even ratio; beats_up_to is 0 when even one scripted ship is not beaten.
    """
    if match is None:
        def match(network_size, scripted_size, count):
            return play(game_config, policy, scripted_config, (network_size, scripted_size), count, generator, **options)["wins"]

    report = {}
    start = None
    for network_size in sorted(sizes):
        matches = {}

        def win_rate(scripted_size):
            if scripted_size not in matches:
                count = games(network_size + scripted_size) if callable(games) else games
                wins = match(network_size, scripted_size, count)
                matches[scripted_size] = {"wins": wins, "games": count, "win_rate": wins / count}
            return matches[scripted_size]["win_rate"]

        limit = maximum_total_ships - network_size
        count = min(max(1, start if start is not None else network_size), limit)
        if win_rate(count) >= 0.5:
            while count < limit and win_rate(count + 1) >= 0.5:
                count += 1
            beats_up_to = count
        else:
            while count > 1 and win_rate(count - 1) < 0.5:
                count -= 1
            beats_up_to = count - 1
        first_lost = beats_up_to + 1 if beats_up_to + 1 in matches else None
        report[network_size] = {
            "beats_up_to": beats_up_to,
            "beats_up_to_win_rate": matches[beats_up_to]["win_rate"] if beats_up_to in matches else None,
            "crossover": first_lost,
            "crossover_win_rate": matches[first_lost]["win_rate"] if first_lost is not None else None,
            "ratio": beats_up_to / network_size,
            "matches": dict(sorted(matches.items())),
        }
        start = beats_up_to + 1
    return {"sizes": report}
