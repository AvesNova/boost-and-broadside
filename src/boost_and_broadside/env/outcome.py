"""Authoritative match-outcome helpers shared by training and evaluation."""

import torch

from boost_and_broadside.config import MatchResult
from boost_and_broadside.env.state import TensorState


def outcome_masks(
    state: TensorState,
    terminal: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Team 0 win, Team 1 win, and draw masks from ``match_result``.

    ``ship_alive`` is deliberately absent from this contract. Frontline ships
    respawn immediately, so survival is not an outcome signal in that mode.
    """

    if terminal is None:
        terminal = state.match_result != int(MatchResult.ONGOING)
    team0_won = terminal & (state.match_result == int(MatchResult.TEAM0_WIN))
    team1_won = terminal & (state.match_result == int(MatchResult.TEAM1_WIN))
    tied = terminal & (state.match_result == int(MatchResult.DRAW))
    return team0_won, team1_won, tied


def winner_name(state: TensorState, env_index: int = 0) -> str:
    """Return the stable external result name for one environment."""

    result = int(state.match_result[env_index].item())
    if result == int(MatchResult.TEAM0_WIN):
        return "team0"
    if result == int(MatchResult.TEAM1_WIN):
        return "team1"
    return "tie"
