"""Shared tournament allocation and outcome-scoring contracts."""

import numpy as np

from boost_and_broadside.config import EnvConfig, ShipConfig
from boost_and_broadside.evaluation.agents import ResolvedAgent
from boost_and_broadside.evaluation.tournament import Player, Tournament, effective_wins


def _tournament(num_envs: int = 6) -> Tournament:
    players = [
        Player("a", ResolvedAgent("random", None), None, None),
        Player("b", ResolvedAgent("random", None), None, None),
    ]
    env_config = EnvConfig(
        num_ships=8,
        max_bullets=4,
        max_episode_steps=8,
        num_fields=0,
    )
    return Tournament(players, ShipConfig(), env_config, "ego_pass", num_envs, "cpu")


def test_pair_allocation_balances_team_sides():
    tournament = _tournament()
    allocation = np.array([[0, 6], [6, 0]], dtype=np.int64)
    team0, team1 = tournament._assign(allocation)
    assignments = list(zip(team0.tolist(), team1.tolist()))
    assert assignments.count((0, 1)) == 3
    assert assignments.count((1, 0)) == 3


def test_half_win_scoring_is_symmetric_while_decisive_drops_ties():
    wins = np.array([[0.0, 7.0], [3.0, 0.0]])
    ties = np.array([[0.0, 4.0], [0.0, 0.0]])
    assert effective_wins(wins, ties, "half_win").tolist() == [[0.0, 9.0], [5.0, 0.0]]
    assert effective_wins(wins, ties, "decisive") is wins

