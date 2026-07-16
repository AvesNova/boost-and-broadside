"""Tests for information-scheduled league evaluation."""

import pytest
import torch

from boost_and_broadside.train.rl.league_eval import LeagueEvaluator, schedule_eval_pairs
from boost_and_broadside.train.rl.rating import expected_score
from boost_and_broadside.train.rl.roster import LeagueRoster


@pytest.mark.parametrize(
    ("rating", "opponent", "expected"),
    [(0.0, 0.0, 0.5), (400.0, 0.0, 10.0 / 11.0), (0.0, 400.0, 1.0 / 11.0)],
)
def test_expected_score(rating: float, opponent: float, expected: float) -> None:
    assert expected_score(rating, opponent) == pytest.approx(expected)


def test_scheduler_guarantees_live_avg_and_anchor_pairs() -> None:
    agent_ids = ["live", "random", "scripted", "avg", "ckpt_1", "ckpt_2"]
    ratings = dict.fromkeys(agent_ids, 0.0)
    errors = dict.fromkeys(agent_ids, 100.0)
    errors["random"] = 0.0

    pairs = schedule_eval_pairs(agent_ids, ratings, errors, 6, avg_active=True)

    assert any("live" in pair for pair in pairs)
    assert any("avg" in pair for pair in pairs)
    assert any("random" in pair for pair in pairs)


def test_scheduler_can_select_league_vs_league_pair() -> None:
    agent_ids = ["live", "random", "ckpt_1", "ckpt_2"]
    ratings = dict.fromkeys(agent_ids, 0.0)
    errors = {"live": 1.0, "random": 0.0, "ckpt_1": 1_000.0, "ckpt_2": 1_000.0}

    pairs = schedule_eval_pairs(
        agent_ids,
        ratings,
        errors,
        6,
        avg_active=False,
        generator=torch.Generator().manual_seed(0),
    )

    assert any(set(pair) == {"ckpt_1", "ckpt_2"} for pair in pairs)


def test_flush_returns_independent_cpu_snapshot() -> None:
    roster = LeagueRoster(8, "hard", 2.0, 10.0)
    roster.counts.add_pair("live", "random", wins=3.0)
    evaluator = LeagueEvaluator.__new__(LeagueEvaluator)
    evaluator.roster = roster
    evaluator._stream = None

    snapshot = evaluator.flush()
    roster.counts.add_pair("live", "random", wins=1.0)

    assert snapshot.device.type == "cpu"
    assert snapshot.tensor[snapshot.index("live"), snapshot.index("random"), 0] == 3.0
