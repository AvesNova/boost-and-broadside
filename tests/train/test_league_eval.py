"""Tests for information-scheduled league evaluation."""

import pytest
import torch

from boost_and_broadside.config import EnvConfig, LeagueEvalConfig, ModelConfig, ShipConfig
from boost_and_broadside.models.mvp.policy import MVPPolicy
from boost_and_broadside.train.rl.features import build_standard_coordinator
from boost_and_broadside.train.rl.league_eval import LeagueEvaluator, schedule_eval_pairs
from boost_and_broadside.train.rl.rating import expected_score
from boost_and_broadside.train.rl.roster import LeagueRoster


def _make_evaluator(
    max_episode_steps: int, eval_block_rollouts: int
) -> tuple[LeagueEvaluator, LeagueRoster]:
    ship_config = ShipConfig()
    env_config = EnvConfig(num_ships=2, max_bullets=4, max_episode_steps=max_episode_steps)
    model_config = ModelConfig(d_model=32, n_heads=4, n_transformer_blocks=1)
    coordinator = build_standard_coordinator(ship_config)
    policy = MVPPolicy(
        model_config, coordinator, num_value_components=1, num_ships=2, team_pma_k=()
    )
    roster = LeagueRoster(8, "hard", 2.0, 10.0)
    config = LeagueEvalConfig(
        eval_num_envs=4,
        step_interval=1,
        eval_pairs=2,
        eval_slots=1,
        eval_block_rollouts=eval_block_rollouts,
        max_episode_steps=max_episode_steps,
    )
    evaluator = LeagueEvaluator(
        config=config,
        ship_config=ship_config,
        env_config=env_config,
        model_config=model_config,
        coordinator=coordinator,
        num_value_components=1,
        team_pma_k=(),
        device=torch.device("cpu"),
        obstacle_cache=None,
        roster=roster,
        live_policy=policy,
        avg_policy=policy,
        scripted_agent=None,
        num_ships=2,
        num_tokens=2,
        compile_mode=None,
    )
    return evaluator, roster


@pytest.mark.parametrize(
    ("rating", "opponent", "expected"),
    [(0.0, 0.0, 0.5), (400.0, 0.0, 10.0 / 11.0), (0.0, 400.0, 1.0 / 11.0)],
)
def test_expected_score(rating: float, opponent: float, expected: float) -> None:
    assert expected_score(rating, opponent) == pytest.approx(expected)


def test_scheduler_guarantees_gate_rung_and_canary_pairs() -> None:
    agent_ids = ["live", "random", "scripted", "avg", "ckpt_1", "ckpt_2"]
    ratings = dict.fromkeys(agent_ids, 0.0)
    errors = dict.fromkeys(agent_ids, 100.0)
    errors["random"] = 0.0

    pairs = schedule_eval_pairs(
        agent_ids, ratings, errors, 6, avg_active=True, newest_checkpoint_id="ckpt_2"
    )

    assert any({"live", "scripted"} == set(pair) for pair in pairs)
    assert any({"live", "ckpt_2"} == set(pair) for pair in pairs)
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


def test_envs_persist_across_begin_rollout_within_block() -> None:
    evaluator, _ = _make_evaluator(max_episode_steps=64, eval_block_rollouts=4)
    pairs = evaluator.begin_rollout(avg_active=False)
    evaluator.env.state.step_count.zero_()
    evaluator.step(0)
    step_counts = evaluator.env.state.step_count.clone()

    assert evaluator.begin_rollout(avg_active=False) == pairs
    assert torch.equal(evaluator.env.state.step_count, step_counts)


def test_hard_reset_records_from_first_episode() -> None:
    evaluator, roster = _make_evaluator(max_episode_steps=4, eval_block_rollouts=10)
    evaluator.begin_rollout(avg_active=False)

    # The block assignment hard-reset the envs, so the very first completed
    # episode of every env is attributable and recorded.
    assert bool(evaluator._attribution_valid.all())
    for _ in range(4):
        evaluator.step(0)
    assert float(roster.counts.tensor.sum()) == 4.0
    for _ in range(4):
        evaluator.step(0)
    assert float(roster.counts.tensor.sum()) == 8.0


def test_roster_layout_change_forces_reschedule_with_hard_reset() -> None:
    evaluator, roster = _make_evaluator(max_episode_steps=64, eval_block_rollouts=10)
    evaluator.begin_rollout(avg_active=False)
    evaluator.env.state.step_count.fill_(7)

    roster.add_avg(global_step=1, update=1)
    evaluator.begin_rollout(avg_active=False)

    # The layout change triggers an immediate reschedule whose hard reset
    # starts fresh, attributable episodes.
    assert bool(evaluator._attribution_valid.all())
    assert bool((evaluator.env.state.step_count == 0).all())


def test_flush_returns_independent_cpu_snapshots_and_drains_tally() -> None:
    roster = LeagueRoster(8, "hard", 2.0, 10.0)
    roster.record_outcomes(
        torch.tensor([roster.counts.index("live")] * 3),
        torch.tensor([roster.counts.index("random")] * 3),
        torch.tensor([0, 0, 0]),
    )
    evaluator = LeagueEvaluator.__new__(LeagueEvaluator)
    evaluator.roster = roster
    evaluator._stream = None

    counts, tally = evaluator.flush()
    roster.counts.add_pair("live", "random", wins=1.0)

    assert counts.device.type == "cpu"
    assert counts.tensor[counts.index("live"), counts.index("random"), 0] == 3.0
    assert tally.tensor[tally.index("live"), tally.index("random"), 0] == 3.0
    # The tally is drained on flush; lifetime counts persist.
    assert roster.update_tally.tensor.sum() == 0.0
    assert roster.counts.tensor.sum() == 4.0
