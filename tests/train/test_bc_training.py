"""Bounded validation that the corrected BC profile trains as BC (S11).

``tests/config/test_bc_profile.py`` proves the resolved *values*.  This drives
the real ``PPOTrainer`` from the registered BC profile so the objective is
validated where it takes effect: no policy gradient, no league opponent, the
scripted controller supplying targets, and the zero-field gauge RL continues on.

Only launch sizing is reduced -- environment width, rollout length, model width,
budget, and evaluator widths.  Everything the profile means is untouched.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config.core import ModelConfig
from boost_and_broadside.config.defaults import (
    ZERO_FIELD_RANDOM_ELO,
    ZERO_FIELD_REFERENCE_LADDER,
)
from boost_and_broadside.config.resolve import resolve_profile
from boost_and_broadside.config.schema import LaunchSizingSpec, ResolvedTrainConfig
from boost_and_broadside.profiles import PROFILES
from boost_and_broadside.train.rl.ppo import PPOTrainer

_NUM_ENVS = 4
_NUM_STEPS = 8
_UPDATES = 2


def _bounded_bc(checkpoint_dir: str) -> ResolvedTrainConfig:
    """Resolve the registered BC profile at a launch size a CPU test can run."""

    profile = PROFILES["bc"]
    entity_tokens = profile.environment.num_ships + profile.environment.num_fields
    rollout_tokens = _NUM_ENVS * entity_tokens * _NUM_STEPS
    bounded = replace(
        profile,
        model_config=ModelConfig(d_model=32, n_heads=4, n_yemong_blocks=1),
        environment=replace(profile.environment, max_episode_steps=64),
        rollout=replace(
            profile.rollout,
            logical_batch_tokens=rollout_tokens,
            num_steps=_NUM_STEPS,
            num_minibatches=1,
        ),
        launch_defaults=LaunchSizingSpec(num_envs=_NUM_ENVS),
        optimizer=replace(
            profile.optimizer,
            total_timesteps=_NUM_ENVS * _NUM_STEPS * _UPDATES,
            checkpoint_dir=checkpoint_dir,
            histogram_interval=1000,
            log_interval=1,
        ),
        league=replace(
            profile.league,
            elo_eval=replace(
                profile.league.elo_eval,
                envs_per_matchup=2,
                step_interval=1,
                window_size=4,
                min_games_to_freeze=0,
            ),
        ),
    )
    return resolve_profile(bounded)


def _trainer(tmp_path, *, with_scripted: bool = True) -> PPOTrainer:
    resolved = _bounded_bc(str(tmp_path))
    scripted = (
        StochasticScriptedAgent(resolved.ship_config, StochasticAgentConfig())
        if with_scripted
        else None
    )
    return PPOTrainer(
        train_config=resolved.train_config,
        model_config=resolved.model_config,
        ship_config=resolved.ship_config,
        device="cpu",
        use_wandb=False,
        scripted_agent=scripted,
    )


def test_bc_allocates_no_league_envs_despite_declaring_slots(tmp_path) -> None:
    """``league_slots=4`` is inert while the fraction is zero for the whole budget."""

    trainer = _trainer(tmp_path)

    assert trainer.cfg.league_slots == 4
    assert trainer.B_league == 0
    assert trainer.B_self == _NUM_ENVS
    assert trainer._prepare_league_slots(trainer.wrapper.num_ships) == []


def test_bc_requires_the_scripted_controller_it_clones(tmp_path) -> None:
    with pytest.raises(ValueError, match="BC mode"):
        _trainer(tmp_path, with_scripted=False)


def test_bc_rates_on_the_zero_field_gauge(tmp_path) -> None:
    trainer = _trainer(tmp_path)

    rungs = {entry.p_scripted: entry.elo for entry in trainer.roster.entries if entry.fixed}
    for probability, elo in ZERO_FIELD_REFERENCE_LADDER:
        assert rungs[probability] == elo
    assert trainer._random_elo() == ZERO_FIELD_RANDOM_ELO
    assert trainer._training_elo == ZERO_FIELD_RANDOM_ELO

    stationary = [entry for entry in trainer.roster.entries if entry.is_stationary]
    assert len(stationary) == len(ZERO_FIELD_REFERENCE_LADDER) + 2  # + random + scripted
    assert all(entry.fixed for entry in stationary)


def test_bc_takes_no_policy_gradient_across_its_whole_budget(tmp_path) -> None:
    trainer = _trainer(tmp_path)
    budget = PROFILES["bc"].optimizer.total_timesteps

    for step in (0, budget // 2, budget):
        assert trainer.cfg.schedule.policy_gradient_coef(step) == 0.0
        assert trainer.cfg.schedule.league_fraction(step) == 0.0
    assert trainer._policy_gradient_coef == 0.0
    # Full strength before any scripted game has been recorded.
    assert trainer._behavior_cloning_coef == pytest.approx(1.0)
    assert trainer.cfg.next_state_coef == 1.0


def test_bounded_bc_run_learns_from_supervision_and_freezes_no_milestone(tmp_path) -> None:
    trainer = _trainer(tmp_path)
    before = [parameter.clone() for parameter in trainer.policy.parameters()]

    trainer.train()

    after = list(trainer.policy.parameters())
    assert any(
        not torch.equal(one, other) for one, other in zip(before, after, strict=True)
    ), "no policy parameter moved under the behavior-cloning objective"
    assert trainer._global_step == _NUM_ENVS * _NUM_STEPS * _UPDATES
    # Milestones are gated on a live policy gradient, so BC contributes no
    # frozen ladder entry no matter how its rating moves.
    assert trainer.roster.floating_checkpoint() is None
    assert trainer._scripted_elo == trainer.cfg.elo_eval.scripted_elo_init
