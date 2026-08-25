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
from boost_and_broadside.config.defaults import LIVE_REFERENCE_PROBABILITIES
from boost_and_broadside.config.live_elo import (
    LIVE_RANDOM_ELO,
    LIVE_SCRIPTED_ELO,
    live_reference_ladder,
)
from boost_and_broadside.config.resolve import resolve_profile
from boost_and_broadside.config.schema import LaunchSizingSpec, ResolvedTrainConfig
from boost_and_broadside.profiles import PROFILES
from boost_and_broadside.train.rl.ppo import PPOTrainer, _actor_entropy_coef

_NUM_ENVS = 4
_NUM_STEPS = 8
_UPDATES = 2


def _bounded_bc(checkpoint_dir: str) -> ResolvedTrainConfig:
    """Resolve the registered BC profile at a launch size a CPU test can run."""

    profile = PROFILES["bc"]
    entity_tokens = profile.num_ships + profile.num_fields
    rollout_tokens = _NUM_ENVS * entity_tokens * _NUM_STEPS
    bounded = replace(
        profile,
        model_config=ModelConfig(d_model=32, n_heads=4, n_yemong_blocks=1),
        max_episode_steps=64,
        logical_batch_tokens=rollout_tokens,
        num_steps=_NUM_STEPS,
        num_minibatches=1,
        launch=LaunchSizingSpec(num_envs=_NUM_ENVS),
        total_timesteps=_NUM_ENVS * _NUM_STEPS * _UPDATES,
        checkpoint_dir=checkpoint_dir,
        histogram_interval=1000,
        log_interval=1,
        elo_eval=replace(
            profile.elo_eval,
            envs_per_matchup=2,
            step_interval=1,
            window_size=4,
            min_games_to_freeze=0,
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


def test_bc_rates_on_the_derived_live_gauge(tmp_path) -> None:
    trainer = _trainer(tmp_path)

    rungs = {entry.p_scripted: entry.elo for entry in trainer.roster.entries if entry.fixed}
    for probability, elo in live_reference_ladder(LIVE_REFERENCE_PROBABILITIES):
        assert rungs[probability] == elo == pytest.approx(1000.0 * probability)
    assert trainer._random_elo() == LIVE_RANDOM_ELO == 0.0
    assert trainer._live_elo == LIVE_RANDOM_ELO
    scripted = next(entry for entry in trainer.roster.entries if entry.kind == "scripted")
    assert scripted.elo == LIVE_SCRIPTED_ELO == 1000.0
    assert scripted.fixed

    stationary = [entry for entry in trainer.roster.entries if entry.is_stationary]
    assert len(stationary) == len(LIVE_REFERENCE_PROBABILITIES) + 2  # + random + scripted
    assert all(entry.fixed for entry in stationary)


def test_bc_takes_no_policy_gradient_across_its_whole_budget(tmp_path) -> None:
    trainer = _trainer(tmp_path)
    budget = PROFILES["bc"].total_timesteps

    for step in (0, budget // 2, budget):
        assert trainer.cfg.schedule.policy_gradient_coef(step) == 0.0
        assert trainer.cfg.schedule.league_fraction(step) == 0.0
    assert trainer._policy_gradient_coef == 0.0
    # Full strength before any scripted game has been recorded.
    assert trainer._behavior_cloning_coef == pytest.approx(1.0)
    assert trainer.cfg.predictive_state_coef == 1.0
    assert trainer.cfg.predictive_action_coef == 1.0


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
    scripted = next(entry for entry in trainer.roster.entries if entry.kind == "scripted")
    assert scripted.elo == trainer.cfg.elo_eval.scripted_live_elo


class TestEntropyAfterTheCloningCutoff:
    """What the actor is left with once the cloning weight decays to zero.

    ``bc_winrate_target`` zeroes ``_behavior_cloning_coef`` at a 45% scripted win
    rate, and BC's ``policy_gradient_coef`` is zero for its whole budget. Before
    this gate the surviving actor term was ``entropy_coef * ent_loss``, whose
    optimum is the uniform distribution: at a reduced launch width a policy
    cloned to a KL of 1.12 and 60% of maximum action entropy came back to 99.8%
    of maximum and a KL of 2.66 — its untrained value — within 400 updates,
    while a control arm that kept cloning held at 1.10 and 60%.
    """

    def test_entropy_is_dropped_when_no_objective_trains_the_actor(self) -> None:
        assert (
            _actor_entropy_coef(0.005, policy_gradient_coef=0.0, behavior_cloning_coef=0.0) == 0.0
        )

    @pytest.mark.parametrize(
        ("policy_gradient_coef", "behavior_cloning_coef"),
        [(1.0, 0.0), (0.0, 1.0), (1.0, 2.0), (1.0, 1e-9)],
    )
    def test_entropy_is_untouched_while_something_trains_the_actor(
        self, policy_gradient_coef: float, behavior_cloning_coef: float
    ) -> None:
        assert (
            _actor_entropy_coef(
                0.005,
                policy_gradient_coef=policy_gradient_coef,
                behavior_cloning_coef=behavior_cloning_coef,
            )
            == 0.005
        )

    def test_a_bc_run_holds_entropy_until_its_cloning_weight_decays(self, tmp_path) -> None:
        """At the real trainer, through the schedule refresh that sets both."""
        trainer = _trainer(tmp_path)
        metrics: dict = {}
        runtime = trainer._initialize_rollout_runtime()

        trainer._refresh_training_schedule(metrics, runtime.elo_eval)
        scheduled = trainer._schedule_state.entropy_coef
        assert scheduled > 0.0
        assert trainer._behavior_cloning_coef > 0.0
        assert trainer._entropy_coef == scheduled == metrics["schedule/entropy_coef"]

        # The cutoff: a full window at or above the target win rate.
        window = trainer._eval_window_sc
        window.extend([1.0] * window.maxlen)
        trainer._refresh_training_schedule(metrics, runtime.elo_eval)

        assert trainer._behavior_cloning_coef == 0.0
        assert trainer._policy_gradient_coef == 0.0
        assert trainer._entropy_coef == 0.0 == metrics["schedule/entropy_coef"]
        trainer.shutdown()

    def test_an_rl_run_keeps_its_entropy_bonus(self, tmp_path) -> None:
        """The gate must not touch the profile it was not about."""
        from tests.train.test_ppo import _make_trainer

        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        metrics: dict = {}
        runtime = trainer._initialize_rollout_runtime()
        window = trainer._eval_window_sc
        window.extend([1.0] * window.maxlen)

        trainer._refresh_training_schedule(metrics, runtime.elo_eval)

        assert trainer._policy_gradient_coef > 0.0
        assert trainer._behavior_cloning_coef == 0.0  # decayed by the same cutoff
        assert trainer._entropy_coef == trainer._schedule_state.entropy_coef > 0.0
        trainer.shutdown()
