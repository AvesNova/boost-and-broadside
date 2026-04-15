"""End-to-end smoke test for the PPO training loop."""

import torch

from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config import (
    ShipConfig,
    EnvConfig,
    ModelConfig,
    RewardConfig,
    TrainingSchedule,
    TrainConfig,
    ScaleConfig,
    constant,
    stepped,
    linear,
)
from boost_and_broadside.train.rl.ppo import PPOTrainer, _GROUP


def _make_rewards(**overrides) -> RewardConfig:
    defaults = dict(
        ally_damage_weight=0.01,
        enemy_damage_weight=0.01,
        ally_death_weight=0.5,
        enemy_death_weight=0.5,
        ally_win_weight=1.0,
        enemy_win_weight=1.0,
        facing_weight=0.01,
        closing_speed_weight=0.01,
        shoot_quality_weight=0.01,
        kill_shot_weight=0.5,
        kill_assist_weight=0.5,
        damage_taken_weight=0.1,
        damage_dealt_enemy_weight=0.1,
        damage_dealt_ally_weight=0.1,
        death_weight=0.5,
        proximity_radius=300.0,
        shoot_quality_radius=200.0,
        enemy_neg_lambda_components=frozenset(
            {"enemy_damage", "enemy_death", "enemy_win"}
        ),
        ally_zero_components=frozenset({"enemy_damage", "enemy_death", "enemy_win"}),
    )
    defaults.update(overrides)
    return RewardConfig(**defaults)


def _make_schedule(**overrides) -> TrainingSchedule:
    defaults = dict(
        learning_rate=constant(3e-4),
        policy_gradient_coef=constant(1.0),
        entropy_coef=constant(0.01),
        behavior_cloning_coef=constant(0.0),
        value_function_coef=constant(0.5),
        true_reward_scale=constant(1.0),
        global_scale=constant(1.0),
        local_scale=constant(1.0),
        scripted_fraction=constant(0.0),
        avg_model_fraction=constant(0.0),
        league_fraction=constant(0.0),
        allow_avg_model_updates=stepped((0, False)),
        allow_scripted_in_roster=stepped((0, False)),
        elo_eval_games=stepped((0, 16)),
        elo_eval_interval=stepped((0, 0)),
        checkpoint_interval=stepped((0, 0)),
    )
    defaults.update(overrides)
    return TrainingSchedule(**defaults)


def _make_trainer(n_fourier_freqs: int = 4, **reward_overrides) -> PPOTrainer:
    return PPOTrainer(
        train_config=TrainConfig(
            scales=(
                ScaleConfig(
                    env_config=EnvConfig(
                        num_ships=4, max_bullets=8, max_episode_steps=50
                    ),
                    num_envs=4,
                ),
            ),
            schedule=_make_schedule(),
            rewards=_make_rewards(**reward_overrides),
            num_steps=16,
            num_epochs=1,
            num_minibatches=2,
            gamma=0.99,
            gae_lambda=0.95,
            clip_coef=0.2,
            max_grad_norm=0.5,
            total_timesteps=64,
            return_ema_alpha=0.005,
            return_min_span=1.0,
            checkpoint_dir="checkpoints",
            league_size=20,
            league_uniform_sampling=False,
            elo_milestone_gap=50.0,
            elo_k_factor=32.0,
            elo_temperature=200.0,
            scripted_roster_min_steps=0,
        ),
        model_config=ModelConfig(
            d_model=32,
            n_heads=4,
            n_fourier_freqs=n_fourier_freqs,
            n_transformer_blocks=1,
        ),
        ship_config=ShipConfig(),
        device="cpu",
        use_wandb=False,
    )


class TestPPOSmokeTest:
    def test_full_training_loop_runs(self):
        """One complete PPO training run (64 total timesteps) must not raise."""
        trainer = _make_trainer()
        trainer.train()

    def test_encoder_works_with_non_default_n_fourier_freqs(self):
        """Encoder raw dim must adjust when n_fourier_freqs != 8."""
        trainer = _make_trainer(n_fourier_freqs=6)
        trainer.train()

    def test_policy_parameters_change_after_update(self):
        """At least one policy parameter must change after one PPO update."""
        trainer = _make_trainer()
        params_before = [p.clone() for p in trainer.policy.parameters()]

        trainer.train()

        params_after = list(trainer.policy.parameters())
        any_changed = any(
            not torch.equal(b, a) for b, a in zip(params_before, params_after)
        )
        assert any_changed, "No parameters changed after training"


class TestSchedulePrimitives:
    def test_constant_returns_same_value_at_any_step(self):
        fn = constant(3e-4)
        assert fn(0) == 3e-4
        assert fn(1_000_000) == 3e-4
        assert fn(999_999_999) == 3e-4

    def test_linear_warmup_at_start(self):
        fn = linear((0, 1e-7), (1_000_000, 3e-4))
        assert fn(0) == 1e-7

    def test_linear_reaches_target(self):
        fn = linear((0, 1e-7), (1_000_000, 3e-4))
        assert abs(fn(1_000_000) - 3e-4) < 1e-10

    def test_linear_interpolates_midpoint(self):
        fn = linear((0, 0.0), (1_000_000, 1.0))
        assert abs(fn(500_000) - 0.5) < 1e-10

    def test_linear_clamps_before_first_keypoint(self):
        fn = linear((100, 0.0), (200, 1.0))
        assert fn(0) == 0.0

    def test_linear_clamps_after_last_keypoint(self):
        fn = linear((0, 0.0), (1_000_000, 1.0))
        assert fn(99_000_000) == 1.0

    def test_linear_multi_segment(self):
        fn = linear((0, 0.0), (500_000, 1.0), (1_000_000, 0.0))
        assert abs(fn(250_000) - 0.5) < 1e-10
        assert abs(fn(750_000) - 0.5) < 1e-10

    def test_stepped_holds_initial_value(self):
        fn = stepped((0, 0.5))
        assert fn(0) == 0.5
        assert fn(99_999_999) == 0.5

    def test_stepped_changes_at_keypoint(self):
        fn = stepped((0, 0.5), (1_000_000, 0.3))
        assert fn(999_999) == 0.5
        assert fn(1_000_000) == 0.3

    def test_stepped_bool(self):
        fn = stepped((0, False), (5_000_000, True))
        assert fn(0) is False
        assert fn(5_000_000) is True

    def test_stepped_beyond_last_keypoint(self):
        fn = stepped((0, 0.5), (1_000_000, 0.1))
        assert fn(99_000_000) == 0.1

    def test_group_scales_applied_by_trainer(self):
        """After training, effective component weight = group_scale * individual weight."""
        trainer = PPOTrainer(
            train_config=TrainConfig(
                scales=(
                    ScaleConfig(
                        env_config=EnvConfig(
                            num_ships=4, max_bullets=8, max_episode_steps=50
                        ),
                        num_envs=4,
                    ),
                ),
                schedule=_make_schedule(local_scale=constant(0.5)),
                rewards=_make_rewards(closing_speed_weight=0.01),
                num_steps=16,
                num_epochs=1,
                num_minibatches=2,
                gamma=0.99,
                gae_lambda=0.95,
                clip_coef=0.2,
                max_grad_norm=0.5,
                total_timesteps=64,
                return_ema_alpha=0.005,
                return_min_span=1.0,
                checkpoint_dir="checkpoints",
                league_size=20,
                league_uniform_sampling=False,
                elo_milestone_gap=50.0,
                elo_k_factor=32.0,
                elo_temperature=200.0,
                scripted_roster_min_steps=0,
            ),
            model_config=ModelConfig(
                d_model=32, n_heads=4, n_fourier_freqs=4, n_transformer_blocks=1
            ),
            ship_config=ShipConfig(),
            device="cpu",
            use_wandb=False,
        )
        trainer.train()
        for comp in trainer.wrapper._all_components:
            if comp.name == "closing_speed":
                scale_name = _GROUP[comp.name]
                individual_weight = getattr(trainer.cfg.rewards, f"{comp.name}_weight")
                group_scale = getattr(trainer._schedule_state, scale_name)
                expected = individual_weight * group_scale
                actual = comp.weight
                assert abs(actual - expected) < 1e-9


class TestRLSmokeTest:
    """Full RL smoke test using the real runs/shared.py config.

    Exercises the complete training stack with the production reward config
    (including kill_shot and kill_assist) for a small number of updates.
    Uses a scripted opponent to ensure combat happens and kill rewards fire.
    """

    def test_rl_run_with_production_config(self):
        from runs.shared import MODEL_CONFIG, REWARDS, SHIP_CONFIG

        schedule = TrainingSchedule(
            learning_rate=constant(3e-4),
            policy_gradient_coef=constant(1.0),
            entropy_coef=constant(0.01),
            behavior_cloning_coef=constant(0.0),
            value_function_coef=constant(1.0),
            true_reward_scale=constant(1.0),
            global_scale=constant(1.0),
            local_scale=constant(1.0),
            scripted_fraction=constant(0.5),
            avg_model_fraction=constant(0.0),
            league_fraction=constant(0.0),
            allow_avg_model_updates=constant(False),
            allow_scripted_in_roster=constant(True),
            elo_eval_games=constant(0),
            elo_eval_interval=constant(9999),
            checkpoint_interval=constant(9999),
        )
        cfg = TrainConfig(
            scales=(
                ScaleConfig(
                    env_config=EnvConfig(
                        num_ships=4, max_bullets=20, max_episode_steps=64
                    ),
                    num_envs=16,
                ),
            ),
            schedule=schedule,
            rewards=REWARDS,
            num_steps=32,
            num_epochs=1,
            num_minibatches=2,
            gamma=0.99,
            gae_lambda=0.95,
            clip_coef=0.2,
            max_grad_norm=1.0,
            total_timesteps=16 * 32 * 3,  # 3 updates
            return_ema_alpha=0.005,
            return_min_span=1.0,
            checkpoint_dir="checkpoints",
            league_size=5,
            elo_milestone_gap=100.0,
            elo_k_factor=32.0,
            elo_temperature=200.0,
            league_uniform_sampling=False,
            scripted_roster_min_steps=0,
        )
        scripted = StochasticScriptedAgent(SHIP_CONFIG, StochasticAgentConfig())
        trainer = PPOTrainer(
            cfg,
            MODEL_CONFIG,
            SHIP_CONFIG,
            device="cpu",
            use_wandb=False,
            scripted_agent=scripted,
        )
        trainer.train()
