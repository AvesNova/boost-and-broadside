"""End-to-end smoke test for the PPO training loop."""

import torch

from boost_and_broadside.config import (
    ShipConfig, EnvConfig, ModelConfig, RewardConfig, TrainConfig, ScaleConfig,
)
from boost_and_broadside.train.rl.ppo import PPOTrainer


def _make_trainer(n_fourier_freqs: int = 4) -> PPOTrainer:
    return PPOTrainer(
        train_config=TrainConfig(
            scales=(
                ScaleConfig(
                    env_config=EnvConfig(num_ships=4, max_bullets=8, max_episode_steps=50),
                    num_envs=4,
                ),
            ),
            num_steps=16, num_epochs=1, num_minibatches=2,
            learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_coef=0.2,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, total_timesteps=64,
            return_ema_alpha=0.005, return_min_span=1.0,
        ),
        model_config=ModelConfig(
            d_model=32, n_heads=4, n_fourier_freqs=n_fourier_freqs, n_transformer_blocks=1,
        ),
        ship_config=ShipConfig(),
        reward_config=RewardConfig(
            damage_weight=0.01, death_weight=0.5,
            victory_weight=1.0,
            enemy_neg_lambda_components=frozenset({"damage", "death", "victory", "exposure"}),
            positioning_weight=0.05, positioning_radius=400.0,
            facing_weight=0.01, exposure_weight=0.01,
            proximity_weight=0.01, proximity_radius=300.0,
            closing_speed_weight=0.01,
            turn_rate_weight=0.01,
            power_range_weight=0.01, power_range_lo=0.2, power_range_hi=0.8,
            speed_range_weight=0.01, speed_range_lo=40.0, speed_range_hi=120.0,
            shoot_quality_weight=0.01, shoot_quality_radius=200.0,
        ),
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
            not torch.equal(b, a)
            for b, a in zip(params_before, params_after)
        )
        assert any_changed, "No parameters changed after training"


class TestShapingScheduler:
    def _make_trainer_with_schedule(self, **schedule_kwargs):
        from boost_and_broadside.config import TrainConfig, ScaleConfig, EnvConfig
        return PPOTrainer(
            train_config=TrainConfig(
                scales=(
                    ScaleConfig(
                        env_config=EnvConfig(num_ships=4, max_bullets=8, max_episode_steps=50),
                        num_envs=4,
                    ),
                ),
                num_steps=16, num_epochs=1, num_minibatches=2,
                learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_coef=0.2,
                ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, total_timesteps=64,
                return_ema_alpha=0.005, return_min_span=1.0,
                **schedule_kwargs,
            ),
            model_config=ModelConfig(d_model=32, n_heads=4, n_fourier_freqs=4, n_transformer_blocks=1),
            ship_config=ShipConfig(),
            reward_config=RewardConfig(
                damage_weight=0.01, death_weight=0.5, victory_weight=1.0,
                enemy_neg_lambda_components=frozenset({"damage", "death", "victory", "exposure"}),
                positioning_weight=0.05, positioning_radius=400.0,
                facing_weight=0.01, exposure_weight=0.01,
                proximity_weight=0.01, proximity_radius=300.0,
                closing_speed_weight=0.01, turn_rate_weight=0.01,
                power_range_weight=0.01, power_range_lo=0.2, power_range_hi=0.8,
                speed_range_weight=0.01, speed_range_lo=40.0, speed_range_hi=120.0,
                shoot_quality_weight=0.01, shoot_quality_radius=200.0,
            ),
            device="cpu", use_wandb=False,
        )

    def test_weight_constant_during_hold(self):
        """Component weight must not change before hold_steps."""
        trainer = self._make_trainer_with_schedule(
            shaping_schedules=(("closing_speed", 1_000_000, 500_000),),
        )
        comp = next(c for c in trainer.wrapper._all_components if c.name == "closing_speed")
        initial = comp.closing_speed_weight

        _, _ = trainer._shaping_scheduler.step(500_000)
        assert comp.closing_speed_weight == initial

    def test_weight_decays_after_hold(self):
        """Component weight must be strictly between 0 and initial halfway through decay."""
        trainer = self._make_trainer_with_schedule(
            shaping_schedules=(("closing_speed", 0, 1_000_000),),
        )
        comp = next(c for c in trainer.wrapper._all_components if c.name == "closing_speed")
        initial = comp.closing_speed_weight

        _, _ = trainer._shaping_scheduler.step(500_000)
        assert 0.0 < comp.closing_speed_weight < initial

    def test_weight_reaches_zero_after_full_decay(self):
        """Component weight must be 0 after hold + decay steps."""
        trainer = self._make_trainer_with_schedule(
            shaping_schedules=(("closing_speed", 100_000, 500_000),),
        )
        comp = next(c for c in trainer.wrapper._all_components if c.name == "closing_speed")

        _, _ = trainer._shaping_scheduler.step(600_001)
        assert comp.closing_speed_weight == 0.0

    def test_bc_coef_schedule(self):
        """BC coefficient must decay correctly through the scheduler."""
        trainer = self._make_trainer_with_schedule(
            bc_coef=0.1, bc_hold_steps=0, bc_decay_steps=1_000_000,
        )
        _, bc_half = trainer._shaping_scheduler.step(500_000)
        assert abs(bc_half - 0.05) < 1e-6
        _, bc_zero = trainer._shaping_scheduler.step(1_000_001)
        assert bc_zero == 0.0
