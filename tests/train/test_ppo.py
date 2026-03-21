"""End-to-end smoke test for the PPO training loop."""

import torch

from boost_and_broadside.config import (
    ShipConfig, EnvConfig, ModelConfig, RewardConfig, TrainConfig,
)
from boost_and_broadside.train.rl.ppo import PPOTrainer


def _make_trainer(n_fourier_freqs: int = 4) -> PPOTrainer:
    return PPOTrainer(
        train_config=TrainConfig(
            num_envs=4, num_steps=16, num_epochs=1, num_minibatches=2,
            learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_coef=0.2,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, total_timesteps=64,
        ),
        model_config=ModelConfig(d_model=32, n_heads=4, n_fourier_freqs=n_fourier_freqs),
        ship_config=ShipConfig(),
        env_config=EnvConfig(num_ships=4, max_bullets=8, max_episode_steps=50),
        reward_config=RewardConfig(
            damage_weight=0.01, kill_weight=0.5, death_weight=0.5,
            victory_weight=1.0, positioning_weight=0.05, positioning_radius=400.0,
        ),
        device="cpu",
        use_wandb=False,
    )


class TestPPOSmokeTest:
    def test_full_training_loop_runs(self):
        """One complete PPO training run (64 total timesteps) must not raise."""
        trainer = _make_trainer()
        trainer.train()  # completes without exception

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
