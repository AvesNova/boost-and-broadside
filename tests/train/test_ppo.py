"""End-to-end smoke test for the PPO training loop."""

import torch

from boost_and_broadside.config import (
    ShipConfig, EnvConfig, ModelConfig, RewardConfig, TrainConfig,
)
from boost_and_broadside.train.rl.ppo import PPOTrainer, _team_sort_idx, _gather_ships


def _make_trainer(n_fourier_freqs: int = 4) -> PPOTrainer:
    return PPOTrainer(
        train_config=TrainConfig(
            num_envs=4, num_steps=16, num_epochs=1, num_minibatches=2,
            learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_coef=0.2,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, total_timesteps=64,
            avg_policy_warmup_steps=0, avg_policy_opp_fraction=0.5,
        ),
        model_config=ModelConfig(d_model=32, n_heads=4, n_fourier_freqs=n_fourier_freqs),
        ship_config=ShipConfig(),
        env_config=EnvConfig(num_ships=4, max_bullets=8, max_episode_steps=50),
        reward_config=RewardConfig(
            damage_weight=0.01, kill_weight=0.5, death_weight=0.5,
            victory_weight=1.0, positioning_weight=0.05, positioning_radius=400.0,
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


class TestDecayScheduler:
    def test_weight_decays_after_threshold_sustained(self):
        """Decay scheduler must reduce a component weight once the metric EMA stays above threshold."""
        trainer = _make_trainer()
        approach_comp = next(c for c in trainer.wrapper._components if c.name == "closing_speed")
        initial_weight = approach_comp.closing_speed_weight

        # Feed kill metric above the closing_speed threshold (3.0) for more than sustain (30) updates
        for _ in range(35):
            trainer._decay.step({"ep/reward_kill": 5.0})

        assert approach_comp.closing_speed_weight < initial_weight

    def test_weight_does_not_decay_below_threshold(self):
        """Decay must not trigger when metric stays below threshold."""
        trainer = _make_trainer()
        approach_comp = next(c for c in trainer.wrapper._components if c.name == "closing_speed")
        initial_weight = approach_comp.closing_speed_weight

        for _ in range(50):
            trainer._decay.step({"ep/reward_kill": 1.0})  # below threshold of 3.0

        assert approach_comp.closing_speed_weight == initial_weight

    def test_count_does_not_accumulate_with_brief_pulses(self):
        """Short bursts above threshold separated by cool-down gaps should never trigger decay.

        5 on / 35 off / 5 on: EMA memory keeps count rising for ~10 extra steps
        during the off-phase, then count decrements back to zero before the
        second on-pulse — final count stays well below sustain=30.
        """
        trainer = _make_trainer()
        approach_comp = next(c for c in trainer.wrapper._components if c.name == "closing_speed")
        initial_weight = approach_comp.closing_speed_weight

        for _ in range(5):
            trainer._decay.step({"ep/reward_kill": 5.0})   # brief pulse
        for _ in range(35):
            trainer._decay.step({"ep/reward_kill": 0.0})   # long cool-down
        for _ in range(5):
            trainer._decay.step({"ep/reward_kill": 5.0})   # second brief pulse

        assert approach_comp.closing_speed_weight == initial_weight


class TestTeamSplit:
    def test_team_sort_idx_puts_team0_first(self):
        """sort_idx must place team-0 ship indices before team-1 indices."""
        team_id = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.int32)
        sort_idx, unsort_idx = _team_sort_idx(team_id)

        # After sorting, the first N_t=2 positions are team 0
        for b in range(2):
            t0_slots = sort_idx[b, :2]
            t1_slots = sort_idx[b, 2:]
            assert (team_id[b][t0_slots] == 0).all()
            assert (team_id[b][t1_slots] == 1).all()

    def test_unsort_idx_is_inverse_of_sort_idx(self):
        """Applying sort then unsort must recover the original order."""
        team_id = torch.tensor([[1, 0, 1, 0]], dtype=torch.int32)
        sort_idx, unsort_idx = _team_sort_idx(team_id)
        original = torch.arange(4).unsqueeze(0)
        sorted_  = _gather_ships(original, sort_idx, dim=1)
        restored = _gather_ships(sorted_,  unsort_idx, dim=1)
        assert torch.equal(restored, original)

    def test_two_team_forward_produces_full_action_tensor(self):
        """_two_team_forward must return (B, N, 3) actions covering all ship slots."""
        trainer = _make_trainer()
        obs = trainer.wrapper.reset()
        B   = trainer.cfg.num_envs
        N   = trainer.wrapper.num_ships
        N_t = N // 2
        D   = trainer.model_config.d_model
        B_avg = trainer._envs_vs_avg

        hidden     = trainer.policy.initial_hidden(B, N, torch.device("cpu"))
        avg_hidden = trainer.avg_policy.initial_hidden(B_avg, N_t, torch.device("cpu"))
        action, logprob, value, new_hidden, new_avg_hidden = trainer._two_team_forward(
            obs, hidden, avg_hidden
        )

        assert action.shape         == (B, N, 3)
        assert logprob.shape        == (B, N)
        assert value.shape          == (B, N)
        assert new_hidden.shape     == (1, B * N, D)
        assert new_avg_hidden.shape == (1, B_avg * N_t, D)
