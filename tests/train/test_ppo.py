"""End-to-end smoke test for the PPO training loop."""

import torch

from boost_and_broadside.config import (
    ShipConfig,
    EnvConfig,
    ModelConfig,
    PhaseConfig,
    TimelineConfig,
    TrainConfig,
    ScaleConfig,
)
from boost_and_broadside.train.rl.ppo import PPOTrainer, _TimelineScheduler


def _make_base_phase(**overrides) -> PhaseConfig:
    defaults = dict(
        step=0,
        learning_rate=3e-4,
        pg_coef=1.0,
        ent_coef=0.01,
        bc_coef=0.0,
        vf_coef=0.5,
        scripted_frac=0.0,
        avg_model_frac=0.0,
        league_frac=0.0,
        allow_avg_model_updates=False,
        allow_scripted_in_roster=False,
        elo_eval_games=16,
        elo_eval_interval=0,
        checkpoint_interval=0,
        true_reward_scale=1.0,
        important_scale=1.0,
        aux_scale=1.0,
        victory_weight=1.0,
        death_weight=0.5,
        damage_weight=0.01,
        facing_weight=0.01,
        exposure_weight=0.01,
        turn_rate_weight=0.01,
        closing_speed_weight=0.01,
        proximity_weight=0.01,
        positioning_weight=0.05,
        power_range_weight=0.01,
        speed_range_weight=0.01,
        shoot_quality_weight=0.01,
        positioning_radius=400.0,
        proximity_radius=300.0,
        power_range_lo=0.2,
        power_range_hi=0.8,
        speed_range_lo=40.0,
        speed_range_hi=120.0,
        shoot_quality_radius=200.0,
        enemy_neg_lambda_components=frozenset({"damage", "death", "victory", "exposure"}),
        disabled_rewards=frozenset(),
    )
    defaults.update(overrides)
    return PhaseConfig(**defaults)


def _make_trainer(n_fourier_freqs: int = 4, **phase_overrides) -> PPOTrainer:
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
            timeline=TimelineConfig(phases=(_make_base_phase(**phase_overrides),)),
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
            avg_model_min_steps=0,
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


class TestTimelineScheduler:
    def _make_two_phase_timeline(self, **phase1_overrides) -> TimelineConfig:
        base = _make_base_phase()
        phase1 = PhaseConfig(step=1_000_000, **phase1_overrides)
        return TimelineConfig(phases=(base, phase1))

    def test_returns_base_phase_before_first_keyframe(self):
        """Step 0 must return base phase values exactly."""
        tl = self._make_two_phase_timeline(learning_rate=1e-4)
        sched = _TimelineScheduler(tl)
        state = sched.step(0)
        assert state.learning_rate == 3e-4

    def test_float_field_interpolates_between_phases(self):
        """learning_rate must be midpoint at the halfway step."""
        tl = self._make_two_phase_timeline(learning_rate=1e-4)
        sched = _TimelineScheduler(tl)
        state = sched.step(500_000)
        expected = 3e-4 + (1e-4 - 3e-4) * 0.5
        assert abs(state.learning_rate - expected) < 1e-10

    def test_float_field_reaches_target_at_keyframe(self):
        """learning_rate must equal phase1 value exactly at its step."""
        tl = self._make_two_phase_timeline(learning_rate=1e-4)
        sched = _TimelineScheduler(tl)
        state = sched.step(1_000_000)
        assert state.learning_rate == 1e-4

    def test_clamped_at_last_phase_beyond_final_step(self):
        """Steps past the last keyframe must return the last phase's values."""
        tl = self._make_two_phase_timeline(learning_rate=1e-4)
        sched = _TimelineScheduler(tl)
        state = sched.step(99_000_000)
        assert state.learning_rate == 1e-4

    def test_pg_coef_interpolates(self):
        """pg_coef must interpolate linearly between phases."""
        base = _make_base_phase(pg_coef=0.0)
        rl = PhaseConfig(step=1_000_000, pg_coef=1.0)
        tl = TimelineConfig(phases=(base, rl))
        sched = _TimelineScheduler(tl)
        assert abs(sched.step(500_000).pg_coef - 0.5) < 1e-10
        assert sched.step(1_000_000).pg_coef == 1.0

    def test_group_scales_applied_by_trainer(self):
        """After training, effective component weight = group_scale * individual."""
        trainer = _make_trainer(aux_scale=0.5, closing_speed_weight=0.01)
        trainer.train()
        from boost_and_broadside.train.rl.ppo import _GROUP
        state = trainer._phase_state
        for comp in trainer.wrapper._all_components:
            if comp.name == "closing_speed":
                scale_attr = _GROUP.get(comp.name, "aux_scale")
                raw = getattr(state, f"{comp.name}_weight")
                expected = raw * getattr(state, scale_attr)
                actual = getattr(comp, f"{comp.name}_weight")
                assert abs(actual - expected) < 1e-9
