"""End-to-end smoke test for the PPO training loop."""

import pytest
import torch

from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config import (
    EnvConfig,
    LeagueEvalConfig,
    ModelConfig,
    RewardConfig,
    ScaleConfig,
    ShipConfig,
    TrainConfig,
    TrainingSchedule,
    constant,
    exponential,
    join,
    linear,
    stepped,
)
from boost_and_broadside.env.observation import ObsKey
from boost_and_broadside.train.rl.opponents import OpponentSlot
from boost_and_broadside.train.rl.ppo import _GROUP, PPOTrainer


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
        enemy_neg_lambda_components=frozenset({"enemy_damage", "enemy_death", "enemy_win"}),
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
        sigreg_coef=constant(0.0),
        true_reward_scale=constant(1.0),
        global_scale=constant(1.0),
        local_scale=constant(1.0),
        opponent_fraction=constant(0.0),
        checkpoint_interval=stepped((0, 0)),
        num_epochs=constant(1),
        target_kl=constant(None),
        high_elo_threshold=constant(900.0),
        high_elo_target_kl=constant(0.02),
    )
    defaults.update(overrides)
    return TrainingSchedule(**defaults)


def _make_train_config(
    paradigm: str = "ego_pass",
    opponent_fraction: float = 0.0,
    checkpoint_dir: str = "checkpoints",
    **reward_overrides,
) -> TrainConfig:
    return TrainConfig(
        paradigm=paradigm,
        scales=(
            ScaleConfig(
                env_config=EnvConfig(num_ships=4, max_bullets=8, max_episode_steps=50),
                num_envs=4,
            ),
        ),
        schedule=_make_schedule(opponent_fraction=constant(opponent_fraction)),
        rewards=_make_rewards(**reward_overrides),
        num_steps=16,
        num_minibatches=2,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        max_grad_norm=0.5,
        total_timesteps=64,
        return_ema_alpha=0.005,
        return_min_span=1.0,
        checkpoint_dir=checkpoint_dir,
        league_size=20,
        league_k=4,
        league_admission_interval=999,
        opponent_hold_rollouts=1,
        pfsp_mode="hard",
        pfsp_exponent=2.0,
        live_rating_decay=0.9,
        avg_rating_decay=0.995,
        bt_prior_draws=1.0,
        admission_prior_games=10.0,
        league_eval=LeagueEvalConfig(4, 4, 2, 1, 1),
        bc_elo_target=950.0,
        bc_elo_scale=200.0,
        histogram_interval=10,
    )


def _make_trainer(
    paradigm: str = "ego_pass",
    opponent_fraction: float = 0.0,
    checkpoint_dir: str = "checkpoints",
    **reward_overrides,
) -> PPOTrainer:
    ship_config = ShipConfig()
    scripted_agent = (
        StochasticScriptedAgent(ship_config, StochasticAgentConfig())
        if opponent_fraction > 0.0
        else None
    )
    return PPOTrainer(
        train_config=_make_train_config(
            paradigm=paradigm,
            opponent_fraction=opponent_fraction,
            checkpoint_dir=checkpoint_dir,
            **reward_overrides,
        ),
        model_config=ModelConfig(
            d_model=32,
            n_heads=4,
            n_transformer_blocks=1,
        ),
        ship_config=ship_config,
        device="cpu",
        use_wandb=False,
        scripted_agent=scripted_agent,
    )


class TestPPOSmokeTest:
    @pytest.mark.parametrize("paradigm", ["ego_pass", "shared_pass"])
    def test_full_training_loop_runs(self, paradigm, tmp_path):
        """One complete PPO training run (64 total timesteps) must not raise."""
        trainer = _make_trainer(paradigm=paradigm, checkpoint_dir=str(tmp_path))
        trainer.train()

    # test_encoder_works_with_non_default_n_fourier_freqs is removed because
    # n_fourier_freqs is no longer in ModelConfig.

    @pytest.mark.parametrize("paradigm", ["ego_pass", "shared_pass"])
    def test_policy_parameters_change_after_update(self, paradigm, tmp_path):
        """At least one policy parameter must change after one PPO update."""
        trainer = _make_trainer(paradigm=paradigm, checkpoint_dir=str(tmp_path))
        params_before = [p.clone() for p in trainer.policy.parameters()]

        trainer.train()

        params_after = list(trainer.policy.parameters())
        any_changed = any(not torch.equal(b, a) for b, a in zip(params_before, params_after))
        assert any_changed, "No parameters changed after training"


class TestParadigm:
    """Paradigm-specific rollout behavior, verified via the stored actor masks."""

    def test_invalid_paradigm_raises(self):
        with pytest.raises(ValueError, match="paradigm"):
            _make_train_config(paradigm="both_sides")

    def test_ego_pass_actor_mask_covers_only_team0(self, tmp_path):
        """ego_pass: exactly the team 0 ships contribute to the actor loss."""
        trainer = _make_trainer(paradigm="ego_pass", checkpoint_dir=str(tmp_path))
        trainer.train()

        T, N = trainer.cfg.num_steps, trainer.wrapper.num_ships
        team_id = trainer.buffer.obs[ObsKey.TEAM_ID][:T, :, :N].long()  # (T, B, N)
        assert torch.equal(trainer.buffer.actor_masks, team_id == 0)

    def test_shared_pass_actor_mask_covers_both_teams_in_self_play(self, tmp_path):
        """shared_pass self-play: every ship contributes to the actor loss."""
        trainer = _make_trainer(paradigm="shared_pass", checkpoint_dir=str(tmp_path))
        trainer.train()

        assert trainer.buffer.actor_masks.all()

    def test_shared_pass_opponent_envs_exclude_one_full_team(self, tmp_path):
        """shared_pass + scripted opponent: the masked-out ships in each opponent
        env form exactly one complete team (whichever the random flag assigned)."""
        trainer = _make_trainer(
            paradigm="shared_pass", opponent_fraction=0.5, checkpoint_dir=str(tmp_path)
        )
        trainer.train()

        T, N = trainer.cfg.num_steps, trainer.wrapper.num_ships
        opponent_slice = slice(trainer.B_self, trainer.B_self + trainer.B_opp)
        team_id = trainer.buffer.obs[ObsKey.TEAM_ID][:T, opponent_slice, :N].long()
        excluded = ~trainer.buffer.actor_masks[:, opponent_slice]

        excluded_is_team0 = (excluded == (team_id == 0)).all(dim=-1)
        excluded_is_team1 = (excluded == (team_id == 1)).all(dim=-1)
        assert (excluded_is_team0 | excluded_is_team1).all()

    def test_ego_pass_scripted_opponent_always_controls_team1(self, tmp_path):
        """ego_pass + scripted opponent: only team 0 ships train in opponent envs."""
        trainer = _make_trainer(
            paradigm="ego_pass", opponent_fraction=0.5, checkpoint_dir=str(tmp_path)
        )
        trainer.train()

        T, N = trainer.cfg.num_steps, trainer.wrapper.num_ships
        team_id = trainer.buffer.obs[ObsKey.TEAM_ID][:T, :, :N].long()  # (T, B, N)
        assert torch.equal(trainer.buffer.actor_masks, team_id == 0)

    def test_training_outcomes_respect_attribution_watermark(self, tmp_path):
        trainer = _make_trainer(
            paradigm="ego_pass", opponent_fraction=0.5, checkpoint_dir=str(tmp_path)
        )
        random_entry = trainer.roster.entry("random")
        slot = OpponentSlot(random_entry, 2, 4, None, None)
        trainer._opp_count_index.fill_(trainer.roster.counts.index("random"))
        trainer._opp_active_mask.fill_(True)
        info = {
            "team0_won": torch.tensor([False, False, True, False]),
            "team1_won": torch.tensor([False, False, False, True]),
        }
        done = torch.tensor([False, False, True, True])
        live_index = trainer.roster.counts.index("live")
        random_index = trainer.roster.counts.index("random")

        # The first completion after a reassignment only validates attribution —
        # the episode may have been played against a previous opponent.
        trainer._record_training_outcomes(done, [slot], info)
        first_counts = trainer.roster.counts.tensor[live_index, random_index].sum().item()

        # Episodes that began under the validated assignment are recorded.
        trainer._record_training_outcomes(done, [slot], info)
        trainer.shutdown()

        assert first_counts == 0.0
        assert trainer.roster.counts.tensor[live_index, random_index].tolist() == [1.0, 1.0, 0.0]

    def test_opponent_partition_never_creates_empty_slots(self, tmp_path):
        trainer = _make_trainer(
            paradigm="ego_pass", opponent_fraction=0.5, checkpoint_dir=str(tmp_path)
        )
        trainer._avg_update_count = 1
        trainer.roster.add_avg(global_step=1, update=1)

        slots = trainer._prepare_opponents(trainer.wrapper.num_ships)
        trainer.shutdown()

        assert all(slot.end > slot.start for slot in slots)


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

    def test_exponential_endpoints(self):
        fn = exponential((0, 1e-4), (100, 1e-2))
        assert abs(fn(0) - 1e-4) < 1e-12
        assert abs(fn(100) - 1e-2) < 1e-10

    def test_exponential_midpoint_is_geometric_mean(self):
        fn = exponential((0, 1e-4), (100, 1e-2))
        assert abs(fn(50) - 1e-3) < 1e-10

    def test_exponential_clamps_outside_keypoints(self):
        fn = exponential((100, 1.0), (200, 2.0))
        assert fn(0) == 1.0
        assert fn(9_999) == 2.0

    def test_exponential_rejects_nonpositive_values(self):
        with pytest.raises(ValueError, match="values > 0"):
            exponential((0, 0.0), (100, 1.0))

    def test_join_switches_at_activation_step(self):
        fn = join((0, constant(1.0)), (100, constant(2.0)))
        assert fn(99) == 1.0
        assert fn(100) == 2.0

    def test_join_passes_global_step_to_active_segment(self):
        fn = join((0, constant(0.0)), (100, linear((100, 0.0), (200, 1.0))))
        assert abs(fn(150) - 0.5) < 1e-10

    def test_join_rejects_non_ascending_segments(self):
        with pytest.raises(ValueError, match="ascending"):
            join((100, constant(1.0)), (0, constant(2.0)))

    def test_group_scales_applied_by_trainer(self, tmp_path):
        """After training, effective weight = group_scale * individual weight for EVERY
        component (regression: setattr on a per-class attribute name silently missed the
        18 components whose weight lived in a `_weight`-backed property)."""
        group_scales = {"true_reward_scale": 0.25, "global_scale": 2.0, "local_scale": 0.5}
        trainer = PPOTrainer(
            train_config=TrainConfig(
                paradigm="ego_pass",
                scales=(
                    ScaleConfig(
                        env_config=EnvConfig(num_ships=4, max_bullets=8, max_episode_steps=50),
                        num_envs=4,
                    ),
                ),
                schedule=_make_schedule(
                    true_reward_scale=constant(group_scales["true_reward_scale"]),
                    global_scale=constant(group_scales["global_scale"]),
                    local_scale=constant(group_scales["local_scale"]),
                ),
                rewards=_make_rewards(),
                num_steps=16,
                num_minibatches=2,
                gamma=0.99,
                gae_lambda=0.95,
                clip_coef=0.2,
                max_grad_norm=0.5,
                total_timesteps=64,
                return_ema_alpha=0.005,
                return_min_span=1.0,
                checkpoint_dir=str(tmp_path),
                league_size=20,
                league_k=4,
                league_admission_interval=999,
                opponent_hold_rollouts=1,
                pfsp_mode="hard",
                pfsp_exponent=2.0,
                live_rating_decay=0.9,
                avg_rating_decay=0.995,
                bt_prior_draws=1.0,
                admission_prior_games=10.0,
                league_eval=LeagueEvalConfig(4, 4, 2, 1, 1),
                bc_elo_target=950.0,
                bc_elo_scale=200.0,
                histogram_interval=10,
            ),
            model_config=ModelConfig(d_model=32, n_heads=4, n_transformer_blocks=1),
            ship_config=ShipConfig(),
            device="cpu",
            use_wandb=False,
        )
        trainer.train()
        mismatched = {}
        for comp in trainer.wrapper.reward_components:
            individual_weight = getattr(trainer.cfg.rewards, f"{comp.name}_weight")
            expected = individual_weight * group_scales[_GROUP[comp.name]]
            if abs(comp.weight - expected) > 1e-9:
                mismatched[comp.name] = (comp.weight, expected)
        assert not mismatched, f"components with wrong effective weight: {mismatched}"
        # The wrapper's cached (K,) weight tensor must reflect the same scaled weights.
        expected_t = torch.tensor(
            [c.weight for c in trainer.wrapper.active_components], dtype=torch.float32
        )
        assert torch.equal(trainer.wrapper.component_weights.cpu(), expected_t)


class TestWinComponentLambdaMatrix:
    """Regression for the win-component lambda design (audit §1.2): ally_win/enemy_win
    must use the team-based zero-sum lambda path, not the diagonal (self-only) path."""

    def test_win_component_lambda_rows_are_zero_sum(self, tmp_path):
        """In a 2v2 layout, ship 0 aggregates ally_win from its own team (+1) and
        enemy_win from the enemy team (-1), so win/draw/loss are distinguishable."""
        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        names = trainer._active_names
        k_ally, k_enemy = names.index("ally_win"), names.index("enemy_win")

        # Build the per-pair lambda matrix the same way _precompute_lambda_aggregates does.
        teams = torch.tensor([0, 0, 1, 1])
        same_team = teams.unsqueeze(1) == teams.unsqueeze(0)  # (N, N)
        ally_lam = torch.where(trainer.ally_zero_k, 0.0, 1.0)  # (K,)
        enemy_lam = torch.where(trainer.enemy_neg_k, -1.0, 0.0)  # (K,)
        global_lambda = (
            same_team.float().unsqueeze(-1) * ally_lam
            + (~same_team).float().unsqueeze(-1) * enemy_lam
        )  # (N, N, K)
        lam = torch.where(trainer.local_k, torch.eye(4).unsqueeze(-1), global_lambda)

        assert lam[0, :, k_ally].tolist() == [1.0, 1.0, 0.0, 0.0]
        assert lam[0, :, k_enemy].tolist() == [0.0, 0.0, -1.0, -1.0]

    def test_production_config_win_lambda_sets(self):
        """runs/shared.py must agree with rl_obstacles.py and the test configs on
        which win components are zero-sum (enemy_win) vs ally-shared (ally_win)."""
        from runs.shared import REWARDS

        assert "enemy_win" in REWARDS.enemy_neg_lambda_components
        assert "enemy_win" in REWARDS.ally_zero_components
        assert "ally_win" not in REWARDS.enemy_neg_lambda_components
        assert "ally_win" not in REWARDS.ally_zero_components


class TestRLSmokeTest:
    """Full RL smoke test using the real runs/shared.py config.

    Exercises the complete training stack with the production reward config
    (including kill_shot and kill_assist) for a small number of updates.
    Uses a scripted opponent to ensure combat happens and kill rewards fire.
    """

    @pytest.mark.parametrize("paradigm", ["ego_pass", "shared_pass"])
    def test_rl_run_with_production_config(self, paradigm, tmp_path):
        from runs.shared import MODEL_CONFIG, REWARDS, SHIP_CONFIG

        schedule = TrainingSchedule(
            learning_rate=constant(3e-4),
            policy_gradient_coef=constant(1.0),
            entropy_coef=constant(0.01),
            behavior_cloning_coef=constant(0.0),
            value_function_coef=constant(1.0),
            sigreg_coef=constant(0.0),
            true_reward_scale=constant(1.0),
            global_scale=constant(1.0),
            local_scale=constant(1.0),
            opponent_fraction=constant(0.5),
            checkpoint_interval=constant(9999),
            num_epochs=constant(1),
            target_kl=constant(None),
            high_elo_threshold=constant(900.0),
            high_elo_target_kl=constant(0.02),
        )
        cfg = TrainConfig(
            paradigm=paradigm,
            scales=(
                ScaleConfig(
                    env_config=EnvConfig(num_ships=4, max_bullets=20, max_episode_steps=64),
                    num_envs=16,
                ),
            ),
            schedule=schedule,
            rewards=REWARDS,
            num_steps=32,
            num_minibatches=2,
            gamma=0.99,
            gae_lambda=0.95,
            clip_coef=0.2,
            max_grad_norm=1.0,
            total_timesteps=16 * 32 * 3,  # 3 updates
            return_ema_alpha=0.005,
            return_min_span=1.0,
            checkpoint_dir=str(tmp_path),
            league_size=5,
            league_k=4,
            league_admission_interval=999,
            opponent_hold_rollouts=1,
            pfsp_mode="hard",
            pfsp_exponent=2.0,
            live_rating_decay=0.9,
            avg_rating_decay=0.995,
            bt_prior_draws=1.0,
            admission_prior_games=10.0,
            league_eval=LeagueEvalConfig(4, 4, 2, 1, 1),
            bc_elo_target=950.0,
            bc_elo_scale=200.0,
            histogram_interval=10,
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
