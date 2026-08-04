"""End-to-end smoke test for the PPO training loop."""

import math

import pytest
import torch

from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config import (
    EloEvalConfig,
    EnvConfig,
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
from boost_and_broadside.train.rl.ppo import _GROUP, _LOCAL_COMPONENTS, PPOTrainer


def _make_rewards(**overrides) -> RewardConfig:
    defaults = dict(
        ally_combat_damage_weight=0.01,
        enemy_combat_damage_weight=0.01,
        ally_field_damage_weight=0.01,
        enemy_field_damage_weight=0.01,
        ally_combat_death_weight=0.5,
        enemy_combat_death_weight=0.5,
        ally_field_death_weight=0.5,
        enemy_field_death_weight=0.5,
        ally_win_weight=1.0,
        enemy_win_weight=1.0,
        facing_weight=0.01,
        closing_speed_weight=0.01,
        shoot_quality_weight=0.01,
        kill_shot_weight=0.5,
        kill_assist_weight=0.5,
        combat_damage_taken_weight=0.1,
        field_damage_taken_weight=0.1,
        damage_dealt_enemy_weight=0.1,
        damage_dealt_ally_weight=0.1,
        combat_death_weight=0.5,
        field_death_weight=0.5,
        proximity_radius=300.0,
        shoot_quality_radius=200.0,
        enemy_neg_lambda_components=frozenset(
            {
                "enemy_combat_damage",
                "enemy_field_damage",
                "enemy_combat_death",
                "enemy_field_death",
                "enemy_win",
            }
        ),
        ally_zero_components=frozenset(
            {
                "enemy_combat_damage",
                "enemy_field_damage",
                "enemy_combat_death",
                "enemy_field_death",
                "enemy_win",
            }
        ),
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
        scripted_fraction=constant(0.0),
        avg_model_fraction=constant(0.0),
        league_fraction=constant(0.0),
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
    scripted_fraction: float = 0.0,
    avg_model_fraction: float = 0.0,
    league_fraction: float = 0.0,
    checkpoint_dir: str = "checkpoints",
    min_games_to_freeze: int = 0,
    rollouts_per_update: int = 1,
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
        schedule=_make_schedule(
            scripted_fraction=constant(scripted_fraction),
            avg_model_fraction=constant(avg_model_fraction),
            league_fraction=constant(league_fraction),
        ),
        rewards=_make_rewards(**reward_overrides),
        num_steps=16,
        rollouts_per_update=rollouts_per_update,
        num_minibatches=2,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        max_grad_norm=0.5,
        total_timesteps=64 * rollouts_per_update,
        return_ema_alpha=0.005,
        return_min_span=1e-3,
        advantage_min_rms=1e-4,
        checkpoint_dir=checkpoint_dir,
        league_size=20,
        league_uniform_sampling=False,
        elo_milestone_gap=50.0,
        elo_temperature=200.0,
        elo_eval=EloEvalConfig(
            envs_per_matchup=4,
            step_interval=4,
            k_factor=4.0,
            scripted_elo_init=1000.0,
            window_size=100,
            min_games_to_freeze=min_games_to_freeze,
        ),
        bc_winrate_target=0.9,
        histogram_interval=10,
    )


def _make_trainer(
    paradigm: str = "ego_pass",
    scripted_fraction: float = 0.0,
    avg_model_fraction: float = 0.0,
    league_fraction: float = 0.0,
    checkpoint_dir: str = "checkpoints",
    min_games_to_freeze: int = 0,
    rollouts_per_update: int = 1,
    device: str = "cpu",
    model_config: ModelConfig | None = None,
    **reward_overrides,
) -> PPOTrainer:
    ship_config = ShipConfig()
    scripted_agent = (
        StochasticScriptedAgent(ship_config, StochasticAgentConfig())
        if scripted_fraction > 0.0
        else None
    )
    return PPOTrainer(
        train_config=_make_train_config(
            paradigm=paradigm,
            scripted_fraction=scripted_fraction,
            avg_model_fraction=avg_model_fraction,
            league_fraction=league_fraction,
            checkpoint_dir=checkpoint_dir,
            min_games_to_freeze=min_games_to_freeze,
            rollouts_per_update=rollouts_per_update,
            **reward_overrides,
        ),
        model_config=model_config
        or ModelConfig(
            d_model=32,
            n_heads=4,
            n_yemong_blocks=1,
        ),
        ship_config=ship_config,
        device=device,
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

    def test_host_backed_logical_batch_runs_one_update(self, tmp_path):
        trainer = _make_trainer(
            checkpoint_dir=str(tmp_path),
            rollouts_per_update=2,
        )

        trainer.train()

        assert trainer._global_step == 128

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_host_backed_logical_batch_runs_on_cuda(self, tmp_path):
        trainer = _make_trainer(
            checkpoint_dir=str(tmp_path),
            rollouts_per_update=2,
            device="cuda",
        )

        trainer.train()

        assert trainer._global_step == 128


class TestEloLadder:
    """Milestone flow: snapshot → settle → freeze, driven via _maybe_advance_ladder."""

    def test_first_milestone_creates_floating_checkpoint(self, tmp_path):
        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        runtime = trainer._initialize_rollout_runtime()
        trainer._training_elo = 60.0  # above the 50-point milestone gap
        trainer._maybe_advance_ladder(update=1, elo_eval=runtime.elo_eval)
        assert trainer.roster.floating_checkpoint() is not None
        assert runtime.elo_eval.float_pro_agent is not None

    def test_ladder_snapshot_file_is_written(self, tmp_path):
        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        runtime = trainer._initialize_rollout_runtime()
        trainer._training_elo = 60.0
        trainer._maybe_advance_ladder(update=1, elo_eval=runtime.elo_eval)
        floating = trainer.roster.floating_checkpoint()
        ckpt = torch.load(floating.path, map_location="cpu", weights_only=False)
        assert "policy_state_dict" in ckpt

    def test_second_milestone_freezes_first_at_measured_rating(self, tmp_path):
        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        runtime = trainer._initialize_rollout_runtime()
        trainer._training_elo = 60.0
        trainer._maybe_advance_ladder(update=1, elo_eval=runtime.elo_eval)
        first = trainer.roster.floating_checkpoint()
        # Pretend the floating rating settled at 42 rather than a round number.
        trainer.roster.set_floating_elo(42.0)
        runtime.elo_eval.floating_elo.fill_(42.0)
        trainer._training_elo = 120.0
        trainer._maybe_advance_ladder(update=2, elo_eval=runtime.elo_eval)
        assert first.fixed
        assert first.elo == 42.0
        assert trainer.roster.floating_checkpoint() is not first

    def test_milestone_deferred_until_floating_settles(self, tmp_path):
        trainer = _make_trainer(checkpoint_dir=str(tmp_path), min_games_to_freeze=10_000)
        runtime = trainer._initialize_rollout_runtime()
        trainer._training_elo = 60.0
        trainer._maybe_advance_ladder(update=1, elo_eval=runtime.elo_eval)
        first = trainer.roster.floating_checkpoint()
        trainer._training_elo = 120.0
        trainer._floating_games = 0  # rating has not settled yet
        trainer._maybe_advance_ladder(update=2, elo_eval=runtime.elo_eval)
        assert trainer.roster.floating_checkpoint() is first
        assert not first.fixed

    def test_third_milestone_keeps_surviving_anchor_episodes(self, tmp_path):
        """Dropping the oldest anchor restarts only its envs; the surviving
        anchor's slot-0 episodes play on with their assignment shifted down."""
        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        runtime = trainer._initialize_rollout_runtime()
        elo_eval = runtime.elo_eval
        size = elo_eval.matchup_size
        for update, elo in enumerate([60.0, 120.0], start=1):
            trainer._training_elo = elo
            trainer._maybe_advance_ladder(update=update, elo_eval=elo_eval)

        # Anchors are now [random, first-frozen]; split slot 0 across both.
        elo_eval._anchor_idx_live = torch.tensor(
            [0, 1] * (size // 2), device=elo_eval.device, dtype=torch.long
        )
        survivor = elo_eval._anchor_idx_live == 1
        elo_eval._rated[:size] = True
        elo_eval.env.state.step_count[:size] = 7

        trainer._training_elo = 180.0
        trainer._maybe_advance_ladder(update=3, elo_eval=elo_eval)

        step_count = elo_eval.env.state.step_count[:size]
        # The random anchor was dropped, so the survivor shifts from index 1 to 0.
        assert (elo_eval._anchor_idx_live[survivor] == 0).all()
        assert (step_count[survivor] == 7).all(), "surviving episodes were restarted"
        assert elo_eval._rated[:size][survivor].all(), "surviving episodes lost their rated flag"
        assert not (step_count[~survivor] == 7).all(), "dropped-anchor envs were not reseeded"

    def test_fresh_runtime_reloads_ladder_from_disk(self, tmp_path):
        """A new rollout runtime (resume path) rebuilds anchors and the floating
        checkpoint from the roster's saved ladder snapshot files."""
        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        runtime = trainer._initialize_rollout_runtime()
        for update, elo in enumerate([60.0, 120.0, 180.0], start=1):
            trainer._training_elo = elo
            trainer._maybe_advance_ladder(update=update, elo_eval=runtime.elo_eval)
        trainer.roster.evict_all_checkpoint_policies()  # force reload from disk
        fresh = trainer._initialize_rollout_runtime()
        assert fresh.elo_eval.float_pro_agent is not None
        fresh.elo_eval.step(0, avg_active=False)

    def test_evaluator_steps_with_ladder_anchors(self, tmp_path):
        """After several milestones the eval battery holds policy anchors plus a
        floating checkpoint; full eval steps and a flush must run cleanly."""
        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        runtime = trainer._initialize_rollout_runtime()
        for update, elo in enumerate([60.0, 120.0, 180.0], start=1):
            trainer._training_elo = elo
            trainer._maybe_advance_ladder(update=update, elo_eval=runtime.elo_eval)
        for _ in range(8):
            runtime.elo_eval.step(0, avg_active=False)
        snapshot = runtime.elo_eval.flush(avg_active=False)
        assert snapshot.floating_elo is not None

    def test_milestones_land_on_an_absolute_grid(self, tmp_path):
        """A snapshot that fires late claims its grid point, not its own rating,
        so the next one is still due at the following multiple of the gap."""
        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        runtime = trainer._initialize_rollout_runtime()
        # Fires well past the 50 grid point; the grid must not ratchet to 90+50.
        trainer._training_elo = 90.0
        trainer._maybe_advance_ladder(update=1, elo_eval=runtime.elo_eval)
        assert trainer._elo_milestone == 50.0
        first = trainer.roster.floating_checkpoint()

        trainer._training_elo = 105.0  # past the 100 grid point
        trainer._maybe_advance_ladder(update=2, elo_eval=runtime.elo_eval)
        assert trainer.roster.floating_checkpoint() is not first
        assert trainer._elo_milestone == 100.0

    def test_milestone_grid_point_is_claimed_once(self, tmp_path):
        """Rating is not monotonic: dipping below a claimed grid point and
        recrossing it must not take a second snapshot at the same rung."""
        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        runtime = trainer._initialize_rollout_runtime()
        trainer._training_elo = 60.0
        trainer._maybe_advance_ladder(update=1, elo_eval=runtime.elo_eval)
        floating = trainer.roster.floating_checkpoint()

        trainer._training_elo = 40.0  # dipped back below the 50 rung
        trainer._maybe_advance_ladder(update=2, elo_eval=runtime.elo_eval)
        trainer._training_elo = 70.0  # and recrossed it
        trainer._maybe_advance_ladder(update=3, elo_eval=runtime.elo_eval)
        assert trainer.roster.floating_checkpoint() is floating

    def test_jump_across_several_grid_points_snapshots_once(self, tmp_path):
        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        runtime = trainer._initialize_rollout_runtime()
        trainer._training_elo = 260.0  # crosses 50, 100, 150, 200, 250 at once
        trainer._maybe_advance_ladder(update=1, elo_eval=runtime.elo_eval)
        checkpoints = [e for e in trainer.roster.entries if e.kind == "checkpoint"]
        assert len(checkpoints) == 1
        assert trainer._elo_milestone == 250.0


class TestMatchCounts:
    """Per-opponent win/loss/tie capture for the non-stationary players."""

    def test_counts_are_keyed_by_roster_label(self, tmp_path):
        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        runtime = trainer._initialize_rollout_runtime()
        trainer._training_elo = 60.0
        trainer._maybe_advance_ladder(update=1, elo_eval=runtime.elo_eval)
        floating_label = trainer.roster.floating_checkpoint().label

        elo_eval = runtime.elo_eval
        elo_eval._rated[:] = True
        for _ in range(120):  # long enough for 50-step episodes to resolve
            elo_eval.step(0, avg_active=False)
        counts = elo_eval.flush(avg_active=False).match_counts

        assert "random" in counts, "games against the random anchor were not recorded"
        assert floating_label in counts, "games against the floating checkpoint were lost"
        assert all(len(wlt) == 3 for wlt in counts.values())
        assert any(sum(wlt) > 0 for wlt in counts.values())

    def test_counts_reset_between_flushes(self, tmp_path):
        """Each update's record must describe only that update's games — the live
        policy is a different player at every one."""
        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        elo_eval = trainer._initialize_rollout_runtime().elo_eval
        elo_eval._rated[:] = True
        for _ in range(120):
            elo_eval.step(0, avg_active=False)
        assert sum(sum(wlt) for wlt in elo_eval.flush(avg_active=False).match_counts.values()) > 0
        second = elo_eval.flush(avg_active=False).match_counts
        assert sum(sum(wlt) for wlt in second.values()) == 0

    def test_ties_are_recorded_separately_from_wins(self, tmp_path):
        """Tie counts must survive as their own column. The Elo update collapses
        them into a half-win, but draw frequency here is level-dependent, so the
        post-hoc fit needs the raw three-way split to model it at all."""
        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        elo_eval = trainer._initialize_rollout_runtime().elo_eval
        size = elo_eval.matchup_size
        won = torch.zeros(5 * size, dtype=torch.bool, device=elo_eval.device)
        tied = torch.zeros_like(won)
        tied[:size] = True  # every slot-0 episode timed out as a draw
        elo_eval._accumulate_match_counts(won, won, tied, torch.ones_like(won), avg_active=False)
        counts = elo_eval.flush(avg_active=False).match_counts
        assert counts["random"] == (0, 0, size)


class TestAvgModelTrigger:
    """Avg-model accumulation starts exactly where the BC aux loss ends."""

    def _drive(self, trainer, win_rate: float, updates: int) -> None:
        window = trainer._eval_window_sc
        window.clear()
        window.extend([win_rate] * window.maxlen)
        for _ in range(updates):
            trainer._refresh_training_schedule({}, trainer._elo_eval_stub)

    def test_starts_once_scripted_target_holds(self, tmp_path):
        trainer = _make_trainer(checkpoint_dir=str(tmp_path), avg_model_fraction=0.25)
        trainer._elo_eval_stub = trainer._initialize_rollout_runtime().elo_eval
        # bc_winrate_target is 0.9 in the test config.
        self._drive(trainer, win_rate=0.95, updates=3)
        assert trainer._avg_update_count > 0

    def test_below_target_never_starts(self, tmp_path):
        trainer = _make_trainer(checkpoint_dir=str(tmp_path), avg_model_fraction=0.25)
        trainer._elo_eval_stub = trainer._initialize_rollout_runtime().elo_eval
        self._drive(trainer, win_rate=0.5, updates=20)
        assert trainer._avg_update_count == 0

    def test_single_lucky_update_does_not_latch(self, tmp_path):
        """The gate is permanent, so one window above target must not trip it."""
        trainer = _make_trainer(checkpoint_dir=str(tmp_path), avg_model_fraction=0.25)
        trainer._elo_eval_stub = trainer._initialize_rollout_runtime().elo_eval
        self._drive(trainer, win_rate=0.95, updates=1)
        assert trainer._avg_update_count == 0
        self._drive(trainer, win_rate=0.5, updates=1)  # streak broken
        assert trainer._bc_cutoff_streak == 0

    def test_partial_window_does_not_count(self, tmp_path):
        """A window with only a handful of games is too noisy to latch on."""
        trainer = _make_trainer(checkpoint_dir=str(tmp_path), avg_model_fraction=0.25)
        trainer._elo_eval_stub = trainer._initialize_rollout_runtime().elo_eval
        trainer._eval_window_sc.clear()
        trainer._eval_window_sc.extend([1.0] * 5)
        for _ in range(5):
            trainer._refresh_training_schedule({}, trainer._elo_eval_stub)
        assert trainer._bc_cutoff_streak == 0
        assert trainer._avg_update_count == 0


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
            paradigm="shared_pass", scripted_fraction=0.5, checkpoint_dir=str(tmp_path)
        )
        trainer.train()

        T, N = trainer.cfg.num_steps, trainer.wrapper.num_ships
        sc = slice(trainer.B_self, trainer.B_self + trainer.B_sc)
        team_id = trainer.buffer.obs[ObsKey.TEAM_ID][:T, sc, :N].long()  # (T, B_sc, N)
        excluded = ~trainer.buffer.actor_masks[:, sc]  # (T, B_sc, N)

        excluded_is_team0 = (excluded == (team_id == 0)).all(dim=-1)  # (T, B_sc)
        excluded_is_team1 = (excluded == (team_id == 1)).all(dim=-1)  # (T, B_sc)
        assert (excluded_is_team0 | excluded_is_team1).all()

    def test_ego_pass_scripted_opponent_always_controls_team1(self, tmp_path):
        """ego_pass + scripted opponent: only team 0 ships train in opponent envs."""
        trainer = _make_trainer(
            paradigm="ego_pass", scripted_fraction=0.5, checkpoint_dir=str(tmp_path)
        )
        trainer.train()

        T, N = trainer.cfg.num_steps, trainer.wrapper.num_ships
        team_id = trainer.buffer.obs[ObsKey.TEAM_ID][:T, :, :N].long()  # (T, B, N)
        assert torch.equal(trainer.buffer.actor_masks, team_id == 0)


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
                rollouts_per_update=1,
                num_minibatches=2,
                gamma=0.99,
                gae_lambda=0.95,
                clip_coef=0.2,
                max_grad_norm=0.5,
                total_timesteps=64,
                return_ema_alpha=0.005,
                return_min_span=1e-3,
                advantage_min_rms=1e-4,
                checkpoint_dir=str(tmp_path),
                league_size=20,
                league_uniform_sampling=False,
                elo_milestone_gap=50.0,
                elo_temperature=200.0,
                elo_eval=EloEvalConfig(4, 4, 4.0, 1000.0, 100),
                bc_winrate_target=0.9,
                histogram_interval=10,
            ),
            model_config=ModelConfig(d_model=32, n_heads=4, n_yemong_blocks=1),
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
        """runs/shared.py must agree with the field profile and test configs on
        which win components are zero-sum (enemy_win) vs ally-shared (ally_win)."""
        from runs.shared import REWARDS

        assert "enemy_win" in REWARDS.enemy_neg_lambda_components
        assert "enemy_win" in REWARDS.ally_zero_components
        assert "ally_win" not in REWARDS.enemy_neg_lambda_components
        assert "ally_win" not in REWARDS.ally_zero_components


class TestLocalComponentRegistry:
    """Regression for AUDIT-014: the self-only (diagonal-lambda) set and the
    group-scale classification must not drift apart."""

    def test_local_components_match_local_scale_group(self):
        """A component uses diagonal (self-only) lambda iff it is in the
        `local_scale` group. If a new reward is classified in one registry but
        not the other, its lambda aggregation would be silently wrong."""
        local_scale_group = {name for name, group in _GROUP.items() if group == "local_scale"}
        assert _LOCAL_COMPONENTS == local_scale_group

    def test_every_reward_component_is_classified(self):
        """Every registered reward component must appear in _GROUP, or
        `_refresh_training_schedule` would KeyError on the first update."""
        from boost_and_broadside.env.rewards import REWARD_COMPONENT_NAMES

        assert set(REWARD_COMPONENT_NAMES) == set(_GROUP)


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
            scripted_fraction=constant(0.5),
            avg_model_fraction=constant(0.0),
            league_fraction=constant(0.0),
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
            rollouts_per_update=1,
            num_minibatches=2,
            gamma=0.99,
            gae_lambda=0.95,
            clip_coef=0.2,
            max_grad_norm=1.0,
            total_timesteps=16 * 32 * 3,  # 3 updates
            return_ema_alpha=0.005,
            return_min_span=1e-3,
            advantage_min_rms=1e-4,
            checkpoint_dir=str(tmp_path),
            league_size=5,
            elo_milestone_gap=100.0,
            elo_temperature=200.0,
            league_uniform_sampling=False,
            elo_eval=EloEvalConfig(4, 4, 4.0, 1000.0, 100),
            bc_winrate_target=0.9,
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


class TestNumericalPrecision:
    """Parameter storage stays fp32 so the optimizer can actually move it.

    bf16 carries 8 mantissa bits, so its ulp at 1.0 is ~0.0039 while an Adam
    step is ~lr (3e-4). A bf16 parameter sitting at 1.0 — where every RMSNorm
    gain initializes — absorbs every update as a rounding no-op and never
    trains. These run on CUDA because the cast that caused this was guarded on
    ``weight.is_cuda``, so a CPU-only suite cannot observe it.
    """

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    @pytest.mark.parametrize("module_attr", ["_policy_module", "_avg_policy_module"])
    def test_no_low_precision_parameters(self, module_attr, tmp_path):
        trainer = _make_trainer(
            checkpoint_dir=str(tmp_path), avg_model_fraction=0.25, device="cuda"
        )
        offenders = {
            name: str(param.dtype)
            for name, param in getattr(trainer, module_attr).named_parameters()
            if param.dtype != torch.float32
        }
        assert not offenders, f"{module_attr} has non-fp32 parameters: {offenders}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_rmsnorm_gains_are_trainable(self, tmp_path):
        """Every RMSNorm gain must move under a run of optimizer steps."""
        trainer = _make_trainer(checkpoint_dir=str(tmp_path), device="cuda")
        gains = [
            (name, module.weight)
            for name, module in trainer._policy_module.named_modules()
            if isinstance(module, torch.nn.RMSNorm)
        ]
        assert gains, "no RMSNorm gains found"

        before = {name: weight.detach().clone() for name, weight in gains}
        for _ in range(50):
            for _, weight in gains:
                weight.grad = torch.randn_like(weight) * 0.01
            trainer.optim.step()
            trainer.optim.zero_grad()

        stuck = [name for name, weight in gains if torch.equal(weight.detach(), before[name])]
        assert not stuck, f"RMSNorm gains never moved: {stuck}"


class TestNonFiniteGradientGuard:
    """A single overflowing gradient element must not poison the run.

    clip_grad_norm_ scales by max_norm/inf == 0 when the total norm is
    non-finite, which turns that element into NaN (inf * 0) and zeroes the rest.
    Adam folds the NaN into exp_avg, so the parameter stays NaN forever and the
    policy emits NaN logits until a sampler trips a CUDA assert somewhere
    unrelated.
    """

    def test_inf_gradient_does_not_poison_parameters(self, tmp_path):
        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        params = list(trainer._policy_module.parameters())
        for param in params:
            param.grad = torch.randn_like(param) * 0.01
        params[0].grad.view(-1)[0] = float("inf")

        torch.nn.utils.clip_grad_norm_(params, trainer.cfg.max_grad_norm)
        for param in params:
            if param.grad is not None:
                torch.nan_to_num_(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
        trainer.optim.step()

        assert all(torch.isfinite(p).all() for p in params), "an inf gradient reached a parameter"
        for param in params:
            state = trainer.optim.state.get(param, {})
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state:
                    assert torch.isfinite(state[key]).all(), f"Adam {key} went non-finite"

    def test_training_reports_the_skip_fraction(self, tmp_path):
        """The counter has to reach the metrics or a real overflow stays silent."""
        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        runtime = trainer._initialize_rollout_runtime()
        dones = trainer._collect_rollout(runtime, False)
        trainer._compute_rollout_gae(runtime, dones)

        metrics = trainer._update_epochs(
            all_buffers=[trainer.buffer] + trainer.aux_buffers, record_histograms=False
        )

        assert "train/nonfinite_grad_fraction" in metrics
        assert metrics["train/nonfinite_grad_fraction"] == 0.0, (
            "healthy gradients must not be reported as scrubbed"
        )

    def test_update_with_histograms_handles_bf16_returns(self, tmp_path):
        """record_histograms=True must not choke on the bf16-stored returns buffer.

        The histogram diagnostic path calls .numpy() on mb_returns, and numpy has no
        bfloat16 dtype — so the buffer's reduced-precision storage requires an explicit
        upcast. record_histograms=False paths never exercise this, hence the guard here.
        """
        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        assert trainer.buffer.returns.dtype == torch.bfloat16  # precondition for the bug
        runtime = trainer._initialize_rollout_runtime()
        dones = trainer._collect_rollout(runtime, False)
        trainer._compute_rollout_gae(runtime, dones)

        # Must not raise "Got unsupported ScalarType BFloat16".
        metrics = trainer._update_epochs(
            all_buffers=[trainer.buffer] + trainer.aux_buffers, record_histograms=True
        )
        assert "train/epochs_completed" in metrics


class TestUpdateEpochsMetricKeys:
    """AUDIT-015: the accumulator-to-output copy in _update_epochs is table-driven
    for direct renames; this pins the full output key set so a future edit can't
    silently drop or rename one — the exact failure mode a hand-written block of
    25 near-identical lines invites."""

    _EXPECTED_KEYS = frozenset(
        {
            "loss/total",
            "loss/policy_gradient",
            "loss/value",
            "loss/entropy",
            "loss/behavioral_cloning",
            "loss/behavioral_cloning_kl",
            "loss/scripted_entropy",
            "loss/sigreg",
            "loss/next_state",
            "loss/next_state_cont",
            "loss/windowed_ns",
            "loss_proxy/policy_gradient",
            "loss_proxy/value",
            "loss_proxy/entropy",
            "loss_proxy/behavioral_cloning",
            "loss_proxy/sigreg",
            "loss_proxy/next_state",
            "policy/kl",
            "policy/clip_fraction",
            "policy/ratio_mean",
            "policy/ratio_max",
            "policy/entropy_power",
            "policy/entropy_turn",
            "policy/entropy_shoot",
            "returns/aggregate",
            "returns/aggregate_std",
            "returns/advantage_std",
            "episode/alive_fraction",
            "train/gradient_norm",
            "train/nonfinite_grad_fraction",
        }
    )

    def test_all_expected_metric_keys_are_present_and_finite(self, tmp_path):
        trainer = _make_trainer(checkpoint_dir=str(tmp_path))
        runtime = trainer._initialize_rollout_runtime()
        dones = trainer._collect_rollout(runtime, False)
        trainer._compute_rollout_gae(runtime, dones)

        metrics = trainer._update_epochs(
            all_buffers=[trainer.buffer] + trainer.aux_buffers, record_histograms=False
        )

        assert self._EXPECTED_KEYS <= metrics.keys()
        for key in self._EXPECTED_KEYS:
            assert math.isfinite(metrics[key]), f"{key} is not finite: {metrics[key]}"
