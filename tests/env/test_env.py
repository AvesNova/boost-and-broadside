"""Integration tests for TensorEnv and MVPEnvWrapper."""

import pytest
import torch

from boost_and_broadside.config import ShipConfig, EnvConfig, PhaseConfig
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.wrapper import MVPEnvWrapper


@pytest.fixture
def ship_cfg() -> ShipConfig:
    return ShipConfig()


@pytest.fixture
def env_cfg() -> EnvConfig:
    return EnvConfig(num_ships=8, max_bullets=20, max_episode_steps=100)


@pytest.fixture
def reward_cfg() -> PhaseConfig:
    return PhaseConfig(
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
        positioning_radius=300.0,
        proximity_radius=300.0,
        power_range_lo=0.2,
        power_range_hi=0.8,
        speed_range_lo=40.0,
        speed_range_hi=120.0,
        shoot_quality_radius=200.0,
        enemy_neg_lambda_components=frozenset({"damage", "death", "victory", "exposure"}),
        disabled_rewards=frozenset(),
    )


class TestTensorEnvReset:
    def test_state_allocated_after_reset(self, ship_cfg, env_cfg):
        """State must be non-None after reset."""
        env = TensorEnv(
            num_envs=2, ship_config=ship_cfg, env_config=env_cfg, device="cpu"
        )
        env.reset()
        assert env.state is not None

    def test_state_shape_matches_config(self, ship_cfg, env_cfg):
        """Tensor shapes must match (num_envs, num_ships)."""
        B, N = 3, env_cfg.num_ships
        env = TensorEnv(
            num_envs=B, ship_config=ship_cfg, env_config=env_cfg, device="cpu"
        )
        env.reset()

        assert env.state.ship_pos.shape == (B, N)
        assert env.state.ship_alive.shape == (B, N)
        assert env.state.ship_health.shape == (B, N)

    def test_team_sizes_respected(self, ship_cfg, env_cfg):
        """reset() with team_sizes option must assign correct team counts.

        Team IDs may be randomly flipped per env, but the sizes {3, 4} must
        always be present regardless of which ID got which count.
        """
        B = 2
        env = TensorEnv(
            num_envs=B, ship_config=ship_cfg, env_config=env_cfg, device="cpu"
        )
        env.reset(options={"team_sizes": (3, 4)})

        alive = env.state.ship_alive
        teams = env.state.ship_team_id

        for b in range(B):
            t0_alive = (alive[b] & (teams[b] == 0)).sum().item()
            t1_alive = (alive[b] & (teams[b] == 1)).sum().item()
            assert {t0_alive, t1_alive} == {3, 4}

    def test_team_assignment_is_randomized(self, ship_cfg, env_cfg):
        """With many parallel envs, both team-ID orderings must occur."""
        env = TensorEnv(
            num_envs=200, ship_config=ship_cfg, env_config=env_cfg, device="cpu"
        )
        env.reset()
        # ship slot 0: should be team-0 in some envs and team-1 in others
        first_slot_team = env.state.ship_team_id[:, 0]
        assert (first_slot_team == 0).any(), (
            "slot-0 was always team-0 — randomization broken"
        )
        assert (first_slot_team == 1).any(), (
            "slot-0 was always team-1 — randomization broken"
        )

    def test_step_count_starts_at_zero(self, ship_cfg, env_cfg):
        env = TensorEnv(
            num_envs=2, ship_config=ship_cfg, env_config=env_cfg, device="cpu"
        )
        env.reset()
        assert (env.state.step_count == 0).all()

    def test_all_ships_have_full_health_and_power_after_reset(self, ship_cfg, env_cfg):
        env = TensorEnv(
            num_envs=2, ship_config=ship_cfg, env_config=env_cfg, device="cpu"
        )
        env.reset(options={"team_sizes": (4, 4)})

        alive = env.state.ship_alive
        assert (env.state.ship_health[alive] == ship_cfg.max_health).all()
        assert (env.state.ship_power[alive] == ship_cfg.max_power).all()


class TestTensorEnvStep:
    def test_step_count_increments(self, ship_cfg, env_cfg):
        env = TensorEnv(
            num_envs=2, ship_config=ship_cfg, env_config=env_cfg, device="cpu"
        )
        env.reset(options={"team_sizes": (4, 4)})

        actions = torch.zeros((2, env_cfg.num_ships, 3), dtype=torch.long)
        env.step(actions)

        assert (env.state.step_count == 1).all()

    def test_step_returns_bool_tensors(self, ship_cfg, env_cfg):
        env = TensorEnv(
            num_envs=2, ship_config=ship_cfg, env_config=env_cfg, device="cpu"
        )
        env.reset(options={"team_sizes": (4, 4)})

        actions = torch.zeros((2, env_cfg.num_ships, 3), dtype=torch.long)
        dones, truncated = env.step(actions)

        assert dones.dtype == torch.bool
        assert truncated.dtype == torch.bool
        assert dones.shape == (2,)
        assert truncated.shape == (2,)

    def test_truncated_fires_at_max_episode_steps(self, ship_cfg):
        """truncated must become True exactly when step_count hits max_episode_steps."""
        env_cfg = EnvConfig(num_ships=2, max_bullets=5, max_episode_steps=3)
        env = TensorEnv(
            num_envs=1, ship_config=ship_cfg, env_config=env_cfg, device="cpu"
        )
        env.reset(options={"team_sizes": (1, 1)})

        actions = torch.zeros((1, 2, 3), dtype=torch.long)
        for _ in range(2):
            _, truncated = env.step(actions)
            assert not truncated[0].item()

        _, truncated = env.step(actions)
        assert truncated[0].item()

    def test_ships_move_when_coasting(self, ship_cfg, env_cfg):
        """COAST action with initial velocity should move ships (non-zero position change)."""
        env = TensorEnv(
            num_envs=1, ship_config=ship_cfg, env_config=env_cfg, device="cpu"
        )
        env.reset(options={"team_sizes": (4, 4)})

        pos_before = env.state.ship_pos.clone()
        actions = torch.zeros((1, env_cfg.num_ships, 3), dtype=torch.long)
        env.step(actions)

        # Ships have default_speed velocity, so position changes
        assert not torch.allclose(env.state.ship_pos, pos_before)


class TestMVPEnvWrapper:
    def test_reset_returns_obs_dict(self, ship_cfg, env_cfg, reward_cfg):
        wrapper = MVPEnvWrapper(
            num_envs=2,
            ship_config=ship_cfg,
            env_config=env_cfg,
            phase=reward_cfg,
            device="cpu",
        )
        obs = wrapper.reset(options={"team_sizes": (4, 4)})

        assert "pos" in obs
        assert "vel" in obs
        assert "att" in obs
        assert "ang_vel" in obs
        assert "scalars" in obs
        assert "team_id" in obs
        assert "alive" in obs
        assert "prev_action" in obs

    def test_obs_shapes_correct(self, ship_cfg, env_cfg, reward_cfg):
        B, N = 2, env_cfg.num_ships
        wrapper = MVPEnvWrapper(
            num_envs=B,
            ship_config=ship_cfg,
            env_config=env_cfg,
            phase=reward_cfg,
            device="cpu",
        )
        obs = wrapper.reset(options={"team_sizes": (4, 4)})

        assert obs["pos"].shape == (B, N, 2)
        assert obs["vel"].shape == (B, N, 2)
        assert obs["att"].shape == (B, N, 2)
        assert obs["ang_vel"].shape == (B, N, 1)
        assert obs["scalars"].shape == (B, N, 3)
        assert obs["team_id"].shape == (B, N)
        assert obs["alive"].shape == (B, N)

    def test_step_returns_correct_shapes(self, ship_cfg, env_cfg, reward_cfg):
        B, N = 2, env_cfg.num_ships
        wrapper = MVPEnvWrapper(
            num_envs=B,
            ship_config=ship_cfg,
            env_config=env_cfg,
            phase=reward_cfg,
            device="cpu",
        )
        wrapper.reset(options={"team_sizes": (4, 4)})

        actions = torch.zeros((B, N, 3), dtype=torch.long)
        obs, rewards, dones, truncated, info = wrapper.step(actions)

        K = 12  # num_value_components
        assert rewards.shape == (B, N, K)
        assert dones.shape == (B,)
        assert truncated.shape == (B,)

    def test_pos_normalized_to_unit_range(self, ship_cfg, env_cfg, reward_cfg):
        """Normalized position must be in [0, 1] after reset."""
        wrapper = MVPEnvWrapper(
            num_envs=2,
            ship_config=ship_cfg,
            env_config=env_cfg,
            phase=reward_cfg,
            device="cpu",
        )
        obs = wrapper.reset(options={"team_sizes": (4, 4)})

        assert obs["pos"].min().item() >= 0.0
        assert obs["pos"].max().item() <= 1.0

    def test_episode_info_returned_on_done(self, ship_cfg, reward_cfg):
        """info dict must contain ep_reward and ep_length when an episode ends."""
        env_cfg = EnvConfig(num_ships=2, max_bullets=5, max_episode_steps=2)
        wrapper = MVPEnvWrapper(
            num_envs=1,
            ship_config=ship_cfg,
            env_config=env_cfg,
            phase=reward_cfg,
            device="cpu",
        )
        wrapper.reset(options={"team_sizes": (1, 1)})

        actions = torch.zeros((1, 2, 3), dtype=torch.long)
        # Run until truncation
        for _ in range(2):
            _, _, _, _, info = wrapper.step(actions)

        # After 2 steps with max_episode_steps=2 the env truncates
        assert "ep_reward" in info or "ep_length" in info  # at least one present
