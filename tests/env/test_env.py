"""Integration tests for TensorEnv and YemongEnvWrapper."""

import pytest
import torch

from boost_and_broadside.config import EnvConfig, RewardConfig, ShipConfig
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.observation import observation_from_state
from boost_and_broadside.env.wrapper import YemongEnvWrapper


@pytest.fixture
def ship_cfg() -> ShipConfig:
    return ShipConfig()


@pytest.fixture
def env_cfg() -> EnvConfig:
    return EnvConfig(num_ships=8, max_bullets=20, max_episode_steps=100)


@pytest.fixture
def reward_cfg() -> RewardConfig:
    return RewardConfig(
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


class TestTensorEnvReset:
    def test_state_allocated_after_reset(self, ship_cfg, env_cfg):
        """State must be non-None after reset."""
        env = TensorEnv(num_envs=2, ship_config=ship_cfg, env_config=env_cfg, device="cpu")
        env.reset()
        assert env.state is not None

    def test_state_shape_matches_config(self, ship_cfg, env_cfg):
        """Tensor shapes must match (num_envs, num_ships)."""
        B, N = 3, env_cfg.num_ships
        env = TensorEnv(num_envs=B, ship_config=ship_cfg, env_config=env_cfg, device="cpu")
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
        env = TensorEnv(num_envs=B, ship_config=ship_cfg, env_config=env_cfg, device="cpu")
        env.reset(options={"team_sizes": (3, 4)})

        alive = env.state.ship_alive
        teams = env.state.ship_team_id

        for b in range(B):
            t0_alive = (alive[b] & (teams[b] == 0)).sum().item()
            t1_alive = (alive[b] & (teams[b] == 1)).sum().item()
            assert {t0_alive, t1_alive} == {3, 4}

    def test_team_sizes_summing_past_num_ships_raises(self, ship_cfg, env_cfg):
        """team_sizes that overflow num_ships must fail fast, not silently write
        past the alive-slot range (AUDIT-005)."""
        env = TensorEnv(num_envs=2, ship_config=ship_cfg, env_config=env_cfg, device="cpu")
        with pytest.raises(ValueError, match="team_sizes"):
            env.reset(options={"team_sizes": (env_cfg.num_ships, 1)})

    def test_negative_team_size_raises(self, ship_cfg, env_cfg):
        """A negative team count must fail fast rather than corrupt team ids."""
        env = TensorEnv(num_envs=2, ship_config=ship_cfg, env_config=env_cfg, device="cpu")
        with pytest.raises(ValueError, match="team_sizes"):
            env.reset(options={"team_sizes": (-1, 4)})

    def test_team_assignment_is_randomized(self, ship_cfg, env_cfg):
        """With many parallel envs, both team-ID orderings must occur."""
        env = TensorEnv(num_envs=200, ship_config=ship_cfg, env_config=env_cfg, device="cpu")
        env.reset()
        # ship slot 0: should be team-0 in some envs and team-1 in others
        first_slot_team = env.state.ship_team_id[:, 0]
        assert (first_slot_team == 0).any(), "slot-0 was always team-0 — randomization broken"
        assert (first_slot_team == 1).any(), "slot-0 was always team-1 — randomization broken"

    def test_step_count_starts_at_zero(self, ship_cfg, env_cfg):
        env = TensorEnv(num_envs=2, ship_config=ship_cfg, env_config=env_cfg, device="cpu")
        env.reset()
        assert (env.state.step_count == 0).all()

    def test_all_ships_have_full_health_and_power_after_reset(self, ship_cfg, env_cfg):
        env = TensorEnv(num_envs=2, ship_config=ship_cfg, env_config=env_cfg, device="cpu")
        env.reset(options={"team_sizes": (4, 4)})

        alive = env.state.ship_alive
        assert (env.state.ship_health[alive] == ship_cfg.max_health).all()
        assert (env.state.ship_power[alive] == ship_cfg.max_power).all()


class TestTensorEnvStep:
    def test_step_count_increments(self, ship_cfg, env_cfg):
        env = TensorEnv(num_envs=2, ship_config=ship_cfg, env_config=env_cfg, device="cpu")
        env.reset(options={"team_sizes": (4, 4)})

        actions = torch.zeros((2, env_cfg.num_ships, 3), dtype=torch.long)
        env.step(actions)

        assert (env.state.step_count == 1).all()

    def test_step_returns_bool_tensors(self, ship_cfg, env_cfg):
        env = TensorEnv(num_envs=2, ship_config=ship_cfg, env_config=env_cfg, device="cpu")
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
        env = TensorEnv(num_envs=1, ship_config=ship_cfg, env_config=env_cfg, device="cpu")
        env.reset(options={"team_sizes": (1, 1)})

        actions = torch.zeros((1, 2, 3), dtype=torch.long)
        for _ in range(2):
            _, truncated = env.step(actions)
            assert not truncated[0].item()

        _, truncated = env.step(actions)
        assert truncated[0].item()

    def test_ships_move_when_coasting(self, ship_cfg, env_cfg):
        """COAST action with initial velocity should move ships (non-zero position change)."""
        env = TensorEnv(num_envs=1, ship_config=ship_cfg, env_config=env_cfg, device="cpu")
        env.reset(options={"team_sizes": (4, 4)})

        pos_before = env.state.ship_pos.clone()
        actions = torch.zeros((1, env_cfg.num_ships, 3), dtype=torch.long)
        env.step(actions)

        # Ships have default_speed velocity, so position changes
        assert not torch.allclose(env.state.ship_pos, pos_before)


class TestYemongEnvWrapper:
    def test_reset_returns_obs_dict(self, ship_cfg, env_cfg, reward_cfg):
        wrapper = YemongEnvWrapper(
            num_envs=2,
            ship_config=ship_cfg,
            env_config=env_cfg,
            rewards=reward_cfg,
            device="cpu",
        )
        obs = wrapper.reset(options={"team_sizes": (4, 4)})

        assert "pos" in obs
        assert "vel" in obs
        assert "att" in obs
        assert "ang_vel" in obs
        assert "health" in obs
        assert "power" in obs
        assert "cooldown" in obs
        assert "team_id" in obs
        assert "alive" in obs
        assert "previous_action" in obs

    def test_obs_shapes_correct(self, ship_cfg, env_cfg, reward_cfg):
        B, N = 2, env_cfg.num_ships
        wrapper = YemongEnvWrapper(
            num_envs=B,
            ship_config=ship_cfg,
            env_config=env_cfg,
            rewards=reward_cfg,
            device="cpu",
        )
        obs = wrapper.reset(options={"team_sizes": (4, 4)})

        assert obs["pos"].shape == (B, N, 2)
        assert obs["vel"].shape == (B, N, 2)
        assert obs["att"].shape == (B, N, 2)
        assert obs["ang_vel"].shape == (B, N, 1)
        assert obs["health"].shape == (B, N, 1)
        assert obs["power"].shape == (B, N, 1)
        assert obs["cooldown"].shape == (B, N, 1)
        assert obs["team_id"].shape == (B, N)
        assert obs["alive"].shape == (B, N)
        assert obs["previous_action"].shape == (B, N, 3)

    def test_observation_from_state_matches_wrapper(self, ship_cfg, env_cfg, reward_cfg):
        """The standalone builder and training wrapper must emit the same raw tensors."""
        wrapper = YemongEnvWrapper(
            num_envs=2,
            ship_config=ship_cfg,
            env_config=env_cfg,
            rewards=reward_cfg,
            device="cpu",
        )
        wrapper_obs = wrapper.reset(options={"team_sizes": (4, 4)})
        standalone_obs = observation_from_state(wrapper.state, ship_cfg)

        assert all(
            torch.equal(wrapper_obs[key], standalone_obs[key]) for key in standalone_obs.data
        )

    def test_observation_from_state_copies_obstacle_radius(self, ship_cfg):
        """The standalone builder must retain dynamic obstacle geometry."""
        env = TensorEnv(
            num_envs=2,
            ship_config=ship_cfg,
            env_config=EnvConfig(
                num_ships=2,
                max_bullets=20,
                max_episode_steps=100,
                num_obstacles=1,
            ),
            device="cpu",
        )
        env.reset(options={"team_sizes": (1, 1)})
        env.state.obstacle_radius.fill_(42.0)
        obs = observation_from_state(env.state, ship_cfg)

        assert torch.equal(obs.radius[:, -1, 0], torch.full((2,), 42.0))

    def test_step_returns_correct_shapes(self, ship_cfg, env_cfg, reward_cfg):
        B, N = 2, env_cfg.num_ships
        wrapper = YemongEnvWrapper(
            num_envs=B,
            ship_config=ship_cfg,
            env_config=env_cfg,
            rewards=reward_cfg,
            device="cpu",
        )
        wrapper.reset(options={"team_sizes": (4, 4)})

        actions = torch.zeros((B, N, 3), dtype=torch.long)
        obs, rewards, dones, truncated, info = wrapper.step(actions)

        assert rewards.shape == (B, N, wrapper.num_active_components)
        assert dones.shape == (B,)
        assert truncated.shape == (B,)

    def test_pos_within_world_bounds(self, ship_cfg, env_cfg, reward_cfg):
        """Raw positions must be within [0, world_w] x [0, world_h] after reset."""
        wrapper = YemongEnvWrapper(
            num_envs=2,
            ship_config=ship_cfg,
            env_config=env_cfg,
            rewards=reward_cfg,
            device="cpu",
        )
        obs = wrapper.reset(options={"team_sizes": (4, 4)})
        world_w, world_h = ship_cfg.world_size

        assert obs["pos"][..., 0].min().item() >= 0.0
        assert obs["pos"][..., 0].max().item() <= world_w
        assert obs["pos"][..., 1].min().item() >= 0.0
        assert obs["pos"][..., 1].max().item() <= world_h

    def test_episode_stats_accumulated_on_done(self, ship_cfg, reward_cfg):
        """pop_episode_stats must report finished episodes, then reset."""
        env_cfg = EnvConfig(num_ships=2, max_bullets=5, max_episode_steps=2)
        wrapper = YemongEnvWrapper(
            num_envs=1,
            ship_config=ship_cfg,
            env_config=env_cfg,
            rewards=reward_cfg,
            device="cpu",
        )
        wrapper.reset(options={"team_sizes": (1, 1)})

        actions = torch.zeros((1, 2, 3), dtype=torch.long)
        # Run until truncation (max_episode_steps=2)
        for _ in range(2):
            wrapper.step(actions)

        stats = wrapper.pop_episode_stats()
        assert stats["episodes"].item() == 1
        assert stats["length_sum"].item() == 2

        # Accumulators must be cleared by the pop
        stats_after = wrapper.pop_episode_stats()
        assert stats_after["episodes"].item() == 0
