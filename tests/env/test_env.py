"""Integration tests for TensorEnv and YemongEnvWrapper."""

import pytest
import torch

from boost_and_broadside.config import (
    EnvConfig,
    InterfaceDamageLevel,
    RefractiveIndexLevel,
    RewardConfig,
    ShipConfig,
)
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.field_cache import FieldMapCache
from boost_and_broadside.env.observation import observation_from_state
from boost_and_broadside.env.wrapper import YemongEnvWrapper
from tests.conftest import activate_bullet


@pytest.fixture
def ship_cfg() -> ShipConfig:
    return ShipConfig()


@pytest.fixture
def env_cfg() -> EnvConfig:
    return EnvConfig(num_ships=8, max_bullets=20, max_episode_steps=100)


@pytest.fixture
def reward_cfg() -> RewardConfig:
    return RewardConfig(
        ally_combat_damage_weight=0.01,
        enemy_combat_damage_weight=0.01,
        ally_field_damage_weight=0.01,
        enemy_field_damage_weight=0.01,
        ally_combat_death_weight=0.5,
        enemy_combat_death_weight=0.5,
        ally_field_death_weight=0.5,
        enemy_field_death_weight=0.5,
        win_weight=1.0,
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
                "win",
            }
        ),
        ally_zero_components=frozenset(
            {
                "enemy_combat_damage",
                "enemy_field_damage",
                "enemy_combat_death",
                "enemy_field_death",
            }
        ),
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

    def test_none_max_episode_steps_disables_time_truncation(self, ship_cfg):
        env_cfg = EnvConfig(num_ships=1, max_bullets=0, max_episode_steps=None)
        env = TensorEnv(num_envs=1, ship_config=ship_cfg, env_config=env_cfg, device="cpu")
        env.reset()

        actions = torch.zeros((1, 1, 3), dtype=torch.long)
        for _ in range(10):
            _, truncated = env.step(actions)
            assert not truncated.item()

        assert env.state.step_count.item() == 10

    def test_ships_move_when_coasting(self, ship_cfg, env_cfg):
        """COAST action with initial velocity should move ships (non-zero position change)."""
        env = TensorEnv(num_envs=1, ship_config=ship_cfg, env_config=env_cfg, device="cpu")
        env.reset(options={"team_sizes": (4, 4)})

        pos_before = env.state.ship_pos.clone()
        actions = torch.zeros((1, env_cfg.num_ships, 3), dtype=torch.long)
        env.step(actions)

        # Ships have default_speed velocity, so position changes
        assert not torch.allclose(env.state.ship_pos, pos_before)

    def test_unlimited_resources_survive_lethal_hit_and_refill_power(self, ship_cfg):
        env_cfg = EnvConfig(num_ships=2, max_bullets=1, max_episode_steps=100)
        env = TensorEnv(num_envs=1, ship_config=ship_cfg, env_config=env_cfg, device="cpu")
        env.reset(options={"team_sizes": (1, 1)})

        env.state.ship_pos[0] = torch.tensor([100.0 + 100.0j, 200.0 + 200.0j])
        env.state.ship_health[0, 1] = 1.0
        env.state.ship_power.zero_()
        activate_bullet(
            env.state,
            ship_cfg,
            position=env.state.ship_pos[0, 1],
            velocity=-env.state.ship_attitude[0, 1],
            lifetime=1.0,
            damage=ship_cfg.max_health,
        )

        actions = torch.zeros((1, 2, 3), dtype=torch.long)
        dones, _ = env.step(actions, unlimited_resources=True)

        assert not env.state.bullet_active[0, 0, 0]
        assert not dones.item()
        assert env.state.ship_alive.all()
        assert torch.equal(
            env.state.ship_health,
            torch.full_like(env.state.ship_health, ship_cfg.max_health),
        )
        assert torch.equal(
            env.state.ship_power,
            torch.full_like(env.state.ship_power, ship_cfg.max_power),
        )
        assert not env.state.ship_field_damage.any()
        assert not env.state.ship_combat_damage.any()
        assert not env.state.ship_field_death.any()
        assert not env.state.ship_combat_death.any()

    def test_combat_source_bookkeeping_caps_lethal_overkill(self, ship_cfg):
        env_cfg = EnvConfig(num_ships=2, max_bullets=1, max_episode_steps=100)
        env = TensorEnv(num_envs=1, ship_config=ship_cfg, env_config=env_cfg, device="cpu")
        env.reset(options={"team_sizes": (1, 1)})
        env.state.ship_pos[0] = torch.tensor([100.0 + 100.0j, 200.0 + 200.0j])
        env.state.ship_vel.zero_()
        activate_bullet(
            env.state,
            ship_cfg,
            position=env.state.ship_pos[0, 1],
            velocity=-env.state.ship_attitude[0, 1],
            lifetime=1.0,
            damage=2.0 * ship_cfg.max_health / ship_cfg.bullet_min_damage_frac,
        )

        env.step(torch.zeros((1, 2, 3), dtype=torch.long))

        assert env.state.ship_combat_damage[0, 1].item() == ship_cfg.max_health
        assert env.state.ship_combat_death[0, 1].item()
        assert not env.state.ship_field_death.any()
        assert env.state.ship_health[0, 1].item() == 0.0


class TestActionRepeat:
    """One wrapper step is one decision, held for action_repeat physics ticks."""

    @staticmethod
    def _wrapper(ship_cfg, reward_cfg, repeat: int, num_envs: int = 4):
        return YemongEnvWrapper(
            num_envs=num_envs,
            ship_config=ship_cfg,
            env_config=EnvConfig(
                num_ships=8, max_bullets=20, max_episode_steps=1000, action_repeat=repeat
            ),
            rewards=reward_cfg,
            device="cpu",
        )

    def test_step_advances_physics_by_the_repeat(self, ship_cfg, reward_cfg):
        wrapper = self._wrapper(ship_cfg, reward_cfg, repeat=3)
        wrapper.reset(options={"team_sizes": (4, 4)})
        actions = torch.zeros(4, 8, 3, dtype=torch.int32)

        wrapper.step(actions)

        assert (wrapper.env.state.step_count == 3).all()

    def test_repeat_one_is_the_unrepeated_path(self, ship_cfg, reward_cfg):
        """Repeat 1 must reproduce single-tick behaviour exactly."""
        torch.manual_seed(5)
        one = self._wrapper(ship_cfg, reward_cfg, repeat=1)
        one.reset(seed=17, options={"team_sizes": (4, 4)})
        actions = torch.ones(4, 8, 3, dtype=torch.int32)
        torch.manual_seed(9)
        _, reward_a, _, _, _ = one.step(actions)

        torch.manual_seed(5)
        two = self._wrapper(ship_cfg, reward_cfg, repeat=1)
        two.reset(seed=17, options={"team_sizes": (4, 4)})
        torch.manual_seed(9)
        _, reward_b, _, _, _ = two.step(actions)

        assert torch.equal(reward_a, reward_b)

    def test_rewards_sum_over_held_ticks(self, ship_cfg, reward_cfg):
        """Summing is what keeps the reward scale invariant to the repeat.

        Over a fixed span of game time both the dense per-tick terms and the
        one-off event terms total the same, so RewardConfig's component ratios
        are untouched by the tick rate.
        """
        actions = torch.ones(4, 8, 3, dtype=torch.int32)

        torch.manual_seed(3)
        stepwise = self._wrapper(ship_cfg, reward_cfg, repeat=1)
        stepwise.reset(seed=23, options={"team_sizes": (4, 4)})
        torch.manual_seed(11)
        total = sum(stepwise.step(actions)[1] for _ in range(3))

        torch.manual_seed(3)
        held = self._wrapper(ship_cfg, reward_cfg, repeat=3)
        held.reset(seed=23, options={"team_sizes": (4, 4)})
        torch.manual_seed(11)
        _, repeated, _, _, _ = held.step(actions)

        assert torch.allclose(total, repeated, atol=1e-5)

    def test_episode_length_counts_physics_ticks(self, ship_cfg, reward_cfg):
        """Lengths stay in ticks so they are comparable across tick rates."""
        wrapper = self._wrapper(ship_cfg, reward_cfg, repeat=3)
        wrapper.reset(options={"team_sizes": (4, 4)})
        actions = torch.zeros(4, 8, 3, dtype=torch.int32)
        for _ in range(4):
            wrapper.step(actions)
        assert (wrapper._ep_length == 12).all()

    def test_finished_env_stops_earning_mid_hold(self, ship_cfg, reward_cfg):
        """An env that ends partway through the hold contributes nothing after.

        Its remaining ticks still simulate — masking the physics would cost more
        than the wasted ticks — but nothing they produce is read.
        """
        wrapper = self._wrapper(ship_cfg, reward_cfg, repeat=3, num_envs=2)
        wrapper.reset(options={"team_sizes": (4, 4)})
        # Truncate env 0 on the first tick of the next hold.
        wrapper.env.state.step_count[0] = wrapper.env_config.max_episode_steps - 1
        actions = torch.zeros(2, 8, 3, dtype=torch.int32)

        _, rewards, _, truncated = wrapper.step(actions)[:4]

        assert truncated[0] and not truncated[1]
        # One tick of dense shaping for env 0 against three for env 1.
        assert wrapper._ep_length[1] == 3


class TestDecisionRateIsGlobal:
    """Every consumer of the env must see the same decision rate.

    Regression: action_repeat lived only in YemongEnvWrapper, so training ran at
    the configured rate while the Elo battery and every evaluation mode stepped
    TensorEnv directly at one tick per action. A policy trained to hold an action
    for N ticks then turns a fraction of its intended amount per decision,
    mistimes every lead, and advances its recurrent state N times too fast for
    the game clock — it still plays, just far worse, and no metric says why.
    """

    @staticmethod
    def _env(ship_cfg, repeat):
        return TensorEnv(
            4,
            ship_cfg,
            EnvConfig(num_ships=4, max_bullets=8, max_episode_steps=1000, action_repeat=repeat),
            "cpu",
        )

    def test_step_advances_by_action_repeat(self, ship_cfg):
        env = self._env(ship_cfg, 3)
        env.reset()
        env.step(torch.zeros(4, 4, 3, dtype=torch.int32))
        assert (env.state.step_count == 3).all()

    def test_tick_is_always_one_physics_step(self, ship_cfg):
        """The wrapper opts out of the repeat because it accumulates per tick."""
        env = self._env(ship_cfg, 3)
        env.reset()
        env.tick(torch.zeros(4, 4, 3, dtype=torch.int32))
        assert (env.state.step_count == 1).all()

    def test_repeat_one_leaves_step_and_tick_equivalent(self, ship_cfg):
        for method in ("step", "tick"):
            env = self._env(ship_cfg, 1)
            env.reset()
            getattr(env, method)(torch.zeros(4, 4, 3, dtype=torch.int32))
            assert (env.state.step_count == 1).all(), method

    def test_step_flags_are_sticky_across_the_hold(self, ship_cfg):
        """An env finishing mid-hold must still be reported as finished."""
        env = self._env(ship_cfg, 3)
        env.reset()
        env.state.step_count[0] = env.env_config.max_episode_steps - 1
        _, truncated = env.step(torch.zeros(4, 4, 3, dtype=torch.int32))
        assert truncated[0] and not truncated[1:].any()


class TestSpawnResourceSpread:
    def test_zero_spread_spawns_at_full_resources(self, ship_cfg, reward_cfg):
        env = TensorEnv(
            64, ship_cfg, EnvConfig(num_ships=4, max_bullets=8, max_episode_steps=100), "cpu"
        )
        env.reset()
        assert (env.state.ship_health == ship_cfg.max_health).all()
        assert (env.state.ship_power == ship_cfg.max_power).all()
        assert (env.state.ship_cooldown == 0.0).all()

    def test_spread_randomizes_within_bounds(self, ship_cfg, reward_cfg):
        env = TensorEnv(
            256,
            ship_cfg,
            EnvConfig(
                num_ships=4,
                max_bullets=8,
                max_episode_steps=100,
                spawn_resource_spread=0.25,
            ),
            "cpu",
        )
        env.reset()
        health, power = env.state.ship_health, env.state.ship_power
        assert (health >= 0.75 * ship_cfg.max_health).all()
        assert (health <= ship_cfg.max_health).all()
        assert (power >= 0.75 * ship_cfg.max_power).all()
        assert (power <= ship_cfg.max_power).all()
        assert (env.state.ship_cooldown >= 0.0).all()
        assert (env.state.ship_cooldown <= ship_cfg.firing_cooldown).all()
        assert health.std() > 0.0, "spread did not randomize"

    def test_teams_start_with_equal_expected_resources(self, ship_cfg, reward_cfg):
        """A lopsided spawn would put outcome variance into the win signal that
        no policy could have influenced."""
        env = TensorEnv(
            4096,
            ship_cfg,
            EnvConfig(
                num_ships=8,
                max_bullets=8,
                max_episode_steps=100,
                spawn_resource_spread=0.25,
            ),
            "cpu",
        )
        env.reset()
        team0 = env.state.ship_team_id == 0
        team1 = env.state.ship_team_id == 1
        mean0 = (env.state.ship_health * team0).sum() / team0.sum()
        mean1 = (env.state.ship_health * team1).sum() / team1.sum()
        assert abs((mean0 - mean1).item()) < 0.5


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

    def test_observation_from_state_copies_field_geometry(self, ship_cfg):
        """The standalone builder retains static field radius and transition width."""
        field_map = FieldMapCache(
            pos=torch.tensor([[100.0 + 100.0j]]),
            radius=torch.tensor([[42.0]]),
            transition_width=torch.tensor([[20.0]]),
            index_level=torch.tensor([[RefractiveIndexLevel.HIGH]], dtype=torch.int8),
            damage_level=torch.tensor([[InterfaceDamageLevel.NONE]], dtype=torch.int8),
            ship_config=ship_cfg,
        )
        env = TensorEnv(
            num_envs=2,
            ship_config=ship_cfg,
            env_config=EnvConfig(
                num_ships=2,
                max_bullets=20,
                max_episode_steps=100,
                num_fields=1,
            ),
            device="cpu",
            field_map=field_map,
        )
        env.reset(options={"team_sizes": (1, 1)})
        obs = observation_from_state(env.state, ship_cfg)

        assert torch.equal(obs.radius[:, -1, 0], torch.full((2,), 42.0))
        assert torch.equal(obs["field_transition_width"][:, -1, 0], torch.full((2,), 20.0))

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
        assert not stats_after["source_stats"].any()

    def test_source_metrics_accumulate_without_waiting_for_episode_end(self, ship_cfg, reward_cfg):
        env_cfg = EnvConfig(num_ships=2, max_bullets=1, max_episode_steps=100)
        wrapper = YemongEnvWrapper(
            num_envs=1,
            ship_config=ship_cfg,
            env_config=env_cfg,
            rewards=reward_cfg,
            device="cpu",
        )
        wrapper.reset(options={"team_sizes": (1, 1)})
        wrapper.state.ship_pos[0] = torch.tensor([100.0 + 100.0j, 200.0 + 200.0j])
        wrapper.state.ship_vel.zero_()
        activate_bullet(
            wrapper.state,
            ship_cfg,
            position=wrapper.state.ship_pos[0, 1],
            velocity=-wrapper.state.ship_attitude[0, 1],
            lifetime=1.0,
            damage=10.0 / ship_cfg.bullet_min_damage_frac,
        )

        wrapper.step(torch.zeros((1, 2, 3), dtype=torch.long))
        source = wrapper.pop_episode_stats()["source_stats"]

        assert source[0].item() == 0.0
        assert source[1].item() == pytest.approx(10.0)
        assert source[6].item() == 2.0

    def test_death_auto_resets_an_unlimited_episode(self, ship_cfg, reward_cfg):
        env_cfg = EnvConfig(num_ships=2, max_bullets=0, max_episode_steps=None)
        wrapper = YemongEnvWrapper(
            num_envs=1,
            ship_config=ship_cfg,
            env_config=env_cfg,
            rewards=reward_cfg,
            device="cpu",
        )
        wrapper.reset(options={"team_sizes": (1, 1)})
        wrapper.state.ship_alive[0, 0] = False
        wrapper.state.ship_health[0, 0] = 0.0

        actions = torch.zeros((1, 2, 3), dtype=torch.long)
        _, _, done, truncated, _ = wrapper.step(actions)

        assert done.item()
        assert not truncated.item()
        assert wrapper.state.ship_alive.all()
        assert wrapper.state.step_count.item() == 0
