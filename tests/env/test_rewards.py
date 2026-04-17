"""Unit tests for reward components.

Tests reward signs and magnitudes under controlled scenarios.
No compute_rewards() — per-ship signals are tested directly; zero-sum accounting
(lambda aggregation) lives in the PPO trainer, not the reward components.
"""

import pytest
import torch

from boost_and_broadside.config import ShipConfig, RewardConfig
from boost_and_broadside.env.rewards import (
    AllyDamageReward,
    EnemyDamageReward,
    AllyDeathReward,
    EnemyDeathReward,
    AllyWinReward,
    EnemyWinReward,
    FacingReward,
    ClosingSpeedReward,
    KillShotReward,
    KillAssistReward,
    LocalDamageTakenReward,
    LocalDamageDealtEnemyReward,
    LocalDamageDealtAllyReward,
    LocalDeathReward,
    REWARD_COMPONENT_NAMES,
    compute_per_component_rewards,
    build_reward_components,
)
from tests.conftest import make_state


@pytest.fixture
def cfg() -> ShipConfig:
    return ShipConfig()


@pytest.fixture
def reward_cfg() -> RewardConfig:
    return RewardConfig(
        ally_damage_weight=0.1,
        enemy_damage_weight=0.1,
        ally_death_weight=0.5,
        enemy_death_weight=0.5,
        ally_win_weight=1.0,
        enemy_win_weight=1.0,
        facing_weight=1.0,
        closing_speed_weight=1.0,
        shoot_quality_weight=1.0,
        kill_shot_weight=1.0,
        kill_assist_weight=1.0,
        damage_taken_weight=1.0,
        damage_dealt_enemy_weight=1.0,
        damage_dealt_ally_weight=1.0,
        death_weight=1.0,
        bullet_death_weight=0.0,
        obstacle_death_weight=0.0,
        proximity_radius=500.0,
        shoot_quality_radius=200.0,
        enemy_neg_lambda_components=frozenset(
            {"enemy_damage", "enemy_death", "enemy_win"}
        ),
        ally_zero_components=frozenset({"enemy_damage", "enemy_death", "enemy_win"}),
    )


def _make_4ship_state(cfg):
    """Team 0: ships 0,1. Team 1: ships 2,3."""
    state = make_state(num_envs=2, max_ships=4, ship_config=cfg)
    state.ship_team_id[:, 0] = 0
    state.ship_team_id[:, 1] = 0
    state.ship_team_id[:, 2] = 1
    state.ship_team_id[:, 3] = 1
    return state


# ---------------------------------------------------------------------------
# Component registry
# ---------------------------------------------------------------------------


class TestRewardComponentNames:
    def test_k_equals_17(self):
        assert len(REWARD_COMPONENT_NAMES) == 17

    def test_ally_damage_is_index_0(self):
        assert REWARD_COMPONENT_NAMES[0] == "ally_damage"

    def test_enemy_damage_is_index_1(self):
        assert REWARD_COMPONENT_NAMES[1] == "enemy_damage"

    def test_kill_shot_is_index_9(self):
        assert REWARD_COMPONENT_NAMES[9] == "kill_shot"

    def test_kill_assist_is_index_10(self):
        assert REWARD_COMPONENT_NAMES[10] == "kill_assist"

    def test_damage_taken_is_index_11(self):
        assert REWARD_COMPONENT_NAMES[11] == "damage_taken"

    def test_damage_dealt_enemy_is_index_12(self):
        assert REWARD_COMPONENT_NAMES[12] == "damage_dealt_enemy"

    def test_damage_dealt_ally_is_index_13(self):
        assert REWARD_COMPONENT_NAMES[13] == "damage_dealt_ally"

    def test_death_is_index_14(self):
        assert REWARD_COMPONENT_NAMES[14] == "death"

    def test_no_duplicates(self):
        assert len(set(REWARD_COMPONENT_NAMES)) == len(REWARD_COMPONENT_NAMES)


class TestComputePerComponentRewards:
    def test_output_shape(self, cfg, reward_cfg):
        state = _make_4ship_state(cfg)
        components = build_reward_components(reward_cfg, cfg)
        result = compute_per_component_rewards(
            components,
            state,
            torch.zeros(2, 4, 3),
            state,
            torch.zeros(2, dtype=torch.bool),
        )
        B, N, K = 2, 4, len(REWARD_COMPONENT_NAMES)
        assert result.shape == (B, N, K)


# ---------------------------------------------------------------------------
# Damage rewards (ally/enemy split)
# ---------------------------------------------------------------------------


class TestAllyDamageReward:
    def test_damaged_ship_gets_negative_reward(self, cfg):
        prev = _make_4ship_state(cfg)
        next_ = _make_4ship_state(cfg)
        next_.ship_health[0, 0] = prev.ship_health[0, 0] - 10.0

        r = AllyDamageReward(weight=1.0)
        reward = r.compute(
            prev, torch.zeros(2, 4, 3), next_, torch.zeros(2, dtype=torch.bool)
        )

        assert reward[0, 0].item() < 0

    def test_undamaged_ships_get_zero_reward(self, cfg):
        prev = _make_4ship_state(cfg)
        next_ = _make_4ship_state(cfg)
        next_.ship_health[0, 0] = prev.ship_health[0, 0] - 10.0

        r = AllyDamageReward(weight=1.0)
        reward = r.compute(
            prev, torch.zeros(2, 4, 3), next_, torch.zeros(2, dtype=torch.bool)
        )

        assert reward[0, 1].item() == pytest.approx(0.0)
        assert reward[0, 2].item() == pytest.approx(0.0)
        assert reward[0, 3].item() == pytest.approx(0.0)

    def test_zero_damage_gives_zero_reward(self, cfg):
        state = _make_4ship_state(cfg)

        r = AllyDamageReward(weight=1.0)
        reward = r.compute(
            state, torch.zeros(2, 4, 3), state, torch.zeros(2, dtype=torch.bool)
        )

        assert reward.abs().max().item() == 0.0


class TestEnemyDamageReward:
    def test_damaged_ship_gets_negative_reward(self, cfg):
        """EnemyDamageReward also returns negative for the damaged ship.
        Zero-sum inversion (lambda=-1) happens at PPO aggregation time."""
        prev = _make_4ship_state(cfg)
        next_ = _make_4ship_state(cfg)
        next_.ship_health[0, 2] = prev.ship_health[0, 2] - 15.0

        r = EnemyDamageReward(weight=1.0)
        reward = r.compute(
            prev, torch.zeros(2, 4, 3), next_, torch.zeros(2, dtype=torch.bool)
        )

        assert reward[0, 2].item() < 0

    def test_other_ships_unaffected(self, cfg):
        prev = _make_4ship_state(cfg)
        next_ = _make_4ship_state(cfg)
        next_.ship_health[0, 2] = prev.ship_health[0, 2] - 15.0

        r = EnemyDamageReward(weight=1.0)
        reward = r.compute(
            prev, torch.zeros(2, 4, 3), next_, torch.zeros(2, dtype=torch.bool)
        )

        assert reward[0, 0].item() == pytest.approx(0.0)
        assert reward[0, 1].item() == pytest.approx(0.0)
        assert reward[0, 3].item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Death rewards (ally/enemy split)
# ---------------------------------------------------------------------------


class TestAllyDeathReward:
    def test_dying_ship_gets_penalty(self, cfg):
        prev = _make_4ship_state(cfg)
        next_ = _make_4ship_state(cfg)
        next_.ship_alive[0, 0] = False

        r = AllyDeathReward(weight=5.0)
        reward = r.compute(
            prev, torch.zeros(2, 4, 3), next_, torch.zeros(2, dtype=torch.bool)
        )

        assert reward[0, 0].item() == pytest.approx(-1.0, rel=1e-5)

    def test_surviving_ships_get_zero(self, cfg):
        prev = _make_4ship_state(cfg)
        next_ = _make_4ship_state(cfg)
        next_.ship_alive[0, 0] = False

        r = AllyDeathReward(weight=5.0)
        reward = r.compute(
            prev, torch.zeros(2, 4, 3), next_, torch.zeros(2, dtype=torch.bool)
        )

        assert reward[0, 1].item() == pytest.approx(0.0)
        assert reward[0, 2].item() == pytest.approx(0.0)
        assert reward[0, 3].item() == pytest.approx(0.0)

    def test_no_death_gives_zero_reward(self, cfg):
        state = _make_4ship_state(cfg)

        r = AllyDeathReward(weight=5.0)
        reward = r.compute(
            state, torch.zeros(2, 4, 3), state, torch.zeros(2, dtype=torch.bool)
        )

        assert reward.abs().max().item() == 0.0


class TestEnemyDeathReward:
    def test_dying_enemy_gets_negative_own_reward(self, cfg):
        """EnemyDeathReward returns -1 for the dying ship itself.
        Lambda=-1 at PPO time inverts this to +1 benefit for enemies."""
        prev = _make_4ship_state(cfg)
        next_ = _make_4ship_state(cfg)
        next_.ship_alive[0, 2] = False

        r = EnemyDeathReward(weight=5.0)
        reward = r.compute(
            prev, torch.zeros(2, 4, 3), next_, torch.zeros(2, dtype=torch.bool)
        )

        assert reward[0, 2].item() == pytest.approx(-1.0, rel=1e-5)

    def test_other_ships_get_zero(self, cfg):
        prev = _make_4ship_state(cfg)
        next_ = _make_4ship_state(cfg)
        next_.ship_alive[0, 2] = False

        r = EnemyDeathReward(weight=5.0)
        reward = r.compute(
            prev, torch.zeros(2, 4, 3), next_, torch.zeros(2, dtype=torch.bool)
        )

        assert reward[0, 0].item() == pytest.approx(0.0)
        assert reward[0, 1].item() == pytest.approx(0.0)
        assert reward[0, 3].item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Win rewards (ally/enemy split)
# ---------------------------------------------------------------------------


class TestAllyWinReward:
    def test_winning_team_gets_positive_reward(self, cfg):
        """AllyWinReward gives +1 to each ship on the team that won."""
        prev = _make_4ship_state(cfg)
        next_ = _make_4ship_state(cfg)
        next_.ship_alive[0, 2] = False
        next_.ship_alive[0, 3] = False  # team 1 eliminated
        dones = torch.tensor([True, False], dtype=torch.bool)

        r = AllyWinReward(weight=1.0)
        reward = r.compute(prev, torch.zeros(2, 4, 3), next_, dones)

        assert reward[0, 0].item() == pytest.approx(1.0, rel=1e-5)
        assert reward[0, 1].item() == pytest.approx(1.0, rel=1e-5)

    def test_losing_team_gets_zero(self, cfg):
        """AllyWinReward gives 0 (not -1) to the losing team; lambda handles sign."""
        prev = _make_4ship_state(cfg)
        next_ = _make_4ship_state(cfg)
        next_.ship_alive[0, 2] = False
        next_.ship_alive[0, 3] = False
        dones = torch.tensor([True, False], dtype=torch.bool)

        r = AllyWinReward(weight=1.0)
        reward = r.compute(prev, torch.zeros(2, 4, 3), next_, dones)

        assert reward[0, 2].item() == pytest.approx(0.0)
        assert reward[0, 3].item() == pytest.approx(0.0)

    def test_non_terminal_gives_zero_reward(self, cfg):
        prev = _make_4ship_state(cfg)
        next_ = _make_4ship_state(cfg)
        dones = torch.zeros(2, dtype=torch.bool)

        r = AllyWinReward(weight=1.0)
        reward = r.compute(prev, torch.zeros(2, 4, 3), next_, dones)

        assert reward.abs().max().item() == 0.0


class TestEnemyWinReward:
    def test_winning_team_gets_positive_reward(self, cfg):
        """EnemyWinReward also gives +1 to winning-team ships.
        Lambda=-1 at PPO time means allies benefit when enemies get 0 here."""
        prev = _make_4ship_state(cfg)
        next_ = _make_4ship_state(cfg)
        next_.ship_alive[0, 2] = False
        next_.ship_alive[0, 3] = False
        dones = torch.tensor([True, False], dtype=torch.bool)

        r = EnemyWinReward(weight=1.0)
        reward = r.compute(prev, torch.zeros(2, 4, 3), next_, dones)

        assert reward[0, 0].item() == pytest.approx(1.0, rel=1e-5)
        assert reward[0, 1].item() == pytest.approx(1.0, rel=1e-5)

    def test_losing_team_gets_zero(self, cfg):
        prev = _make_4ship_state(cfg)
        next_ = _make_4ship_state(cfg)
        next_.ship_alive[0, 2] = False
        next_.ship_alive[0, 3] = False
        dones = torch.tensor([True, False], dtype=torch.bool)

        r = EnemyWinReward(weight=1.0)
        reward = r.compute(prev, torch.zeros(2, 4, 3), next_, dones)

        assert reward[0, 2].item() == pytest.approx(0.0)
        assert reward[0, 3].item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Shaping rewards
# ---------------------------------------------------------------------------


def _facing_state(cfg):
    """Two ships pointing at each other, 100 units apart."""
    state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
    state.ship_team_id[0, 0] = 0
    state.ship_team_id[0, 1] = 1
    state.ship_pos[0, 0] = 0.0 + 0j
    state.ship_pos[0, 1] = 100.0 + 0j
    state.ship_attitude[0, 0] = 1.0 + 0j  # team-0 pointing toward team-1
    state.ship_attitude[0, 1] = -1.0 + 0j  # team-1 pointing toward team-0
    return state


class TestFacingReward:
    def test_both_ships_get_positive_facing_reward(self, cfg):
        state = _facing_state(cfg)
        comp = FacingReward(facing_weight=1.0, radius=500.0, world_size=cfg.world_size)
        reward = comp.compute(
            state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool)
        )

        assert reward[0, 0].item() > 0, "team-0 should get positive facing reward"
        assert reward[0, 1].item() > 0, "team-1 should get positive facing reward"

    def test_ship_not_facing_enemy_gets_lower_reward(self, cfg):
        state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
        state.ship_team_id[0, 0] = 0
        state.ship_team_id[0, 1] = 1
        state.ship_pos[0, 0] = 0.0 + 0j
        state.ship_pos[0, 1] = 100.0 + 0j

        comp = FacingReward(facing_weight=1.0, radius=500.0, world_size=cfg.world_size)

        state.ship_attitude[0, 0] = 1.0 + 0j
        r_facing = comp.compute(
            state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool)
        )

        state.ship_attitude[0, 0] = -1.0 + 0j
        r_away = comp.compute(
            state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool)
        )

        assert r_facing[0, 0].item() > r_away[0, 0].item()


class TestClosingSpeedReward:
    def test_moving_toward_enemy_gives_positive_reward(self, cfg):
        state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
        state.ship_team_id[0, 0] = 0
        state.ship_team_id[0, 1] = 1
        state.ship_pos[0, 0] = 0.0 + 0j
        state.ship_pos[0, 1] = 100.0 + 0j
        state.ship_vel[0, 0] = 50.0 + 0j  # moving east toward enemy

        comp = ClosingSpeedReward(
            closing_speed_weight=1.0, world_size=cfg.world_size, max_speed=cfg.max_speed
        )
        reward = comp.compute(
            state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool)
        )

        assert reward[0, 0].item() > 0

    def test_moving_away_from_enemy_gives_zero_reward(self, cfg):
        state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
        state.ship_team_id[0, 0] = 0
        state.ship_team_id[0, 1] = 1
        state.ship_pos[0, 0] = 0.0 + 0j
        state.ship_pos[0, 1] = 100.0 + 0j
        state.ship_vel[0, 0] = -50.0 + 0j  # moving away

        comp = ClosingSpeedReward(
            closing_speed_weight=1.0, world_size=cfg.world_size, max_speed=cfg.max_speed
        )
        reward = comp.compute(
            state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool)
        )

        assert reward[0, 0].item() == 0.0

    def test_dead_ship_gets_zero_reward(self, cfg):
        state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
        state.ship_team_id[0, 0] = 0
        state.ship_team_id[0, 1] = 1
        state.ship_alive[0, 0] = False
        state.ship_pos[0, 0] = 0.0 + 0j
        state.ship_pos[0, 1] = 100.0 + 0j
        state.ship_vel[0, 0] = 50.0 + 0j

        comp = ClosingSpeedReward(
            closing_speed_weight=1.0, world_size=cfg.world_size, max_speed=cfg.max_speed
        )
        reward = comp.compute(
            state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool)
        )

        assert reward[0, 0].item() == 0.0


# ---------------------------------------------------------------------------
# Kill rewards
# ---------------------------------------------------------------------------


def _kill_state(cfg):
    """2v2 (ships 0,1 vs ships 2,3). All alive, all at max health."""
    state = make_state(num_envs=1, max_ships=4, ship_config=cfg)
    state.ship_team_id[0, 0] = 0
    state.ship_team_id[0, 1] = 0
    state.ship_team_id[0, 2] = 1
    state.ship_team_id[0, 3] = 1
    return state


class TestKillShotReward:
    def test_sole_damage_dealer_gets_full_credit(self, cfg):
        """Ship 0 deals all step damage to dying ship 2; ship 0 gets full kill credit."""
        prev = _kill_state(cfg)
        next_ = _kill_state(cfg)
        next_.ship_alive[0, 2] = False
        next_.damage_matrix[0, 0, 2] = 30.0  # ship 0 dealt 30 to ship 2

        r = KillShotReward(weight=1.0)
        reward = r.compute(
            prev, torch.zeros(1, 4, 3), next_, torch.zeros(1, dtype=torch.bool)
        )

        assert reward[0, 0].item() == pytest.approx(1.0)
        assert reward[0, 1].item() == pytest.approx(0.0)
        assert reward[0, 2].item() == pytest.approx(0.0)
        assert reward[0, 3].item() == pytest.approx(0.0)

    def test_proportional_split_between_two_shooters(self, cfg):
        """Ships 0 and 1 both hit dying ship 2 this step; credit splits proportionally."""
        prev = _kill_state(cfg)
        next_ = _kill_state(cfg)
        next_.ship_alive[0, 2] = False
        next_.damage_matrix[0, 0, 2] = 10.0
        next_.damage_matrix[0, 1, 2] = 30.0  # ship 1 dealt 3× more

        r = KillShotReward(weight=1.0)
        reward = r.compute(
            prev, torch.zeros(1, 4, 3), next_, torch.zeros(1, dtype=torch.bool)
        )

        assert reward[0, 0].item() == pytest.approx(0.25, rel=1e-5)
        assert reward[0, 1].item() == pytest.approx(0.75, rel=1e-5)

    def test_equal_damage_splits_evenly(self, cfg):
        """Ships 0 and 1 dealt equal damage to dying ship 2; each gets 0.5."""
        prev = _kill_state(cfg)
        next_ = _kill_state(cfg)
        next_.ship_alive[0, 2] = False
        next_.damage_matrix[0, 0, 2] = 20.0
        next_.damage_matrix[0, 1, 2] = 20.0

        r = KillShotReward(weight=1.0)
        reward = r.compute(
            prev, torch.zeros(1, 4, 3), next_, torch.zeros(1, dtype=torch.bool)
        )

        assert reward[0, 0].item() == pytest.approx(0.5, rel=1e-5)
        assert reward[0, 1].item() == pytest.approx(0.5, rel=1e-5)

    def test_no_death_gives_zero_reward(self, cfg):
        state = _kill_state(cfg)

        r = KillShotReward(weight=1.0)
        reward = r.compute(
            state, torch.zeros(1, 4, 3), state, torch.zeros(1, dtype=torch.bool)
        )

        assert reward.abs().max().item() == 0.0

    def test_friendly_fire_gives_penalty(self, cfg):
        """Ship 0 dealt all damage to dying teammate ship 1; ship 0 gets -1 penalty."""
        prev = _kill_state(cfg)
        next_ = _kill_state(cfg)
        next_.ship_alive[0, 1] = False  # teammate of ship 0 died
        next_.damage_matrix[0, 0, 1] = 40.0  # ship 0 caused the death

        r = KillShotReward(weight=1.0)
        reward = r.compute(
            prev, torch.zeros(1, 4, 3), next_, torch.zeros(1, dtype=torch.bool)
        )

        assert reward[0, 0].item() == pytest.approx(-1.0)  # friendly fire penalty
        assert reward[0, 1].item() == pytest.approx(0.0)
        assert reward[0, 2].item() == pytest.approx(0.0)
        assert reward[0, 3].item() == pytest.approx(0.0)


class TestKillAssistReward:
    def test_sole_damage_dealer_gets_full_credit(self, cfg):
        """Ship 0 is the only one that damaged dying ship 2; gets 1.0 assist credit."""
        prev = _kill_state(cfg)
        next_ = _kill_state(cfg)
        next_.ship_alive[0, 2] = False
        next_.cumulative_damage_matrix[0, 0, 2] = 50.0

        r = KillAssistReward(weight=1.0)
        reward = r.compute(
            prev, torch.zeros(1, 4, 3), next_, torch.zeros(1, dtype=torch.bool)
        )

        assert reward[0, 0].item() == pytest.approx(1.0)
        assert reward[0, 1].item() == pytest.approx(0.0)

    def test_proportional_split_between_two_damage_dealers(self, cfg):
        """Ships 0 and 1 both damaged dying ship 2; credit splits 25%/75%."""
        prev = _kill_state(cfg)
        next_ = _kill_state(cfg)
        next_.ship_alive[0, 2] = False
        next_.cumulative_damage_matrix[0, 0, 2] = 25.0
        next_.cumulative_damage_matrix[0, 1, 2] = 75.0

        r = KillAssistReward(weight=1.0)
        reward = r.compute(
            prev, torch.zeros(1, 4, 3), next_, torch.zeros(1, dtype=torch.bool)
        )

        assert reward[0, 0].item() == pytest.approx(0.25, rel=1e-5)
        assert reward[0, 1].item() == pytest.approx(0.75, rel=1e-5)

    def test_no_death_gives_zero_reward(self, cfg):
        state = _kill_state(cfg)

        r = KillAssistReward(weight=1.0)
        reward = r.compute(
            state, torch.zeros(1, 4, 3), state, torch.zeros(1, dtype=torch.bool)
        )

        assert reward.abs().max().item() == 0.0

    def test_multiple_kills_accumulate_per_ship(self, cfg):
        """If two enemies die, a ship that damaged both accumulates credit for each."""
        prev = _kill_state(cfg)
        next_ = _kill_state(cfg)
        next_.ship_alive[0, 2] = False
        next_.ship_alive[0, 3] = False
        # Ship 0 was the sole damage dealer for both kills
        next_.cumulative_damage_matrix[0, 0, 2] = 40.0
        next_.cumulative_damage_matrix[0, 0, 3] = 60.0

        r = KillAssistReward(weight=1.0)
        reward = r.compute(
            prev, torch.zeros(1, 4, 3), next_, torch.zeros(1, dtype=torch.bool)
        )

        assert reward[0, 0].item() == pytest.approx(2.0)  # 1.0 per kill


# ---------------------------------------------------------------------------
# Local damage rewards
# ---------------------------------------------------------------------------


class TestLocalDamageTakenReward:
    def test_damaged_ship_gets_negative_reward(self, cfg):
        prev = _make_4ship_state(cfg)
        next_ = _make_4ship_state(cfg)
        next_.ship_health[0, 0] = prev.ship_health[0, 0] - 10.0

        r = LocalDamageTakenReward(weight=1.0)
        reward = r.compute(
            prev, torch.zeros(2, 4, 3), next_, torch.zeros(2, dtype=torch.bool)
        )

        assert reward[0, 0].item() == pytest.approx(-10.0)

    def test_undamaged_ships_get_zero(self, cfg):
        prev = _make_4ship_state(cfg)
        next_ = _make_4ship_state(cfg)
        next_.ship_health[0, 0] = prev.ship_health[0, 0] - 10.0

        r = LocalDamageTakenReward(weight=1.0)
        reward = r.compute(
            prev, torch.zeros(2, 4, 3), next_, torch.zeros(2, dtype=torch.bool)
        )

        assert reward[0, 1].item() == pytest.approx(0.0)
        assert reward[0, 2].item() == pytest.approx(0.0)
        assert reward[0, 3].item() == pytest.approx(0.0)

    def test_dead_ship_gets_zero(self, cfg):
        """Dead ships receive no reward even if health dropped this step."""
        prev = _make_4ship_state(cfg)
        next_ = _make_4ship_state(cfg)
        next_.ship_health[0, 0] = 0.0
        next_.ship_alive[0, 0] = False

        r = LocalDamageTakenReward(weight=1.0)
        reward = r.compute(
            prev, torch.zeros(2, 4, 3), next_, torch.zeros(2, dtype=torch.bool)
        )

        assert reward[0, 0].item() == pytest.approx(0.0)

    def test_zero_damage_gives_zero_reward(self, cfg):
        state = _make_4ship_state(cfg)

        r = LocalDamageTakenReward(weight=1.0)
        reward = r.compute(
            state, torch.zeros(2, 4, 3), state, torch.zeros(2, dtype=torch.bool)
        )

        assert reward.abs().max().item() == 0.0


class TestLocalDamageDealtEnemyReward:
    def test_ship_that_dealt_enemy_damage_gets_positive_reward(self, cfg):
        """Ship 0 dealt 20 damage to enemy ship 2; ship 0 gets +20."""
        state = _make_4ship_state(cfg)
        state.damage_matrix[0, 0, 2] = 20.0

        r = LocalDamageDealtEnemyReward(weight=1.0)
        reward = r.compute(
            state, torch.zeros(2, 4, 3), state, torch.zeros(2, dtype=torch.bool)
        )

        assert reward[0, 0].item() == pytest.approx(20.0)

    def test_ships_that_dealt_no_damage_get_zero(self, cfg):
        state = _make_4ship_state(cfg)
        state.damage_matrix[0, 0, 2] = 20.0

        r = LocalDamageDealtEnemyReward(weight=1.0)
        reward = r.compute(
            state, torch.zeros(2, 4, 3), state, torch.zeros(2, dtype=torch.bool)
        )

        assert reward[0, 1].item() == pytest.approx(0.0)
        assert reward[0, 2].item() == pytest.approx(0.0)
        assert reward[0, 3].item() == pytest.approx(0.0)

    def test_damage_to_multiple_enemies_accumulates(self, cfg):
        """Ship 0 dealt damage to both enemy ships; rewards sum."""
        state = _make_4ship_state(cfg)
        state.damage_matrix[0, 0, 2] = 15.0
        state.damage_matrix[0, 0, 3] = 10.0

        r = LocalDamageDealtEnemyReward(weight=1.0)
        reward = r.compute(
            state, torch.zeros(2, 4, 3), state, torch.zeros(2, dtype=torch.bool)
        )

        assert reward[0, 0].item() == pytest.approx(25.0)

    def test_friendly_fire_ignored(self, cfg):
        """Damage dealt to a teammate must not contribute to enemy damage reward."""
        state = _make_4ship_state(cfg)
        state.damage_matrix[0, 0, 1] = 30.0  # ship 0 hit ally ship 1

        r = LocalDamageDealtEnemyReward(weight=1.0)
        reward = r.compute(
            state, torch.zeros(2, 4, 3), state, torch.zeros(2, dtype=torch.bool)
        )

        assert reward[0, 0].item() == pytest.approx(0.0)

    def test_dead_ship_gets_zero(self, cfg):
        state = _make_4ship_state(cfg)
        state.damage_matrix[0, 0, 2] = 20.0
        state.ship_alive[0, 0] = False

        r = LocalDamageDealtEnemyReward(weight=1.0)
        reward = r.compute(
            state, torch.zeros(2, 4, 3), state, torch.zeros(2, dtype=torch.bool)
        )

        assert reward[0, 0].item() == pytest.approx(0.0)


class TestLocalDamageDealtAllyReward:
    def test_friendly_fire_gives_negative_reward(self, cfg):
        """Ship 0 dealt 30 damage to ally ship 1; ship 0 gets -30."""
        state = _make_4ship_state(cfg)
        state.damage_matrix[0, 0, 1] = 30.0

        r = LocalDamageDealtAllyReward(weight=1.0)
        reward = r.compute(
            state, torch.zeros(2, 4, 3), state, torch.zeros(2, dtype=torch.bool)
        )

        assert reward[0, 0].item() == pytest.approx(-30.0)

    def test_enemy_damage_ignored(self, cfg):
        """Damage to enemies must not contribute to the friendly-fire penalty."""
        state = _make_4ship_state(cfg)
        state.damage_matrix[0, 0, 2] = 20.0  # ship 0 hit enemy ship 2

        r = LocalDamageDealtAllyReward(weight=1.0)
        reward = r.compute(
            state, torch.zeros(2, 4, 3), state, torch.zeros(2, dtype=torch.bool)
        )

        assert reward[0, 0].item() == pytest.approx(0.0)

    def test_no_friendly_fire_gives_zero(self, cfg):
        state = _make_4ship_state(cfg)

        r = LocalDamageDealtAllyReward(weight=1.0)
        reward = r.compute(
            state, torch.zeros(2, 4, 3), state, torch.zeros(2, dtype=torch.bool)
        )

        assert reward.abs().max().item() == 0.0

    def test_dead_ship_gets_zero(self, cfg):
        state = _make_4ship_state(cfg)
        state.damage_matrix[0, 0, 1] = 30.0
        state.ship_alive[0, 0] = False

        r = LocalDamageDealtAllyReward(weight=1.0)
        reward = r.compute(
            state, torch.zeros(2, 4, 3), state, torch.zeros(2, dtype=torch.bool)
        )

        assert reward[0, 0].item() == pytest.approx(0.0)


class TestLocalDeathReward:
    def test_dying_ship_gets_minus_one(self, cfg):
        """Fires on the exact step of death using just_died, not next_state.ship_alive."""
        prev = _make_4ship_state(cfg)
        next_ = _make_4ship_state(cfg)
        next_.ship_alive[0, 0] = False

        r = LocalDeathReward(weight=1.0)
        reward = r.compute(
            prev, torch.zeros(2, 4, 3), next_, torch.zeros(2, dtype=torch.bool)
        )

        assert reward[0, 0].item() == pytest.approx(-1.0)

    def test_surviving_ships_get_zero(self, cfg):
        prev = _make_4ship_state(cfg)
        next_ = _make_4ship_state(cfg)
        next_.ship_alive[0, 0] = False

        r = LocalDeathReward(weight=1.0)
        reward = r.compute(
            prev, torch.zeros(2, 4, 3), next_, torch.zeros(2, dtype=torch.bool)
        )

        assert reward[0, 1].item() == pytest.approx(0.0)
        assert reward[0, 2].item() == pytest.approx(0.0)
        assert reward[0, 3].item() == pytest.approx(0.0)

    def test_already_dead_ship_does_not_fire_again(self, cfg):
        """A ship dead in both prev and next (subsequent steps) must not re-trigger."""
        prev = _make_4ship_state(cfg)
        next_ = _make_4ship_state(cfg)
        prev.ship_alive[0, 0] = False  # already dead last step
        next_.ship_alive[0, 0] = False  # still dead

        r = LocalDeathReward(weight=1.0)
        reward = r.compute(
            prev, torch.zeros(2, 4, 3), next_, torch.zeros(2, dtype=torch.bool)
        )

        assert reward[0, 0].item() == pytest.approx(0.0)

    def test_no_deaths_gives_zero_reward(self, cfg):
        state = _make_4ship_state(cfg)

        r = LocalDeathReward(weight=1.0)
        reward = r.compute(
            state, torch.zeros(2, 4, 3), state, torch.zeros(2, dtype=torch.bool)
        )

        assert reward.abs().max().item() == 0.0
