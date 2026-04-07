"""Unit tests for reward components.

Tests reward signs and magnitudes under controlled scenarios.
No compute_rewards() — per-ship signals are tested directly; zero-sum accounting
(lambda aggregation) lives in the PPO trainer, not the reward components.
"""

import pytest
import torch

from boost_and_broadside.config import ShipConfig, RewardConfig
from boost_and_broadside.env.rewards import (
    DamageReward,
    DeathReward,
    VictoryReward,
    PositioningReward,
    FacingReward,
    ExposureReward,
    SpeedRangeReward,
    PowerRangeReward,
    ClosingSpeedReward,
    TurnRateReward,
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
        damage_weight=1.0,
        death_weight=5.0,
        victory_weight=10.0,
        enemy_neg_lambda_components=frozenset(
            {"damage", "death", "victory", "exposure"}
        ),
        positioning_weight=1.0,
        positioning_radius=500.0,
        facing_weight=1.0,
        exposure_weight=1.0,
        proximity_weight=1.0,
        proximity_radius=500.0,
        closing_speed_weight=1.0,
        turn_rate_weight=1.0,
        power_range_weight=1.0,
        power_range_lo=0.2,
        power_range_hi=0.8,
        speed_range_weight=1.0,
        speed_range_lo=40.0,
        speed_range_hi=120.0,
        shoot_quality_weight=1.0,
        shoot_quality_radius=200.0,
    )


def _make_4ship_state(cfg):
    """Team 0: ships 0,1. Team 1: ships 2,3."""
    state = make_state(num_envs=2, max_ships=4, ship_config=cfg)
    state.ship_team_id[:, 0] = 0
    state.ship_team_id[:, 1] = 0
    state.ship_team_id[:, 2] = 1
    state.ship_team_id[:, 3] = 1
    return state


class TestRewardComponentNames:
    def test_k_equals_12(self):
        assert len(REWARD_COMPONENT_NAMES) == 12

    def test_damage_is_index_0(self):
        assert REWARD_COMPONENT_NAMES[0] == "damage"

    def test_death_is_index_1(self):
        assert REWARD_COMPONENT_NAMES[1] == "death"

    def test_victory_is_index_2(self):
        assert REWARD_COMPONENT_NAMES[2] == "victory"

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

    def test_damage_component_correct_index(self, cfg):
        """Damage taken populates slot k=0 (damage index) in the output."""
        prev = _make_4ship_state(cfg)
        next_ = _make_4ship_state(cfg)
        next_.ship_health[0, 0] = prev.ship_health[0, 0] - 10.0
        components = [DamageReward(damage_weight=1.0)]
        result = compute_per_component_rewards(
            components,
            prev,
            torch.zeros(2, 4, 3),
            next_,
            torch.zeros(2, dtype=torch.bool),
        )
        damage_k = REWARD_COMPONENT_NAMES.index("damage")
        assert result[0, 0, damage_k].item() < 0


class TestDamageReward:
    def test_damaged_ship_gets_negative_reward(self, cfg):
        """The ship that takes damage gets a negative reward."""
        prev = _make_4ship_state(cfg)
        next_ = _make_4ship_state(cfg)
        next_.ship_health[0, 0] = prev.ship_health[0, 0] - 10.0

        r = DamageReward(damage_weight=1.0)
        reward = r.compute(
            prev, torch.zeros(2, 4, 3), next_, torch.zeros(2, dtype=torch.bool)
        )

        assert reward[0, 0].item() < 0

    def test_undamaged_ships_get_zero_reward(self, cfg):
        """Ships that did not take damage get zero reward (including enemies)."""
        prev = _make_4ship_state(cfg)
        next_ = _make_4ship_state(cfg)
        next_.ship_health[0, 0] = prev.ship_health[0, 0] - 10.0

        r = DamageReward(damage_weight=1.0)
        reward = r.compute(
            prev, torch.zeros(2, 4, 3), next_, torch.zeros(2, dtype=torch.bool)
        )

        # Ships 1, 2, 3 did not take damage — zero reward each
        assert reward[0, 1].item() == pytest.approx(0.0)
        assert reward[0, 2].item() == pytest.approx(0.0)
        assert reward[0, 3].item() == pytest.approx(0.0)

    def test_enemy_taking_damage_gives_enemy_negative_reward(self, cfg):
        """When an enemy ship takes damage, only that ship gets a negative reward."""
        prev = _make_4ship_state(cfg)
        next_ = _make_4ship_state(cfg)
        next_.ship_health[0, 2] = (
            prev.ship_health[0, 2] - 20.0
        )  # ship 2 (enemy) took damage

        r = DamageReward(damage_weight=1.0)
        reward = r.compute(
            prev, torch.zeros(2, 4, 3), next_, torch.zeros(2, dtype=torch.bool)
        )

        assert reward[0, 2].item() < 0  # ship 2 itself gets penalised
        assert reward[0, 0].item() == pytest.approx(0.0)  # allied ships unaffected

    def test_zero_damage_gives_zero_reward(self, cfg):
        """No damage → zero reward from damage component."""
        prev = _make_4ship_state(cfg)
        next_ = _make_4ship_state(cfg)

        r = DamageReward(damage_weight=1.0)
        reward = r.compute(
            prev, torch.zeros(2, 4, 3), next_, torch.zeros(2, dtype=torch.bool)
        )

        assert reward.abs().max().item() == 0.0

    def test_damage_weight_exposed_via_property(self, cfg):
        """damage_weight is accessible via the weight property (used by lambda aggregation)."""
        r1 = DamageReward(damage_weight=1.0)
        r2 = DamageReward(damage_weight=2.0)
        assert r1.weight == pytest.approx(1.0)
        assert r2.weight == pytest.approx(2.0)


class TestDeathReward:
    def test_dying_ship_gets_penalty(self, cfg):
        """A ship that just died should receive a negative reward."""
        prev = _make_4ship_state(cfg)
        next_ = _make_4ship_state(cfg)
        next_.ship_alive[0, 0] = False  # ship 0 died

        r = DeathReward(death_weight=5.0)
        reward = r.compute(
            prev, torch.zeros(2, 4, 3), next_, torch.zeros(2, dtype=torch.bool)
        )

        assert reward[0, 0].item() == pytest.approx(-1.0, rel=1e-5)

    def test_surviving_ships_get_zero_death_reward(self, cfg):
        """When ship 0 dies, surviving ships (including enemies) get zero death reward."""
        prev = _make_4ship_state(cfg)
        next_ = _make_4ship_state(cfg)
        next_.ship_alive[0, 0] = False

        r = DeathReward(death_weight=5.0)
        reward = r.compute(
            prev, torch.zeros(2, 4, 3), next_, torch.zeros(2, dtype=torch.bool)
        )

        assert reward[0, 1].item() == pytest.approx(0.0)
        assert reward[0, 2].item() == pytest.approx(0.0)
        assert reward[0, 3].item() == pytest.approx(0.0)

    def test_enemy_dying_gets_penalty(self, cfg):
        """An enemy ship that dies receives its own death penalty (no kill bonus for allies)."""
        prev = _make_4ship_state(cfg)
        next_ = _make_4ship_state(cfg)
        next_.ship_alive[0, 2] = False  # ship 2 (team 1) died

        r = DeathReward(death_weight=5.0)
        reward = r.compute(
            prev, torch.zeros(2, 4, 3), next_, torch.zeros(2, dtype=torch.bool)
        )

        assert reward[0, 2].item() == pytest.approx(-1.0, rel=1e-5)
        assert reward[0, 0].item() == pytest.approx(0.0)  # no kill bonus for allies

    def test_no_death_gives_zero_reward(self, cfg):
        """No ships died → zero death reward."""
        prev = _make_4ship_state(cfg)
        next_ = _make_4ship_state(cfg)

        r = DeathReward(death_weight=5.0)
        reward = r.compute(
            prev, torch.zeros(2, 4, 3), next_, torch.zeros(2, dtype=torch.bool)
        )

        assert reward.abs().max().item() == 0.0


class TestVictoryReward:
    def test_winning_team_gets_positive_reward(self, cfg):
        """When team 1 is eliminated, team-0 ships get +victory_weight."""
        prev = _make_4ship_state(cfg)
        next_ = _make_4ship_state(cfg)
        next_.ship_alive[0, 2] = False
        next_.ship_alive[0, 3] = False  # team 1 eliminated
        dones = torch.tensor([True, False], dtype=torch.bool)

        r = VictoryReward(victory_weight=10.0)
        reward = r.compute(prev, torch.zeros(2, 4, 3), next_, dones)

        assert reward[0, 0].item() == pytest.approx(1.0, rel=1e-5)
        assert reward[0, 1].item() == pytest.approx(1.0, rel=1e-5)

    def test_losing_team_gets_negative_reward(self, cfg):
        """Losing team ships get -victory_weight from their own perspective."""
        prev = _make_4ship_state(cfg)
        next_ = _make_4ship_state(cfg)
        next_.ship_alive[0, 2] = False
        next_.ship_alive[0, 3] = False
        dones = torch.tensor([True, False], dtype=torch.bool)

        r = VictoryReward(victory_weight=10.0)
        reward = r.compute(prev, torch.zeros(2, 4, 3), next_, dones)

        assert reward[0, 2].item() == pytest.approx(-1.0, rel=1e-5)
        assert reward[0, 3].item() == pytest.approx(-1.0, rel=1e-5)

    def test_non_terminal_gives_zero_reward(self, cfg):
        """No victory reward when done=False."""
        prev = _make_4ship_state(cfg)
        next_ = _make_4ship_state(cfg)
        dones = torch.zeros(2, dtype=torch.bool)

        r = VictoryReward(victory_weight=10.0)
        reward = r.compute(prev, torch.zeros(2, 4, 3), next_, dones)

        assert reward.abs().max().item() == 0.0


class TestPositioningReward:
    def test_pointing_at_enemy_gives_positive_reward(self, cfg):
        """A ship pointing directly at an enemy within range should get positive reward."""
        state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
        state.ship_team_id[0, 0] = 0
        state.ship_team_id[0, 1] = 1

        state.ship_pos[0, 0] = 0.0 + 0j
        state.ship_pos[0, 1] = 100.0 + 0j
        state.ship_attitude[0, 0] = 1.0 + 0j  # pointing East = toward ship 1

        r = PositioningReward(
            positioning_weight=1.0, positioning_radius=500.0, world_size=cfg.world_size
        )
        reward = r.compute(
            state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool)
        )

        assert reward[0, 0].item() > 0

    def test_ships_outside_radius_get_zero_reward(self, cfg):
        """Ships far beyond positioning_radius should get zero reward."""
        state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
        state.ship_team_id[0, 0] = 0
        state.ship_team_id[0, 1] = 1

        state.ship_pos[0, 0] = 0.0 + 0j
        state.ship_pos[0, 1] = 200.0 + 0j

        r = PositioningReward(
            positioning_weight=1.0, positioning_radius=100.0, world_size=cfg.world_size
        )
        reward = r.compute(
            state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool)
        )

        assert abs(reward[0, 0].item()) < 1e-6

    def test_dead_ships_get_zero_positioning_reward(self, cfg):
        """Dead ships must receive zero positioning reward."""
        state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
        state.ship_team_id[0, 0] = 0
        state.ship_team_id[0, 1] = 1
        state.ship_alive[0, 0] = False

        state.ship_pos[0, 0] = 0.0 + 0j
        state.ship_pos[0, 1] = 100.0 + 0j
        state.ship_attitude[0, 0] = 1.0 + 0j

        r = PositioningReward(
            positioning_weight=1.0, positioning_radius=500.0, world_size=cfg.world_size
        )
        reward = r.compute(
            state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool)
        )

        assert reward[0, 0].item() == 0.0


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
    """FacingReward must incentivise every ship to face its enemies."""

    def test_both_ships_get_positive_facing_reward(self, cfg):
        """Both ships pointing at each other should get positive facing reward directly."""
        state = _facing_state(cfg)
        comp = FacingReward(facing_weight=1.0, radius=500.0, world_size=cfg.world_size)
        reward = comp.compute(
            state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool)
        )
        assert reward[0, 0].item() > 0, "team-0 should get positive facing reward"
        assert reward[0, 1].item() > 0, "team-1 should get positive facing reward"

    def test_ship_not_facing_enemy_gets_lower_reward(self, cfg):
        """Ship pointing away from enemy gets less (or zero) reward than one facing it."""
        state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
        state.ship_team_id[0, 0] = 0
        state.ship_team_id[0, 1] = 1
        state.ship_pos[0, 0] = 0.0 + 0j
        state.ship_pos[0, 1] = 100.0 + 0j

        comp = FacingReward(facing_weight=1.0, radius=500.0, world_size=cfg.world_size)

        state.ship_attitude[0, 0] = 1.0 + 0j  # facing enemy
        r_facing = comp.compute(
            state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool)
        )

        state.ship_attitude[0, 0] = -1.0 + 0j  # facing away
        r_away = comp.compute(
            state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool)
        )

        assert r_facing[0, 0].item() > r_away[0, 0].item()


class TestExposureReward:
    """ExposureReward must penalise each ship for being in enemies' crosshairs."""

    def test_both_ships_get_negative_exposure_reward(self, cfg):
        """Both ships facing each other = both exposed; both get negative reward."""
        state = _facing_state(cfg)
        comp = ExposureReward(
            exposure_weight=1.0, radius=500.0, world_size=cfg.world_size
        )
        reward = comp.compute(
            state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool)
        )
        assert reward[0, 0].item() < 0, "team-0 should be penalised for being exposed"
        assert reward[0, 1].item() < 0, "team-1 should be penalised for being exposed"

    def test_ship_not_in_crosshairs_gets_less_penalty(self, cfg):
        """Ship not aimed at gets less exposure penalty than one in direct crosshairs."""
        state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
        state.ship_team_id[0, 0] = 0
        state.ship_team_id[0, 1] = 1
        # Ship 0 at origin, ship 1 at (100, 0). Ship 0 is to the WEST of ship 1.
        state.ship_pos[0, 0] = 0.0 + 0j
        state.ship_pos[0, 1] = 100.0 + 0j

        comp = ExposureReward(
            exposure_weight=1.0, radius=500.0, world_size=cfg.world_size
        )

        state.ship_attitude[0, 1] = 1.0 + 0j  # ship 1 pointing east = away from ship 0
        r_safe = comp.compute(
            state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool)
        )

        state.ship_attitude[0, 1] = -1.0 + 0j  # ship 1 pointing west = toward ship 0
        r_exposed = comp.compute(
            state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool)
        )

        assert r_safe[0, 0].item() >= r_exposed[0, 0].item()


class TestSpeedRangeReward:
    """SpeedRangeReward must reward each ship for staying in the target range."""

    def test_both_ships_positive_when_in_range(self, cfg):
        state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
        state.ship_team_id[0, 0] = 0
        state.ship_team_id[0, 1] = 1
        state.ship_vel[0, 0] = 80.0 + 0j
        state.ship_vel[0, 1] = 80.0 + 0j

        comp = SpeedRangeReward(
            speed_range_weight=1.0, speed_range_lo=40.0, speed_range_hi=120.0
        )
        reward = comp.compute(
            state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool)
        )
        assert reward[0, 0].item() > 0, "team-0 should be rewarded for in-range speed"
        assert reward[0, 1].item() > 0, "team-1 should be rewarded for in-range speed"

    def test_out_of_range_gives_less_reward_than_in_range(self, cfg):
        """The trapezoidal function gives less reward when speed is outside [lo, hi]."""
        comp = SpeedRangeReward(
            speed_range_weight=1.0, speed_range_lo=40.0, speed_range_hi=120.0
        )

        in_state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
        in_state.ship_team_id[0, 0] = 0
        in_state.ship_team_id[0, 1] = 1
        in_state.ship_vel[0, 0] = 80.0 + 0j
        in_state.ship_vel[0, 1] = 80.0 + 0j

        out_state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
        out_state.ship_team_id[0, 0] = 0
        out_state.ship_team_id[0, 1] = 1
        out_state.ship_vel[0, 0] = 0.0 + 0j
        out_state.ship_vel[0, 1] = 0.0 + 0j

        r_in = comp.compute(
            in_state, torch.zeros(1, 2, 3), in_state, torch.zeros(1, dtype=torch.bool)
        )
        r_out = comp.compute(
            out_state, torch.zeros(1, 2, 3), out_state, torch.zeros(1, dtype=torch.bool)
        )

        assert r_in[0, 0].item() > r_out[0, 0].item()
        assert r_in[0, 1].item() > r_out[0, 1].item()


class TestPowerRangeReward:
    """PowerRangeReward must reward each ship for keeping power in range."""

    def test_both_ships_positive_when_in_range(self, cfg):
        state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
        state.ship_team_id[0, 0] = 0
        state.ship_team_id[0, 1] = 1
        state.ship_power[0, 0] = 50.0
        state.ship_power[0, 1] = 50.0

        comp = PowerRangeReward(
            power_range_weight=1.0,
            power_range_lo=0.2,
            power_range_hi=0.8,
            max_power=cfg.max_power,
        )
        reward = comp.compute(
            state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool)
        )
        assert reward[0, 0].item() > 0, "team-0 should be rewarded for in-range power"
        assert reward[0, 1].item() > 0, "team-1 should be rewarded for in-range power"


class TestClosingSpeedReward:
    def test_moving_toward_enemy_gives_positive_reward(self, cfg):
        """Ship moving directly toward enemy should get positive closing speed reward."""
        state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
        state.ship_team_id[0, 0] = 0
        state.ship_team_id[0, 1] = 1
        state.ship_pos[0, 0] = 0.0 + 0j
        state.ship_pos[0, 1] = 100.0 + 0j
        state.ship_vel[0, 0] = 50.0 + 0j  # moving east
        state.ship_vel[0, 1] = 0.0 + 0j

        comp = ClosingSpeedReward(closing_speed_weight=1.0, world_size=cfg.world_size)
        reward = comp.compute(
            state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool)
        )

        assert reward[0, 0].item() > 0

    def test_moving_away_from_enemy_gives_zero_reward(self, cfg):
        """Ship moving away from enemy gets zero (clamped) reward."""
        state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
        state.ship_team_id[0, 0] = 0
        state.ship_team_id[0, 1] = 1
        state.ship_pos[0, 0] = 0.0 + 0j
        state.ship_pos[0, 1] = 100.0 + 0j
        state.ship_vel[0, 0] = -50.0 + 0j  # moving away

        comp = ClosingSpeedReward(closing_speed_weight=1.0, world_size=cfg.world_size)
        reward = comp.compute(
            state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool)
        )

        assert reward[0, 0].item() == 0.0

    def test_dead_ship_gets_zero_reward(self, cfg):
        """Dead ships must receive zero closing speed reward."""
        state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
        state.ship_team_id[0, 0] = 0
        state.ship_team_id[0, 1] = 1
        state.ship_alive[0, 0] = False
        state.ship_pos[0, 0] = 0.0 + 0j
        state.ship_pos[0, 1] = 100.0 + 0j
        state.ship_vel[0, 0] = 50.0 + 0j

        comp = ClosingSpeedReward(closing_speed_weight=1.0, world_size=cfg.world_size)
        reward = comp.compute(
            state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool)
        )

        assert reward[0, 0].item() == 0.0


class TestTurnRateReward:
    def test_turning_toward_enemy_gives_positive_reward(self, cfg):
        """Ship rotating counterclockwise with enemy to the left gets positive reward."""
        state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
        state.ship_team_id[0, 0] = 0
        state.ship_team_id[0, 1] = 1
        state.ship_pos[0, 0] = 0.0 + 0j
        state.ship_pos[0, 1] = 0.0 + 100j  # enemy is directly north
        state.ship_attitude[0, 0] = 1.0 + 0j  # heading east
        state.ship_ang_vel[0, 0] = 1.0  # counterclockwise = toward enemy

        comp = TurnRateReward(turn_rate_weight=1.0, world_size=cfg.world_size)
        reward = comp.compute(
            state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool)
        )

        assert reward[0, 0].item() > 0

    def test_turning_away_from_enemy_gives_negative_reward(self, cfg):
        """Ship rotating clockwise with enemy to the left gets negative reward."""
        state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
        state.ship_team_id[0, 0] = 0
        state.ship_team_id[0, 1] = 1
        state.ship_pos[0, 0] = 0.0 + 0j
        state.ship_pos[0, 1] = 0.0 + 100j
        state.ship_attitude[0, 0] = 1.0 + 0j
        state.ship_ang_vel[0, 0] = -1.0  # clockwise = away from enemy

        comp = TurnRateReward(turn_rate_weight=1.0, world_size=cfg.world_size)
        reward = comp.compute(
            state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool)
        )

        assert reward[0, 0].item() < 0

    def test_dead_ship_gets_zero_reward(self, cfg):
        """Dead ships must receive zero turn rate reward."""
        state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
        state.ship_team_id[0, 0] = 0
        state.ship_team_id[0, 1] = 1
        state.ship_alive[0, 0] = False
        state.ship_pos[0, 0] = 0.0 + 0j
        state.ship_pos[0, 1] = 0.0 + 100j
        state.ship_attitude[0, 0] = 1.0 + 0j
        state.ship_ang_vel[0, 0] = 1.0

        comp = TurnRateReward(turn_rate_weight=1.0, world_size=cfg.world_size)
        reward = comp.compute(
            state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool)
        )

        assert reward[0, 0].item() == 0.0


class TestDisabledRewards:
    def test_disabled_component_is_excluded_from_list(self, cfg):
        """build_reward_components must omit any component named in disabled_rewards."""
        rc = RewardConfig(
            damage_weight=0.01,
            death_weight=0.5,
            victory_weight=1.0,
            enemy_neg_lambda_components=frozenset(
                {"damage", "death", "victory", "exposure"}
            ),
            positioning_weight=0.05,
            positioning_radius=300.0,
            facing_weight=0.01,
            exposure_weight=0.01,
            proximity_weight=0.01,
            proximity_radius=300.0,
            closing_speed_weight=0.01,
            turn_rate_weight=0.01,
            power_range_weight=0.01,
            power_range_lo=0.2,
            power_range_hi=0.8,
            speed_range_weight=0.01,
            speed_range_lo=40.0,
            speed_range_hi=120.0,
            shoot_quality_weight=0.01,
            shoot_quality_radius=200.0,
            disabled_rewards=frozenset({"facing", "turn_rate"}),
        )
        components = build_reward_components(rc, cfg)
        names = {c.name for c in components}

        assert "facing" not in names
        assert "turn_rate" not in names
        assert "closing_speed" in names  # not disabled
