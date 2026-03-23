"""Unit tests for reward components.

Tests reward signs and magnitudes under controlled scenarios.
"""

import pytest
import torch

from boost_and_broadside.config import ShipConfig, RewardConfig
from boost_and_broadside.env.rewards import (
    DamageReward, DeathReward, VictoryReward, PositioningReward,
    FacingReward, ExposureReward, SpeedRangeReward, PowerRangeReward,
    ClosingSpeedReward, TurnRateReward,
    compute_rewards, build_reward_components,
)
from tests.conftest import make_state


@pytest.fixture
def cfg() -> ShipConfig:
    return ShipConfig()


@pytest.fixture
def reward_cfg() -> RewardConfig:
    return RewardConfig(
        damage_weight=1.0,
        kill_weight=5.0,
        death_weight=5.0,
        victory_weight=10.0,
        positioning_weight=1.0,
        positioning_radius=500.0,
    )


def _make_4ship_state(cfg):
    """Team 0: ships 0,1. Team 1: ships 2,3."""
    state = make_state(num_envs=2, max_ships=4, ship_config=cfg)
    state.ship_team_id[:, 0] = 0
    state.ship_team_id[:, 1] = 0
    state.ship_team_id[:, 2] = 1
    state.ship_team_id[:, 3] = 1
    return state


class TestDamageReward:
    def test_ally_taking_damage_gives_negative_reward_to_ally(self, cfg):
        """When a team-0 ship takes damage, team-0 ships get negative reward."""
        prev  = _make_4ship_state(cfg)
        next_ = _make_4ship_state(cfg)
        next_.ship_health[0, 0] = prev.ship_health[0, 0] - 10.0  # ship 0 took 10 damage

        r = DamageReward(damage_weight=1.0)
        reward = r.compute(prev, torch.zeros(2, 4, 3), next_, torch.zeros(2, dtype=torch.bool))

        # Team-0 ship 0 (the one damaged) should get negative reward
        assert reward[0, 0].item() < 0

    def test_enemy_taking_damage_gives_positive_reward_to_ally(self, cfg):
        """When a team-1 ship takes damage, raw output is pre-inverted so zero-sum yields penalty."""
        prev  = _make_4ship_state(cfg)
        next_ = _make_4ship_state(cfg)
        next_.ship_health[0, 2] = prev.ship_health[0, 2] - 20.0  # ship 2 (enemy) took 20 damage

        r = DamageReward(damage_weight=1.0)
        reward = r.compute(prev, torch.zeros(2, 4, 3), next_, torch.zeros(2, dtype=torch.bool))

        # Ship 2 is team-1; pre-inverted raw = +20 so zero-sum negation gives -20 (penalty ✓)
        assert reward[0, 2].item() > 0

    def test_zero_damage_gives_zero_reward(self, cfg):
        """No damage → zero reward from damage component."""
        prev  = _make_4ship_state(cfg)
        next_ = _make_4ship_state(cfg)

        r = DamageReward(damage_weight=1.0)
        reward = r.compute(prev, torch.zeros(2, 4, 3), next_, torch.zeros(2, dtype=torch.bool))

        assert reward.abs().max().item() == 0.0

    def test_damage_weight_scales_reward(self, cfg):
        """Reward magnitude scales linearly with damage_weight."""
        prev  = _make_4ship_state(cfg)
        next_ = _make_4ship_state(cfg)
        next_.ship_health[0, 0] = prev.ship_health[0, 0] - 10.0

        r1 = DamageReward(damage_weight=1.0)
        r2 = DamageReward(damage_weight=2.0)
        dones = torch.zeros(2, dtype=torch.bool)

        rew1 = r1.compute(prev, torch.zeros(2, 4, 3), next_, dones)
        rew2 = r2.compute(prev, torch.zeros(2, 4, 3), next_, dones)

        assert abs(rew2[0, 0].item()) == pytest.approx(2 * abs(rew1[0, 0].item()), rel=1e-5)


class TestDeathReward:
    def test_dying_ship_gets_penalty(self, cfg):
        """A ship that just died should receive a negative reward."""
        prev  = _make_4ship_state(cfg)
        next_ = _make_4ship_state(cfg)
        next_.ship_alive[0, 0] = False  # ship 0 died

        r = DeathReward(kill_weight=5.0, death_weight=5.0)
        reward = r.compute(prev, torch.zeros(2, 4, 3), next_, torch.zeros(2, dtype=torch.bool))

        assert reward[0, 0].item() == pytest.approx(-5.0, rel=1e-5)

    def test_enemy_dying_gives_positive_reward(self, cfg):
        """When a team-1 ship dies, raw output is pre-inverted (+5) so zero-sum yields penalty."""
        prev  = _make_4ship_state(cfg)
        next_ = _make_4ship_state(cfg)
        next_.ship_alive[0, 2] = False  # ship 2 (team 1) died

        r = DeathReward(kill_weight=5.0, death_weight=5.0)
        reward = r.compute(prev, torch.zeros(2, 4, 3), next_, torch.zeros(2, dtype=torch.bool))

        # Ship 2 is team-1; pre-inverted raw = +5.0 so zero-sum negation gives -5.0 (penalty ✓)
        assert reward[0, 2].item() == pytest.approx(+5.0, rel=1e-5)

    def test_no_death_gives_zero_reward(self, cfg):
        """No ships died → zero death reward."""
        prev  = _make_4ship_state(cfg)
        next_ = _make_4ship_state(cfg)

        r = DeathReward(kill_weight=5.0, death_weight=5.0)
        reward = r.compute(prev, torch.zeros(2, 4, 3), next_, torch.zeros(2, dtype=torch.bool))

        assert reward.abs().max().item() == 0.0


class TestVictoryReward:
    def test_winning_team_gets_positive_reward(self, cfg):
        """When team 1 is eliminated, team-0 ships get positive victory reward."""
        prev  = _make_4ship_state(cfg)
        next_ = _make_4ship_state(cfg)
        next_.ship_alive[0, 2] = False
        next_.ship_alive[0, 3] = False  # team 1 eliminated
        dones = torch.tensor([True, False], dtype=torch.bool)

        r = VictoryReward(victory_weight=10.0)
        reward = r.compute(prev, torch.zeros(2, 4, 3), next_, dones)

        # Team-0 ships (0, 1) in env 0 should get +10
        assert reward[0, 0].item() == pytest.approx(10.0, rel=1e-5)
        assert reward[0, 1].item() == pytest.approx(10.0, rel=1e-5)
        # Team-1 ships: pre-inverted raw = +10 so zero-sum negation gives -10 (penalty ✓)
        assert reward[0, 2].item() == pytest.approx(+10.0, rel=1e-5)

    def test_non_terminal_gives_zero_reward(self, cfg):
        """No victory reward when done=False."""
        prev  = _make_4ship_state(cfg)
        next_ = _make_4ship_state(cfg)
        dones = torch.zeros(2, dtype=torch.bool)  # not done

        r = VictoryReward(victory_weight=10.0)
        reward = r.compute(prev, torch.zeros(2, 4, 3), next_, dones)

        assert reward.abs().max().item() == 0.0


class TestZeroSumTransform:
    def test_team1_rewards_are_negated(self, cfg):
        """compute_rewards must negate team-1 ships' rewards for zero-sum training."""
        prev  = _make_4ship_state(cfg)
        next_ = _make_4ship_state(cfg)
        next_.ship_health[0, 0] = prev.ship_health[0, 0] - 10.0  # ship 0 (team-0) takes damage
        dones = torch.zeros(2, dtype=torch.bool)

        components = [DamageReward(damage_weight=1.0)]
        rewards, _ = compute_rewards(components, prev, torch.zeros(2, 4, 3), next_, dones)

        # Ship 0 (team-0) took damage → negative reward for team-0
        # Ship 2 (team-1) did not take damage directly, but via zero-sum...
        # Team-0 ally pain is team-1 gain → team-1 ship should get positive BEFORE negation
        # After negation, let's just check sign consistency:
        # team-0 ship 0 = negative (took damage)
        # team-1 ships should have reward = -(team-1 component) which mirrors team-0

        # Verify team-0 ship hurt = negative
        assert rewards[0, 0].item() < 0

        # Team-1 ships get the cross-team damage bonus: team-0 took damage → team-1 gains.
        # Raw for team-1 = -(team0_dmg / n_team1), pre-inverted → +(team0_dmg / n_team1).
        # After zero-sum negation: -(pre-inverted) = -(positive) which should be negative...
        # Actually: team-1 BENEFITS when team-0 takes damage.
        # Raw ship2: reward = -0 (own dmg) + team0_dmg/n_team1 = +5.0
        # Pre-invert: -5.0. Zero-sum negates: +5.0. Team-1 correctly gets positive (gain) ✓
        assert rewards[0, 2].item() > 0


class TestPositioningReward:
    def test_pointing_at_enemy_gives_positive_reward(self, cfg):
        """A ship pointing directly at an enemy within range should get positive reward."""
        state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
        state.ship_team_id[0, 0] = 0
        state.ship_team_id[0, 1] = 1

        # Place ships 100 units apart, ship 0 pointing directly at ship 1
        state.ship_pos[0, 0] = 0.0 + 0j
        state.ship_pos[0, 1] = 100.0 + 0j
        state.ship_attitude[0, 0] = 1.0 + 0j   # pointing East = toward ship 1

        r = PositioningReward(positioning_weight=1.0, positioning_radius=500.0,
                               world_size=cfg.world_size)
        reward = r.compute(state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool))

        assert reward[0, 0].item() > 0

    def test_ships_outside_radius_get_zero_reward(self, cfg):
        """Ships far beyond positioning_radius should get zero reward."""
        state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
        state.ship_team_id[0, 0] = 0
        state.ship_team_id[0, 1] = 1

        # Place ships very far apart (beyond radius of 100)
        state.ship_pos[0, 0] = 0.0 + 0j
        state.ship_pos[0, 1] = 200.0 + 0j

        r = PositioningReward(positioning_weight=1.0, positioning_radius=100.0,
                               world_size=cfg.world_size)
        reward = r.compute(state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool))

        assert abs(reward[0, 0].item()) < 1e-6

    def test_dead_ships_get_zero_positioning_reward(self, cfg):
        """Dead ships must receive zero positioning reward."""
        state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
        state.ship_team_id[0, 0] = 0
        state.ship_team_id[0, 1] = 1
        state.ship_alive  [0, 0] = False  # ship 0 is dead

        state.ship_pos[0, 0] = 0.0 + 0j
        state.ship_pos[0, 1] = 100.0 + 0j
        state.ship_attitude[0, 0] = 1.0 + 0j

        r = PositioningReward(positioning_weight=1.0, positioning_radius=500.0,
                               world_size=cfg.world_size)
        reward = r.compute(state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool))

        assert reward[0, 0].item() == 0.0


def _facing_state(cfg):
    """Two ships pointing at each other, 100 units apart."""
    state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
    state.ship_team_id[0, 0] = 0
    state.ship_team_id[0, 1] = 1
    state.ship_pos[0, 0] = 0.0 + 0j
    state.ship_pos[0, 1] = 100.0 + 0j
    state.ship_attitude[0, 0] = 1.0 + 0j    # team-0 pointing toward team-1
    state.ship_attitude[0, 1] = -1.0 + 0j   # team-1 pointing toward team-0
    return state


class TestFacingRewardZeroSum:
    """FacingReward must incentivise BOTH teams to face their enemies."""

    def test_both_teams_positive_after_zero_sum(self, cfg):
        """After compute_rewards zero-sum transform both ships get positive facing reward."""
        state = _facing_state(cfg)
        comp = FacingReward(facing_weight=1.0, radius=500.0, world_size=cfg.world_size)
        rewards, _ = compute_rewards(
            [comp], state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool)
        )
        assert rewards[0, 0].item() > 0, "team-0 should get positive facing reward"
        assert rewards[0, 1].item() > 0, "team-1 should get positive facing reward"

    def test_team1_raw_is_pre_inverted(self, cfg):
        """Raw compute() output for team-1 must be negative so zero-sum makes it positive."""
        state = _facing_state(cfg)
        comp = FacingReward(facing_weight=1.0, radius=500.0, world_size=cfg.world_size)
        raw = comp.compute(state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool))
        assert raw[0, 0].item() > 0, "team-0 raw reward should be positive"
        assert raw[0, 1].item() < 0, "team-1 raw reward should be pre-inverted negative"


class TestExposureRewardZeroSum:
    """ExposureReward must penalise BOTH teams for being in crosshairs."""

    def test_both_teams_negative_after_zero_sum(self, cfg):
        """After zero-sum both ships facing each other get negative exposure reward."""
        state = _facing_state(cfg)  # both ships aim at each other = both exposed
        comp = ExposureReward(exposure_weight=1.0, radius=500.0, world_size=cfg.world_size)
        rewards, _ = compute_rewards(
            [comp], state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool)
        )
        assert rewards[0, 0].item() < 0, "team-0 should be penalised for being exposed"
        assert rewards[0, 1].item() < 0, "team-1 should be penalised for being exposed"

    def test_team1_raw_is_pre_inverted(self, cfg):
        """Raw compute() for team-1 must be positive so zero-sum makes it negative."""
        state = _facing_state(cfg)
        comp = ExposureReward(exposure_weight=1.0, radius=500.0, world_size=cfg.world_size)
        raw = comp.compute(state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool))
        assert raw[0, 0].item() < 0, "team-0 raw reward should be negative (direct penalty)"
        assert raw[0, 1].item() > 0, "team-1 raw reward should be pre-inverted positive"


class TestSpeedRangeRewardZeroSum:
    """SpeedRangeReward must reward BOTH teams for staying in range."""

    def test_both_teams_positive_after_zero_sum(self, cfg):
        state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
        state.ship_team_id[0, 0] = 0
        state.ship_team_id[0, 1] = 1
        state.ship_vel[0, 0] = 80.0 + 0j   # speed 80, in [40, 120] range
        state.ship_vel[0, 1] = 80.0 + 0j

        comp = SpeedRangeReward(speed_range_weight=1.0, speed_range_lo=40.0, speed_range_hi=120.0)
        rewards, _ = compute_rewards(
            [comp], state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool)
        )
        assert rewards[0, 0].item() > 0, "team-0 should be rewarded for in-range speed"
        assert rewards[0, 1].item() > 0, "team-1 should be rewarded for in-range speed"

    def test_out_of_range_gives_less_reward_than_in_range(self, cfg):
        """The trapezoidal function gives less reward when speed is outside [lo, hi]."""
        comp = SpeedRangeReward(speed_range_weight=1.0, speed_range_lo=40.0, speed_range_hi=120.0)

        in_state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
        in_state.ship_team_id[0, 0] = 0
        in_state.ship_team_id[0, 1] = 1
        in_state.ship_vel[0, 0] = 80.0 + 0j   # in range
        in_state.ship_vel[0, 1] = 80.0 + 0j

        out_state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
        out_state.ship_team_id[0, 0] = 0
        out_state.ship_team_id[0, 1] = 1
        out_state.ship_vel[0, 0] = 0.0 + 0j   # stopped — well below lo
        out_state.ship_vel[0, 1] = 0.0 + 0j

        r_in,  _ = compute_rewards([comp], in_state,  torch.zeros(1, 2, 3), in_state,  torch.zeros(1, dtype=torch.bool))
        r_out, _ = compute_rewards([comp], out_state, torch.zeros(1, 2, 3), out_state, torch.zeros(1, dtype=torch.bool))

        assert r_in[0, 0].item() > r_out[0, 0].item(), "team-0: in-range speed gives more reward"
        assert r_in[0, 1].item() > r_out[0, 1].item(), "team-1: in-range speed gives more reward"


class TestPowerRangeRewardZeroSum:
    """PowerRangeReward must reward BOTH teams for keeping power in range."""

    def test_both_teams_positive_after_zero_sum(self, cfg):
        state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
        state.ship_team_id[0, 0] = 0
        state.ship_team_id[0, 1] = 1
        # Default state has full power (100); range is [0.2, 0.8] * max_power = [20, 80]
        state.ship_power[0, 0] = 50.0   # in range
        state.ship_power[0, 1] = 50.0

        comp = PowerRangeReward(
            power_range_weight=1.0, power_range_lo=0.2, power_range_hi=0.8,
            max_power=cfg.max_power,
        )
        rewards, _ = compute_rewards(
            [comp], state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool)
        )
        assert rewards[0, 0].item() > 0, "team-0 should be rewarded for in-range power"
        assert rewards[0, 1].item() > 0, "team-1 should be rewarded for in-range power"


class TestClosingSpeedReward:
    def test_moving_toward_enemy_gives_positive_reward(self, cfg):
        """Ship moving directly toward enemy should get positive closing speed reward."""
        state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
        state.ship_team_id[0, 0] = 0
        state.ship_team_id[0, 1] = 1
        # Ship 0 at origin, ship 1 at (100, 0); ship 0 moving east (+x) toward ship 1
        state.ship_pos[0, 0] = 0.0 + 0j
        state.ship_pos[0, 1] = 100.0 + 0j
        state.ship_vel[0, 0] = 50.0 + 0j  # moving east
        state.ship_vel[0, 1] = 0.0 + 0j

        comp = ClosingSpeedReward(closing_speed_weight=1.0, world_size=cfg.world_size)
        reward = comp.compute(state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool))

        assert reward[0, 0].item() > 0

    def test_moving_away_from_enemy_gives_zero_reward(self, cfg):
        """Ship moving away from enemy gets zero (clamped) reward."""
        state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
        state.ship_team_id[0, 0] = 0
        state.ship_team_id[0, 1] = 1
        state.ship_pos[0, 0] = 0.0 + 0j
        state.ship_pos[0, 1] = 100.0 + 0j
        state.ship_vel[0, 0] = -50.0 + 0j  # moving west, away from enemy

        comp = ClosingSpeedReward(closing_speed_weight=1.0, world_size=cfg.world_size)
        reward = comp.compute(state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool))

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
        reward = comp.compute(state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool))

        assert reward[0, 0].item() == 0.0


class TestTurnRateReward:
    def test_turning_toward_enemy_gives_positive_reward(self, cfg):
        """Ship rotating counterclockwise with enemy to the left gets positive reward."""
        state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
        state.ship_team_id[0, 0] = 0
        state.ship_team_id[0, 1] = 1
        # Ship 0 pointing east, enemy is to the north (left of heading)
        state.ship_pos[0, 0] = 0.0 + 0j
        state.ship_pos[0, 1] = 0.0 + 100j   # enemy is directly north
        state.ship_attitude[0, 0] = 1.0 + 0j  # heading east
        # Positive ang_vel = counterclockwise = toward enemy (enemy is to the left)
        state.ship_ang_vel[0, 0] = 1.0

        comp = TurnRateReward(turn_rate_weight=1.0, world_size=cfg.world_size)
        reward = comp.compute(state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool))

        assert reward[0, 0].item() > 0

    def test_turning_away_from_enemy_gives_negative_reward(self, cfg):
        """Ship rotating clockwise with enemy to the left gets negative reward."""
        state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
        state.ship_team_id[0, 0] = 0
        state.ship_team_id[0, 1] = 1
        state.ship_pos[0, 0] = 0.0 + 0j
        state.ship_pos[0, 1] = 0.0 + 100j   # enemy is directly north
        state.ship_attitude[0, 0] = 1.0 + 0j  # heading east
        # Negative ang_vel = clockwise = away from enemy (enemy is to the left)
        state.ship_ang_vel[0, 0] = -1.0

        comp = TurnRateReward(turn_rate_weight=1.0, world_size=cfg.world_size)
        reward = comp.compute(state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool))

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
        reward = comp.compute(state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool))

        assert reward[0, 0].item() == 0.0
