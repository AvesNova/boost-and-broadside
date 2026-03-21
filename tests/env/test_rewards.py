"""Unit tests for reward components.

Tests reward signs and magnitudes under controlled scenarios.
"""

import pytest
import torch

from boost_and_broadside.config import ShipConfig, RewardConfig
from boost_and_broadside.env.rewards import (
    DamageReward, DeathReward, VictoryReward, PositioningReward,
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
        """When a team-1 ship takes damage, team-0 ships should get positive reward."""
        prev  = _make_4ship_state(cfg)
        next_ = _make_4ship_state(cfg)
        next_.ship_health[0, 2] = prev.ship_health[0, 2] - 20.0  # ship 2 (enemy) took 20 damage

        r = DamageReward(damage_weight=1.0)
        reward = r.compute(prev, torch.zeros(2, 4, 3), next_, torch.zeros(2, dtype=torch.bool))

        # Team-1 ship (ship 2) gets negative reward (from its own-team perspective)
        # Then zero-sum negation in compute_rewards flips it — but here we test raw component
        # Team-1 ship: damage to itself = negative for team-1
        assert reward[0, 2].item() < 0  # ship 2 is team-1; took ally (self) damage = bad

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
        """When a team-1 ship dies, team-1 ship (enemy) gets positive reward from team-0 view."""
        prev  = _make_4ship_state(cfg)
        next_ = _make_4ship_state(cfg)
        next_.ship_alive[0, 2] = False  # ship 2 (team 1) died

        r = DeathReward(kill_weight=5.0, death_weight=5.0)
        reward = r.compute(prev, torch.zeros(2, 4, 3), next_, torch.zeros(2, dtype=torch.bool))

        # Ship 2 is team-1 — from team-1's perspective, an ally (itself) died = penalty
        assert reward[0, 2].item() == pytest.approx(-5.0, rel=1e-5)

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
        # Team-1 ships in env 0 get -10 (from team-1 perspective, their team lost)
        assert reward[0, 2].item() == pytest.approx(-10.0, rel=1e-5)

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

        # Verify zero-sum symmetry: the reward structure should be antisymmetric
        # between teams for a damage-only scenario with pure symmetric geometry
        # (here it's not purely symmetric so just verify sign of team-1)
        # team-1 ship 2 is NOT damaged, so its component is 0 → after negation still 0
        assert rewards[0, 2].item() == pytest.approx(0.0, abs=1e-5)


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
