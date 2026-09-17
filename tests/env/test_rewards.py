"""Unit tests for reward components.

Tests reward signs and magnitudes under controlled scenarios.
No compute_rewards() — per-ship signals are tested directly; zero-sum accounting
(lambda aggregation) lives in the PPO trainer, not the reward components.
"""

import pytest
import torch

from boost_and_broadside.config import MatchResult, RewardConfig, ShipConfig, ZoneRole
from boost_and_broadside.env.rewards import (
    REWARD_COMPONENT_NAMES,
    AllyCombatDamageReward,
    AllyCombatDeathReward,
    AllyWinReward,
    CaptureProgressReward,
    ClosingSpeedReward,
    EnemyCombatDamageReward,
    EnemyCombatDeathReward,
    EnemyWinReward,
    FacingReward,
    FrontAdvanceReward,
    KillAllyAssistReward,
    KillAllyShotReward,
    KillAssistReward,
    KillShotReward,
    LocalCombatDamageTakenReward,
    LocalCombatDeathReward,
    LocalDamageDealtAllyReward,
    LocalDamageDealtEnemyReward,
    ShootingPenaltyReward,
    ShootQualityReward,
    SpeedReward,
    build_reward_components,
    component_weights,
    compute_per_component_rewards,
)
from tests.conftest import make_state


@pytest.fixture
def cfg() -> ShipConfig:
    return ShipConfig()


@pytest.fixture
def reward_cfg() -> RewardConfig:
    return RewardConfig(
        win_weight=1.0,
        death_weight=1.0,
        damage_weight=1.0,
        kill_shot_fraction=0.5,
        facing_weight=1.0,
        closing_speed_weight=1.0,
        shoot_quality_weight=1.0,
        proximity_radius=500.0,
        shoot_quality_radius=200.0,
        enemy_neg_lambda_components=frozenset(
            {
                "enemy_combat_damage",
                "enemy_combat_death",
                "enemy_win",
            }
        ),
        ally_zero_components=frozenset(
            {
                "enemy_combat_damage",
                "enemy_combat_death",
                "enemy_win",
            }
        ),
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
    def test_registry_matches_builder(self):
        names = [
            c.name
            for c in build_reward_components(TestComponentWeightDerivation._cfg(), ShipConfig())
        ]
        assert set(names) == set(REWARD_COMPONENT_NAMES)
        assert len(set(names)) == len(names)


class TestComponentWeightDerivation:
    """The balance rule: an event pays one side what it charges the other.

    Every case here leaves ``kill_payout_ratio`` at its default 1.0, which is
    where the rule holds exactly. ``TestKillPayoutRatio`` covers the one tier
    that is allowed to break it.
    """

    @staticmethod
    def _cfg(**kw):
        base = dict(
            win_weight=1.0,
            death_weight=0.4,
            damage_weight=0.3,
            kill_shot_fraction=0.5,
            facing_weight=0.06,
            closing_speed_weight=0.09,
            proximity_radius=400.0,
            shoot_quality_radius=200.0,
            enemy_neg_lambda_components=frozenset(),
            ally_zero_components=frozenset(),
        )
        base.update(kw)
        return RewardConfig(**base)

    def test_a_combat_death_is_paid_for_exactly(self):
        """Charged to the victim, paid to whoever shot it, same total."""
        w = component_weights(self._cfg())
        assert w["kill_shot"] + w["kill_assist"] == pytest.approx(w["combat_death"])

    def test_friendly_fire_mirrors_the_offensive_side(self):
        w = component_weights(self._cfg())
        assert w["kill_ally_shot"] == pytest.approx(w["kill_shot"])
        assert w["kill_ally_assist"] == pytest.approx(w["kill_assist"])
        assert w["damage_dealt_ally"] == pytest.approx(w["damage_dealt_enemy"])

    def test_killing_a_teammate_costs_the_team_twice(self):
        """The ally is charged for dying and the shooter for causing it, and the
        enemy is paid nothing -- so friendly fire is twice as expensive as an
        ordinary death, with no special case saying so."""
        w = component_weights(self._cfg())
        friendly = w["combat_death"] + w["kill_ally_shot"] + w["kill_ally_assist"]
        assert friendly == pytest.approx(2 * w["combat_death"])

    def test_double_charging_components_stay_at_zero(self):
        """Their events are already paid for by the local and dealer-attributed
        components; turning them on would charge the same event twice."""
        w = component_weights(self._cfg())
        for name in (
            "ally_combat_damage",
            "enemy_combat_damage",
            "ally_combat_death",
            "enemy_combat_death",
        ):
            assert w[name] == 0.0

    def test_kill_split_moves_only_within_the_death_budget(self):
        for fraction in (0.0, 0.25, 0.5, 0.9, 1.0):
            w = component_weights(self._cfg(kill_shot_fraction=fraction))
            assert w["kill_shot"] + w["kill_assist"] == pytest.approx(w["combat_death"])
            assert w["kill_shot"] == pytest.approx(w["combat_death"] * fraction)

    def test_every_registered_component_gets_a_weight(self):
        w = component_weights(self._cfg())
        assert set(w) == set(REWARD_COMPONENT_NAMES)

    def test_pre_derivation_checkpoints_read_their_recorded_weights(self):
        """An older run stored one weight per component; those are the record, and
        it still has to load for inference."""
        w = component_weights(
            {"ally_win_weight": 1.5, "kill_shot_weight": 1.0, "facing_weight": 0.1}
        )
        assert w["ally_win"] == 1.5
        assert w["kill_shot"] == 1.0
        assert w["combat_death"] == 0.0
        assert set(w) == set(REWARD_COMPONENT_NAMES)


class TestKillPayoutRatio:
    """The kill tier is allowed to pay more than it charges. Nothing else is."""

    _cfg = staticmethod(TestComponentWeightDerivation._cfg)

    def test_default_is_the_balance_rule(self):
        """Unset, the knob must not move a single weight."""
        assert self._cfg().kill_payout_ratio == 1.0
        assert component_weights(self._cfg()) == component_weights(self._cfg(kill_payout_ratio=1.0))

    def test_the_kill_side_is_paid_the_ratio_times_the_charge(self):
        w = component_weights(self._cfg(death_weight=0.4, kill_payout_ratio=2.0))
        assert w["combat_death"] == pytest.approx(0.4)
        assert w["kill_shot"] + w["kill_assist"] == pytest.approx(0.8)

    def test_the_split_still_divides_the_payout_evenly(self):
        """f partitions the payout, not the charge, so the two knobs stay
        independent: changing one must not move the other's total."""
        for fraction in (0.0, 0.25, 0.5, 1.0):
            w = component_weights(self._cfg(kill_shot_fraction=fraction, kill_payout_ratio=2.0))
            payout = w["kill_shot"] + w["kill_assist"]
            assert payout == pytest.approx(2 * w["combat_death"])
            assert w["kill_shot"] == pytest.approx(payout * fraction)

    def test_friendly_fire_follows_the_payout(self):
        """Blame for a teammate's death is priced at the same rate as credit for
        an enemy's, so raising the payout does not make friendly fire cheap."""
        w = component_weights(self._cfg(kill_payout_ratio=2.0))
        assert w["kill_ally_shot"] == pytest.approx(w["kill_shot"])
        assert w["kill_ally_assist"] == pytest.approx(w["kill_assist"])

    def test_damage_and_win_are_untouched(self):
        """The asymmetry is evidenced for the kill tier only."""
        balanced = component_weights(self._cfg())
        paid = component_weights(self._cfg(kill_payout_ratio=2.0))
        for name in (
            "ally_win",
            "enemy_win",
            "combat_death",
            "combat_damage_taken",
            "damage_dealt_enemy",
            "damage_dealt_ally",
            "facing",
            "closing_speed",
        ):
            assert paid[name] == pytest.approx(balanced[name])

    def test_zero_pays_the_kill_side_nothing(self):
        w = component_weights(self._cfg(kill_payout_ratio=0.0))
        assert w["kill_shot"] == 0.0
        assert w["kill_assist"] == 0.0
        assert w["combat_death"] == pytest.approx(0.4)

    @pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
    def test_a_nonsense_ratio_is_refused(self, bad):
        with pytest.raises(ValueError, match="kill_payout_ratio"):
            self._cfg(kill_payout_ratio=bad)


class TestDamagePayoutRatio:
    """The damage tier gets the same exception, on separate and weaker evidence."""

    _cfg = staticmethod(TestComponentWeightDerivation._cfg)

    def test_default_is_the_balance_rule(self):
        assert self._cfg().damage_payout_ratio == 1.0
        assert component_weights(self._cfg()) == component_weights(
            self._cfg(damage_payout_ratio=1.0)
        )

    def test_damage_dealt_is_paid_the_ratio_times_the_charge(self):
        w = component_weights(self._cfg(damage_weight=0.3, damage_payout_ratio=2.0))
        assert w["combat_damage_taken"] == pytest.approx(0.3)
        assert w["damage_dealt_enemy"] == pytest.approx(0.6)

    def test_friendly_fire_follows_the_payout(self):
        """Damaging a teammate is priced at the rate damaging an enemy pays, so
        raising the payout does not quietly make friendly fire cheap."""
        w = component_weights(self._cfg(damage_payout_ratio=2.0))
        assert w["damage_dealt_ally"] == pytest.approx(w["damage_dealt_enemy"])

    def test_kill_and_win_are_untouched(self):
        """The two ratios are independent knobs on independent tiers."""
        balanced = component_weights(self._cfg())
        paid = component_weights(self._cfg(damage_payout_ratio=2.0))
        for name in (
            "ally_win",
            "enemy_win",
            "combat_death",
            "kill_shot",
            "kill_assist",
            "kill_ally_shot",
            "kill_ally_assist",
            "combat_damage_taken",
            "facing",
            "closing_speed",
        ):
            assert paid[name] == pytest.approx(balanced[name])

    def test_the_two_ratios_compose(self):
        w = component_weights(
            self._cfg(
                death_weight=0.4, damage_weight=0.3, kill_payout_ratio=2.0, damage_payout_ratio=3.0
            )
        )
        assert w["kill_shot"] + w["kill_assist"] == pytest.approx(0.8)
        assert w["damage_dealt_enemy"] == pytest.approx(0.9)

    @pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
    def test_a_nonsense_ratio_is_refused(self, bad):
        with pytest.raises(ValueError, match="damage_payout_ratio"):
            self._cfg(damage_payout_ratio=bad)


# The frontline arena's own objective. It postdates both reconstruction targets
# below, so the reconstruction tests hold it separately rather than counting it
# as a change to the combat balance.
# The strategic tier: the sparse completion pair and the dense progress pair that
# pays the capture leading to it. None of the four is touched by the balance rule
# -- each is exactly the free number that names it.
# The whole strategic tier, including the single undiscounted result stream that
# is carried at a token weight purely so its value head can be watched.
FRONTLINE_COMPONENTS = frozenset({"capture_progress", "front_advance", "outcome"})


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
# Source-split global damage/death rewards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("component_cls", "source_attr"),
    [
        (AllyCombatDamageReward, "ship_combat_damage"),
        (EnemyCombatDamageReward, "ship_combat_damage"),
    ],
)
def test_source_damage_rewards_read_only_their_applied_source(cfg, component_cls, source_attr):
    prev = _make_4ship_state(cfg)
    next_ = _make_4ship_state(cfg)
    getattr(next_, source_attr)[0, 2] = 15.0

    reward = component_cls(weight=1.0).compute(
        prev, torch.zeros(2, 4, 3), next_, torch.zeros(2, dtype=torch.bool)
    )

    assert reward[0, 2].item() == pytest.approx(-15.0)
    assert reward[0, [0, 1, 3]].abs().sum().item() == 0.0


@pytest.mark.parametrize(
    ("component_cls", "source_attr"),
    [
        (AllyCombatDeathReward, "ship_combat_death"),
        (EnemyCombatDeathReward, "ship_combat_death"),
    ],
)
def test_source_death_rewards_read_only_their_exact_source(cfg, component_cls, source_attr):
    prev = _make_4ship_state(cfg)
    next_ = _make_4ship_state(cfg)
    getattr(next_, source_attr)[0, 2] = True

    reward = component_cls(weight=1.0).compute(
        prev, torch.zeros(2, 4, 3), next_, torch.zeros(2, dtype=torch.bool)
    )

    assert reward[0, 2].item() == pytest.approx(-1.0)
    assert reward[0, [0, 1, 3]].abs().sum().item() == 0.0


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
        next_.match_result[0] = int(MatchResult.TEAM0_WIN)
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
        next_.match_result[0] = int(MatchResult.TEAM0_WIN)
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
        next_.match_result[0] = int(MatchResult.TEAM0_WIN)
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
        next_.match_result[0] = int(MatchResult.TEAM0_WIN)
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
        comp = FacingReward(weight=1.0, radius=500.0, world_size=cfg.world_size)
        reward = comp.compute(state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool))

        assert reward[0, 0].item() > 0, "team-0 should get positive facing reward"
        assert reward[0, 1].item() > 0, "team-1 should get positive facing reward"

    def test_ship_not_facing_enemy_gets_lower_reward(self, cfg):
        state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
        state.ship_team_id[0, 0] = 0
        state.ship_team_id[0, 1] = 1
        state.ship_pos[0, 0] = 0.0 + 0j
        state.ship_pos[0, 1] = 100.0 + 0j

        comp = FacingReward(weight=1.0, radius=500.0, world_size=cfg.world_size)

        state.ship_attitude[0, 0] = 1.0 + 0j
        r_facing = comp.compute(
            state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool)
        )

        state.ship_attitude[0, 0] = -1.0 + 0j
        r_away = comp.compute(state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool))

        assert r_facing[0, 0].item() > r_away[0, 0].item()


class TestClosingSpeedReward:
    def test_moving_toward_enemy_gives_positive_reward(self, cfg):
        state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
        state.ship_team_id[0, 0] = 0
        state.ship_team_id[0, 1] = 1
        state.ship_pos[0, 0] = 0.0 + 0j
        state.ship_pos[0, 1] = 100.0 + 0j
        state.ship_vel[0, 0] = 50.0 + 0j  # moving east toward enemy

        comp = ClosingSpeedReward(weight=1.0, world_size=cfg.world_size, max_speed=cfg.max_speed)
        reward = comp.compute(state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool))

        assert reward[0, 0].item() > 0

    def test_moving_away_from_enemy_gives_zero_reward(self, cfg):
        state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
        state.ship_team_id[0, 0] = 0
        state.ship_team_id[0, 1] = 1
        state.ship_pos[0, 0] = 0.0 + 0j
        state.ship_pos[0, 1] = 100.0 + 0j
        state.ship_vel[0, 0] = -50.0 + 0j  # moving away

        comp = ClosingSpeedReward(weight=1.0, world_size=cfg.world_size, max_speed=cfg.max_speed)
        reward = comp.compute(state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool))

        assert reward[0, 0].item() == 0.0

    def test_dead_ship_gets_zero_reward(self, cfg):
        state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
        state.ship_team_id[0, 0] = 0
        state.ship_team_id[0, 1] = 1
        state.ship_alive[0, 0] = False
        state.ship_pos[0, 0] = 0.0 + 0j
        state.ship_pos[0, 1] = 100.0 + 0j
        state.ship_vel[0, 0] = 50.0 + 0j

        comp = ClosingSpeedReward(weight=1.0, world_size=cfg.world_size, max_speed=cfg.max_speed)
        reward = comp.compute(state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool))

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
        reward = r.compute(prev, torch.zeros(1, 4, 3), next_, torch.zeros(1, dtype=torch.bool))

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
        reward = r.compute(prev, torch.zeros(1, 4, 3), next_, torch.zeros(1, dtype=torch.bool))

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
        reward = r.compute(prev, torch.zeros(1, 4, 3), next_, torch.zeros(1, dtype=torch.bool))

        assert reward[0, 0].item() == pytest.approx(0.5, rel=1e-5)
        assert reward[0, 1].item() == pytest.approx(0.5, rel=1e-5)

    def test_no_death_gives_zero_reward(self, cfg):
        state = _kill_state(cfg)

        r = KillShotReward(weight=1.0)
        reward = r.compute(state, torch.zeros(1, 4, 3), state, torch.zeros(1, dtype=torch.bool))

        assert reward.abs().max().item() == 0.0

    def test_friendly_kills_are_not_folded_in(self, cfg):
        """A teammate's death is kill_ally's business, not kill_shot's.

        Folded together, one critic head had to predict the sum of a positive
        enemy-kill signal and a negative friendly-kill one, and the friendly
        half could be neither weighted nor seen in any diagnostic.
        """
        prev = _kill_state(cfg)
        next_ = _kill_state(cfg)
        next_.ship_alive[0, 1] = False  # teammate of ship 0 died
        next_.damage_matrix[0, 0, 1] = 40.0  # ship 0 caused the death

        r = KillShotReward(weight=1.0)
        reward = r.compute(prev, torch.zeros(1, 4, 3), next_, torch.zeros(1, dtype=torch.bool))

        assert reward.abs().max().item() == 0.0


class TestKillAllyRewards:
    """Friendly-kill blame, mirroring the enemy kill_shot/kill_assist pair."""

    def test_sole_damage_dealer_takes_full_blame(self, cfg):
        prev = _kill_state(cfg)
        next_ = _kill_state(cfg)
        next_.ship_alive[0, 1] = False  # teammate of ship 0 died
        next_.cumulative_damage_matrix[0, 0, 1] = 40.0

        r = KillAllyAssistReward(weight=1.0)
        reward = r.compute(prev, torch.zeros(1, 4, 3), next_, torch.zeros(1, dtype=torch.bool))

        assert reward[0, 0].item() == pytest.approx(-1.0)
        assert reward[0, 1].item() == pytest.approx(0.0)
        assert reward[0, 2].item() == pytest.approx(0.0)
        assert reward[0, 3].item() == pytest.approx(0.0)

    def test_blame_splits_by_cumulative_damage(self, cfg):
        """Whoever landed last does not matter; the whole contribution does."""
        prev = _kill_state(cfg)
        next_ = _kill_state(cfg)
        next_.ship_alive[0, 1] = False
        next_.cumulative_damage_matrix[0, 0, 1] = 30.0
        next_.cumulative_damage_matrix[0, 2, 1] = 10.0  # enemy of ship 1, not a teammate

        r = KillAllyAssistReward(weight=1.0)
        reward = r.compute(prev, torch.zeros(1, 4, 3), next_, torch.zeros(1, dtype=torch.bool))

        # Only the teammate is blamed; the enemy's damage is an ordinary kill.
        assert reward[0, 0].item() == pytest.approx(-1.0)
        assert reward[0, 2].item() == pytest.approx(0.0)

    def test_enemy_kills_produce_no_blame(self, cfg):
        prev = _kill_state(cfg)
        next_ = _kill_state(cfg)
        next_.ship_alive[0, 2] = False  # an enemy of ship 0
        next_.cumulative_damage_matrix[0, 0, 2] = 50.0

        r = KillAllyAssistReward(weight=1.0)
        reward = r.compute(prev, torch.zeros(1, 4, 3), next_, torch.zeros(1, dtype=torch.bool))

        assert reward.abs().max().item() == 0.0

    def test_a_ship_is_not_blamed_for_its_own_death(self, cfg):
        prev = _kill_state(cfg)
        next_ = _kill_state(cfg)
        next_.ship_alive[0, 0] = False
        next_.cumulative_damage_matrix[0, 0, 0] = 100.0

        r = KillAllyAssistReward(weight=1.0)
        reward = r.compute(prev, torch.zeros(1, 4, 3), next_, torch.zeros(1, dtype=torch.bool))

        assert reward.abs().max().item() == 0.0

    def test_no_death_gives_zero(self, cfg):
        state = _kill_state(cfg)

        r = KillAllyAssistReward(weight=1.0)
        reward = r.compute(state, torch.zeros(1, 4, 3), state, torch.zeros(1, dtype=torch.bool))

        assert reward.abs().max().item() == 0.0

    def test_step_level_blame_uses_this_step_only(self, cfg):
        """kill_ally_shot mirrors kill_shot: who was firing when the ally died."""
        prev = _kill_state(cfg)
        next_ = _kill_state(cfg)
        next_.ship_alive[0, 1] = False
        next_.damage_matrix[0, 0, 1] = 40.0
        next_.cumulative_damage_matrix[0, 2, 1] = 500.0  # an enemy, and not this step

        r = KillAllyShotReward(weight=1.0)
        reward = r.compute(prev, torch.zeros(1, 4, 3), next_, torch.zeros(1, dtype=torch.bool))

        assert reward[0, 0].item() == pytest.approx(-1.0)
        assert reward[0, 2].item() == pytest.approx(0.0)

    def test_the_two_horizons_can_disagree(self, cfg):
        """A ship that chipped an ally early but did not fire the fatal shot is
        blamed by the cumulative component and not by the step-level one — the
        same split the enemy pair already draws."""
        prev = _kill_state(cfg)
        next_ = _kill_state(cfg)
        next_.ship_alive[0, 1] = False
        next_.damage_matrix[0, 0, 1] = 10.0  # ship 0 fired the fatal shot
        next_.cumulative_damage_matrix[0, 0, 1] = 10.0
        next_.cumulative_damage_matrix[0, 3, 1] = 30.0  # enemy of ship 1: not blamed

        actions, dones = torch.zeros(1, 4, 3), torch.zeros(1, dtype=torch.bool)
        step = KillAllyShotReward(weight=1.0).compute(prev, actions, next_, dones)
        cumulative = KillAllyAssistReward(weight=1.0).compute(prev, actions, next_, dones)

        assert step[0, 0].item() == pytest.approx(-1.0)
        assert cumulative[0, 0].item() == pytest.approx(-1.0)
        assert step[0, 3].item() == pytest.approx(0.0)
        assert cumulative[0, 3].item() == pytest.approx(0.0)

    def test_extraction_conserves_the_old_combined_signal(self, cfg):
        """kill_assist + kill_ally reproduces what kill_assist alone used to emit.

        The split changes how the signal is weighted and learned, not what the
        environment reports.
        """
        prev = _kill_state(cfg)
        next_ = _kill_state(cfg)
        next_.ship_alive[0, 1] = False  # teammate of ship 0
        next_.ship_alive[0, 2] = False  # enemy of ship 0
        next_.cumulative_damage_matrix[0, 0, 1] = 40.0  # ship 0 killed its teammate
        next_.cumulative_damage_matrix[0, 0, 2] = 25.0  # and took a quarter of the enemy
        next_.cumulative_damage_matrix[0, 1, 2] = 75.0

        actions, dones = torch.zeros(1, 4, 3), torch.zeros(1, dtype=torch.bool)
        combined = KillAssistReward(weight=1.0).compute(
            prev, actions, next_, dones
        ) + KillAllyAssistReward(weight=1.0).compute(prev, actions, next_, dones)

        # Ship 0: full blame for the teammate, a quarter of the enemy kill.
        assert combined[0, 0].item() == pytest.approx(-1.0 + 0.25)
        assert combined[0, 1].item() == pytest.approx(0.75)  # three quarters, no blame
        assert combined[0, 3].item() == pytest.approx(0.0)  # ship 2's own teammate


class TestKillAssistReward:
    def test_sole_damage_dealer_gets_full_credit(self, cfg):
        """Ship 0 is the only one that damaged dying ship 2; gets 1.0 assist credit."""
        prev = _kill_state(cfg)
        next_ = _kill_state(cfg)
        next_.ship_alive[0, 2] = False
        next_.cumulative_damage_matrix[0, 0, 2] = 50.0

        r = KillAssistReward(weight=1.0)
        reward = r.compute(prev, torch.zeros(1, 4, 3), next_, torch.zeros(1, dtype=torch.bool))

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
        reward = r.compute(prev, torch.zeros(1, 4, 3), next_, torch.zeros(1, dtype=torch.bool))

        assert reward[0, 0].item() == pytest.approx(0.25, rel=1e-5)
        assert reward[0, 1].item() == pytest.approx(0.75, rel=1e-5)

    def test_no_death_gives_zero_reward(self, cfg):
        state = _kill_state(cfg)

        r = KillAssistReward(weight=1.0)
        reward = r.compute(state, torch.zeros(1, 4, 3), state, torch.zeros(1, dtype=torch.bool))

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
        reward = r.compute(prev, torch.zeros(1, 4, 3), next_, torch.zeros(1, dtype=torch.bool))

        assert reward[0, 0].item() == pytest.approx(2.0)  # 1.0 per kill


# ---------------------------------------------------------------------------
# Local damage rewards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("component_cls", "source_attr"),
    [
        (LocalCombatDamageTakenReward, "ship_combat_damage"),
    ],
)
def test_local_source_damage_rewards_are_exact(cfg, component_cls, source_attr):
    prev = _make_4ship_state(cfg)
    next_ = _make_4ship_state(cfg)
    getattr(next_, source_attr)[0, 0] = 10.0

    reward = component_cls(weight=1.0).compute(
        prev, torch.zeros(2, 4, 3), next_, torch.zeros(2, dtype=torch.bool)
    )

    assert reward[0, 0].item() == pytest.approx(-10.0)
    assert reward[0, 1:].abs().sum().item() == 0.0


class TestLocalDamageDealtEnemyReward:
    def test_ship_that_dealt_enemy_damage_gets_positive_reward(self, cfg):
        """Ship 0 dealt 20 damage to enemy ship 2; ship 0 gets +20."""
        state = _make_4ship_state(cfg)
        state.damage_matrix[0, 0, 2] = 20.0

        r = LocalDamageDealtEnemyReward(weight=1.0)
        state.ship_combat_damage = state.damage_matrix.sum(1)
        reward = r.compute(state, torch.zeros(2, 4, 3), state, torch.zeros(2, dtype=torch.bool))

        assert reward[0, 0].item() == pytest.approx(20.0)

    def test_ships_that_dealt_no_damage_get_zero(self, cfg):
        state = _make_4ship_state(cfg)
        state.damage_matrix[0, 0, 2] = 20.0

        r = LocalDamageDealtEnemyReward(weight=1.0)
        state.ship_combat_damage = state.damage_matrix.sum(1)
        reward = r.compute(state, torch.zeros(2, 4, 3), state, torch.zeros(2, dtype=torch.bool))

        assert reward[0, 1].item() == pytest.approx(0.0)
        assert reward[0, 2].item() == pytest.approx(0.0)
        assert reward[0, 3].item() == pytest.approx(0.0)

    def test_damage_to_multiple_enemies_accumulates(self, cfg):
        """Ship 0 dealt damage to both enemy ships; rewards sum."""
        state = _make_4ship_state(cfg)
        state.damage_matrix[0, 0, 2] = 15.0
        state.damage_matrix[0, 0, 3] = 10.0

        r = LocalDamageDealtEnemyReward(weight=1.0)
        state.ship_combat_damage = state.damage_matrix.sum(1)
        reward = r.compute(state, torch.zeros(2, 4, 3), state, torch.zeros(2, dtype=torch.bool))

        assert reward[0, 0].item() == pytest.approx(25.0)

    def test_friendly_fire_ignored(self, cfg):
        """Damage dealt to a teammate must not contribute to enemy damage reward."""
        state = _make_4ship_state(cfg)
        state.damage_matrix[0, 0, 1] = 30.0  # ship 0 hit ally ship 1

        r = LocalDamageDealtEnemyReward(weight=1.0)
        state.ship_combat_damage = state.damage_matrix.sum(1)
        reward = r.compute(state, torch.zeros(2, 4, 3), state, torch.zeros(2, dtype=torch.bool))

        assert reward[0, 0].item() == pytest.approx(0.0)

    def test_dead_shooter_keeps_projectile_credit(self, cfg):
        state = _make_4ship_state(cfg)
        state.damage_matrix[0, 0, 2] = 20.0
        state.ship_alive[0, 0] = False

        r = LocalDamageDealtEnemyReward(weight=1.0)
        state.ship_combat_damage = state.damage_matrix.sum(1)
        reward = r.compute(state, torch.zeros(2, 4, 3), state, torch.zeros(2, dtype=torch.bool))

        assert reward[0, 0].item() == pytest.approx(20.0)


class TestLocalDamageDealtAllyReward:
    def test_friendly_fire_gives_negative_reward(self, cfg):
        """Ship 0 dealt 30 damage to ally ship 1; ship 0 gets -30."""
        state = _make_4ship_state(cfg)
        state.damage_matrix[0, 0, 1] = 30.0

        r = LocalDamageDealtAllyReward(weight=1.0)
        state.ship_combat_damage = state.damage_matrix.sum(1)
        reward = r.compute(state, torch.zeros(2, 4, 3), state, torch.zeros(2, dtype=torch.bool))

        assert reward[0, 0].item() == pytest.approx(-30.0)

    def test_enemy_damage_ignored(self, cfg):
        """Damage to enemies must not contribute to the friendly-fire penalty."""
        state = _make_4ship_state(cfg)
        state.damage_matrix[0, 0, 2] = 20.0  # ship 0 hit enemy ship 2

        r = LocalDamageDealtAllyReward(weight=1.0)
        state.ship_combat_damage = state.damage_matrix.sum(1)
        reward = r.compute(state, torch.zeros(2, 4, 3), state, torch.zeros(2, dtype=torch.bool))

        assert reward[0, 0].item() == pytest.approx(0.0)

    def test_no_friendly_fire_gives_zero(self, cfg):
        state = _make_4ship_state(cfg)

        r = LocalDamageDealtAllyReward(weight=1.0)
        state.ship_combat_damage = state.damage_matrix.sum(1)
        reward = r.compute(state, torch.zeros(2, 4, 3), state, torch.zeros(2, dtype=torch.bool))

        assert reward.abs().max().item() == 0.0

    def test_dead_shooter_keeps_projectile_credit(self, cfg):
        state = _make_4ship_state(cfg)
        state.damage_matrix[0, 0, 1] = 30.0
        state.ship_alive[0, 0] = False

        r = LocalDamageDealtAllyReward(weight=1.0)
        state.ship_combat_damage = state.damage_matrix.sum(1)
        reward = r.compute(state, torch.zeros(2, 4, 3), state, torch.zeros(2, dtype=torch.bool))

        assert reward[0, 0].item() == pytest.approx(-30.0)


@pytest.mark.parametrize(
    ("component_cls", "source_attr"),
    [
        (LocalCombatDeathReward, "ship_combat_death"),
    ],
)
def test_local_source_death_rewards_are_exact(cfg, component_cls, source_attr):
    prev = _make_4ship_state(cfg)
    next_ = _make_4ship_state(cfg)
    getattr(next_, source_attr)[0, 0] = True

    reward = component_cls(weight=1.0).compute(
        prev, torch.zeros(2, 4, 3), next_, torch.zeros(2, dtype=torch.bool)
    )

    assert reward[0, 0].item() == pytest.approx(-1.0)
    assert reward[0, 1:].abs().sum().item() == 0.0


# ---------------------------------------------------------------------------
# Shoot-quality shaping reward
# ---------------------------------------------------------------------------


def _shoot_state(cfg, *, attitude, shooting):
    """Ship 0 (team 0) at origin, enemy ship 1 (team 1) 50 units east."""
    state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
    state.ship_team_id[0, 0] = 0
    state.ship_team_id[0, 1] = 1
    state.ship_pos[0, 0] = 0.0 + 0j
    state.ship_pos[0, 1] = 50.0 + 0j
    state.ship_attitude[0, 0] = attitude
    state.ship_is_shooting[0, 0] = shooting
    return state


class TestShootQualityReward:
    def test_close_aimed_shot_scores_positive(self, cfg):
        """Firing while aimed at a nearby enemy (inside the radius) is rewarded."""
        state = _shoot_state(cfg, attitude=1.0 + 0j, shooting=True)
        comp = ShootQualityReward(weight=1.0, radius=200.0, world_size=cfg.world_size)
        reward = comp.compute(state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool))

        assert reward[0, 0].item() > 0

    def test_unaimed_shot_scores_negative(self, cfg):
        """Firing while pointing away from the enemy is penalised."""
        state = _shoot_state(cfg, attitude=-1.0 + 0j, shooting=True)
        comp = ShootQualityReward(weight=1.0, radius=200.0, world_size=cfg.world_size)
        reward = comp.compute(state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool))

        assert reward[0, 0].item() < 0

    def test_not_shooting_scores_zero(self, cfg):
        """A ship that does not fire gets no shoot-quality signal, aim notwithstanding."""
        state = _shoot_state(cfg, attitude=1.0 + 0j, shooting=False)
        comp = ShootQualityReward(weight=1.0, radius=200.0, world_size=cfg.world_size)
        reward = comp.compute(state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool))

        assert reward[0, 0].item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Shooting penalty & speed shaping rewards
# ---------------------------------------------------------------------------


class TestShootingPenaltyReward:
    def test_firing_ship_gets_negative_reward(self, cfg):
        state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
        state.ship_is_shooting[0, 0] = True

        r = ShootingPenaltyReward(weight=1.0)
        reward = r.compute(state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool))

        assert reward[0, 0].item() == pytest.approx(-1.0)

    def test_non_firing_ship_gets_zero(self, cfg):
        state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
        state.ship_is_shooting[0, 0] = True

        r = ShootingPenaltyReward(weight=1.0)
        reward = r.compute(state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool))

        assert reward[0, 1].item() == pytest.approx(0.0)

    def test_dead_firing_ship_gets_zero(self, cfg):
        state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
        state.ship_is_shooting[0, 0] = True
        state.ship_alive[0, 0] = False

        r = ShootingPenaltyReward(weight=1.0)
        reward = r.compute(state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool))

        assert reward[0, 0].item() == pytest.approx(0.0)


class TestSpeedReward:
    def test_stationary_ship_gets_full_penalty(self, cfg):
        """Speed 0 is the worst case: penalty saturates at -1."""
        state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
        state.ship_vel[0, 0] = 0.0 + 0j

        r = SpeedReward(weight=1.0, min_speed=40.0)
        reward = r.compute(state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool))

        assert reward[0, 0].item() == pytest.approx(-1.0)

    def test_ship_at_min_speed_gets_zero(self, cfg):
        """At or above min_speed there is no penalty."""
        state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
        state.ship_vel[0, 0] = 40.0 + 0j

        r = SpeedReward(weight=1.0, min_speed=40.0)
        reward = r.compute(state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool))

        assert reward[0, 0].item() == pytest.approx(0.0)

    def test_slow_ship_gets_partial_penalty(self, cfg):
        """Between 0 and min_speed the penalty is linear: speed 10 → -0.75."""
        state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
        state.ship_vel[0, 0] = 10.0 + 0j

        r = SpeedReward(weight=1.0, min_speed=40.0)
        reward = r.compute(state, torch.zeros(1, 2, 3), state, torch.zeros(1, dtype=torch.bool))

        assert reward[0, 0].item() == pytest.approx(-0.75)


class TestZoneCredit:
    """Credit for a meter goes to who held the point, blame to who did not.

    The tier is a team objective but its credit is not team-wide: a meter moves
    because ships stood on the point and against you because ships did not. The
    favoured side splits the payment among its ships inside the zone; the other
    side splits the charge among its ships elsewhere, dead ones included. Ships
    of the losing side who are *inside* are contesting and losing, which is not
    the failure being priced, so they are charged nothing.
    """

    ZONES = 5
    T0_DEFENSE = 1
    T1_DEFENSE = 3
    RADIUS = 50.0
    HERE = complex(100.0, 100.0)
    AWAY = complex(600.0, 600.0)

    def _states(self, before, after, *, inside, teams, alive=None, captured=(False, False)):
        """Two states differing only in the meters, with placement controlled."""
        made = []
        for progress in (before, after):
            st = make_state(num_envs=1, max_ships=len(teams))
            st.zone_roles = torch.full((1, self.ZONES), int(ZoneRole.NEUTRAL), dtype=torch.int8)
            st.zone_roles[0, self.T0_DEFENSE] = int(ZoneRole.TEAM0_DEFENSE)
            st.zone_roles[0, self.T1_DEFENSE] = int(ZoneRole.TEAM1_DEFENSE)
            st.zone_pos = torch.full((1, self.ZONES), self.AWAY, dtype=torch.complex64)
            st.zone_pos[0, self.T0_DEFENSE] = self.HERE
            st.zone_pos[0, self.T1_DEFENSE] = self.HERE
            st.zone_radius = torch.full((1, self.ZONES), self.RADIUS)
            st.zone_capture_progress = torch.zeros((1, self.ZONES))
            for index, value in progress.items():
                st.zone_capture_progress[0, index] = value
            st.ship_team_id = torch.tensor([teams], dtype=torch.int32)
            st.ship_pos = torch.tensor(
                [[self.HERE if here else self.AWAY for here in inside]], dtype=torch.complex64
            )
            st.ship_alive = torch.tensor([alive if alive else [True] * len(teams)])
            st.team0_captured = torch.tensor([captured[0]])
            st.team1_captured = torch.tensor([captured[1]])
            made.append(st)
        return made

    def _reward(self, component, **kwargs):
        prev, nxt = self._states(**kwargs)
        n = prev.ship_team_id.shape[1]
        return component.compute(
            prev, torch.zeros((1, n, 3), dtype=torch.long), nxt, torch.zeros(1, dtype=torch.bool)
        )[0]

    def _progress(self, ratio=1.0):
        return CaptureProgressReward(weight=1.0, payout_ratio=ratio, world_size=(1024.0, 1024.0))

    def test_the_present_attackers_split_the_payment(self):
        # Team 0 attacks T1_DEFENSE; two of its ships are on the point, one is not.
        r = self._reward(
            self._progress(),
            before={self.T1_DEFENSE: 0.2},
            after={self.T1_DEFENSE: 0.5},
            inside=[True, True, False, False],
            teams=[0, 0, 1, 1],
        )
        # 0.3 of movement split between the two who held it.
        assert r[0].item() == pytest.approx(0.15)
        assert r[1].item() == pytest.approx(0.15)

    def test_the_absent_defenders_split_the_charge(self):
        r = self._reward(
            self._progress(),
            before={self.T1_DEFENSE: 0.2},
            after={self.T1_DEFENSE: 0.5},
            inside=[True, True, False, False],
            teams=[0, 0, 1, 1],
        )
        assert r[2].item() == pytest.approx(-0.15)
        assert r[3].item() == pytest.approx(-0.15)

    def test_a_defender_who_showed_up_is_not_charged(self):
        """Contesting and losing is not the failure being priced."""
        r = self._reward(
            self._progress(),
            before={self.T1_DEFENSE: 0.2},
            after={self.T1_DEFENSE: 0.5},
            inside=[True, True, True, False],
            teams=[0, 0, 1, 1],
        )
        assert r[2].item() == pytest.approx(0.0)  # inside, losing, not charged
        assert r[3].item() == pytest.approx(-0.3)  # the only one absent takes it all

    def test_a_dead_ship_counts_as_absent_and_is_charged(self):
        """Identity survives death and respawn is immediate, so the charge lands
        on a ship that still exists -- and a dead ship is precisely one not
        holding the point."""
        r = self._reward(
            self._progress(),
            before={self.T1_DEFENSE: 0.2},
            after={self.T1_DEFENSE: 0.5},
            inside=[True, True, True, False],
            teams=[0, 0, 1, 1],
            alive=[True, True, True, False],
        )
        assert r[3].item() == pytest.approx(-0.3)

    def test_the_payout_ratio_tilts_toward_holding_the_point(self):
        r = self._reward(
            self._progress(ratio=2.0),
            before={self.T1_DEFENSE: 0.2},
            after={self.T1_DEFENSE: 0.5},
            inside=[True, False, False, False],
            teams=[0, 0, 1, 1],
        )
        assert r[0].item() == pytest.approx(0.6)  # paid 2x
        assert r[2].item() == pytest.approx(-0.15)  # charged 1x, split two ways
        assert r[3].item() == pytest.approx(-0.15)

    def test_the_charged_set_cannot_be_empty(self):
        """A meter favours a side only when it has strictly more ships on the
        point, so the other side always has at least one ship elsewhere."""
        r = self._reward(
            self._progress(),
            before={self.T1_DEFENSE: 0.2},
            after={self.T1_DEFENSE: 0.5},
            inside=[True, True, True, False],
            teams=[0, 0, 1, 1],
        )
        assert r.sum().item() == pytest.approx(0.0)  # still zero-sum at ratio 1

    def test_defending_your_own_point_pays_the_same_as_taking_theirs(self):
        # Progress on T0_DEFENSE falling means team 0 pushed the attacker back.
        r = self._reward(
            self._progress(),
            before={self.T0_DEFENSE: 0.5},
            after={self.T0_DEFENSE: 0.2},
            inside=[True, False, False, False],
            teams=[0, 0, 1, 1],
        )
        assert r[0].item() == pytest.approx(0.3)

    def test_the_completion_tick_is_left_to_front_advance(self):
        """A capture resets the meter 1 -> 0, which read naively is the largest
        loss the component can report, on the exact tick a team succeeded."""
        r = self._reward(
            self._progress(),
            before={self.T1_DEFENSE: 0.99},
            after={self.T1_DEFENSE: 0.0},
            inside=[True, True, False, False],
            teams=[0, 0, 1, 1],
            captured=(True, False),
        )
        assert r.abs().max().item() == pytest.approx(0.0)

    def test_front_advance_pays_the_ships_that_finished_it(self):
        """The finished zone is found by its new role: taking a point steps the
        front, which rotates the roles, so it now reads as the captor's own
        defense."""
        component = FrontAdvanceReward(weight=1.0, payout_ratio=1.0, world_size=(1024.0, 1024.0))
        prev, nxt = self._states(
            before={}, after={}, inside=[True, False, False, False], teams=[0, 0, 1, 1]
        )
        nxt.team0_captured = torch.tensor([True])
        # After a team-0 capture the taken zone carries TEAM0_DEFENSE.
        nxt.zone_roles[0, self.T0_DEFENSE] = int(ZoneRole.TEAM0_DEFENSE)
        nxt.zone_pos[0, self.T0_DEFENSE] = self.HERE
        r = component.compute(
            prev, torch.zeros((1, 4, 3), dtype=torch.long), nxt, torch.zeros(1, dtype=torch.bool)
        )[0]
        assert r[0].item() == pytest.approx(1.0)
        assert r[1].item() == pytest.approx(0.0)
        assert r[2].item() == pytest.approx(-0.5)
        assert r[3].item() == pytest.approx(-0.5)
