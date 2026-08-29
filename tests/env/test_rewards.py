"""Unit tests for reward components.

Tests reward signs and magnitudes under controlled scenarios.
No compute_rewards() — per-ship signals are tested directly; zero-sum accounting
(lambda aggregation) lives in the PPO trainer, not the reward components.
"""

import dataclasses

import pytest
import torch

from boost_and_broadside.config import RewardConfig, ShipConfig
from boost_and_broadside.config.defaults import REWARDS
from boost_and_broadside.env.rewards import (
    REWARD_COMPONENT_NAMES,
    AllyCombatDamageReward,
    AllyCombatDeathReward,
    AllyFieldDamageReward,
    AllyFieldDeathReward,
    AllyWinReward,
    ClosingSpeedReward,
    EnemyCombatDamageReward,
    EnemyCombatDeathReward,
    EnemyFieldDamageReward,
    EnemyFieldDeathReward,
    EnemyWinReward,
    FacingReward,
    KillAllyAssistReward,
    KillAllyShotReward,
    KillAssistReward,
    KillShotReward,
    LocalCombatDamageTakenReward,
    LocalCombatDeathReward,
    LocalDamageDealtAllyReward,
    LocalDamageDealtEnemyReward,
    LocalFieldDamageTakenReward,
    LocalFieldDeathReward,
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
    def test_k_equals_25(self):
        assert len(REWARD_COMPONENT_NAMES) == 25

    def test_source_split_starts_the_registry(self):
        assert REWARD_COMPONENT_NAMES[:8] == (
            "ally_combat_damage",
            "enemy_combat_damage",
            "ally_field_damage",
            "enemy_field_damage",
            "ally_combat_death",
            "enemy_combat_death",
            "ally_field_death",
            "enemy_field_death",
        )

    def test_kill_shot_is_index_13(self):
        assert REWARD_COMPONENT_NAMES[13] == "kill_shot"

    def test_kill_assist_is_index_14(self):
        assert REWARD_COMPONENT_NAMES[14] == "kill_assist"

    def test_friendly_kill_pair_mirrors_the_enemy_pair(self):
        assert REWARD_COMPONENT_NAMES[15:17] == ("kill_ally_shot", "kill_ally_assist")

    def test_source_split_local_damage_is_registered(self):
        assert REWARD_COMPONENT_NAMES[17:19] == (
            "combat_damage_taken",
            "field_damage_taken",
        )

    def test_damage_dealt_enemy_is_index_19(self):
        assert REWARD_COMPONENT_NAMES[19] == "damage_dealt_enemy"

    def test_damage_dealt_ally_is_index_20(self):
        assert REWARD_COMPONENT_NAMES[20] == "damage_dealt_ally"

    def test_source_split_local_death_is_registered(self):
        assert REWARD_COMPONENT_NAMES[21:23] == ("combat_death", "field_death")

    def test_no_duplicates(self):
        assert len(set(REWARD_COMPONENT_NAMES)) == len(REWARD_COMPONENT_NAMES)


class TestComponentWeightDerivation:
    """The balance rule: an event pays one side what it charges the other.

    Every case here leaves ``kill_payout_ratio`` at its default 1.0, which is
    where the rule holds exactly. ``TestKillPayoutRatio`` covers the one tier
    that is allowed to break it.
    """

    @staticmethod
    def _cfg(**kw):
        base = dict(
            win_weight=1.0, death_weight=0.4, damage_weight=0.3, kill_shot_fraction=0.5,
            facing_weight=0.06, closing_speed_weight=0.09,
            proximity_radius=400.0, shoot_quality_radius=200.0,
            enemy_neg_lambda_components=frozenset({"enemy_field_damage", "enemy_field_death",
                                                   "enemy_win"}),
            ally_zero_components=frozenset({"enemy_field_damage", "enemy_field_death",
                                            "enemy_win"}),
        )
        base.update(kw)
        return RewardConfig(**base)

    def test_a_combat_death_is_paid_for_exactly(self):
        """Charged to the victim, paid to whoever shot it, same total."""
        w = component_weights(self._cfg())
        assert w["kill_shot"] + w["kill_assist"] == pytest.approx(w["combat_death"])

    def test_a_field_death_is_paid_for_exactly(self):
        """kill_shot cannot fire on a field death, so enemy_field_death makes up
        precisely its share -- otherwise field kills would pay less than combat ones."""
        w = component_weights(self._cfg())
        assert w["kill_assist"] + w["enemy_field_death"] == pytest.approx(w["field_death"])
        assert w["enemy_field_death"] == pytest.approx(w["kill_shot"])

    def test_damage_is_paid_for_exactly_from_both_sources(self):
        w = component_weights(self._cfg())
        assert w["damage_dealt_enemy"] == pytest.approx(w["combat_damage_taken"])
        assert w["enemy_field_damage"] == pytest.approx(w["field_damage_taken"])

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
        for name in ("ally_combat_damage", "enemy_combat_damage", "ally_field_damage",
                     "ally_combat_death", "enemy_combat_death", "ally_field_death"):
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
        w = component_weights({"ally_win_weight": 1.5, "kill_shot_weight": 1.0,
                               "facing_weight": 0.1})
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
        assert component_weights(self._cfg()) == component_weights(
            self._cfg(kill_payout_ratio=1.0)
        )

    def test_the_kill_side_is_paid_the_ratio_times_the_charge(self):
        w = component_weights(self._cfg(death_weight=0.4, kill_payout_ratio=2.0))
        assert w["combat_death"] == pytest.approx(0.4)
        assert w["kill_shot"] + w["kill_assist"] == pytest.approx(0.8)

    def test_the_split_still_divides_the_payout_evenly(self):
        """f partitions the payout, not the charge, so the two knobs stay
        independent: changing one must not move the other's total."""
        for fraction in (0.0, 0.25, 0.5, 1.0):
            w = component_weights(
                self._cfg(kill_shot_fraction=fraction, kill_payout_ratio=2.0)
            )
            payout = w["kill_shot"] + w["kill_assist"]
            assert payout == pytest.approx(2 * w["combat_death"])
            assert w["kill_shot"] == pytest.approx(payout * fraction)

    def test_friendly_fire_follows_the_payout(self):
        """Blame for a teammate's death is priced at the same rate as credit for
        an enemy's, so raising the payout does not make friendly fire cheap."""
        w = component_weights(self._cfg(kill_payout_ratio=2.0))
        assert w["kill_ally_shot"] == pytest.approx(w["kill_shot"])
        assert w["kill_ally_assist"] == pytest.approx(w["kill_assist"])

    def test_a_field_kill_still_pays_what_a_combat_kill_pays(self):
        """enemy_field_death exists to cover the shot the field did not fire, so
        it has to track the payout rather than the charge."""
        w = component_weights(self._cfg(kill_payout_ratio=2.0))
        assert w["enemy_field_death"] == pytest.approx(w["kill_shot"])
        assert w["kill_assist"] + w["enemy_field_death"] == pytest.approx(
            2 * w["field_death"]
        )

    def test_damage_and_win_are_untouched(self):
        """The asymmetry is evidenced for the kill tier only."""
        balanced = component_weights(self._cfg())
        paid = component_weights(self._cfg(kill_payout_ratio=2.0))
        for name in ("ally_win", "enemy_win", "combat_death", "field_death",
                     "combat_damage_taken", "field_damage_taken",
                     "damage_dealt_enemy", "damage_dealt_ally",
                     "enemy_field_damage", "facing", "closing_speed"):
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

    def test_a_checkpoint_without_the_field_reads_as_balanced(self):
        """Runs recorded before the knob existed derived at 1:1, and reloading
        one must not silently re-price its rewards."""
        stored = {"win_weight": 1.0, "death_weight": 0.4, "damage_weight": 0.3,
                  "kill_shot_fraction": 0.5}
        w = component_weights(stored)
        assert w["kill_shot"] + w["kill_assist"] == pytest.approx(w["combat_death"])


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
        assert w["field_damage_taken"] == pytest.approx(0.3)
        assert w["damage_dealt_enemy"] == pytest.approx(0.6)

    def test_field_damage_follows_the_payout(self):
        """enemy_field_damage supplies the offensive side of damage nobody dealt,
        so it tracks what dealing damage pays, not what taking it charges."""
        w = component_weights(self._cfg(damage_payout_ratio=2.0))
        assert w["enemy_field_damage"] == pytest.approx(w["damage_dealt_enemy"])

    def test_friendly_fire_follows_the_payout(self):
        """Damaging a teammate is priced at the rate damaging an enemy pays, so
        raising the payout does not quietly make friendly fire cheap."""
        w = component_weights(self._cfg(damage_payout_ratio=2.0))
        assert w["damage_dealt_ally"] == pytest.approx(w["damage_dealt_enemy"])

    def test_kill_and_win_are_untouched(self):
        """The two ratios are independent knobs on independent tiers."""
        balanced = component_weights(self._cfg())
        paid = component_weights(self._cfg(damage_payout_ratio=2.0))
        for name in ("ally_win", "enemy_win", "combat_death", "field_death",
                     "kill_shot", "kill_assist", "kill_ally_shot",
                     "kill_ally_assist", "enemy_field_death",
                     "combat_damage_taken", "field_damage_taken",
                     "facing", "closing_speed"):
            assert paid[name] == pytest.approx(balanced[name])

    def test_the_two_ratios_compose(self):
        w = component_weights(
            self._cfg(death_weight=0.4, damage_weight=0.3,
                      kill_payout_ratio=2.0, damage_payout_ratio=3.0)
        )
        assert w["kill_shot"] + w["kill_assist"] == pytest.approx(0.8)
        assert w["damage_dealt_enemy"] == pytest.approx(0.9)

    @pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
    def test_a_nonsense_ratio_is_refused(self, bad):
        with pytest.raises(ValueError, match="damage_payout_ratio"):
            self._cfg(damage_payout_ratio=bad)

    def test_a_checkpoint_without_the_field_reads_as_balanced(self):
        stored = {"win_weight": 1.0, "death_weight": 0.4, "damage_weight": 0.3,
                  "kill_shot_fraction": 0.5}
        w = component_weights(stored)
        assert w["damage_dealt_enemy"] == pytest.approx(w["combat_damage_taken"])


class TestRun719Reconstruction:
    """The derivation can rebuild run 719's reward vector, and the profile says
    where it deliberately departs from it.

    719 is the strongest policy measured over a full budget and the only one
    whose weights were not derived. Solving the free numbers against measured
    gradient share was tried in runs 721 and 724 and lost both times, so the
    profile copies 719 rather than targeting a share — with exactly one intended
    departure at a time, which is what the second half of this class pins.

    The target is what 719 *trained under*, not what its config file said. Its
    lambda rows were normalized after the component weight was applied, which
    divided the weight back out of every global component: ``ally_win`` and
    ``enemy_win`` came out at an effective total of 1.0 against a configured
    1.5. Local components were below the clamp and passed through untouched.
    """

    # Run 719's effective weights. Global pair at 1.0, everything else as configured.
    EFFECTIVE_719 = {
        "ally_win": 1.0, "enemy_win": 1.0,
        "combat_death": 1.0, "field_death": 1.0,
        "kill_shot": 1.0, "kill_assist": 1.0,
        "combat_damage_taken": 0.5, "field_damage_taken": 0.5,
        "damage_dealt_enemy": 0.5, "damage_dealt_ally": 0.5,
        "facing": 0.1, "closing_speed": 0.1,
    }

    # The profile's numbers with both tiers balanced: run 725's vector, which
    # calibrated to parity with 719 at 133M and 154M.
    @staticmethod
    def _as_run_725():
        return dataclasses.replace(REWARDS, damage_payout_ratio=1.0)

    def test_every_component_719_carried_is_reproduced_exactly(self):
        w = component_weights(self._as_run_725())
        for name, expected in self.EFFECTIVE_719.items():
            assert w[name] == pytest.approx(expected), name

    def test_the_only_additions_are_the_ones_the_rule_requires(self):
        """719 had no source-split offensive components at all, so a field death
        paid its killers nothing. Anything else appearing here is a difference
        nobody argued for."""
        w = component_weights(self._as_run_725())
        added = {
            name for name, weight in w.items()
            if weight != 0.0 and name not in self.EFFECTIVE_719
        }
        assert added == {"enemy_field_death", "enemy_field_damage",
                         "kill_ally_shot", "kill_ally_assist"}

    def test_the_friendly_kill_pair_matches_the_penalty_719_folded_in(self):
        """719 had no kill_ally_* components, but it did penalize friendly kills:
        the term lived inside KillShotReward at an unscaled -1.0 share, under
        kill_shot's own weight. Extracting it changed what is weightable and
        visible, not how much friendly fire cost."""
        w = component_weights(self._as_run_725())
        assert w["kill_ally_shot"] == pytest.approx(self.EFFECTIVE_719["kill_shot"])
        assert w["kill_ally_assist"] == pytest.approx(self.EFFECTIVE_719["kill_assist"])

    def test_the_kill_ratio_is_what_makes_719_reachable(self):
        """Under the plain balance rule no setting of the free numbers reaches
        719, because it paid a kill 2.0 while charging a death 1.0. This fails
        if the kill knob is ever quietly returned to 1.0."""
        assert REWARDS.kill_payout_ratio == 2.0
        balanced = component_weights(
            dataclasses.replace(REWARDS, kill_payout_ratio=1.0, damage_payout_ratio=1.0)
        )
        assert balanced["kill_shot"] != pytest.approx(self.EFFECTIVE_719["kill_shot"])

    def test_the_shipped_vector_departs_from_719_in_the_damage_tier_only(self):
        """The profile is 725's vector plus one change. 725 matched 719 and did
        not beat it; 720 did, and its damage tier was tilted toward damage dealt.
        This names that departure so a second one cannot arrive unannounced."""
        assert REWARDS.damage_payout_ratio == 2.0
        shipped = component_weights(REWARDS)
        matched = component_weights(self._as_run_725())
        moved = {name for name in shipped if shipped[name] != pytest.approx(matched[name])}
        assert moved == {"damage_dealt_enemy", "damage_dealt_ally", "enemy_field_damage"}

    def test_the_departure_pays_damage_dealt_twice_what_damage_taken_charges(self):
        w = component_weights(REWARDS)
        assert w["damage_dealt_enemy"] == pytest.approx(2 * w["combat_damage_taken"])
        assert w["combat_damage_taken"] == pytest.approx(
            self.EFFECTIVE_719["combat_damage_taken"]
        )


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
        (AllyFieldDamageReward, "ship_field_damage"),
        (EnemyFieldDamageReward, "ship_field_damage"),
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
        (AllyFieldDeathReward, "ship_field_death"),
        (EnemyFieldDeathReward, "ship_field_death"),
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

    def test_field_final_blow_keeps_cumulative_combat_credit(self, cfg):
        prev = _kill_state(cfg)
        next_ = _kill_state(cfg)
        next_.ship_alive[0, 2] = False
        next_.ship_field_death[0, 2] = True
        next_.cumulative_damage_matrix[0, 0, 2] = 30.0
        next_.cumulative_damage_matrix[0, 1, 2] = 10.0

        reward = KillAssistReward(weight=1.0).compute(
            prev, torch.zeros(1, 4, 3), next_, torch.zeros(1, dtype=torch.bool)
        )

        assert reward[0, 0].item() == pytest.approx(0.75)
        assert reward[0, 1].item() == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Local damage rewards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("component_cls", "source_attr"),
    [
        (LocalCombatDamageTakenReward, "ship_combat_damage"),
        (LocalFieldDamageTakenReward, "ship_field_damage"),
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
        reward = r.compute(state, torch.zeros(2, 4, 3), state, torch.zeros(2, dtype=torch.bool))

        assert reward[0, 0].item() == pytest.approx(20.0)

    def test_ships_that_dealt_no_damage_get_zero(self, cfg):
        state = _make_4ship_state(cfg)
        state.damage_matrix[0, 0, 2] = 20.0

        r = LocalDamageDealtEnemyReward(weight=1.0)
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
        reward = r.compute(state, torch.zeros(2, 4, 3), state, torch.zeros(2, dtype=torch.bool))

        assert reward[0, 0].item() == pytest.approx(25.0)

    def test_friendly_fire_ignored(self, cfg):
        """Damage dealt to a teammate must not contribute to enemy damage reward."""
        state = _make_4ship_state(cfg)
        state.damage_matrix[0, 0, 1] = 30.0  # ship 0 hit ally ship 1

        r = LocalDamageDealtEnemyReward(weight=1.0)
        reward = r.compute(state, torch.zeros(2, 4, 3), state, torch.zeros(2, dtype=torch.bool))

        assert reward[0, 0].item() == pytest.approx(0.0)

    def test_dead_ship_gets_zero(self, cfg):
        state = _make_4ship_state(cfg)
        state.damage_matrix[0, 0, 2] = 20.0
        state.ship_alive[0, 0] = False

        r = LocalDamageDealtEnemyReward(weight=1.0)
        reward = r.compute(state, torch.zeros(2, 4, 3), state, torch.zeros(2, dtype=torch.bool))

        assert reward[0, 0].item() == pytest.approx(0.0)


class TestLocalDamageDealtAllyReward:
    def test_friendly_fire_gives_negative_reward(self, cfg):
        """Ship 0 dealt 30 damage to ally ship 1; ship 0 gets -30."""
        state = _make_4ship_state(cfg)
        state.damage_matrix[0, 0, 1] = 30.0

        r = LocalDamageDealtAllyReward(weight=1.0)
        reward = r.compute(state, torch.zeros(2, 4, 3), state, torch.zeros(2, dtype=torch.bool))

        assert reward[0, 0].item() == pytest.approx(-30.0)

    def test_enemy_damage_ignored(self, cfg):
        """Damage to enemies must not contribute to the friendly-fire penalty."""
        state = _make_4ship_state(cfg)
        state.damage_matrix[0, 0, 2] = 20.0  # ship 0 hit enemy ship 2

        r = LocalDamageDealtAllyReward(weight=1.0)
        reward = r.compute(state, torch.zeros(2, 4, 3), state, torch.zeros(2, dtype=torch.bool))

        assert reward[0, 0].item() == pytest.approx(0.0)

    def test_no_friendly_fire_gives_zero(self, cfg):
        state = _make_4ship_state(cfg)

        r = LocalDamageDealtAllyReward(weight=1.0)
        reward = r.compute(state, torch.zeros(2, 4, 3), state, torch.zeros(2, dtype=torch.bool))

        assert reward.abs().max().item() == 0.0

    def test_dead_ship_gets_zero(self, cfg):
        state = _make_4ship_state(cfg)
        state.damage_matrix[0, 0, 1] = 30.0
        state.ship_alive[0, 0] = False

        r = LocalDamageDealtAllyReward(weight=1.0)
        reward = r.compute(state, torch.zeros(2, 4, 3), state, torch.zeros(2, dtype=torch.bool))

        assert reward[0, 0].item() == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("component_cls", "source_attr"),
    [
        (LocalCombatDeathReward, "ship_combat_death"),
        (LocalFieldDeathReward, "ship_field_death"),
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
