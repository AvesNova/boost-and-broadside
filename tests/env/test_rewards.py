"""The five reward levels: their event rules, payees and zero-sum invariant."""

from dataclasses import replace

import pytest
import torch

from boost_and_broadside.config import MatchResult, RewardConfig, ShipConfig
from boost_and_broadside.config.core import EnvConfig
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.rewards import (
    REWARD_COMPONENT_NAMES,
    DamageReward,
    KillDeathReward,
    OutcomeReward,
    ZoneCaptureReward,
    build_reward_components,
    component_payout_ratios,
    component_weights,
)
from boost_and_broadside.profiles import PROFILES
from tests.conftest import make_state

REWARDS = RewardConfig(
    win_weight=1.0,
    zone_capture_weight=2.0,
    zone_progress_weight=2.0,
    death_weight=0.3,
    damage_weight=0.2,
    kill_payout_ratio=2.0,
    damage_payout_ratio=1.5,
    capture_payout_ratio=3.0,
)
DONE = torch.zeros(1, dtype=torch.bool)


def _duel_state():
    """Ships 0, 1 on team 0 and ships 2, 3 on team 1."""
    state = make_state(num_envs=1, max_ships=4)
    state.ship_team_id = torch.tensor([[0, 0, 1, 1]], dtype=torch.int32)
    return state


def _events(component, state):
    return component.events(state, None, state, DONE)[0]  # (N, E)


def _column(component, name: str) -> int:
    return component.event_names.index(name)


# ---------------------------------------------------------------------------
# Registry and weights
# ---------------------------------------------------------------------------


def test_the_builder_returns_one_component_per_level_in_order():
    built = [c.name for c in build_reward_components(REWARDS, ShipConfig())]
    assert built == list(REWARD_COMPONENT_NAMES)


def test_component_weights_are_the_level_weights():
    assert component_weights(REWARDS) == {
        "outcome": 1.0,
        "zone_capture": 2.0,
        "zone_progress": 2.0,
        "kill_death": 0.3,
        "damage": 0.2,
    }


@pytest.mark.parametrize("bias", [0.0, 0.5, 1.0])
def test_the_offensive_bias_anneals_every_ratio_toward_one(bias):
    ratios = component_payout_ratios(REWARDS, bias)
    assert ratios["kill_death"] == pytest.approx(1.0 + bias * 1.0)
    assert ratios["damage"] == pytest.approx(1.0 + bias * 0.5)
    assert ratios["zone_capture"] == pytest.approx(1.0 + bias * 2.0)


def test_compute_is_the_weighted_sum_of_the_events():
    state = _duel_state()
    state.ship_combat_damage[0, 2] = 10.0
    state.damage_matrix[0, 0, 2] = 10.0
    component = DamageReward(weight=0.2, payout_ratio=1.5)
    expected = 0.2 * _events(component, state).sum(-1)
    assert torch.allclose(component.compute(state, None, state, DONE)[0], expected)


# ---------------------------------------------------------------------------
# Outcome
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (MatchResult.TEAM0_WIN, [1.0, 1.0, -1.0, -1.0]),
        (MatchResult.TEAM1_WIN, [-1.0, -1.0, 1.0, 1.0]),
        (MatchResult.DRAW, [0.0, 0.0, 0.0, 0.0]),
    ],
)
def test_the_outcome_is_broadcast_to_each_team(result, expected):
    state = _duel_state()
    state.match_result[:] = int(result)
    events = OutcomeReward(1.0).events(state, None, state, torch.ones(1, dtype=torch.bool))
    assert events[0, :, 0].tolist() == expected


def test_no_outcome_before_the_terminal_transition():
    state = _duel_state()
    state.match_result[:] = int(MatchResult.TEAM0_WIN)
    assert not OutcomeReward(1.0).events(state, None, state, DONE).any()


# ---------------------------------------------------------------------------
# Damage
# ---------------------------------------------------------------------------


def test_an_enemy_hit_charges_the_target_and_pays_the_shooter_the_ratio():
    state = _duel_state()
    state.ship_combat_damage[0, 2] = 10.0
    state.damage_matrix[0, 0, 2] = 10.0
    component = DamageReward(1.0, payout_ratio=1.5)
    events = _events(component, state)
    assert events[2, _column(component, "taken")] == -10.0
    assert events[0, _column(component, "dealt")] == 15.0


def test_simultaneous_hits_share_only_the_applied_damage():
    state = _duel_state()
    state.ship_combat_damage[0, 2] = 6.0  # shield had 6 left of 10 + 10 incoming
    state.damage_matrix[0, 0, 2] = 10.0
    state.damage_matrix[0, 1, 2] = 10.0
    component = DamageReward(1.0, payout_ratio=1.0)
    dealt = _events(component, state)[:, _column(component, "dealt")]
    assert dealt.tolist() == pytest.approx([3.0, 3.0, 0.0, 0.0])


def test_friendly_fire_blames_the_shooter_and_pays_the_other_team_twice():
    state = _duel_state()
    state.ship_combat_damage[0, 1] = 10.0
    state.damage_matrix[0, 0, 1] = 10.0
    component = DamageReward(1.0, payout_ratio=1.5)
    events = _events(component, state)
    assert events[0, _column(component, "friendly_fire")] == -15.0
    assert events[2:, _column(component, "unforced")].tolist() == [15.0, 15.0]


def test_boundary_damage_pays_the_opposing_team():
    state = _duel_state()
    state.ship_boundary_damage[0, 3] = 4.0
    component = DamageReward(1.0, payout_ratio=1.5)
    events = _events(component, state)
    assert events[3, _column(component, "taken")] == -4.0
    assert events[:2, _column(component, "unforced")].tolist() == [3.0, 3.0]


def test_recharge_pays_the_ship_and_charges_back_its_attacker():
    state = _duel_state()
    state.ship_shield_recharge[0, 2] = 5.0
    state.recharge_matrix[0, 0, 2] = 5.0
    component = DamageReward(1.0, payout_ratio=2.0)
    events = _events(component, state)
    assert events[2, _column(component, "recharge")] == 5.0
    assert events[0, _column(component, "charge_back")] == -10.0


def test_recharge_of_damage_nobody_caused_charges_the_opposing_team():
    state = _duel_state()
    state.ship_shield_recharge[0, 2] = 4.0
    state.ship_unattributed_recharge[0, 2] = 4.0
    component = DamageReward(1.0, payout_ratio=1.0)
    unforced = _events(component, state)[:, _column(component, "unforced")]
    assert unforced.tolist() == [-2.0, -2.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Kill and death
# ---------------------------------------------------------------------------


def test_a_death_charges_the_dying_ship_one():
    state = _duel_state()
    state.ship_combat_death[0, 2] = True
    state.damage_matrix[0, 0, 2] = 10.0
    component = KillDeathReward(1.0, payout_ratio=2.0)
    assert _events(component, state)[2, _column(component, "death")] == -1.0


def test_a_sole_killer_takes_the_whole_ratio():
    state = _duel_state()
    state.ship_combat_death[0, 2] = True
    state.damage_matrix[0, 0, 2] = 10.0
    state.cumulative_damage_matrix[0, 0, 2] = 90.0
    component = KillDeathReward(1.0, payout_ratio=2.0)
    assert _events(component, state)[0, _column(component, "kill")] == pytest.approx(2.0)


def test_the_share_is_half_final_impact_and_half_outstanding_damage():
    state = _duel_state()
    state.ship_combat_death[0, 2] = True
    state.damage_matrix[0, 0, 2] = 10.0  # landed the killing hit
    state.cumulative_damage_matrix[0, 1, 2] = 100.0  # did all the damage before
    component = KillDeathReward(1.0, payout_ratio=1.0)
    kill = _events(component, state)[:, _column(component, "kill")]
    assert kill[:2].tolist() == pytest.approx([0.5, 0.5])


def test_without_outstanding_damage_the_final_hit_takes_the_whole_share():
    state = _duel_state()
    state.ship_combat_death[0, 2] = True
    state.damage_matrix[0, 0, 2] = 10.0
    component = KillDeathReward(1.0, payout_ratio=1.0)
    assert _events(component, state)[0, _column(component, "kill")] == pytest.approx(1.0)


def test_a_boundary_death_pays_the_opposing_team():
    state = _duel_state()
    state.ship_boundary_death[0, 2] = True
    state.cumulative_damage_matrix[0, 0, 2] = 50.0
    component = KillDeathReward(1.0, payout_ratio=2.0)
    events = _events(component, state)
    assert events[:2, _column(component, "unforced")].tolist() == [1.0, 1.0]
    assert events[0, _column(component, "kill")] == 0.0


def test_an_allied_kill_is_blamed_and_the_other_team_paid_twice():
    state = _duel_state()
    state.ship_combat_death[0, 1] = True
    state.damage_matrix[0, 2, 1] = 10.0  # the enemy landed the final hit
    state.cumulative_damage_matrix[0, 0, 1] = 80.0  # the ally did the damage
    component = KillDeathReward(1.0, payout_ratio=1.0)
    events = _events(component, state)
    assert events[0, _column(component, "ally_kill")] == pytest.approx(-0.5)
    assert events[2, _column(component, "kill")] == pytest.approx(0.5)
    assert events[2:, _column(component, "unforced")].tolist() == pytest.approx([0.5, 0.5])


# ---------------------------------------------------------------------------
# Zone capture
# ---------------------------------------------------------------------------


def test_a_capture_pays_the_ratio_and_charges_one():
    state = _duel_state()
    state.ship_capture_gain = torch.tensor([[0.75, 0.25, 0.0, 0.0]])
    state.ship_capture_loss = torch.tensor([[0.0, 0.0, 0.5, 0.5]])
    component = ZoneCaptureReward(1.0, payout_ratio=3.0)
    events = _events(component, state)
    assert events[:, 0].tolist() == [2.25, 0.75, 0.0, 0.0]
    assert events[:, 1].tolist() == [0.0, 0.0, -0.5, -0.5]


# ---------------------------------------------------------------------------
# The zero-sum invariant on real play
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def even_rollout():
    """Per-level rewards at every ratio 1 over scripted Frontline play.

    Captures are shortened so the zone levels fire inside a short test.
    """
    from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
    from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
    from boost_and_broadside.env.perception import team_visibility_from_state

    profile = PROFILES["rl"]
    frontline = replace(profile.frontline, capture_seconds=1.0)
    config = EnvConfig(
        num_ships=10,
        num_fields=4,
        max_bullets=profile.max_bullets,
        max_episode_steps=profile.max_episode_steps,
        frontline=frontline,
        vision_range=profile.vision_range,
    )
    env = TensorEnv(16, profile.ship_config, config, "cpu")
    env.reset(seed=5)
    agent = StochasticScriptedAgent(profile.ship_config, StochasticAgentConfig())
    even = replace(
        profile.rewards, kill_payout_ratio=1.0, damage_payout_ratio=1.0, capture_payout_ratio=1.0
    )
    components = build_reward_components(even, profile.ship_config)
    rewards = []
    for _ in range(1000):
        visibility = team_visibility_from_state(env.state, profile.ship_config, config, False)
        actions = agent.get_actions(env.state, visibility.ship)
        previous = env.state.clone()
        dones, truncated = env.tick(actions)
        rewards.append(
            torch.stack(
                [c.compute(previous, actions, env.state, dones | truncated) for c in components],
                dim=-1,
            )
        )
        finished = dones | truncated
        if bool(finished.any()):
            env.reset_envs(finished)
    return torch.stack(rewards)  # (T, B, N, K)


@pytest.mark.parametrize("level", REWARD_COMPONENT_NAMES)
def test_each_level_sums_to_zero_over_both_teams_at_ratio_one(even_rollout, level):
    index = REWARD_COMPONENT_NAMES.index(level)
    per_env = even_rollout[..., index].sum(-1)  # (T, B)
    assert per_env.abs().max().item() < 1e-4


@pytest.mark.parametrize("level", ["zone_capture", "zone_progress", "kill_death", "damage"])
def test_each_level_fires_in_the_zero_sum_rollout(even_rollout, level):
    index = REWARD_COMPONENT_NAMES.index(level)
    assert even_rollout[..., index].abs().sum().item() > 0.0


def test_each_completed_capture_pays_out_exactly_one_share_per_side(even_rollout):
    """Presence and ledger halves each sum to one, so a capture pays w once."""
    index = REWARD_COMPONENT_NAMES.index("zone_capture")
    weight = PROFILES["rl"].rewards.zone_capture_weight
    paid = even_rollout[..., index].clamp(min=0.0).sum(-1) / weight  # (T, B)
    fired = paid[paid > 0]
    assert torch.allclose(fired, fired.round(), atol=1e-4)
