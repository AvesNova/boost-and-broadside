"""State-derived Frontline strategy and exact dogfighter endpoint contracts."""

import copy

import pytest
import torch

from boost_and_broadside.agents.frontline_strategy import frontline_strategy
from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config import ShipConfig, ZoneRole
from tests.conftest import make_state


def _frontline_state(*, teams: list[int] | None = None):
    teams = teams or [0, 0, 1, 1]
    config = ShipConfig(world_size=(16384.0, 16384.0))
    state = make_state(
        num_envs=1,
        max_ships=len(teams),
        max_bullets=0,
        ship_config=config,
    )
    state.ship_team_id[0] = torch.tensor(teams, dtype=torch.int32)
    state.zone_pos = torch.tensor(
        [[100.0 + 100.0j, 3000.0 + 100.0j, 6000.0 + 100.0j, 9000.0 + 100.0j, 12000.0 + 100.0j]],
        dtype=torch.complex64,
    )
    state.zone_radius = torch.full((1, 5), 200.0)
    state.zone_roles = torch.tensor(
        [
            [
                int(ZoneRole.NEUTRAL),
                int(ZoneRole.TEAM0_SPAWN),
                int(ZoneRole.TEAM0_DEFENSE),
                int(ZoneRole.TEAM1_DEFENSE),
                int(ZoneRole.TEAM1_SPAWN),
            ]
        ],
        dtype=torch.int8,
    )
    state.zone_capture_progress = torch.zeros((1, 5))
    state.zone_capture_direction = torch.zeros((1, 5), dtype=torch.int8)
    return config, state


def _scenario(size=4):
    ship, state = _frontline_state(teams=[0] * size + [1] * size)
    state.ship_pos[0, :size] = torch.linspace(5500, 6500, size).to(torch.complex64) + 100j
    state.ship_pos[0, size:] = torch.linspace(8000, 9500, size).to(torch.complex64) + 200j
    visibility = torch.ones((1, 2, 2 * size), dtype=torch.bool)
    return ship, state, visibility


def test_valid_defender_reduces_redundant_demand():
    ship, state, visibility = _scenario()
    state.ship_pos[0, 1] = 12000 + 100j
    before = frontline_strategy(state, ship, StochasticAgentConfig(), visibility)
    state.ship_pos[0, 1] = 6000 + 100j
    after = frontline_strategy(state, ship, StochasticAgentConfig(), visibility)
    assert after.zone_need[0, 0, 2] < before.zone_need[0, 0, 2]


def test_sole_defender_has_greater_marginal_need_than_remote_ally():
    ship, state, visibility = _scenario(2)
    state.ship_pos[0, :2] = torch.tensor([6000 + 100j, 3000 + 100j])
    result = frontline_strategy(state, ship, StochasticAgentConfig(), visibility)
    assert result.zone_need[0, 0, 2] > result.zone_need[0, 1, 2]


def test_enemy_pressure_recruits_reinforcements():
    ship, state, visibility = _scenario()
    before = frontline_strategy(state, ship, StochasticAgentConfig(), visibility)
    state.ship_pos[0, 4] = 6000 + 100j
    after = frontline_strategy(state, ship, StochasticAgentConfig(), visibility)
    assert torch.all(after.zone_need[0, :4, 2] > before.zone_need[0, :4, 2])


def test_aggression_does_not_remove_preemptive_defense_demand():
    ship, state, visibility = _scenario()
    state.ship_pos[0, :4] = 3000 + 100j
    defensive = frontline_strategy(
        state, ship, StochasticAgentConfig(frontline_aggression=1), visibility
    )
    assert torch.all(defensive.zone_need[0, :4, 2] > 0.5)


def test_capture_progress_recruits_defenders_through_occlusion():
    ship, state, visibility = _scenario()
    visibility[:, 0, 4:] = False
    before = frontline_strategy(state, ship, StochasticAgentConfig(), visibility)
    state.zone_capture_direction[0, 2] = -1
    state.zone_capture_progress[0, 2] = 0.5
    after = frontline_strategy(state, ship, StochasticAgentConfig(), visibility)
    assert torch.all(after.zone_need[0, :4, 2] > before.zone_need[0, :4, 2])
    assert torch.all(after.zone_preference[0, :4, 2] > before.zone_preference[0, :4, 2])


def test_aggression_monotonically_biases_combat_and_objectives():
    ship, state, visibility = _scenario()
    results = [
        frontline_strategy(state, ship, StochasticAgentConfig(frontline_aggression=a), visibility)
        for a in (-1, 0, 1)
    ]
    for lower, upper in zip(results, results[1:]):
        assert torch.all(upper.combat_score >= lower.combat_score)
        assert torch.all(upper.zone_preference[0, :4, 3] > lower.zone_preference[0, :4, 3])


def test_hidden_enemy_changes_do_not_change_labels():
    ship, state, visibility = _scenario()
    visibility[:, 0, 4:] = False
    agent = StochasticScriptedAgent(ship, StochasticAgentConfig())
    before = agent.get_actions_and_probs(state, visibility)[1][:, :4]
    state.ship_pos[:, 4:] = state.ship_pos[:, :4] + 1j
    state.ship_health[:, 4:] = 1
    state.ship_vel[:, 4:] = 100j
    state.ship_alive[:, 4:] = False
    after = agent.get_actions_and_probs(state, visibility)[1][:, :4]
    assert torch.equal(before, after)


@pytest.mark.parametrize("flat", [False, True])
def test_healthy_close_range_is_exact_legacy_dogfighter(flat):
    ship, state, visibility = _scenario(1)
    state.ship_pos[0] = torch.tensor([6000 + 100j, 6100 + 110j])
    state.ship_health[:] = ship.max_health
    agent = StochasticScriptedAgent(
        ship, StochasticAgentConfig(flat_action_sampling=flat, team_target_distance_prob=(0.2, 0.8))
    )
    frontline = agent.get_actions_and_probs(state, visibility)[1]
    state.zone_pos = state.zone_pos[:, :0]
    legacy = agent.get_actions_and_probs(state, visibility)[1]
    assert torch.equal(frontline, legacy)


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_logit_blend_endpoints_are_exact_and_numerically_continuous(alpha):
    old = torch.tensor([[[0.0, 0.2, 0.8]]])
    new = torch.tensor([[[0.7, 0.3, 0.0]]])
    endpoint = StochasticScriptedAgent._blend_probs(old, new, torch.tensor([[alpha]]))
    nearby = StochasticScriptedAgent._blend_probs(
        old, new, torch.tensor([[1e-7 if alpha == 0 else 1 - 1e-7]])
    )
    assert torch.equal(endpoint, old if alpha == 0 else new)
    torch.testing.assert_close(endpoint, nearby, atol=1e-6, rtol=1e-5)


def test_interior_blends_logits_not_probabilities():
    old = torch.tensor([[[0.1, 0.9]]])
    new = torch.tensor([[[0.6, 0.4]]])
    expected = (old * new).sqrt()
    expected /= expected.sum(-1, keepdim=True)
    actual = StochasticScriptedAgent._blend_probs(old, new, torch.tensor([[0.5]]))
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("size", [4, 16, 64])
def test_permutation_equivariance_and_finite_distributions(size):
    ship, state, visibility = _scenario(size)
    agent = StochasticScriptedAgent(ship, StochasticAgentConfig())
    before = agent.get_actions_and_probs(state, visibility)[1]
    permutation = torch.randperm(2 * size)
    changed = copy.deepcopy(state)
    for name, value in vars(state).items():
        if name.startswith("ship_") and isinstance(value, torch.Tensor) and value.ndim >= 2:
            setattr(changed, name, value[:, permutation])
    after = agent.get_actions_and_probs(changed, visibility[:, :, permutation])[1]
    torch.testing.assert_close(after, before[:, permutation], atol=1e-6, rtol=1e-5)
    assert torch.isfinite(after).all() and (after >= 0).all()
    for head in after.split([3, 7, 2], -1):
        torch.testing.assert_close(head.sum(-1), torch.ones_like(head[..., 0]))


def test_recovery_increases_smoothly_as_health_falls():
    ship, state, visibility = _scenario()
    state.ship_pos[:] = state.ship_pos[:, :1] + torch.arange(state.max_ships) * 30
    values = []
    for health in (100, 75, 50, 25, 1):
        state.ship_health[:, :4] = health
        values.append(
            frontline_strategy(state, ship, StochasticAgentConfig(), visibility).recovery[:, :4]
        )
    assert all(torch.all(a < b) for a, b in zip(values, values[1:]))


def test_separation_pushes_apart_and_stays_bounded_in_dense_fleet():
    ship, state, visibility = _scenario(64)
    state.ship_pos[0, :64] = torch.linspace(6000, 6010, 64).to(torch.complex64) + 100j
    result = frontline_strategy(state, ship, StochasticAgentConfig(), visibility)
    assert result.separation[0, 0].real < 0 < result.separation[0, 63].real
    assert torch.all(result.separation.abs() <= 1)


@pytest.mark.parametrize(
    "overrides",
    [
        {"frontline_combat_radius": 200},
        {"frontline_zone_radius": 0},
        {"frontline_aggression": float("nan")},
        {"frontline_zone_margin": -1},
        {"frontline_separation_radius": -1},
        {"frontline_recovery_health": 0},
    ],
)
def test_invalid_strategy_configuration_rejected(overrides):
    with pytest.raises(ValueError):
        StochasticAgentConfig(**overrides)


def test_missing_visibility_cannot_reveal_enemies():
    ship, state, _ = _scenario(1)
    agent = StochasticScriptedAgent(ship, StochasticAgentConfig())
    before = agent.get_actions_and_probs(state)[1][:, :1]
    state.ship_pos[0, 1] = state.ship_pos[0, 0] + 1j
    after = agent.get_actions_and_probs(state)[1][:, :1]
    assert torch.equal(before, after)


def test_distant_extra_ships_do_not_dilute_local_separation():
    ship, small, visibility = _scenario(2)
    small.ship_pos[0, :2] = torch.tensor([6000 + 100j, 6020 + 100j])
    expected = frontline_strategy(small, ship, StochasticAgentConfig(), visibility)
    _, large, large_visibility = _scenario(64)
    large.ship_pos[:] = 13000 + 8000j
    large.ship_pos[0, :2] = small.ship_pos[0, :2]
    large.ship_pos[0, 64:66] = small.ship_pos[0, 2:]
    actual = frontline_strategy(large, ship, StochasticAgentConfig(), large_visibility)
    torch.testing.assert_close(actual.separation[:, :2], expected.separation[:, :2])
    torch.testing.assert_close(actual.combat_score[:, :2], expected.combat_score[:, :2])
    torch.testing.assert_close(actual.zone_need[:, :2], expected.zone_need[:, :2])


def test_dead_ships_have_no_strategic_influence_and_take_noop():
    ship, state, visibility = _scenario()
    state.ship_alive[0, 1] = False
    agent = StochasticScriptedAgent(ship, StochasticAgentConfig())
    before = agent.get_actions_and_probs(state, visibility)[1]
    state.ship_pos[0, 1] = 6000 + 100j
    state.ship_health[0, 1] = 1
    actions, after = agent.get_actions_and_probs(state, visibility)
    torch.testing.assert_close(before[:, 0], after[:, 0])
    assert torch.equal(actions[:, 1], torch.zeros_like(actions[:, 1]))


def test_outer_range_is_exact_strategy_on_all_heads():
    ship, state, visibility = _scenario(1)
    state.ship_pos[0] = torch.tensor([6000 + 100j, 6800 + 100j])
    config = StochasticAgentConfig()
    agent = StochasticScriptedAgent(ship, config)
    result = frontline_strategy(state, ship, config, visibility)
    power, turn, _ = agent._compute_action_probs(
        state, result.distance, result.bearing, torch.zeros_like(result.bearing), state.ship_alive
    )
    expected = torch.cat((power, turn, torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])), -1)
    assert torch.equal(agent.get_actions_and_probs(state, visibility)[1], expected)
