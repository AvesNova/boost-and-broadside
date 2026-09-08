"""Focused tests for the Gate-1 frontline scripted-controller priorities."""

import pytest
import torch

from boost_and_broadside.agents.scripted_utils import select_targets
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


def _targeting(agent: StochasticScriptedAgent, state):
    closest_dist, target_idx, has_target, _ = select_targets(state, agent.ship_config)
    return agent._frontline_targets(state, closest_dist, target_idx, has_target)


def _bearing(source: complex, target: complex, world_size: tuple[float, float]) -> complex:
    width, height = world_size
    dx = (target.real - source.real + width / 2.0) % width - width / 2.0
    dy = (target.imag - source.imag + height / 2.0) % height - height / 2.0
    displacement = complex(dx, dy)
    return displacement / abs(displacement)


def test_low_health_engages_instead_of_healing_with_enemy_nearby() -> None:
    config, state = _frontline_state()
    state.ship_pos[0] = torch.tensor(
        [1000.0 + 100.0j, 2000.0 + 100.0j, 1100.0 + 100.0j, 10000.0 + 100.0j]
    )
    state.ship_health[0, 0] = 49.0
    agent = StochasticScriptedAgent(
        config,
        StochasticAgentConfig(
            frontline_heal_health_fraction=0.5,
            frontline_enemy_engage_distance=500.0,
        ),
    )

    distance, bearing, engage = _targeting(agent, state)

    assert engage[0, 0]
    assert distance[0, 0].item() == pytest.approx(100.0)
    assert bearing[0, 0].item() == pytest.approx(1.0 + 0.0j)


def test_low_health_returns_to_spawn_when_safe_and_defenses_uncontested() -> None:
    config, state = _frontline_state()
    state.ship_pos[0] = torch.tensor(
        [1000.0 + 100.0j, 2000.0 + 100.0j, 9000.0 + 100.0j, 10000.0 + 100.0j]
    )
    state.ship_health[0, 0] = 49.0
    agent = StochasticScriptedAgent(config, StochasticAgentConfig())

    distance, bearing, engage = _targeting(agent, state)

    assert not engage[0, 0]
    assert distance[0, 0].item() == pytest.approx(2000.0)
    assert bearing[0, 0].item() == pytest.approx(1.0 + 0.0j)


def test_low_health_does_not_heal_while_a_defense_is_contested() -> None:
    config, state = _frontline_state()
    state.ship_pos[0] = torch.tensor(
        [1000.0 + 100.0j, 9000.0 + 100.0j, 9000.0 + 100.0j, 12000.0 + 100.0j]
    )
    state.ship_health[0, 0] = 49.0
    agent = StochasticScriptedAgent(config, StochasticAgentConfig())

    distance, bearing, engage = _targeting(agent, state)

    assert not engage[0, 0]
    assert distance[0, 0].item() == pytest.approx(8000.0)
    assert bearing[0, 0].item() == pytest.approx(1.0 + 0.0j)


def test_healthy_ship_engages_nearby_enemy_over_objective_with_toroidal_bearing() -> None:
    config, state = _frontline_state()
    state.ship_pos[0] = torch.tensor(
        [16370.0 + 100.0j, 4000.0 + 100.0j, 10.0 + 100.0j, 10000.0 + 100.0j]
    )
    agent = StochasticScriptedAgent(
        config,
        StochasticAgentConfig(frontline_enemy_engage_distance=500.0),
    )

    distance, bearing, engage = _targeting(agent, state)

    assert engage[0, 0]
    assert distance[0, 0].item() == pytest.approx(24.0)
    assert bearing[0, 0].item() == pytest.approx(1.0 + 0.0j)


def test_stable_within_team_slots_split_between_attack_and_defense() -> None:
    config, state = _frontline_state(teams=[0, 1, 0, 1])
    state.ship_pos[0] = torch.tensor(
        [1000.0 + 100.0j, 10000.0 + 100.0j, 2000.0 + 100.0j, 11000.0 + 100.0j]
    )
    agent = StochasticScriptedAgent(
        config,
        StochasticAgentConfig(frontline_enemy_engage_distance=0.0),
    )

    _, bearing, engage = _targeting(agent, state)

    expected_targets = [
        9000.0 + 100.0j,  # first team-0 slot attacks team-1 defense
        6000.0 + 100.0j,  # first team-1 slot attacks team-0 defense
        6000.0 + 100.0j,  # second team-0 slot defends team-0 defense
        9000.0 + 100.0j,  # second team-1 slot defends team-1 defense
    ]
    expected = torch.tensor(
        [
            _bearing(state.ship_pos[0, i].item(), target, config.world_size)
            for i, target in enumerate(expected_targets)
        ],
        dtype=torch.complex64,
    )
    assert not engage.any()
    assert torch.allclose(bearing[0], expected)


def test_four_ship_team_sends_three_attackers_and_one_defender() -> None:
    config, state = _frontline_state(teams=[0, 0, 0, 0, 1, 1, 1, 1])
    state.ship_pos[0] = torch.tensor(
        [
            1000.0 + 100.0j,
            1400.0 + 100.0j,
            1800.0 + 100.0j,
            2200.0 + 100.0j,
            10000.0 + 100.0j,
            10400.0 + 100.0j,
            10800.0 + 100.0j,
            11200.0 + 100.0j,
        ]
    )
    agent = StochasticScriptedAgent(
        config,
        StochasticAgentConfig(frontline_enemy_engage_distance=0.0),
    )

    _, bearing, engage = _targeting(agent, state)

    expected_targets = [9000.0 + 100.0j] * 3 + [6000.0 + 100.0j]
    expected = torch.tensor(
        [
            _bearing(state.ship_pos[0, i].item(), target, config.world_size)
            for i, target in enumerate(expected_targets)
        ],
        dtype=torch.complex64,
    )
    assert not engage.any()
    assert torch.allclose(bearing[0, :4], expected)


def test_navigation_targets_never_shoot() -> None:
    config, state = _frontline_state()
    state.ship_pos[0] = torch.tensor(
        [1000.0 + 100.0j, 2000.0 + 100.0j, 10000.0 + 100.0j, 11000.0 + 100.0j]
    )
    agent = StochasticScriptedAgent(
        config,
        StochasticAgentConfig(frontline_enemy_engage_distance=0.0),
    )

    actions, probs = agent.get_actions_and_probs(state)

    assert torch.equal(probs[..., -2], torch.ones_like(probs[..., -2]))
    assert torch.equal(probs[..., -1], torch.zeros_like(probs[..., -1]))
    assert not actions[..., 2].any()


def test_zero_zone_combat_mode_never_enters_frontline_path(monkeypatch) -> None:
    config = ShipConfig()
    state = make_state(num_envs=1, max_ships=2, max_bullets=0, ship_config=config)
    state.ship_team_id[0] = torch.tensor([0, 1], dtype=torch.int32)
    state.ship_pos[0] = torch.tensor([100.0 + 100.0j, 200.0 + 100.0j])
    agent = StochasticScriptedAgent(config, StochasticAgentConfig())

    def fail(_state):
        raise AssertionError("legacy combat mode entered the frontline controller")

    monkeypatch.setattr(agent, "_get_frontline_actions_and_probs", fail)
    actions, probs = agent.get_actions_and_probs(state)

    assert actions.shape == (1, 2, 3)
    assert probs.shape == (1, 2, 12)


@pytest.mark.parametrize(
    "overrides",
    [
        {"frontline_heal_health_fraction": -0.01},
        {"frontline_heal_health_fraction": 1.01},
        {"frontline_enemy_engage_distance": -1.0},
    ],
)
def test_frontline_scripted_config_rejects_invalid_values(overrides: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        StochasticAgentConfig(**overrides)
