"""Team-shared range and opaque-core LOS perception tests."""

from dataclasses import replace

import pytest
import torch

from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config import EnvConfig, ShipConfig
from boost_and_broadside.config.defaults import REWARDS
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.frontline import frontline_ship_config, roles_from_front
from boost_and_broadside.env.observation import ObsKey, perceived_observation_from_state
from boost_and_broadside.env.perception import team_visibility_from_state
from boost_and_broadside.env.wrapper import SOURCE_STAT_NAMES, YemongEnvWrapper
from tests.conftest import make_state


def _config(*, vision_range: float | None = 300.0, num_fields: int = 0) -> EnvConfig:
    return EnvConfig(
        num_ships=4,
        max_bullets=2,
        max_episode_steps=60,
        num_fields=num_fields,
        vision_range=vision_range,
    )


def _state(*, num_fields: int = 0):
    ship = ShipConfig()
    state = make_state(
        num_envs=1,
        max_ships=4,
        max_bullets=2,
        ship_config=ship,
        num_fields=num_fields,
    )
    state.ship_team_id[0] = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
    return ship, state


def test_team_sharing_exposes_an_enemy_seen_by_only_one_ally() -> None:
    ship, state = _state()
    state.ship_pos[0] = torch.tensor([100 + 100j, 700 + 700j, 350 + 100j, 700 + 350j])

    sight = team_visibility_from_state(state, ship, _config())

    assert sight.observer_ship[0, 0, 2]
    assert not sight.observer_ship[0, 1, 2]
    assert sight.ship[0, 0, 2]
    assert not sight.ship[0, 0, 3]
    assert sight.ship[0, 0, :2].all()
    assert sight.ship[0, 1, 2:].all()


def test_successful_shot_reveals_firing_ship_beyond_range_and_los() -> None:
    ship, state = _state(num_fields=1)
    state.ship_pos[0] = torch.tensor([100 + 100j, 120 + 100j, 800 + 100j, 820 + 100j])
    state.field_pos[0, 0] = 450 + 100j
    state.field_radius[0, 0] = 100.0
    state.field_transition_width[0, 0] = 40.0
    config = _config(vision_range=200.0, num_fields=1)

    hidden = team_visibility_from_state(state, ship, config)
    assert not hidden.ship[0, 0, 2]
    assert not hidden.range_only_ship[0, 0, 2]
    assert not hidden.los_ship[0, 0, 2]

    state.ship_is_shooting[0, 2] = True
    revealed = team_visibility_from_state(state, ship, config)

    assert revealed.ship[0, 0, 2]
    assert revealed.observer_ship[0, :2, 2].all()
    assert not revealed.range_only_ship[0, 0, 2]
    assert not revealed.los_ship[0, 0, 2]


def test_range_uses_shortest_toroidal_displacement() -> None:
    ship, state = _state()
    state.ship_pos[0] = torch.tensor([20 + 200j, 500 + 500j, 1000 + 200j, 500 + 900j])

    sight = team_visibility_from_state(state, ship, _config(vision_range=50.0))

    assert sight.observer_ship[0, 0, 2]
    assert sight.ship[0, 0, 2]


def test_field_core_blocks_a_clear_range_sighting() -> None:
    ship, state = _state(num_fields=1)
    state.ship_pos[0] = torch.tensor([100 + 100j, 100 + 700j, 500 + 100j, 700 + 700j])
    state.field_pos[0, 0] = 300 + 100j
    state.field_radius[0, 0] = 90.0
    state.field_transition_width[0, 0] = 40.0

    sight = team_visibility_from_state(state, ship, _config(vision_range=500.0, num_fields=1))

    assert sight.range_only_observer_ship[0, 0, 2]
    assert not sight.observer_ship[0, 0, 2]
    assert not sight.ship[0, 0, 2]


def test_a_ship_inside_a_field_cannot_see_out_of_it() -> None:
    ship, state = _state(num_fields=1)
    state.ship_pos[0] = torch.tensor([300 + 100j, 100 + 700j, 500 + 100j, 700 + 700j])
    state.field_pos[0, 0] = 300 + 100j
    state.field_radius[0, 0] = 90.0
    state.field_transition_width[0, 0] = 40.0

    sight = team_visibility_from_state(state, ship, _config(vision_range=500.0, num_fields=1))

    assert sight.range_only_observer_ship[0, 0, 2]
    assert not sight.observer_ship[0, 0, 2]


def test_a_ship_outside_a_field_cannot_see_into_it() -> None:
    ship, state = _state(num_fields=1)
    state.ship_pos[0] = torch.tensor([100 + 100j, 100 + 700j, 300 + 100j, 700 + 700j])
    state.field_pos[0, 0] = 300 + 100j
    state.field_radius[0, 0] = 90.0
    state.field_transition_width[0, 0] = 40.0

    sight = team_visibility_from_state(state, ship, _config(vision_range=500.0, num_fields=1))

    assert sight.range_only_observer_ship[0, 0, 2]
    assert not sight.observer_ship[0, 0, 2]
    assert not sight.ship[0, 0, 2]


def test_two_ships_sharing_one_field_still_see_each_other() -> None:
    """The single exemption: a line that never leaves the core it starts in."""

    ship, state = _state(num_fields=1)
    state.ship_pos[0] = torch.tensor([260 + 100j, 100 + 700j, 340 + 100j, 700 + 700j])
    state.field_pos[0, 0] = 300 + 100j
    state.field_radius[0, 0] = 90.0
    state.field_transition_width[0, 0] = 40.0

    sight = team_visibility_from_state(state, ship, _config(vision_range=500.0, num_fields=1))

    assert sight.observer_ship[0, 0, 2]
    assert sight.ship[0, 0, 2]


def test_the_transparent_transition_band_does_not_block() -> None:
    """Only the flat core is opaque; the graded interface is see-through."""

    ship, state = _state(num_fields=1)
    # Core radius is 90 - 20 = 70, so a target 80 px out sits in the band. It
    # is on the near side, so reaching it never enters the core.
    state.ship_pos[0] = torch.tensor([100 + 100j, 100 + 700j, 220 + 100j, 700 + 700j])
    state.field_pos[0, 0] = 300 + 100j
    state.field_radius[0, 0] = 90.0
    state.field_transition_width[0, 0] = 40.0

    sight = team_visibility_from_state(state, ship, _config(vision_range=500.0, num_fields=1))

    assert sight.observer_ship[0, 0, 2]


def test_enemy_bullets_use_range_and_los_while_own_bullets_are_known() -> None:
    ship, state = _state()
    state.ship_pos[0] = torch.tensor([100 + 100j, 100 + 700j, 500 + 100j, 700 + 700j])
    state.bullet_active[0, 0, 0] = True
    state.bullet_pos[0, 0, 0] = 100 + 500j
    state.bullet_active[0, 2, 0] = True
    state.bullet_pos[0, 2, 0] = 350 + 100j

    sight = team_visibility_from_state(state, ship, _config())

    assert sight.bullet[0, 0, 0, 0]  # own projectile, even outside sensor range
    assert sight.bullet[0, 0, 2, 0]  # enemy projectile seen by the first ally
    assert not sight.bullet[0, 1, 0, 0]  # enemy projectile outside team-1 range


def test_none_range_is_explicit_omniscient_compatibility_mode() -> None:
    ship, state = _state(num_fields=1)
    state.ship_pos[0] = torch.tensor([100 + 100j, 100 + 700j, 500 + 100j, 700 + 700j])
    state.field_pos[0, 0] = 300 + 100j
    state.field_radius[0, 0] = 90.0
    state.field_transition_width[0, 0] = 40.0

    sight = team_visibility_from_state(state, ship, _config(vision_range=None, num_fields=1))

    assert sight.ship.all()


def test_hidden_enemy_channels_and_projectiles_are_zeroed_before_policy() -> None:
    ship, state = _state()
    state.ship_pos[0] = torch.tensor([100 + 100j, 100 + 200j, 700 + 700j, 800 + 800j])
    state.ship_vel[0, 2:] = torch.tensor([13 + 17j, -11 + 19j])
    state.ship_health[0, 2:] = torch.tensor([37.0, 0.0])
    state.ship_power[0, 2:] = torch.tensor([29.0, 81.0])
    state.ship_cooldown[0, 2:] = torch.tensor([0.07, 0.03])
    state.ship_alive[0, 3] = False
    state.prev_action[0, 2:] = torch.tensor([[2, 6, 1], [1, 4, 1]])
    state.ship_local_index[0, 2:] = torch.tensor([2.0, 0.5])
    state.ship_field_gradient[0, 2:] = torch.tensor([1 + 2j, 3 + 4j])
    state.bullet_active[0, 2, 0] = True
    state.bullet_pos[0, 2, 0] = 750 + 750j

    obs, _ = perceived_observation_from_state(
        state, ship, _config(vision_range=100.0), include_bullets=True
    )

    assert not obs.visible[0, 2:].any()
    for key, value in obs.items():
        if key is ObsKey.VISIBLE:
            continue
        hidden = value[0, 2:4] if value.dim() == 2 else value[0, 2:4, :]
        assert not hidden.any(), key
    assert obs.bullets is not None
    assert not obs.bullets[next(k for k in obs.bullets if k.value == "bullet_active")][0, 4]
    assert not obs.bullets[next(k for k in obs.bullets if k.value == "bullet_pos")][0, 4].any()


def test_team_views_are_independent_not_label_swaps_of_hidden_truth() -> None:
    ship, state = _state()
    state.ship_pos[0] = torch.tensor([100 + 100j, 100 + 200j, 700 + 700j, 800 + 800j])

    obs, _ = perceived_observation_from_state(state, ship, _config(vision_range=120.0))
    team1 = obs.for_team(1)

    assert obs.visible[0, :2].all()
    assert not obs.visible[0, 2:].any()
    assert team1.visible[0, 2:].all()
    assert not team1.visible[0, :2].any()


def test_enemy_pending_actions_stay_private_while_ship_is_visible() -> None:
    ship, state = _state()
    state.ship_pos[0] = torch.tensor([100 + 100j, 120 + 100j, 140 + 100j, 160 + 100j])
    state.prev_action[0] = torch.tensor(
        [[2, 6, 1], [1, 4, 0], [2, 2, 1], [0, 3, 0]], dtype=torch.long
    )

    obs, _ = perceived_observation_from_state(state, ship, _config(vision_range=300.0))
    team1 = obs.for_team(1)

    assert obs.visible[0, :4].all()
    assert torch.equal(obs.previous_action[0, :2], state.prev_action[0, :2])
    assert not obs.previous_action[0, 2:4].any()
    assert team1.visible[0, :4].all()
    assert not team1.previous_action[0, :2].any()
    assert torch.equal(team1.previous_action[0, 2:4], state.prev_action[0, 2:4])


def test_frontline_scripted_team_does_not_target_hidden_enemy_truth() -> None:
    from boost_and_broadside.modes.interactive import PLAY_ENV_CONFIG

    ship = ShipConfig(world_size=(16384.0, 16384.0), field_radius_max=750.0)
    state = make_state(
        num_envs=1,
        max_ships=4,
        max_bullets=0,
        ship_config=ship,
        num_fields=0,
    )
    state.ship_team_id[0] = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
    state.ship_pos[0] = torch.tensor([100 + 100j, 200 + 100j, 5000 + 5000j, 5200 + 5000j])
    # Supply the minimum Frontline map state used by the strategy.
    state.zone_pos = torch.tensor(
        [[1000 + 1000j, 2000 + 1000j, 3000 + 1000j, 4000 + 1000j, 5000 + 1000j]],
        dtype=torch.complex64,
    )
    state.zone_radius = torch.full((1, 5), 330.0)
    state.zone_roles = torch.tensor([[2, 0, 1, 3, 4]], dtype=torch.int8)
    state.zone_capture_progress = torch.zeros((1, 5))
    state.zone_capture_direction = torch.zeros((1, 5), dtype=torch.int8)
    env_config = EnvConfig(
        num_ships=4,
        max_bullets=0,
        max_episode_steps=9000,
        frontline=PLAY_ENV_CONFIG.frontline,
        vision_range=300.0,
    )
    visibility = team_visibility_from_state(state, ship, env_config).ship

    changed = state.clone()
    changed.ship_pos[0, 2:] = torch.tensor([8000 + 8000j, 8200 + 8000j])
    changed.ship_vel[0, 2:] = torch.tensor([100 + 0j, -100 + 20j])
    changed.ship_health[0, 2:] = torch.tensor([5.0, 91.0])
    changed.prev_action[0, 2:] = torch.tensor([[2, 6, 1], [1, 4, 1]])
    changed_visibility = team_visibility_from_state(changed, ship, env_config).ship

    torch.manual_seed(91)
    first = StochasticScriptedAgent(ship, StochasticAgentConfig())
    _, first_probs = first.get_actions_and_probs(state, visibility)
    torch.manual_seed(91)
    second = StochasticScriptedAgent(ship, StochasticAgentConfig())
    _, second_probs = second.get_actions_and_probs(changed, changed_visibility)

    assert torch.equal(first_probs[0, :2], second_probs[0, :2])


def test_frontline_observation_has_typed_field_zone_and_boundary_tokens() -> None:
    from boost_and_broadside.modes.interactive import PLAY_ENV_CONFIG

    ship = frontline_ship_config(ShipConfig())
    env = TensorEnv(2, ship, PLAY_ENV_CONFIG, "cpu")
    env.reset(seed=43)
    obs, _ = perceived_observation_from_state(env.state, ship, PLAY_ENV_CONFIG)
    num_ships = PLAY_ENV_CONFIG.num_ships
    num_fields = PLAY_ENV_CONFIG.num_fields
    zone_start = num_ships + num_fields
    boundary = PLAY_ENV_CONFIG.num_entity_tokens - 1

    assert obs.pos.shape == (2, PLAY_ENV_CONFIG.num_entity_tokens, 2)
    assert (obs[ObsKey.OBJECT_TYPE][:, :num_ships] == 0).all()
    assert (obs[ObsKey.OBJECT_TYPE][:, num_ships:zone_start] == 1).all()
    assert (obs[ObsKey.OBJECT_TYPE][:, zone_start:boundary] == 2).all()
    assert (obs[ObsKey.OBJECT_TYPE][:, boundary] == 3).all()
    assert torch.equal(obs[ObsKey.ZONE_ROLE][:, zone_start:boundary], env.state.zone_roles)
    assert obs.alive[:, num_ships:].all()
    assert obs.visible[:, num_ships:].all()
    assert torch.equal(obs.pos[:, boundary, 0], env.state.map_center.real)
    assert torch.equal(obs.radius[:, boundary, 0], env.state.playable_boundary_radius)


def test_team_canonicalization_flips_strategic_semantics_as_well_as_labels() -> None:
    from boost_and_broadside.modes.interactive import PLAY_ENV_CONFIG

    ship = frontline_ship_config(ShipConfig())
    env = TensorEnv(1, ship, PLAY_ENV_CONFIG, "cpu")
    env.reset(seed=44)
    env.state.front_position.fill_(2)
    env.state.zone_roles = roles_from_front(env.state.front_position)
    env.state.zone_capture_direction[0] = torch.tensor([1, -1, 0, 1, -1], dtype=torch.int8)
    obs, _ = perceived_observation_from_state(env.state, ship, PLAY_ENV_CONFIG)
    team1 = obs.for_team(1).flip_team(PLAY_ENV_CONFIG.num_ships)
    zone_start = PLAY_ENV_CONFIG.num_ships + PLAY_ENV_CONFIG.num_fields
    boundary = PLAY_ENV_CONFIG.num_entity_tokens - 1

    expected_roles = torch.where(
        env.state.zone_roles == 0,
        4,
        torch.where(
            env.state.zone_roles == 4,
            0,
            torch.where(
                env.state.zone_roles == 1,
                3,
                torch.where(env.state.zone_roles == 3, 1, env.state.zone_roles),
            ),
        ),
    )
    assert torch.equal(team1[ObsKey.ZONE_ROLE][:, zone_start:boundary], expected_roles)
    zone_team_ids = obs.team_id[:, zone_start:boundary]
    expected_team_ids = torch.where(
        zone_team_ids == 0,
        torch.ones_like(zone_team_ids),
        torch.where(zone_team_ids == 1, torch.zeros_like(zone_team_ids), zone_team_ids),
    )
    assert torch.equal(team1.team_id[:, zone_start:boundary], expected_team_ids)
    assert team1[ObsKey.FRONT_POSITION][0, boundary, 0] == -2
    assert torch.equal(
        team1[ObsKey.CAPTURE_DIRECTION][0, zone_start:boundary, 0],
        -env.state.zone_capture_direction[0].float(),
    )


def test_wrapper_accumulates_never_seen_hidden_age_and_reacquisition_on_device() -> None:
    ship = ShipConfig()
    env_config = _config(vision_range=150.0)
    wrapper = YemongEnvWrapper(1, ship, env_config, REWARDS, "cpu")
    wrapper.reset(seed=8)
    wrapper.pop_episode_stats()
    state = wrapper.state
    state.ship_team_id[0] = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
    close = torch.tensor([100 + 100j, 100 + 120j, 200 + 100j, 200 + 120j])
    state.ship_pos[0] = close
    wrapper._reset_perception(torch.tensor([True]))
    wrapper._get_obs()  # initially seen
    state.ship_pos[0, 2:] = torch.tensor([700 + 700j, 720 + 700j])
    for _ in range(3):
        wrapper._get_obs()
    state.ship_pos[0] = close
    wrapper._get_obs()  # reacquired after a three-decision hidden run

    stats = wrapper.pop_episode_stats()
    source = dict(zip(SOURCE_STAT_NAMES, stats["source_stats"], strict=True))
    assert source["perception_enemy_slots"] == 20
    assert source["perception_reacquisitions"] == 4
    assert source["perception_hidden_samples"] == 12
    assert source["perception_hidden_age_sum"] == 24
    assert stats["occlusion_hist"].sum() == 4


# ---------------------------------------------------------------------------
# Rule 1 — one team, one set of eyes
# ---------------------------------------------------------------------------


def test_an_ally_with_clear_sight_reveals_what_a_blocked_ally_cannot_see() -> None:
    ship, state = _state(num_fields=1)
    # Ally 0 is looking straight through the field at the enemy; ally 1 is off
    # to the side with an unobstructed line.
    state.ship_pos[0] = torch.tensor([100 + 100j, 500 + 400j, 500 + 100j, 900 + 900j])
    state.field_pos[0, 0] = 300 + 100j
    state.field_radius[0, 0] = 90.0
    state.field_transition_width[0, 0] = 40.0
    config = _config(vision_range=500.0, num_fields=1)

    sight = team_visibility_from_state(state, ship, config)

    assert not sight.observer_ship[0, 0, 2]
    assert sight.observer_ship[0, 1, 2]
    assert sight.ship[0, 0, 2]


def test_a_dead_ally_stops_contributing_its_sight() -> None:
    ship, state = _state()
    state.ship_pos[0] = torch.tensor([100 + 100j, 700 + 700j, 350 + 100j, 900 + 900j])
    config = _config()

    assert team_visibility_from_state(state, ship, config).ship[0, 0, 2]

    state.ship_alive[0, 0] = False
    assert not team_visibility_from_state(state, ship, config).ship[0, 0, 2]


# ---------------------------------------------------------------------------
# Rule 2 — opaque cores break sight lines
# ---------------------------------------------------------------------------


def test_occlusion_is_symmetric_across_random_layouts() -> None:
    """If A cannot see B then B cannot see A, whatever the geometry.

    Run on the production world at the production range. Occluders are located
    by minimum image from the observer, which picks the same copy from both
    ends of a sight line only while ``vision_range + core_radius`` stays inside
    half a world; Frontline clears that by a factor of four.
    """

    ship = frontline_ship_config(ShipConfig())
    torch.manual_seed(7)
    state = make_state(num_envs=64, max_ships=6, max_bullets=0, ship_config=ship, num_fields=8)
    width, height = ship.world_size
    state.ship_team_id[:] = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32)
    # A tight cluster, so most pairs are in range and most lines meet a core.
    state.ship_pos = torch.complex(torch.rand(64, 6) * 2600.0, torch.rand(64, 6) * 2600.0).to(
        torch.complex64
    )
    state.field_pos = torch.complex(torch.rand(64, 8) * 2600.0, torch.rand(64, 8) * 2600.0).to(
        torch.complex64
    )
    state.field_radius = 200.0 + torch.rand(64, 8) * 550.0
    state.field_transition_width = torch.full((64, 8), 40.0)
    config = EnvConfig(
        num_ships=6, max_bullets=0, max_episode_steps=60, num_fields=8, vision_range=1024.0
    )

    sight = team_visibility_from_state(state, ship, config)

    # A meaningful sample of both verdicts, not a vacuous all-true matrix.
    assert sight.observer_ship.float().mean() < 0.9
    assert torch.equal(sight.observer_ship, sight.observer_ship.transpose(1, 2))


def test_every_ship_sees_itself_from_inside_or_outside_a_core() -> None:
    ship, state = _state(num_fields=1)
    state.ship_pos[0] = torch.tensor([300 + 100j, 100 + 700j, 500 + 100j, 700 + 700j])
    state.field_pos[0, 0] = 300 + 100j
    state.field_radius[0, 0] = 90.0
    state.field_transition_width[0, 0] = 40.0

    sight = team_visibility_from_state(state, ship, _config(vision_range=500.0, num_fields=1))

    assert sight.observer_ship[0].diagonal().all()


def test_a_core_blocks_the_wrapped_sight_line_it_straddles() -> None:
    ship, state = _state(num_fields=1)
    width, _ = ship.world_size
    # Observer and target face each other across the seam, 120 px apart the
    # short way, each outside a core that sits on the seam between them.
    state.ship_pos[0] = torch.tensor([(width - 60) + 100j, 100 + 700j, 60 + 100j, 700 + 700j])
    state.field_pos[0, 0] = 0 + 100j
    state.field_radius[0, 0] = 40.0
    state.field_transition_width[0, 0] = 0.0

    sight = team_visibility_from_state(state, ship, _config(vision_range=200.0, num_fields=1))

    assert sight.range_only_observer_ship[0, 0, 2]
    assert not sight.observer_ship[0, 0, 2]


def test_an_enemy_bullet_inside_a_field_is_hidden_from_outside() -> None:
    ship, state = _state(num_fields=1)
    state.ship_pos[0] = torch.tensor([100 + 100j, 100 + 700j, 900 + 900j, 700 + 700j])
    state.bullet_active[0, 2, 0] = True
    state.bullet_pos[0, 2, 0] = 300 + 100j  # dead centre of the field
    state.bullet_active[0, 2, 1] = True
    state.bullet_pos[0, 2, 1] = 200 + 100j  # short of the core, in the clear
    state.field_pos[0, 0] = 300 + 100j
    state.field_radius[0, 0] = 90.0
    state.field_transition_width[0, 0] = 40.0

    sight = team_visibility_from_state(state, ship, _config(vision_range=500.0, num_fields=1))

    assert not sight.bullet[0, 0, 2, 0]
    assert sight.bullet[0, 0, 2, 1]


# ---------------------------------------------------------------------------
# Rule 2 (optional) — zones as occluders
# ---------------------------------------------------------------------------


def _with_one_zone(state, position: complex, radius: float) -> None:
    state.zone_pos = torch.tensor([[position]], dtype=torch.complex64)
    state.zone_radius = torch.tensor([[radius]])
    state.zone_roles = torch.zeros((1, 1), dtype=torch.int8)
    state.zone_capture_progress = torch.zeros((1, 1))
    state.zone_capture_direction = torch.zeros((1, 1), dtype=torch.int8)


def test_zones_are_transparent_unless_the_environment_makes_them_opaque() -> None:
    ship, state = _state()
    state.ship_pos[0] = torch.tensor([100 + 100j, 100 + 700j, 500 + 100j, 700 + 700j])
    _with_one_zone(state, 300 + 100j, 90.0)
    config = _config(vision_range=500.0)

    assert team_visibility_from_state(state, ship, config).observer_ship[0, 0, 2]

    opaque = replace(config, zones_occlude=True)
    sight = team_visibility_from_state(state, ship, opaque)
    assert sight.range_only_observer_ship[0, 0, 2]
    assert not sight.observer_ship[0, 0, 2]


def test_an_opaque_zone_uses_its_whole_radius_and_keeps_the_shared_exemption() -> None:
    ship, state = _state()
    # Ally 0 and enemy 2 are both inside the zone; enemy 3 is just outside it.
    state.ship_pos[0] = torch.tensor([260 + 100j, 100 + 700j, 340 + 100j, 420 + 100j])
    _with_one_zone(state, 300 + 100j, 90.0)
    config = replace(_config(vision_range=500.0), zones_occlude=True)

    sight = team_visibility_from_state(state, ship, config)

    assert sight.observer_ship[0, 0, 2]
    assert not sight.observer_ship[0, 0, 3]


# ---------------------------------------------------------------------------
# Rule 3 — a circular sight radius
# ---------------------------------------------------------------------------


def test_the_sight_radius_is_a_circle_not_a_bounding_box() -> None:
    ship, state = _state()
    reach = 300.0
    diagonal = reach * 0.72  # inside the square, outside the circle
    state.ship_pos[0] = torch.tensor(
        [
            500 + 500j,
            900 + 900j,
            (500 + reach) + 500j,  # exactly at the rim
            (500 + diagonal) + (500 + diagonal) * 1j,
        ]
    )

    sight = team_visibility_from_state(state, ship, _config(vision_range=reach))

    assert sight.observer_ship[0, 0, 2]
    assert not sight.observer_ship[0, 0, 3]


def test_range_is_measured_in_world_pixels_from_the_configured_value() -> None:
    ship, state = _state()
    state.ship_pos[0] = torch.tensor([100 + 100j, 900 + 900j, 100 + 400j, 900 + 100j])

    near = team_visibility_from_state(state, ship, _config(vision_range=299.0))
    far = team_visibility_from_state(state, ship, _config(vision_range=301.0))

    assert not near.observer_ship[0, 0, 2]
    assert far.observer_ship[0, 0, 2]


# ---------------------------------------------------------------------------
# Rule 4 — firing is an observable event
# ---------------------------------------------------------------------------


def test_firing_reveals_the_shooter_to_both_teams_at_once() -> None:
    ship, state = _state()
    state.ship_pos[0] = torch.tensor([100 + 100j, 120 + 100j, 900 + 900j, 880 + 900j])
    state.ship_is_shooting[0, 2] = True

    sight = team_visibility_from_state(state, ship, _config(vision_range=100.0))

    assert sight.ship[0, 0, 2]
    assert sight.ship[0, 1, 2]
    assert not sight.ship[0, 0, 3]


def test_firing_reveals_the_shooter_but_not_the_ally_beside_it() -> None:
    ship, state = _state()
    state.ship_pos[0] = torch.tensor([100 + 100j, 120 + 100j, 900 + 900j, 880 + 900j])
    state.ship_is_shooting[0, 2] = True

    sight = team_visibility_from_state(state, ship, _config(vision_range=100.0))

    assert sight.ship[0, 0, 2]
    assert not sight.ship[0, 0, 3]


def test_a_dead_slot_is_never_revealed_by_a_stale_shooting_flag() -> None:
    ship, state = _state()
    state.ship_pos[0] = torch.tensor([100 + 100j, 120 + 100j, 900 + 900j, 880 + 900j])
    state.ship_is_shooting[0, 2] = True
    state.ship_alive[0, 2] = False

    sight = team_visibility_from_state(state, ship, _config(vision_range=100.0))

    assert not sight.ship[0, 0, 2]


# ---------------------------------------------------------------------------
# Declared projectile perception
# ---------------------------------------------------------------------------


def test_declining_bullet_perception_omits_it_without_changing_ship_sight() -> None:
    ship, state = _state(num_fields=1)
    state.ship_pos[0] = torch.tensor([100 + 100j, 100 + 700j, 500 + 100j, 700 + 700j])
    state.bullet_active[0, 2, 0] = True
    state.bullet_pos[0, 2, 0] = 150 + 100j
    state.field_pos[0, 0] = 300 + 100j
    state.field_radius[0, 0] = 90.0
    state.field_transition_width[0, 0] = 40.0
    config = _config(vision_range=500.0, num_fields=1)

    full = team_visibility_from_state(state, ship, config)
    lean = team_visibility_from_state(state, ship, config, perceive_bullets=False)

    assert full.bullet is not None
    assert lean.bullet is None
    assert torch.equal(full.ship, lean.ship)
    assert torch.equal(full.observer_ship, lean.observer_ship)


def test_bullet_observations_cannot_be_requested_without_bullet_perception() -> None:
    ship, state = _state()
    state.ship_pos[0] = torch.tensor([100 + 100j, 100 + 700j, 500 + 100j, 700 + 700j])

    with pytest.raises(ValueError, match="bullet perception"):
        perceived_observation_from_state(
            state, ship, _config(), include_bullets=True, perceive_bullets=False
        )


# ---------------------------------------------------------------------------
# What the policy actually attends to
# ---------------------------------------------------------------------------


def test_an_occluded_enemy_is_masked_out_of_policy_attention() -> None:
    """BELIEF_VALID is the key mask the trunk attends over, so it must drop."""

    ship, state = _state(num_fields=1)
    state.ship_pos[0] = torch.tensor([100 + 100j, 100 + 140j, 500 + 100j, 540 + 100j])
    state.field_pos[0, 0] = 300 + 100j
    state.field_radius[0, 0] = 200.0
    state.field_transition_width[0, 0] = 40.0
    config = _config(vision_range=800.0, num_fields=1)

    blocked, blocked_sight = perceived_observation_from_state(state, ship, config)
    state.field_pos[0, 0] = 900 + 900j  # same field, nowhere near the sight line
    clear, sight = perceived_observation_from_state(state, ship, config)

    assert sight.ship[0, 0, 2:].all()
    assert clear[ObsKey.BELIEF_VALID][0, 2:4].all()
    assert not blocked_sight.ship[0, 0, 2:].any()
    assert not blocked[ObsKey.BELIEF_VALID][0, 2:4].any()
    assert not blocked[ObsKey.VISIBLE][0, 2:4].any()
    # Field tokens stay attendable: terrain is not what fog hides.
    assert blocked[ObsKey.BELIEF_VALID][0, 4:].all()
