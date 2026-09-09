"""Team-shared range and refractive-field LOS perception tests."""

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

    sight = team_visibility_from_state(
        state, ship, _config(vision_range=500.0, num_fields=1)
    )

    assert sight.range_only_observer_ship[0, 0, 2]
    assert not sight.observer_ship[0, 0, 2]
    assert not sight.ship[0, 0, 2]


def test_field_containing_an_endpoint_does_not_self_blind() -> None:
    ship, state = _state(num_fields=1)
    state.ship_pos[0] = torch.tensor([300 + 100j, 100 + 700j, 500 + 100j, 700 + 700j])
    state.field_pos[0, 0] = 300 + 100j
    state.field_radius[0, 0] = 90.0
    state.field_transition_width[0, 0] = 40.0

    sight = team_visibility_from_state(
        state, ship, _config(vision_range=500.0, num_fields=1)
    )

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

    sight = team_visibility_from_state(
        state, ship, _config(vision_range=None, num_fields=1)
    )

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
    assert not obs.bullets[next(k for k in obs.bullets if k.value == "bullet_active")][
        0, 4
    ]
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
