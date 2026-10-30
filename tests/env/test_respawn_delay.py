"""The Frontline respawn wait, ``FrontlineConfig.respawn_delay``.

A destroyed ship is out of the match for the configured delay and re-enters at
whatever its team's spawn is when the wait ends -- not where it was when the
ship died. That second half is the point of the mechanic: a capture flips the
spawns, and a ship killed in the capture fight has to come back from the new
one. Everything else here is the rest of the contract a waiting slot needs: it
is parked, public to both teams, carries its countdown in the observation, and
is excluded from next-state supervision while staying in the critic's mask.
"""

from dataclasses import replace

import pytest
import torch

from boost_and_broadside.config import EnvConfig, FrontlineConfig, ShipConfig, ZoneRole
from boost_and_broadside.config.defaults import REWARDS
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.frontline import (
    FRONTLINE_WORLD_SIZE,
    respawn_delay_ticks,
    roles_from_front,
    toroidal_displacement,
)
from boost_and_broadside.env.observation import ObsKey
from boost_and_broadside.env.perception import team_visibility_from_state
from boost_and_broadside.env.wrapper import YemongEnvWrapper
from boost_and_broadside.train.rl.features import (
    RESPAWN_TIME_HARMONICS,
    RESPAWN_TIME_HORIZON,
    build_standard_coordinator,
    respawn_time_features,
)
from boost_and_broadside.train.rl.ppo import valued_ships

_FRONTLINE = FrontlineConfig(
    zone_radius=330.0,
    zone_ring_radius=1200.0,
    playable_radius=2600.0,
    capture_seconds=20.0,
    respawn_health=25.0,
    respawn_power=20.0,
    respawn_speed=30.0,
    shield_recharge_delay=4.0,
    shield_recharge_per_second=20.0,
    # Lethal the instant a ship is outside, so a death lands on a chosen tick.
    boundary_damage_per_second=6000.0,
    boundary_damage_per_pixel_second=0.0,
    front_win_threshold=5,
    respawn_delay=0.5,
)
_SHIP = ShipConfig(world_size=FRONTLINE_WORLD_SIZE)


def _env_config(frontline: FrontlineConfig = _FRONTLINE) -> EnvConfig:
    return EnvConfig(
        num_ships=4,
        max_bullets=0,
        max_episode_steps=600,
        vision_range=512.0,
        spawn_reveal=True,
        frontline=frontline,
    )


def _env(frontline: FrontlineConfig = _FRONTLINE) -> TensorEnv:
    env = TensorEnv(2, _SHIP, _env_config(frontline), "cpu")
    env.reset(seed=7)
    return env


def _hold(num_envs: int = 2) -> torch.Tensor:
    return torch.zeros((num_envs, 4, 3), dtype=torch.int32)


def _kill_outside(env: TensorEnv, slot: int) -> None:
    """Put one ship outside the playable circle on no health: it dies next tick."""
    env.state.ship_health[:, slot] = 0.0
    env.state.ship_pos[:, slot] = env.state.map_center + complex(
        _FRONTLINE.playable_radius + 10.0, 0.0
    )


def _spawn_centre(env: TensorEnv, slot: int) -> torch.Tensor:
    """(B,) complex centre of ``slot``'s team spawn under the current roles."""
    team = env.state.ship_team_id[:, slot]
    role = torch.where(team == 0, int(ZoneRole.TEAM0_SPAWN), int(ZoneRole.TEAM1_SPAWN))
    zone = (env.state.zone_roles == role.unsqueeze(1).to(torch.int8)).long().argmax(dim=1)
    return env.state.zone_pos.gather(1, zone.unsqueeze(1)).squeeze(1)


def test_a_dead_ship_waits_out_the_delay_parked_where_it_died() -> None:
    env = _env()
    ticks = respawn_delay_ticks(_FRONTLINE, _SHIP)
    assert ticks > 1
    slot = 0
    _kill_outside(env, slot)
    env.tick(_hold())
    assert not env.state.ship_alive[:, slot].any(), "the ship died"
    assert (env.state.ship_respawn_ticks[:, slot] == ticks).all(), "and starts a full wait"
    wreck = env.state.ship_pos[:, slot].clone()

    for remaining in range(ticks - 1, 0, -1):
        env.tick(_hold())
        assert not env.state.ship_alive[:, slot].any(), "still down during the wait"
        assert not env.state.ship_respawned[:, slot].any()
        assert (env.state.ship_respawn_ticks[:, slot] == remaining).all()
        assert torch.equal(env.state.ship_pos[:, slot], wreck), "a waiting ship does not drift"
    others = [index for index in range(4) if index != slot]
    assert env.state.ship_alive[:, others].all(), "nobody else was touched"

    env.tick(_hold())
    assert env.state.ship_alive[:, slot].all(), "the wait ends on exactly the delay"
    assert env.state.ship_respawned[:, slot].all()
    assert (env.state.ship_respawn_ticks[:, slot] == 0).all()


def test_a_waiting_ship_is_neither_a_shooter_nor_a_target_for_shot_labels() -> None:
    """The learner's shot re-simulation reads the launch copy's ``alive``: a
    parked wreck must not fire a labelled shot or absorb one fired at it."""
    env = TensorEnv(2, _SHIP, replace(_env_config(), max_bullets=4, shot_labels=True), "cpu")
    env.reset(seed=7)
    slot = 0
    _kill_outside(env, slot)
    shoot = _hold()
    shoot[..., 2] = 1  # everyone pulls the trigger, the wreck included
    env.tick(shoot)  # the copy precedes this tick's deaths: a shot fired as it dies is real
    assert not env.state.ship_alive[:, slot].any()
    for _ in range(respawn_delay_ticks(_FRONTLINE, _SHIP) - 1):
        env.tick(shoot)
        assert not env.shot_launch.alive[:, slot].any()
        assert env.shot_launch.alive[:, 1:].all()


def test_a_zero_delay_still_respawns_on_the_tick_of_death() -> None:
    env = _env(replace(_FRONTLINE, respawn_delay=0.0))
    _kill_outside(env, 1)
    env.tick(_hold())
    assert env.state.ship_alive[:, 1].all()
    assert env.state.ship_respawned[:, 1].all()
    assert (env.state.ship_respawn_ticks == 0).all()


def test_the_spawn_is_read_when_the_wait_ends_not_at_death() -> None:
    """A capture during the wait moves the spawn, and the ship follows it."""
    env = _env()
    slot = 0
    _kill_outside(env, slot)
    env.tick(_hold())
    old_spawn = _spawn_centre(env, slot)

    # Advance the front one capture in the dead ship's team's favour, as a
    # capture completing mid-wait would.
    team = env.state.ship_team_id[:, slot]
    env.state.front_position = torch.where(team == 0, 1, -1).to(env.state.front_position)
    env.state.zone_roles = roles_from_front(env.state.front_position)
    new_spawn = _spawn_centre(env, slot)
    assert not torch.allclose(old_spawn, new_spawn)

    for _ in range(respawn_delay_ticks(_FRONTLINE, _SHIP)):
        env.tick(_hold())
    assert env.state.ship_alive[:, slot].all()

    position = env.state.ship_pos[:, slot]
    to_new = toroidal_displacement(position - new_spawn, _SHIP.world_size).abs()
    to_old = toroidal_displacement(position - old_spawn, _SHIP.world_size).abs()
    assert (to_new < _FRONTLINE.zone_radius).all(), "re-entered at the post-capture spawn"
    assert (to_old > _FRONTLINE.zone_radius).all()


def test_a_waiting_ship_is_public_to_both_teams_with_its_countdown() -> None:
    wrapper = YemongEnvWrapper(2, _SHIP, _env_config(), REWARDS, "cpu")
    wrapper.reset(seed=7)
    slot = 0
    _kill_outside(wrapper.env, slot)
    wrapper.step(_hold())
    wrapper.step(_hold())  # spend any reveal latch; the ship is still down

    state = wrapper.env.state
    assert not state.ship_alive[:, slot].any()
    visibility = team_visibility_from_state(state, _SHIP, wrapper.env_config, False)
    assert visibility.ship[:, :, slot].all(), "both teams see the slot is down"

    observation = wrapper.observe()
    for data in (observation.data, observation.team1_data):
        expected = state.ship_respawn_ticks[:, slot].float() * _SHIP.dt
        assert torch.allclose(data[ObsKey.RESPAWN_TIME][:, slot, 0], expected)
        assert (data[ObsKey.RESPAWN_TIME][:, slot, 0] > 0).all()
        assert not data[ObsKey.ALIVE][:, slot].any(), "and reads it as down, not remembered"
        others = [index for index in range(4) if index != slot]
        assert (data[ObsKey.RESPAWN_TIME][:, others] == 0).all()


def test_the_wait_breaks_transitions_but_keeps_the_value_mask() -> None:
    wrapper = YemongEnvWrapper(2, _SHIP, _env_config(), REWARDS, "cpu")
    wrapper.reset(seed=7)
    slot = 2
    others = [index for index in range(4) if index != slot]
    _kill_outside(wrapper.env, slot)
    *_, info = wrapper.step(_hold())
    assert not info["transition_contiguous"][:, slot].any(), "the death tick is not a transition"

    # Rollout storage is time-major and may carry one extra bootstrap step.
    observation = wrapper.observe()
    respawn_time = observation[ObsKey.RESPAWN_TIME].clone()
    stored = {ObsKey.RESPAWN_TIME: respawn_time.unsqueeze(0).repeat(2, 1, 1, 1)}
    alive = observation[ObsKey.ALIVE][:, :4].bool().clone().unsqueeze(0)
    *_, info = wrapper.step(_hold())
    assert not info["transition_contiguous"][:, slot].any(), "nor is any tick of the wait"
    assert info["transition_contiguous"][:, others].all()

    # Without a belief tracker an enemy out of sight also reads not-alive; only
    # the waiting slot is the critic mask's business here.
    valued = valued_ships(stored, alive)
    assert not alive[..., slot].any()
    assert valued[..., slot].all(), "a waiting ship is still valued"
    assert torch.equal(valued[..., others], alive[..., others]), "and nothing else changes"
    assert torch.equal(valued_ships({}, alive), alive), "without the channel, alive alone"


@pytest.mark.parametrize("delay", [-1.0, float("inf")])
def test_the_delay_must_be_finite_and_non_negative(delay: float) -> None:
    with pytest.raises(ValueError, match="respawn_delay"):
        replace(_FRONTLINE, respawn_delay=delay)


def test_a_config_recorded_before_the_delay_means_immediate_respawn() -> None:
    recorded = {
        name: getattr(_FRONTLINE, name)
        for name in _FRONTLINE.__dataclass_fields__
        if name != "respawn_delay"
    }
    config = EnvConfig(num_ships=4, max_bullets=0, max_episode_steps=600, frontline=recorded)
    assert config.frontline.respawn_delay == 0.0


def test_the_encoder_reads_the_wait_as_a_normalized_time_and_fourier_basis() -> None:
    coordinator = build_standard_coordinator(_SHIP)
    (feature,) = [f for f in coordinator.features if f.name == "respawn_time"]
    width = 1 + 2 * RESPAWN_TIME_HARMONICS
    seconds = torch.tensor([[[0.0], [4.0], [RESPAWN_TIME_HORIZON]]])
    encoded = feature.input_encoder(seconds)
    assert encoded.shape == (1, 3, width)
    assert torch.equal(encoded, respawn_time_features(seconds[..., 0]))
    living, waiting, horizon = encoded[0]
    assert living[0] == 0 and torch.allclose(living[1 + RESPAWN_TIME_HARMONICS :], torch.ones(4))
    assert torch.allclose(waiting[0], torch.tensor(0.25))
    assert torch.allclose(horizon[0], torch.tensor(1.0))
    # The linear term is what tells the ends of the period apart.
    assert torch.allclose(horizon[1:], living[1:], atol=1e-5)
    assert not torch.allclose(waiting[1:], living[1:])
