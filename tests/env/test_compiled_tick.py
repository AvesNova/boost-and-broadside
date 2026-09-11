"""What of a physics tick may be compiled, and why the rest may not.

The environment is a large share of training wall clock and almost all of it is
CPU dispatch -- hundreds of small kernels over a wide, shallow token axis -- so
fusing it looks like the obvious win. It is not available, and the reason is
structural rather than incidental: ``update_ships``, ``advance_bullets`` and
``perceived_observation_from_state`` all communicate by writing results back
onto objects the caller owns (the state, or the reusable observation buffers),
and dynamo does not replay those writes. Compiled, they run and produce nothing.

The measurements are in docs/engineering/rl-throughput.md. These tests exist so
that a later attempt fails loudly here instead of silently training on a world
where nobody ever fires.
"""

import pytest
import torch

from boost_and_broadside.config import EnvConfig, FrontlineConfig, ShipConfig
from boost_and_broadside.config.defaults import REWARDS
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.frontline import FRONTLINE_WORLD_SIZE
from boost_and_broadside.env.wrapper import YemongEnvWrapper

NUM_ENVS = 16
NUM_SHIPS = 4
SEED = 20260911


def _frontline() -> FrontlineConfig:
    return FrontlineConfig(
        zone_radius=330.0,
        zone_ring_radius=1200.0,
        playable_radius=2600.0,
        capture_seconds=20.0,
        defense_damage_per_second=2.0,
        respawn_health=25.0,
        spawn_heal_per_second=12.0,
        enemy_spawn_damage_per_second=8.0,
        boundary_damage_per_second=5.0,
        boundary_damage_per_pixel_second=0.05,
        front_win_threshold=5,
    )


def _env_config() -> EnvConfig:
    return EnvConfig(
        num_ships=NUM_SHIPS,
        max_bullets=4,
        max_episode_steps=4000,
        num_fields=6,
        frontline=_frontline(),
        vision_range=1600.0,
    )


def _env(compile_mode: str | None) -> TensorEnv:
    torch.manual_seed(SEED)
    env = TensorEnv(
        NUM_ENVS,
        ShipConfig(world_size=FRONTLINE_WORLD_SIZE),
        _env_config(),
        torch.device("cuda"),
        compile_mode,
    )
    env.reset()
    return env


def _wrapper(compile_mode: str | None) -> YemongEnvWrapper:
    torch.manual_seed(SEED)
    return YemongEnvWrapper(
        num_envs=NUM_ENVS,
        ship_config=ShipConfig(world_size=FRONTLINE_WORLD_SIZE),
        env_config=_env_config(),
        rewards=REWARDS,
        device="cuda",
        collision_compile_mode=compile_mode,
    )


def _shoot_everything() -> torch.Tensor:
    actions = torch.zeros(NUM_ENVS, NUM_SHIPS, 3, device="cuda", dtype=torch.int32)
    actions[..., 2] = 1
    return actions


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_a_compiled_launch_simulates_the_same_world() -> None:
    """Only the pure collision kernel is compiled, so the two ticks agree.

    A shoot-everything action is the sharpest probe available: the shot gate is
    discrete, so a stage whose writes were dropped shows up as *no shots at all*
    rather than as a small numeric drift.
    """
    eager, compiled = _env(None), _env("default")
    assert torch.equal(eager.state.ship_pos, compiled.state.ship_pos), "reset already differs"

    actions = _shoot_everything()
    for _ in range(8):
        # The tick draws for bullet spread, so both must see the same stream or
        # their trajectories decorrelate for reasons that are not the point here.
        rng = torch.cuda.get_rng_state()
        eager.tick(actions)
        torch.cuda.set_rng_state(rng)
        compiled.tick(actions)

    assert eager.state.ship_is_shooting.sum() > 0, "the probe never fired; it proves nothing"
    assert torch.equal(eager.state.ship_is_shooting, compiled.state.ship_is_shooting)
    assert torch.equal(eager.state.ship_alive, compiled.state.ship_alive)
    for name, atol in (
        ("ship_pos", 1e-2),
        ("ship_vel", 1e-3),
        ("ship_health", 1e-3),
        ("ship_power", 1e-3),
    ):
        left, right = getattr(eager.state, name), getattr(compiled.state, name)
        if left.is_complex():
            left, right = torch.view_as_real(left), torch.view_as_real(right)
        torch.testing.assert_close(left, right, rtol=0, atol=atol, msg=f"{name} diverged")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_a_compiled_launch_builds_the_same_observation() -> None:
    """The observation builder writes into reusable buffers, so it stays eager.

    Compiling it was measured at about 5% of a rollout step and is wrong: the
    buffer writes are dropped and positions came back off by 16,135 pixels on a
    16,384-pixel torus.
    """
    eager, compiled = _wrapper(None), _wrapper("default")
    # Reset draws spawn positions, so both have to see the same stream.
    torch.manual_seed(SEED)
    eager_obs = eager.reset()
    torch.manual_seed(SEED)
    compiled_obs = compiled.reset()
    for key in eager_obs.data:
        left, right = eager_obs[key], compiled_obs[key]
        if left.is_complex():
            left, right = torch.view_as_real(left), torch.view_as_real(right)
        torch.testing.assert_close(
            left.float(), right.float(), rtol=0, atol=1e-4, msg=f"{key} differs"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_reusable_buffers_track_the_map_across_resets() -> None:
    """Buffered and unbuffered observation builds must agree after a reset.

    The buffers cache the field-derived rows, which only change when an
    environment draws a new map, so every ``reset_envs`` has to be paired with a
    refresh. Miss one and the affected environments keep observing the map they
    used to be in -- silently, because every other channel stays correct. The
    evaluator builds its observation this way, so this pins the pairing.
    """
    from boost_and_broadside.env.observation import (
        ObservationBuffers,
        perceived_observation_from_state,
    )

    env = _env(None)
    buffers = ObservationBuffers.allocate(
        NUM_ENVS, NUM_SHIPS, env.env_config.num_fields, 5, env.ship_config, env.state.device
    )
    buffers.refresh_field_state_all(env.state)

    reset = torch.zeros(NUM_ENVS, dtype=torch.bool, device="cuda")
    reset[::2] = True
    env.reset_envs(reset)
    buffers.refresh_field_state(env.state, reset)

    buffered, _ = perceived_observation_from_state(
        env.state, env.ship_config, env.env_config, buffers
    )
    fresh, _ = perceived_observation_from_state(env.state, env.ship_config, env.env_config)
    for key in fresh.data:
        left, right = fresh[key], buffered[key]
        if left.is_complex():
            left, right = torch.view_as_real(left), torch.view_as_real(right)
        torch.testing.assert_close(
            left.float(), right.float(), rtol=0, atol=0, msg=f"{key} went stale"
        )
