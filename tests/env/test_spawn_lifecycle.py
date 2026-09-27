"""The spawn-reveal lifecycle, in every path that advances a decision.

``spawn_reveal`` makes a ship entering the world visible to both teams for
exactly one decision. That contract has two halves: the reveal has to survive
every physics tick of a held action, and it has to lapse when the next decision
starts. ``state.ship_spawned`` is the latch that carries it, and the two
decision-level entry points -- :meth:`TensorEnv.step` and the wrapper's
``step``/``step_interactive`` -- each own clearing and re-filling it.

These tests drive the real stepping paths rather than setting the latch by hand,
because the defect they exist to prevent was a missing clear rather than a wrong
visibility rule: with the latch stuck true, ``spawn_reveal`` silently turns into
an omniscient world and every fog-dependent measurement taken through
``TensorEnv`` reads a game nobody is training on.
"""

import torch

from boost_and_broadside.config import EnvConfig, FrontlineConfig, ShipConfig
from boost_and_broadside.config.defaults import REWARDS
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.frontline import FRONTLINE_WORLD_SIZE
from boost_and_broadside.env.perception import team_visibility_from_state
from boost_and_broadside.env.wrapper import YemongEnvWrapper

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
    # Lethal the instant a ship is outside, so a respawn can be provoked on a
    # chosen tick instead of waited for.
    boundary_damage_per_second=6000.0,
    boundary_damage_per_pixel_second=0.0,
    front_win_threshold=5,
)


def _frontline_env(
    *,
    num_envs: int = 2,
    num_ships: int = 4,
    action_repeat: int = 1,
    vision_range: float | None = 512.0,
) -> TensorEnv:
    env = TensorEnv(
        num_envs,
        ShipConfig(world_size=FRONTLINE_WORLD_SIZE),
        EnvConfig(
            num_ships=num_ships,
            max_bullets=0,
            max_episode_steps=600,
            vision_range=vision_range,
            spawn_reveal=True,
            action_repeat=action_repeat,
            frontline=_FRONTLINE,
        ),
        "cpu",
    )
    env.reset(seed=7)
    return env


def _hold(env: TensorEnv) -> torch.Tensor:
    return torch.zeros((env.num_envs, env.env_config.num_ships, 3), dtype=torch.int32)


def _push_outside(env: TensorEnv, slot: int) -> None:
    """Put one ship outside the playable circle on no health, killing it this tick.

    A ship dies on the tick it takes damage while already at zero health, so
    zero rather than one point of health is what makes the respawn land inside
    the next single-tick decision.
    """
    env.state.ship_health[:, slot] = 0.0
    env.state.ship_pos[:, slot] = env.state.map_center + complex(
        _FRONTLINE.playable_radius + 10.0, 0.0
    )


def test_a_reset_latches_the_spawn_and_one_step_clears_it() -> None:
    env = _frontline_env()
    assert env.state.ship_spawned.all(), "a reset is a spawn for every ship"

    env.step(_hold(env))

    assert not env.state.ship_spawned.any(), "the reveal lapses when the next decision starts"


def test_a_respawn_inside_a_step_is_revealed_for_exactly_one_decision() -> None:
    env = _frontline_env()
    env.step(_hold(env))  # spend the opening reveal
    assert not env.state.ship_spawned.any()

    slot = 0
    _push_outside(env, slot)
    env.step(_hold(env))

    revealed = env.state.ship_spawned
    assert revealed[:, slot].all(), "the respawned ship is revealed"
    assert not revealed[:, 1:].any(), "and nothing else is"

    env.step(_hold(env))
    assert not env.state.ship_spawned.any(), "one decision later the reveal is gone"


def test_a_respawn_on_an_early_tick_survives_the_rest_of_the_hold() -> None:
    """``ship_respawned`` is a one-tick flag; the latch is what reaches the obs."""

    env = _frontline_env(action_repeat=4)
    env.step(_hold(env))

    slot = 1
    _push_outside(env, slot)
    env.step(_hold(env))

    assert not env.state.ship_respawned[:, slot].any(), "the physics flag is long gone"
    assert env.state.ship_spawned[:, slot].all(), "the decision-level latch carried it"
    others = [index for index in range(env.env_config.num_ships) if index != slot]
    assert not env.state.ship_spawned[:, others].any(), "and only for the ship that spawned"


def test_direct_stepping_restores_fog_after_the_opening_reveal() -> None:
    """The regression itself: a stuck latch makes ``spawn_reveal`` omniscience.

    Measured as enemy visibility over a stretch of decisions with ships that
    start scattered over a 5200 px circle behind a 512 px sight radius. With the
    latch cleared per decision this has to fall well short of everything-visible.
    """

    env = _frontline_env(num_envs=8, num_ships=6)
    action = _hold(env)
    enemy_seen: list[float] = []
    for _ in range(40):
        env.step(action)
        visibility = team_visibility_from_state(
            env.state, env.ship_config, env.env_config, perceive_bullets=False
        )
        for team in (0, 1):
            enemy = env.state.ship_team_id != team
            enemy_seen.append(float(visibility.ship[:, team][enemy].float().mean()))

    fraction = sum(enemy_seen) / len(enemy_seen)
    assert fraction < 0.9, f"enemies are effectively always visible: {fraction:.3f}"


def test_the_wrapper_and_direct_step_agree_on_the_latch() -> None:
    """One production contract, not two: the two entry points must not diverge."""

    config = EnvConfig(
        num_ships=4,
        max_bullets=0,
        max_episode_steps=600,
        vision_range=512.0,
        spawn_reveal=True,
        frontline=_FRONTLINE,
    )
    ship = ShipConfig(world_size=FRONTLINE_WORLD_SIZE)
    wrapper = YemongEnvWrapper(2, ship, config, REWARDS, "cpu")
    wrapper.reset(seed=7)
    direct = TensorEnv(2, ship, config, "cpu")
    direct.reset(seed=7)

    action = torch.zeros((2, 4, 3), dtype=torch.int32)
    for step_index in range(6):
        if step_index == 2:
            for env in (wrapper.env, direct):
                _push_outside(env, 0)
        wrapper.step(action)
        direct.step(action)
        assert torch.equal(wrapper.state.ship_spawned, direct.state.ship_spawned), (
            f"latch diverged on decision {step_index}"
        )
