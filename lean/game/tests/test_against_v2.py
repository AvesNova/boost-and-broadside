"""Trajectory comparison against v2 (game-audit.md section 5, checks A and B).

Needs v2 on the path (PYTHONPATH=src); skipped otherwise. v2 runs 5v5 with no fields and
the shrink switched off; its ships are pulled toward the centre at tick 0 so that bullets
hit within the run. Check A: one lean step from each v2 state against v2's next state.
Check B: an open-loop lean rollout on v2's actions and spread for 150 ticks.
"""

import dataclasses

import pytest
import torch

api = pytest.importorskip("bnb.game.api")
from bnb.contracts import Action  # noqa: E402

from lean.contract import State  # noqa: E402
from lean.game import GameConfig, step  # noqa: E402
from lean.game.core import fixed_fields, ship_constants  # noqa: E402

ENVS, TICKS, PER_TEAM = 32, 150, 5  # about 6 s on one CPU thread
torch.set_num_threads(1)
# (99.9% bound, bound on every sample) per quantity, section 5.5 check A.
STRICT = {
    "x": (1e-3, 2e-3), "y": (1e-3, 2e-3), "heading_cos": (5e-7, 1e-4), "heading_sin": (5e-7, 1e-4),
    "angle_cos": (5e-7, 1e-4), "angle_sin": (5e-7, 1e-4), "speed": (1e-4, 2e-3), "angle": (3e-7, 5e-4),
    "nose_rate": (5e-6, 5e-3), "energy": (0.06, 0.1), "shield": (5e-3, 2e-2),
}
V2_NAMES = {"speed": "proper_speed", "angle": "angle_of_attack", "angle_cos": "angle_of_attack_cos",
            "angle_sin": "angle_of_attack_sin", "shield_wait": "shield_recharge_wait", "cooldown": "gun_cooldown"}


def v2_setup():
    replace = dataclasses.replace
    game = replace(api.PRESETS["real_scale_elimination"], ships_per_team=PER_TEAM, num_fields=0)
    types = api.ship_types(ship_set=game.ship_set)
    rules = api.rules_of(game, types=types)
    pool = rules.pools[0]
    mode = replace(pool.mode, shrink_hold_seconds=1e9, shrink_end_lead_seconds=0.0)
    rules = replace(rules, pools=(replace(pool, mode=mode),))
    generator = torch.Generator().manual_seed(5)
    state = api.reset(rules, types, api.uniforms(rules, ENVS, generator=generator),
                      fogged=torch.zeros(ENVS, dtype=torch.bool))
    ships = replace(state.ships, x=state.ships.x * 0.03, y=state.ships.y * 0.03)
    return rules, types, replace(state, ships=ships), generator


def to_lean(config, v2_state) -> State:
    ships, bullets = v2_state.ships, v2_state.bullets
    ship_type = ships.ship_type.long()
    envs, slots = ship_type.shape
    values = {}
    for name in ("x", "y", "heading_cos", "heading_sin", "speed", "angle", "angle_cos", "angle_sin",
                 "nose_rate", "energy", "shield", "shield_wait"):
        values[name] = getattr(ships, V2_NAMES.get(name, name))
    return State.pack(
        **values,
        cooldown=ships.gun_cooldown, next_slot=ships.next_slot, alive=ships.alive,
        exists=torch.ones_like(ships.alive), team=(torch.arange(slots) >= PER_TEAM).long().expand(envs, slots),
        type=ship_type, constants=ship_constants(config, ship_type),
        bullet_x=bullets.x, bullet_y=bullets.y, bullet_dir_x=bullets.direction_x,
        bullet_dir_y=bullets.direction_y, bullet_speed=bullets.proper_speed,
        bullet_moves_left=bullets.moves_left, tick=v2_state.match.episode_tick.long(),
        fired=torch.zeros_like(bullets.moves_left[..., 0], dtype=torch.bool),
        bullet_real=bullets.moves_left > 0, bullet_fired_tick=torch.zeros_like(bullets.moves_left),
        log_index=ships.log_refractive_index, gradient_x=ships.log_index_gradient_x,
        gradient_y=ships.log_index_gradient_y, bullet_log_index=bullets.log_refractive_index,
        bullet_gradient_x=bullets.log_index_gradient_x, bullet_gradient_y=bullets.log_index_gradient_y,
        fields=fixed_fields(config, envs, "cpu"),
    )


def lean_result(v2_result, done):
    """v2's result (+1 team 0 wins, -1 team 1, 0 draw) in the lean coding."""
    coded = torch.where(v2_result > 0, 0, torch.where(v2_result < 0, 1, 2))
    return torch.where(done, coded, -1)


def test_checks_a_and_b():
    rules, types, v2_state, generator = v2_setup()
    config = GameConfig(arena_radius=7800.0, clock_ticks=5400)
    open_loop = to_lean(config, v2_state)
    valid = torch.ones(ENVS, dtype=torch.bool)  # environments with no discrete difference yet
    ended = torch.zeros(ENVS, dtype=torch.bool)
    worst = {name: [] for name in STRICT}
    discrete_mismatches, drift = 0, {}
    held = None
    for tick in range(TICKS):
        drawn = torch.stack([torch.randint(0, n, (ENVS, 2 * PER_TEAM), generator=generator) for n in (7, 3, 2)], -1)
        keep = torch.rand((ENVS, 2 * PER_TEAM, 1), generator=generator) < 0.9
        held = drawn if held is None else torch.where(keep, held, drawn)
        spread = torch.randn((ENVS, 2 * PER_TEAM, 2), generator=generator)
        v2_action = Action(power=held[..., 1].int(), stick=held[..., 0].int(), shoot=held[..., 2].int())
        before = to_lean(config, v2_state)
        alive0 = before.alive & ~ended[:, None]
        v2_state, events = api.step(rules, types, v2_state, v2_action, spread,
                                    api.uniforms(rules, ENVS, generator=generator))
        after, out = step(config, before, held, {"spread": spread})
        live = ~ended & ~events.done  # v2 returns the next episode's state where done
        # v2 sets a dying ship's speed to the floor and its nose rate to zero; lean keeps them (NOTES).
        mask = alive0 & live[:, None] & ~events.died
        for name in STRICT:
            reference = getattr(v2_state.ships, V2_NAMES.get(name, name))
            worst[name].append((getattr(after, name) - reference).abs()[mask])
        discrete_mismatches += int((out.died != events.died)[~ended].sum())
        discrete_mismatches += int((out.fired != events.fired)[~ended].sum())
        discrete_mismatches += int((out.done != events.done)[~ended].sum())
        discrete_mismatches += int((out.result != lean_result(events.result, events.done))[~ended].sum())
        discrete_mismatches += int((out.killer != events.killer.long())[~ended].sum())
        discrete_mismatches += int((after.bullet_moves_left != v2_state.bullets.moves_left)[live].sum())
        discrete_mismatches += int((after.cooldown != v2_state.ships.gun_cooldown)[live].sum())
        # Check B.
        open_loop, open_out = step(config, open_loop, held, {"spread": spread})
        valid &= ~ended & (open_out.died == events.died).all(-1) & ~events.done
        ended |= events.done
        if tick + 1 in (30, 150) and valid.any():
            rows = valid[:, None] & open_loop.alive
            drift[tick + 1] = {
                "position": max(float((open_loop.x - v2_state.ships.x).abs()[rows].max()),
                                float((open_loop.y - v2_state.ships.y).abs()[rows].max())),
                "speed": float((open_loop.speed - v2_state.ships.proper_speed).abs()[rows].max()),
                "energy": float((open_loop.energy - v2_state.ships.energy).abs()[rows].max()),
                "shield": float((open_loop.shield - v2_state.ships.shield).abs()[rows].max()),
            }
    report = {}
    for name, (most, every) in STRICT.items():
        errors = torch.cat(worst[name])
        report[name] = (float(errors.quantile(0.999)), float(errors.max()))
    print("check A (99.9%, max):", report)
    print("discrete mismatches:", discrete_mismatches, "; check B drift:", drift,
          "; environments kept:", int(valid.sum()), "of", ENVS)
    for name, (most, every) in STRICT.items():
        assert report[name][0] <= most and report[name][1] <= every, (name, report[name])
    assert discrete_mismatches == 0
    bounds = {30: dict(position=0.05, speed=0.01, energy=5.0, shield=0.1),
              150: dict(position=0.25, speed=0.05, energy=30.0, shield=0.3)}
    for at, limits in ((at, bounds[at]) for at in drift):
        for name, limit in limits.items():
            assert drift[at][name] <= limit, (at, name, drift[at][name])
