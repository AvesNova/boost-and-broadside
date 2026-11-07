"""The observation's acceleration columns (observation-accelerations brief): world-frame
change of the velocity and the nose rate over the last tick, per second, with a flag that
is 1 only for a real tick of motion the team saw at both ticks."""

import math

import pytest
import torch

from bnb.contracts import Action, GameConfig
from bnb.contracts.action import NUM_POWER, NUM_SHOOT, NUM_STICK
from bnb.contracts.obs import GLOBAL, SHIP, SHIP_HIDDEN_ZERO
from bnb.game import api
from bnb.game.state import replace

TYPES = api.ship_types()
F32 = torch.float32
ACCELERATION = ("acceleration_x", "acceleration_y", "nose_acceleration")


def _two_ships(fogged: bool = True):
    """One ship per team, no field cores (sight is range alone), ship 0 at the origin heading
    0.7 rad at 80 m/s with nose rate 1 rad/s, ship 1 (the enemy) 300 m off, in sight, after
    one perception at tick 5."""
    rules = api.rules_of(GameConfig(1, 1, 1800.0))
    state = api.reset(
        rules,
        TYPES,
        api.uniforms(rules, 1, generator=torch.Generator().manual_seed(0)),
        torch.tensor([fogged]),
    )
    state = replace(state, fields=replace(state.fields, radius=torch.zeros_like(state.fields.x)))
    state = _set(
        state,
        x=[0.0, 300.0],
        y=[0.0, 0.0],
        heading=[0.7, 2.0],
        proper_speed=[80.0, 60.0],
        nose_rate=[1.0, -0.5],
        log_refractive_index=[0.0, 0.0],
        alive=[True, True],
        spawned=[False, False],
    )
    state = replace(
        state, match=replace(state.match, episode_tick=torch.tensor([5], dtype=torch.int32))
    )
    state, _, _ = api.perceive(rules, state)
    return rules, state


def _set(state, heading=None, **values):
    ships = state.ships
    changes = {}
    for name, value in values.items():
        dtype = getattr(ships, name).dtype
        changes[name] = torch.tensor([value], dtype=dtype)
    if heading is not None:
        heading = torch.tensor([heading], dtype=F32)
        changes.update(heading_cos=torch.cos(heading), heading_sin=torch.sin(heading))
    if "angle_of_attack" not in changes:
        zero = torch.zeros_like(ships.angle_of_attack)
        changes.update(
            angle_of_attack=zero, angle_of_attack_cos=zero + 1.0, angle_of_attack_sin=zero
        )
    return replace(state, ships=replace(ships, **changes))


def _next(rules, state, **values):
    """Perceive the next tick with the ships' values changed."""
    state = _set(state, **values)
    state = replace(state, match=replace(state.match, episode_tick=state.match.episode_tick + 1))
    return api.perceive(rules, state)


def _columns(rows: torch.Tensor) -> tuple[torch.Tensor, float]:
    """(the three values, the flag) of one row."""
    return rows[..., [SHIP[name] for name in ACCELERATION]], rows[
        ..., SHIP["acceleration_valid"]
    ].item()


def test_a_straight_push_is_the_velocity_change_along_the_heading():
    rules, state = _two_ships()
    rate, dt = 12.0, rules.tick_length
    _, obs, truth = _next(
        rules, state, proper_speed=[80.0 + rate * dt, 60.0], nose_rate=[1.5, -0.5]
    )
    values, flag = _columns(obs.ships[0, 0, 0])
    assert flag == 1.0
    torch.testing.assert_close(
        values[:2], rate * torch.tensor([math.cos(0.7), math.sin(0.7)]), rtol=0, atol=2e-3
    )
    assert values[2].item() == pytest.approx(0.5 / dt, rel=1e-5)
    assert torch.equal(truth.ships[0, 0, 0], obs.ships[0, 0, 0])


@pytest.mark.parametrize("path_rate", [0.8, -0.8])
def test_a_steady_turn_is_across_the_velocity(path_rate):
    """Constant speed, the nose (the flight path at zero angle of attack) turning at
    path_rate: the change is |v| sin(theta) / dt across the velocity (left positive),
    |v| (1 - cos theta) / dt along it, theta the turn in one tick."""
    rules, state = _two_ships()
    dt, speed = rules.tick_length, 80.0
    turn = path_rate * dt
    _, obs, _ = _next(rules, state, heading=[0.7 + turn, 2.0])
    values, flag = _columns(obs.ships[0, 0, 0])
    unit = torch.tensor([math.cos(0.7 + turn), math.sin(0.7 + turn)])
    along = (values[:2] * unit).sum().item()
    across = (values[1] * unit[0] - values[0] * unit[1]).item()
    assert flag == 1.0
    assert across == pytest.approx(speed * math.sin(turn) / dt, rel=1e-3)
    assert across == pytest.approx(speed * path_rate, rel=2e-3)
    assert along == pytest.approx(speed * (1.0 - math.cos(turn)) / dt, abs=5e-2)


def test_seen_at_both_ticks_the_enemy_row_has_its_change():
    rules, state = _two_ships()
    _, obs, truth = _next(rules, state, proper_speed=[80.0, 62.0])
    values, flag = _columns(obs.ships[0, 0, 1])  # team 0's view, the enemy row
    assert flag == 1.0 and obs.ships[0, 0, 1, SHIP["visible"]] == 1.0
    assert torch.equal(obs.ships[0, 0, 1], truth.ships[0, 0, 1])
    torch.testing.assert_close(
        values[:2],
        2.0 / rules.tick_length * torch.tensor([math.cos(2.0), math.sin(2.0)]),
        rtol=1e-4,
        atol=1e-3,
    )


@pytest.mark.parametrize("case", ["dead_now", "dead_before", "spawned", "first_tick"])
def test_an_invalid_tick_is_zero_with_flag_zero_in_the_views_and_the_truth(case):
    rules, state = _two_ships()
    changes = dict(proper_speed=[90.0, 70.0])
    if case == "dead_now":
        changes["alive"] = [False, True]
    elif case == "dead_before":
        state = replace(
            state, match=replace(state.match, previous_alive=torch.tensor([[False, True]]))
        )
    elif case == "spawned":
        changes["spawned"] = [True, False]
    state = _set(state, **changes)
    if case == "first_tick":  # a game's first tick (a reset): the clock at 0
        state = replace(
            state, match=replace(state.match, episode_tick=torch.tensor([-1], dtype=torch.int32))
        )
    _, obs, truth = _next(rules, state)
    for rows in (obs.ships[0, 0, 0], obs.ships[0, 1, 1], truth.ships[0, 0, 0]):
        values, flag = _columns(rows)
        assert flag == 0.0 and (values == 0).all()
    if case not in ("dead_now", "dead_before", "spawned"):
        assert _columns(obs.ships[0, 0, 1])[1] == 0.0  # the other ship's too


def test_a_ship_just_spotted_has_no_acceleration_but_its_truth_has():
    """Hidden at t - 1, visible at t: invalid in the view (nothing of t - 1 leaks), valid in
    the truth; and a ship hidden at t shows none (the hidden block is zero)."""
    rules, state = _two_ships()
    far = rules.vision_range + 500.0
    _, obs, truth = _next(rules, state, x=[0.0, far])  # the enemy leaves sight
    assert obs.ships[0, 0, 1, SHIP["visible"]] == 0.0
    assert (obs.ships[0, 0, 1, :SHIP_HIDDEN_ZERO] == 0).all()
    assert _columns(truth.ships[0, 0, 1])[1] == 1.0
    state = _next(rules, state, x=[0.0, far])[0]
    state, obs, truth = _next(rules, state, x=[0.0, 300.0])  # and comes back
    assert obs.ships[0, 0, 1, SHIP["visible"]] == 1.0
    values, flag = _columns(obs.ships[0, 0, 1])
    assert flag == 0.0 and (values == 0).all()
    assert _columns(truth.ships[0, 0, 1])[1] == 1.0
    _, obs, _ = _next(rules, state, x=[0.0, 300.0])  # seen at both ticks again
    assert _columns(obs.ships[0, 0, 1])[1] == 1.0


def test_without_fog_every_living_ship_is_seen_and_has_its_change():
    rules, state = _two_ships(fogged=False)
    far = rules.vision_range + 500.0
    state, _, _ = _next(rules, state, x=[0.0, far])
    _, obs, _ = _next(rules, state, x=[0.0, far], proper_speed=[80.0, 61.0])
    assert _columns(obs.ships[0, 0, 1])[1] == 1.0


def test_on_a_played_game_the_columns_follow_their_definition():
    """A real fogged game: each view's columns are the change of its own rows' velocity
    columns between consecutive observations, valid exactly when the row was alive and
    seen at both, did not spawn, and the game did not just start."""
    rules = api.rules_of(GameConfig(3, 6, 1800.0, episode_seconds=4.0, respawn_seconds=1.0))
    batch = 16
    generator = torch.Generator().manual_seed(1)
    state = api.reset(rules, TYPES, api.uniforms(rules, batch, generator=generator))
    state, before, before_truth = api.perceive(rules, state)
    sizes = torch.tensor([NUM_POWER, NUM_STICK, NUM_SHOOT], dtype=F32)
    checked = dict(valid=0, just_spotted=0, spawned=0, reset=0)
    for _ in range(400):
        choice = torch.rand(batch, rules.num_ships, 3, generator=generator)
        action = Action.unpack((choice * sizes).to(torch.int32))
        state, _ = api.step(
            rules,
            TYPES,
            state,
            action,
            api.spread(rules, batch, generator=generator),
            api.uniforms(rules, batch, generator=generator),
        )
        state, obs, truth = api.perceive(rules, state)
        for now, previous, sight in ((obs, before, True), (truth, before_truth, False)):
            ships, last = now.ships, previous.ships  # the truth: every row seen

            def column(rows, name):
                return rows[..., SHIP[name]]

            seen = column(ships, "visible") > 0.5
            seen_before = column(last, "visible") > 0.5
            rule = (
                (column(ships, "alive") > 0.5)
                & (column(last, "alive") > 0.5)
                & seen
                & seen_before
                & ~(column(ships, "spawned") > 0.5)
                & ~(obs.globals[..., GLOBAL["reset"], None] > 0.5)
            )
            flag = column(ships, "acceleration_valid") > 0.5
            assert torch.equal(flag, rule)
            change = (
                torch.stack(
                    [
                        column(ships, "velocity_x") - column(last, "velocity_x"),
                        column(ships, "velocity_y") - column(last, "velocity_y"),
                        column(ships, "nose_rate") - column(last, "nose_rate"),
                    ],
                    -1,
                )
                / rules.tick_length
            )
            values = ships[..., [SHIP[name] for name in ACCELERATION]]
            torch.testing.assert_close(values[rule], change[rule], rtol=1e-6, atol=1e-3)
            assert (values[~rule] == 0).all()
            if sight:
                checked["valid"] += int(rule.sum())
                checked["just_spotted"] += int(
                    (
                        seen
                        & ~seen_before
                        & (column(last, "alive") > 0.5)
                        & (column(ships, "alive") > 0.5)
                    ).sum()
                )
                checked["spawned"] += int((column(ships, "spawned") > 0.5).sum())
                checked["reset"] += int((obs.globals[..., GLOBAL["reset"]] > 0.5).sum())
        before, before_truth = obs, truth
    assert all(count > 0 for count in checked.values()), checked
