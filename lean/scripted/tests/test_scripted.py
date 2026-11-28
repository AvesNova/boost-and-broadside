"""The look-ahead scripted opponent and the chaser: well-formed distributions, the candidate
paths against the game's tick, the nose spring's closed form, the reply collapse, the enemies it
scores (roam, cutoff, dead reckoning), the gun, collisions and the boundary, no goal point with
no enemy known, the persona population, and games. Every game number is read from the game."""

import dataclasses
import functools
import math

import pytest
import torch

from lean import game
from lean.contract import JOINT, POWERS, STICKS, join_action, map_fields
from lean.game import card, core
from lean.sandbox.world import World
from lean.scripted import ScriptedConfig, act, chaser_act, draw_uniforms, explain, personas
from lean.scripted.opponent import (
    ENEMIES,
    after_pending,
    collapse,
    dead_reckon,
    nearest_enemies,
)
from lean.scripted.parameters import PARAMETERS, SAMPLED, rows_of, vector
from lean.scripted.paths import CANDIDATES, LOOK_AHEAD, nose_response, predict

CPU = torch.device("cpu")
# The stick indices that turn left (a positive level) and right (card.STICK_LEVELS).
LEFT = [i for i, level in enumerate(card.STICK_LEVELS) if level > 0]
RIGHT = [i for i, level in enumerate(card.STICK_LEVELS) if level < 0]


@functools.cache
def card_values() -> dict:
    """The default game's ship card (its speeds, radii and ranges; core.ship_card): the numbers
    equal for every type and the first type's single rows (the hit model is always rows)."""
    ship_card = core.ship_card(core.GameConfig())
    rows = {
        name: float(ship_card[name].flatten()[0])
        for name, (_, width) in ship_card.columns.items()
        if width == 1
    }
    return {**ship_card.uniform, **rows}


def flight_of(stick: int, power: int) -> int:
    return stick * POWERS + power


def random_state(envs: int, slots: int, seed: int, fog: bool = False, fields: int = 0):
    """A dealt core-game state with ships at random places, headings and speeds inside a 3 km
    square, two teams, and a few dead and absent rows."""
    generator = torch.Generator().manual_seed(seed)
    config = core.GameConfig(field_count=fields, fog=fog)
    sizes = torch.full((envs, 2), slots // 2, dtype=torch.long)
    state = core.reset(config, envs, sizes, generator, CPU)

    def uniform(low: float, high: float) -> torch.Tensor:
        return low + (high - low) * torch.rand((envs, slots), generator=generator)

    heading, angle = uniform(-math.pi, math.pi), uniform(-0.2, 0.2)
    exists = uniform(0.0, 1.0) > 0.15
    return core.with_velocity(
        state.replace(
            x=uniform(-1500.0, 1500.0),
            y=uniform(-1500.0, 1500.0),
            heading_cos=heading.cos(),
            heading_sin=heading.sin(),
            speed=uniform(1.1 * card_values()["stall_speed"], 1.2 * card_values()["cruise_speed"]),
            angle=angle,
            angle_cos=angle.cos(),
            angle_sin=angle.sin(),
            energy=uniform(0.0, 1.0) * state.constants["max_energy"],
            cooldown=uniform(0.0, 3.0).floor(),
            alive=exists & (uniform(0.0, 1.0) > 0.2),
            exists=exists,
        )
    )


def placed(*rows: dict, teams=(0, 1), shields=None):
    """One environment, no fog or fields: ships (x, y, heading in radians, speed in m/s) in
    slot order, the teams' ships in order (team 0 first), straight flight, half store, gun
    ready, coasting straight pending."""
    config = core.GameConfig(spawn="mixed", field_count=0, fog=False)
    sizes = torch.tensor([[list(teams).count(0), list(teams).count(1)]])
    state = core.reset(config, 1, sizes, torch.Generator().manual_seed(0), CPU)
    count = len(rows)
    heading = torch.tensor([[row["heading"] for row in rows]])
    zeros = torch.zeros(1, count)
    shield = state.shield if shields is None else torch.tensor([shields], dtype=torch.float32)
    return core.with_velocity(
        state.replace(
            x=torch.tensor([[row["x"] for row in rows]]),
            y=torch.tensor([[row["y"] for row in rows]]),
            heading_cos=heading.cos(),
            heading_sin=heading.sin(),
            speed=torch.tensor([[row["speed"] for row in rows]]),
            angle=zeros,
            angle_cos=torch.ones(1, count),
            angle_sin=zeros,
            nose_rate=zeros,
            energy=torch.full((1, count), 0.5) * state.constants["max_energy"],
            cooldown=torch.zeros_like(state.cooldown),
            shield=shield,
        )
    )


def ship(x=0.0, y=0.0, heading=0.0, speed=None) -> dict:
    return dict(x=x, y=y, heading=heading, speed=speed or card_values()["corner_speed"])


def choice_of(state, team=0, config=None):
    return explain(config or ScriptedConfig(), state, team, None)


def stick_share(choice, sticks, row=0) -> float:
    """The probability that ship `row` (of environment 0) flies one of the stick indices."""
    return float(choice.flight[0, row].view(STICKS, POWERS)[sticks].sum())


# ------------------------------------------------------------------ well-formed


def test_actions_in_range_and_zero_for_uncontrolled_rows():
    state = random_state(32, 10, seed=1)
    for team in (0, 1, None):
        action = act(ScriptedConfig(), state, team, None, draw_uniforms(32, 10, None, CPU))
        assert action.shape == (32, 10, 3)
        assert action[..., 0].max() < STICKS and action[..., 1].max() < POWERS
        controlled = state.exists & state.alive
        if team is not None:
            controlled = controlled & (state.team == team)
        assert (action[~controlled] == 0).all()
    chased = chaser_act(state, 0)
    assert (chased[~(state.exists & state.alive & (state.team == 0))] == 0).all()


def test_the_joint_is_flight_times_fire_in_the_contracts_order():
    """The teacher's 42-way distribution is the flight softmax (21) times the fire
    probability, indexed as lean.contract.join_action; a proper distribution."""
    state = random_state(16, 10, seed=2, fog=True, fields=4)
    state = core.perceive(state, torch.zeros_like(state.alive))
    choice = choice_of(state, None)
    assert choice.joint.shape == (16, 10, JOINT) and CANDIDATES * POWERS * 2 == JOINT
    rows = choice.controlled
    assert torch.allclose(choice.joint.sum(-1)[rows], torch.ones(()), atol=1e-5)
    assert torch.isfinite(choice.joint).all()
    for stick in range(STICKS):
        for power in range(POWERS):
            for fire in range(2):
                index = int(join_action(torch.tensor([stick, power, fire])))
                chance = choice.fire if fire else 1.0 - choice.fire
                expected = choice.flight[..., flight_of(stick, power)] * chance
                assert torch.allclose(choice.joint[..., index][rows], expected[rows], atol=1e-6)


def test_spread_zero_is_the_yardstick_and_full_rows_decide_alike():
    assert personas(ScriptedConfig(), 4, 10, torch.Generator().manual_seed(0), CPU, 0.0) is None
    state = random_state(8, 10, seed=3)
    table = torch.tensor([vector(ScriptedConfig())])
    rows = rows_of(table, torch.zeros(8, 10, dtype=torch.long))
    assert set(rows) == set(PARAMETERS)
    plain = explain(ScriptedConfig(), state, None, None)
    full = explain(ScriptedConfig(), state, None, rows)
    torch.testing.assert_close(full.joint, plain.joint, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("measure", [0.0, 1.0, 2.0])
def test_the_crowd_rows_interpolate_between_duel_and_crowd(measure):
    """parameters.CROWDED: the crowd ratio changes nothing in a duel (one ship a side) and
    something in a crowd, and numbers and full tensor rows decide alike."""
    crowded = ScriptedConfig().replace(
        crowd_measure=measure, angles_weight_crowd=3.0, temperature_crowd=0.5
    )
    duel = random_state(8, 2, seed=21)
    torch.testing.assert_close(
        explain(crowded, duel, None, None).joint, explain(ScriptedConfig(), duel, None, None).joint
    )
    crowd = random_state(8, 16, seed=22)
    plain = explain(ScriptedConfig(), crowd, None, None).joint
    numbers = explain(crowded, crowd, None, None).joint
    controlled = crowd.exists & crowd.alive
    assert (numbers - plain)[controlled].abs().max() > 1e-3
    table = torch.tensor([vector(crowded)])
    rows = rows_of(table, torch.zeros(8, 16, dtype=torch.long))
    full = explain(crowded, crowd, None, rows).joint
    torch.testing.assert_close(full, numbers, atol=1e-5, rtol=1e-5)


def test_randomness_only_from_sampling():
    state = random_state(8, 10, seed=4)
    persona = personas(ScriptedConfig(), 8, 10, torch.Generator().manual_seed(5), CPU, 1.0)
    uniforms = draw_uniforms(8, 10, torch.Generator().manual_seed(6), CPU)
    first = act(ScriptedConfig(), state, None, persona, uniforms)
    assert torch.equal(first, act(ScriptedConfig(), state, None, persona, uniforms))


def test_one_call_for_both_teams_equals_the_two_single_team_calls():
    state = random_state(8, 10, seed=7)
    uniforms = draw_uniforms(8, 10, torch.Generator().manual_seed(8), CPU)
    both = act(ScriptedConfig(), state, None, None, uniforms)
    split = act(ScriptedConfig(), state, 0, None, uniforms) + act(
        ScriptedConfig(), state, 1, None, uniforms
    )
    assert torch.equal(both, split)


def test_every_sampled_row_has_a_box_and_a_meaning():
    for name in SAMPLED:
        row = PARAMETERS[name]
        assert row.width > 0.0 and row.meaning, name


# ------------------------------------------------------------------ the paths


def test_the_candidate_paths_follow_the_games_tick():
    """Each of the 21 actions held in the real tick (each power's 7 stick paths predicted at
    that power): at the last look-ahead point a whole number of ticks reaches, the predicted
    positions stay near the flown ones (straight coast within a metre; every action's median
    within 25 m)."""
    config = core.GameConfig(field_count=0, fog=False, training_spawn=True)
    generator = torch.Generator().manual_seed(9)
    state = core.reset(config, 16, torch.ones(16, 2, dtype=torch.long), generator, CPU)
    ticks = [time / config.tick_length for time in LOOK_AHEAD]
    last = [k for k, count in enumerate(ticks) if abs(count - round(count)) < 1e-6][-1]
    for power in range(POWERS):
        paths = predict(state.constants, state.fields, state, torch.full_like(state.team, power))
        for stick in range(CANDIDATES):
            action = torch.zeros(16, 2, 3, dtype=torch.long)
            action[..., 0], action[..., 1] = stick, power
            flown = state.replace(pending=action.clone())
            for _ in range(round(ticks[last])):
                noise = core.draw_noise(config, 16, 2, None, CPU)
                flown, _ = core.step(config, flown, action, noise)
            error = torch.hypot(
                paths.x[..., stick, last] - flown.x, paths.y[..., stick, last] - flown.y
            )
            limit = 1.0 if stick == 0 and power == 0 else 25.0
            assert float(error.median()) < limit, (stick, power, float(error.median()))


def test_the_power_follows_the_own_ship_terms():
    """Slow against the preferred speed with a store to spend, the power look prefers the dive
    (store into speed) to the climb; fast with room in the store, the climb. The store's own
    weight (`store_weight`, which values the store held, 0.5 by default) is switched off to
    isolate the speed and energy terms."""
    corner = card_values()["corner_speed"]
    slow = placed(ship(speed=0.6 * corner), ship(x=3000.0, speed=corner))
    fast = placed(ship(speed=1.4 * corner), ship(x=3000.0, speed=corner))
    dive, climb = 1, 2
    config = ScriptedConfig().replace(store_weight=0.0)
    assert float(choice_of(slow, config=config).power_scores[0, 0, dive]) > float(
        choice_of(slow, config=config).power_scores[0, 0, climb]
    )
    assert float(choice_of(fast, config=config).power_scores[0, 0, climb]) > float(
        choice_of(fast, config=config).power_scores[0, 0, dive]
    )


@pytest.mark.parametrize("stiffness", [0.2, 1.0 / 3.0, 1.0])
def test_the_nose_spring_in_closed_form_matches_its_equation(stiffness):
    """e'' + 2 sigma e' + k e = 0 integrated finely against the closed form: over-, critically
    (sigma^2 = 1/3 with k = 1/3: Sustain's spring) and under-damped."""
    sigma, k = torch.tensor([math.sqrt(1.0 / 3.0)]), torch.tensor([stiffness])
    offset, rate = torch.tensor([0.3]), torch.tensor([-0.5])
    e, v, integral, step = 0.3, -0.5, 0.0, 1e-5
    marks = {round(point / step): point for point in LOOK_AHEAD}
    for n in range(1, round(LOOK_AHEAD[-1] / step) + 1):
        a = -2.0 * float(sigma) * v - stiffness * e
        integral += (e + 0.5 * v * step) * step
        e, v = e + v * step + 0.5 * a * step * step, v + a * step
        if n in marks:
            value, derivative, total = nose_response(offset, rate, sigma, k, marks[n])
            assert abs(float(value) - e) < 1e-3 and abs(float(derivative) - v) < 1e-3
            assert abs(float(total) - integral) < 1e-3


def test_collapse_limits():
    """beta -> 0 is the weighted mean (continuous through 0); a very negative beta the worst
    of his replies; a cold tau_him his best reply alone."""
    generator = torch.Generator().manual_seed(10)
    mine = torch.randn(1, 1, 1, CANDIDATES, CANDIDATES, generator=generator)
    his = torch.randn(1, 1, 1, CANDIDATES, CANDIDATES, generator=generator)
    mean = (torch.softmax(his / 0.5, -1) * mine).sum(-1)
    torch.testing.assert_close(collapse(mine, his, 0.5, 0.0), mean)
    # Across the switch to the variance form at |beta| = 1e-3: continuous.
    above, below = collapse(mine, his, 0.5, 1.05e-3), collapse(mine, his, 0.5, 0.95e-3)
    assert torch.allclose(above, below, atol=2e-4)
    assert torch.allclose(collapse(mine, his, 1e6, -200.0), mine.amin(-1), atol=0.05)
    best_reply = his.argmax(-1, keepdim=True)
    torch.testing.assert_close(collapse(mine, his, 1e-4, 0.0), mine.gather(-1, best_reply)[..., 0])


# ------------------------------------------------------------------ the enemies


def ghost_state(distances, seconds, nose_rate=0.0):
    """Fog on, no fields: the own ship (slot 0) at the centre heading +x, one enemy per
    distance on the +x axis heading +y, each last seen `seconds` ago there."""
    config = core.GameConfig(spawn="mixed", field_count=0, fog=True)
    state = core.reset(config, 1, torch.tensor([[1, len(distances)]]), torch.Generator(), CPU)
    zeros = torch.zeros_like(state.x)
    heading = torch.tensor([[0.0] + [math.pi / 2] * len(distances)])
    state = core.with_velocity(
        state.replace(
            x=torch.tensor([[0.0, *distances]]),
            y=zeros,
            heading_cos=heading.cos(),
            heading_sin=heading.sin(),
            speed=torch.full_like(state.x, state.constants["sustained_speed"]),
            angle=zeros,
            angle_cos=torch.ones_like(state.x),
            angle_sin=zeros,
            nose_rate=torch.full_like(state.x, nose_rate),
        )
    )
    state = core.perceive(state, torch.ones_like(state.alive))
    ticks = [round(value / state.constants["tick_length"]) for value in (0.0, *seconds)]
    return state.replace(ticks_since_seen=torch.tensor([ticks]))


def test_a_hidden_enemy_costs_its_roam_and_is_dropped_past_the_cutoff():
    """A hidden enemy's distance grows by its roam (seconds since seen x sustained speed), its
    weight falls linearly to 0 at the cutoff, and beyond it the enemy is not scored."""
    c = ghost_state((600.0, 900.0), (0.0, 0.0)).constants
    cutoff = PARAMETERS["roam_cutoff"].default
    late = (900.0 - 600.0) / c["sustained_speed"]
    for seconds, nearest in ((0.5 * late, 1), (1.5 * late, 2)):
        state = ghost_state((600.0, 900.0), (seconds, 0.0))
        slot, _, _ = nearest_enemies(state, core.enemy_view(state), cutoff)
        assert int(slot[0, 0, 0]) == nearest
    limit = cutoff * c["arena_radius"] / c["sustained_speed"]
    state = ghost_state((600.0,), (0.5 * limit,))
    _, weight, _ = nearest_enemies(state, core.enemy_view(state), cutoff)
    assert weight.shape[-1] == ENEMIES and abs(float(weight[0, 0, 0]) - 0.5) < 0.01
    state = ghost_state((600.0,), (1.1 * limit,))
    _, _, present = nearest_enemies(state, core.enemy_view(state), cutoff)
    assert not present[0, 0].any() and float(choice_of(state).fire[0, 0]) == 0.0


def test_a_hidden_enemy_is_carried_forward_on_his_arc_then_straight():
    """Author 2026-11-26: dead reckoning at his last speed and nose rate for up to 2 s, then
    straight on; a visible ship stays where it is."""
    rate = 0.5
    speed = card_values()["sustained_speed"]
    for seconds in (1.0, 3.0):
        state = ghost_state((1000.0,), (seconds,), nose_rate=rate)
        view = core.enemy_view(state)
        since = state.ticks_since_seen.float() * state.constants["tick_length"]
        moved = dead_reckon(view, since)
        turning = min(seconds, 2.0)
        radius = speed / rate
        # From (1000, 0) heading +y, turning left: the circle's centre is (1000 - r, 0).
        heading = math.pi / 2 + rate * turning
        straight = speed * (seconds - turning)
        expected_x = 1000.0 - radius + radius * math.cos(rate * turning)
        expected_x += straight * math.cos(heading)
        expected_y = radius * math.sin(rate * turning) + straight * math.sin(heading)
        assert abs(float(moved.x[0, 1]) - expected_x) < 0.5
        assert abs(float(moved.y[0, 1]) - expected_y) < 0.5
        assert (float(moved.nose_rate[0, 1]) == 0.0) == (seconds > 2.0)
        assert float(moved.x[0, 0]) == float(view.x[0, 0])


def test_no_goal_point_without_an_enemy():
    """Author 2026-11-26: with no enemy known (never seen) the agent has no goal point: away
    from the edge its distribution does not depend on where it points."""
    config = core.GameConfig(spawn="mixed", field_count=0, fog=True)
    state = core.reset(config, 1, torch.tensor([[1, 1]]), torch.Generator().manual_seed(0), CPU)
    seen = state.seen.clone()
    seen[:, core.SEEN_KNOWN] = 0.0  # never seen
    state = state.replace(seen=seen)
    zeros = torch.zeros(1, 2)

    def pointing(angle: float):
        heading = torch.full((1, 2), angle)
        return core.with_velocity(
            state.replace(
                x=torch.tensor([[500.0, -3000.0]]),
                y=zeros,
                heading_cos=heading.cos(),
                heading_sin=heading.sin(),
                angle=zeros,
                angle_cos=torch.ones(1, 2),
                angle_sin=zeros,
                nose_rate=zeros,
            )
        )

    first, second = choice_of(pointing(0.0)), choice_of(pointing(2.0))
    torch.testing.assert_close(first.flight[0, 0], second.flight[0, 0], atol=1e-4, rtol=1e-3)


# ------------------------------------------------------------------ the fight


def test_turns_toward_an_enemy_on_its_left():
    state = placed(ship(), ship(x=400.0, y=400.0, heading=math.pi / 2))
    assert stick_share(choice_of(state), LEFT) > 0.8


def test_fires_at_a_tail_in_range_and_never_inside_his_nose_cone():
    """Dead astern in range, nose on: it fires. Nose to nose: with the nose armour's rule on
    (the default again since 2026-11-28) it holds fire, even at an empty shield (inside the
    nose cone a hit does nothing; author 2026-11-25); with the rule off it fires there too."""
    distance = 0.5 * card_values()["gun_range"]
    astern = placed(ship(), ship(x=distance))
    assert float(choice_of(astern).fire[0, 0]) > 0.5
    nose_to_nose = placed(ship(), ship(x=distance, heading=math.pi), shields=[100.0, 0.0])
    assert float(choice_of(nose_to_nose).fire[0, 0]) == 0.0
    off = nose_to_nose.replace(constants=core.with_hit_model(nose_to_nose.constants, nose_cone=0.0))
    assert float(choice_of(off).fire[0, 0]) > 0.5


def test_the_trigger_weighs_the_hit_chance():
    """Dead astern near the gun's range, nose on: it fires with the default spread; with a wide
    spread (the game's hit widths unchanged) the expected hit is below its fire worth and it
    holds fire (the hit chance, round 4)."""
    distance = 0.9 * card_values()["gun_range"]
    astern = placed(ship(), ship(x=distance))
    assert float(choice_of(astern).fire[0, 0]) > 0.5
    wide = astern.replace(constants=core.with_hit_model(astern.constants, bullet_spread=60.0))
    assert float(choice_of(wide).fire[0, 0]) < 0.05


@pytest.mark.parametrize("fields", [0, 4])
def test_the_pending_tick_is_the_games_tick(fields):
    """The pose an action chosen now starts from (`after_pending`) is the game's next tick
    under the pending flight action (no pending shot: the firing tick's thrust cut is left
    out)."""
    state = random_state(4, 8, seed=11, fields=fields)
    generator = torch.Generator().manual_seed(12)
    stick = torch.randint(0, STICKS, (4, 8), generator=generator)
    power = torch.randint(0, POWERS, (4, 8), generator=generator)
    state = state.replace(pending=torch.stack([stick, power, torch.zeros_like(stick)], -1))
    config = core.GameConfig(field_count=fields, fog=False)
    noise = {"spread": torch.zeros(4, 8, 2)}
    stepped, _ = core.step(config, state, torch.zeros_like(state.pending), noise)
    ahead = after_pending(state)
    alive = state.alive
    for name in ("x", "y", "heading_cos", "heading_sin", "speed", "nose_rate", "velocity_x"):
        torch.testing.assert_close(getattr(ahead, name)[alive], getattr(stepped, name)[alive])


def test_the_trigger_aims_from_the_pose_after_the_pending_tick():
    """The shot leaves on the next tick: a nose on the lead point now but turning off it over
    the pending tick holds fire; one turning onto it fires."""
    distance = 0.9 * card_values()["gun_range"]
    rate = torch.tensor([[2.0, 0.0]])
    turning = placed(ship(), ship(x=distance)).replace(nose_rate=rate)
    ahead = after_pending(turning)
    turned = math.atan2(float(ahead.heading_sin[0, 0]), float(ahead.heading_cos[0, 0]))
    assert float(choice_of(turning).fire[0, 0]) < 0.1
    onto = placed(ship(heading=-turned), ship(x=distance)).replace(nose_rate=rate)
    assert float(choice_of(onto).fire[0, 0]) > 0.5


def test_the_gun_is_ready_on_the_shots_tick():
    """The gun is read at the shot's tick: one tick of cooldown left counts down in the pending
    tick (ready), a pending shot that fires restarts it (not ready). With a gun slower than one
    shot a tick (the default fires every tick)."""
    distance = 0.5 * card_values()["gun_range"]
    astern = placed(ship(), ship(x=distance))
    constants = astern.constants
    uniform = {**constants.uniform, "fire_interval": 3}
    astern = astern.replace(constants=dataclasses.replace(constants, uniform=uniform))
    waiting = astern.replace(cooldown=torch.tensor([[1, 0]]))
    assert float(choice_of(waiting).fire[0, 0]) > 0.5
    pending = astern.pending.clone()
    pending[0, 0, 2] = 1
    assert float(choice_of(astern.replace(pending=pending)).fire[0, 0]) == 0.0


def test_never_shoots_through_a_team_mate():
    distance = 0.6 * card_values()["gun_range"]
    teams = (0, 0, 1)
    clear = placed(ship(), ship(y=300.0), ship(x=distance), teams=teams)
    assert float(choice_of(clear).fire[0, 0]) > 0.5
    blocked = placed(ship(), ship(x=0.5 * distance), ship(x=distance), teams=teams)
    assert float(choice_of(blocked).fire[0, 0]) == 0.0


def test_avoids_a_head_on_collision():
    """Closing head-on at 300 m on a collision line: flying straight on is not its choice."""
    state = placed(ship(), ship(x=300.0, heading=math.pi))
    assert stick_share(choice_of(state), [0]) < 0.1


def test_turns_away_from_the_edge_flying_outward():
    radius = card_values()["arena_radius"]
    state = placed(ship(x=0.97 * radius), ship(x=-0.5 * radius))
    assert stick_share(choice_of(state), [0]) < 0.2


def test_stickiness_favours_the_pending_action():
    state = placed(ship(), ship(x=2500.0, y=500.0, heading=math.pi))
    flight = flight_of(4, 0)
    pending = state.pending.clone()
    pending[0, 0, 0] = 4
    held = choice_of(state.replace(pending=pending))
    assert float(held.flight[0, 0, flight]) > float(choice_of(state).flight[0, 0, flight])


# ------------------------------------------------------------------ games


def play(envs, ticks, seed, spread=0.0, chaser_team=None):
    """4v4 games of the scripted agent in the default game (or against the chaser on
    `chaser_team`); finished games re-dealt (against the chaser: stop once every environment
    has a result)."""
    generator = torch.Generator().manual_seed(seed)
    config = game.GameConfig()
    sizes = torch.full((envs, 2), 4, dtype=torch.long)
    state = game.reset(config, envs, sizes, generator, CPU)
    slots = state.x.shape[1]
    persona = personas(ScriptedConfig(), envs, slots, generator, CPU, spread)
    farthest = torch.zeros(envs, slots)
    result = torch.full((envs,), -1, dtype=torch.long)
    with torch.no_grad():
        for _ in range(ticks):
            uniforms = draw_uniforms(envs, slots, generator, CPU)
            action = act(ScriptedConfig(), state, None, persona, uniforms)
            if chaser_team is not None:
                chased = chaser_act(state, chaser_team)
                action = torch.where((state.team == chaser_team)[..., None], chased, action)
            living = state.alive & state.exists
            distance = torch.sqrt(state.x * state.x + state.y * state.y)
            farthest = torch.maximum(farthest, torch.where(living, distance, 0.0))
            noise = game.draw_noise(config, envs, slots, generator, CPU)
            state, out = game.step(config, state, action, noise)
            result = torch.where((result < 0) & out.done, out.result, result)
            if chaser_team is None:
                fresh = game.deal(config, envs, slots, sizes, generator, CPU)
                state = game.take_deal(state, out.done, fresh)
            elif bool((result >= 0).all()):
                break
    return dict(farthest=farthest / config.arena_radius, result=result)


@pytest.mark.slow
def test_every_persona_stays_inside_the_damage_ramp():
    config = game.GameConfig()
    farthest = play(envs=16, ticks=900, seed=7, spread=1.0)["farthest"]
    assert float(farthest.max()) < 1.0 + config.boundary_ramp_width / config.arena_radius


@pytest.mark.slow
def test_opponent_beats_chaser():
    wins = losses = 0
    for chaser_team in (0, 1):
        result = play(envs=12, ticks=1800, seed=11 + chaser_team, chaser_team=chaser_team)
        wins += int((result["result"] == 1 - chaser_team).sum())
        losses += int((result["result"] == chaser_team).sum())
    assert wins > 2 * losses


def test_sandbox_flies_the_scripted_opponent():
    world = World(game.GameConfig(), seed=3)
    assert world.opponent == "scripted"
    for _ in range(5):
        world.advance(0, 0, False)
    World(game.GameConfig(), seed=3, opponent="chaser").advance(0, 0, False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the kernel runs on CUDA")
def test_the_pair_kernel_matches_its_plain_twin():
    """lean.scripted.kernels.pair_scores against opponent.pair_tables + collapse: the same
    distributions within float32 rounding (fog, fields, personas at spread 1)."""
    from lean.scripted import kernels

    state = random_state(16, 10, seed=12, fog=True, fields=6)
    state = core.perceive(state, torch.zeros_like(state.alive))
    cuda = torch.device("cuda")
    state = on_device(state, cuda)
    persona = personas(ScriptedConfig(), 16, 10, torch.Generator(cuda).manual_seed(1), cuda, 1.0)
    fused = explain(ScriptedConfig(), state, None, persona)
    kernels.USE_KERNEL = False
    try:
        plain = explain(ScriptedConfig(), state, None, persona)
    finally:
        kernels.USE_KERNEL = True
    rows = plain.controlled
    torch.testing.assert_close(fused.scores[rows], plain.scores[rows], atol=2e-4, rtol=1e-4)


def on_device(state, device):
    """A state with every tensor, its constants too, on `device`."""
    state = map_fields(state, lambda tensor: tensor.to(device))
    return dataclasses.replace(state, constants=map_fields(state.constants, lambda t: t.to(device)))
