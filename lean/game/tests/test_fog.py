"""Fog of war (docs/rewrite/core-decisions.md; docs/questions.md items 2 and 5). Each test names
the rule it guards."""

import dataclasses
import math

import torch

from lean.contract import FIRES, POWERS, REAL, STICKS
from lean.game import GameConfig, card, core, draw_noise, observe, reset, step
from lean.game.tests.values import game_value
from lean.scripted import ScriptedConfig, act, chaser_act, draw_uniforms, personas
from lean.scripted.opponent import nearest_enemies

torch.set_num_threads(1)

FOG = GameConfig(field_count=0)  # the scenes are placed by hand, with no fields unless given
CPU = torch.device("cpu")
# The scenes' distances are set from the game's vision and gun ranges (lean.game.tests.values).
VISION = game_value("vision_range", FOG)
GUN = game_value("gun_range", FOG)


def duel(positions, headings=None, vision=None, fields=None, eccentricity=None):
    """One environment, one ship per entry of positions [(x, y), ...], the first half team 0,
    the rest team 1, nobody seen yet, then one perception at these positions. fields: (1, 5, F)
    as State.fields, or none; eccentricity: every ship's vision eccentricity, or the card's."""
    count = len(positions)
    sizes = torch.tensor([[count // 2, count - count // 2]])
    state = reset(FOG, 1, sizes, torch.Generator().manual_seed(0), CPU)
    if fields is not None:
        state = dataclasses.replace(state, fields=fields)
    x = torch.tensor([[p[0] for p in positions]], dtype=torch.float32)
    y = torch.tensor([[p[1] for p in positions]], dtype=torch.float32)
    angle = torch.tensor([headings if headings is not None else [0.0] * count], dtype=torch.float32)
    constants = state.constants
    if vision is not None:
        constants = with_vision(constants, vision)
    if eccentricity is not None:
        constants = with_eccentricity(constants, eccentricity)
    log_index, gradient_x, gradient_y = core.medium(state.fields, x, y)
    seen = torch.zeros_like(state.seen)
    seen[:, core.SEEN_ALIVE] = 1.0
    state = state.replace(
        x=x,
        y=y,
        heading_cos=angle.cos(),
        heading_sin=angle.sin(),
        constants=constants,
        log_index=log_index,
        gradient_x=gradient_x,
        gradient_y=gradient_y,
        seen=seen,
        ticks_since_seen=torch.zeros_like(state.ticks_since_seen),
    )
    return core.perceive(core.with_velocity(state), torch.zeros_like(state.exists))


def with_vision(constants, vision):
    """The constants with every ship's vision range set to `vision` metres."""
    return dataclasses.replace(constants, uniform={**constants.uniform, "vision_range": vision})


def with_eccentricity(constants, eccentricity):
    """The constants with every ship's vision eccentricity set to `eccentricity`."""
    uniform = {**constants.uniform, "vision_eccentricity": eccentricity}
    return dataclasses.replace(constants, uniform=uniform)


def idle(state, fire=None):
    """(E, N, 3) actions: straight, coast, and fire where `fire` (E, N) bool."""
    action = torch.zeros(state.x.shape + (3,), dtype=torch.long)
    if fire is not None:
        action[..., 2] = fire.long()
    return action


def no_noise(state):
    return {"spread": torch.zeros(state.x.shape + (2,))}


def test_fog_off_is_full_information_and_leaves_the_game_unchanged():
    """Rule: fog off keeps today's full information. The game itself does not depend on fog
    (same states, same events), fog off carries no fog fields, and with fog on and every enemy in
    sight the observation equals the fog-off one (while both teams have a living ship: a wiped
    team has no eyes, so the survivors' rows in its view freeze)."""
    generator = torch.Generator().manual_seed(3)
    sizes = torch.tensor([[3, 3]] * 4)
    plain, fogged = (
        GameConfig(arena_radius=700.0, spawn="mixed", field_count=0, fog=False),
        GameConfig(arena_radius=700.0, spawn="mixed", field_count=0),
    )
    a = reset(plain, 4, sizes, torch.Generator().manual_seed(1), CPU)
    b = reset(fogged, 4, sizes, torch.Generator().manual_seed(1), CPU)
    assert a.seen is None and a.ticks_since_seen is None and b.seen is not None
    b = core.perceive(
        dataclasses.replace(b, constants=with_vision(b.constants, 1e5)), torch.zeros_like(b.exists)
    )
    compared = 0
    for _ in range(40):
        action = torch.stack(
            [torch.randint(0, n, a.x.shape, generator=generator) for n in (STICKS, POWERS, FIRES)],
            -1,
        )
        noise = draw_noise(plain, 4, a.x.shape[1], generator, CPU)
        a, out_a = step(plain, a, action, noise)
        b, out_b = step(fogged, b, action, noise)
        for field in dataclasses.fields(a):
            value = getattr(a, field.name)
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, getattr(b, field.name)), field.name
        assert torch.equal(out_a.damage_taken, out_b.damage_taken) and torch.equal(
            out_a.died, out_b.died
        )
        both = (a.alive & (a.team == 0)).any(-1) & (a.alive & (a.team == 1)).any(-1)
        compared += int(both.sum())
        for team in (0, 1):
            full, seen = observe(plain, a, team), observe(fogged, b, team)
            assert full.visible.all() and (full.since_seen == 0).all()
            for name in ("key_mask", "own"):
                assert torch.equal(getattr(full, name)[both], getattr(seen, name)[both]), name
            # Within rounding: the fogged view reads its values back from a packed record, and
            # elementwise kernels may round a strided slice differently (atan2 by one ulp).
            for name in ("ship", "pos", "heading", "globals"):
                assert torch.allclose(
                    getattr(full, name)[both], getattr(seen, name)[both], rtol=1e-6, atol=1e-6
                ), name
    assert compared > 100


def test_sight_range_and_field_cores():
    """Rule: "in sight" is v2's sight test: within the observer's vision range (the card's),
    all round, blocked only by a field's core (core to band sees, core to outside does not,
    a line through a core is blocked, both in one core see each other)."""
    constants = reset(FOG, 1, torch.tensor([[1, 1]]), None, CPU).constants
    vision = float(torch.as_tensor(constants["vision_range"]).flatten()[0])
    assert abs(vision - card.VISION_RANGE * card.CARD["vision_range_scale"][0]) < 1e-3
    near, far = vision - 1.0, vision + 1.0
    assert core.sighted_by_enemy(duel([(0.0, 0.0), (near, 0.0)], headings=[math.pi, 0.0])).all()
    assert not core.sighted_by_enemy(duel([(0.0, 0.0), (far, 0.0)])).any()
    # One field at the centre: radius 300, band 160, so core 220 and outer disc 380.
    field = torch.tensor([[[0.0], [0.0], [300.0], [160.0], [2.0 * card.LN_SQRT2]]])
    cases = [
        ([(-450.0, 0.0), (450.0, 0.0)], False),  # the line crosses the core
        ([(-450.0, 400.0), (450.0, 400.0)], True),  # passes outside the core
        ([(-450.0, 230.0), (450.0, 230.0)], True),  # passes 230 m from the centre: in the band only
        ([(0.0, 0.0), (500.0, 0.0)], False),  # core to outside
        ([(0.0, 0.0), (300.0, 0.0)], True),  # core to band
        ([(0.0, 0.0), (100.0, 50.0)], True),  # both in the core
        ([(250.0, 0.0), (-250.0, 0.0)], False),  # band to band across the core
    ]
    for positions, expected in cases:
        seen = core.sighted_by_enemy(duel(positions, fields=field))
        assert bool(seen.all()) == expected and bool(seen.any()) == expected, positions


def test_sight_ellipse_at_zero_eccentricity_is_the_circle():
    """Rule: with vision eccentricity 0 the sight ellipse is exactly the old circle d^2 <= p^2
    (p the card's vision range), on random scenes of 3v3 with random headings."""
    generator = torch.Generator().manual_seed(5)
    for _ in range(20):
        positions = (torch.rand(6, 2, generator=generator) * 2.0 - 1.0) * 1.5 * VISION
        headings = (torch.rand(6, generator=generator) * 2.0 * math.pi).tolist()
        state = duel(positions.tolist(), headings=headings, eccentricity=0.0)
        x, y = state.x[0], state.y[0]
        squared = (x[:, None] - x[None]) ** 2 + (y[:, None] - y[None]) ** 2  # (target, observer)
        enemy = state.team[0][:, None] != state.team[0][None]
        expected = ((squared <= VISION * VISION) & enemy).any(-1)
        assert torch.equal(core.sighted_by_enemy(state)[0], expected)


def test_sight_ellipse_reach_ahead_beside_and_behind():
    """Rule: the observer at one focus of an ellipse along its nose, |d| - e (d . nose) <= p:
    reach p / (1 - e) dead ahead, p to the side, p / (1 + e) behind (p the card's vision
    range), in any heading. The target faces away, so only the observer's nose matters."""
    eccentricity = 0.5
    reaches = {
        0.0: VISION / (1.0 - eccentricity),
        0.5 * math.pi: VISION,
        -0.5 * math.pi: VISION,
        math.pi: VISION / (1.0 + eccentricity),
    }
    for heading in (0.0, 1.0, -2.5):
        for bearing, reach in reaches.items():
            for distance, expected in ((reach - 1.0, True), (reach + 1.0, False)):
                angle = heading + bearing
                target = (distance * math.cos(angle), distance * math.sin(angle))
                state = duel(
                    [(0.0, 0.0), target], headings=[heading, angle], eccentricity=eccentricity
                )
                assert bool(core.sighted_by_enemy(state)[0, 1]) == expected, (heading, bearing)


def test_sight_ellipse_keeps_field_cores_and_team_sharing():
    """Rules: the ellipse changes only the reach: a field's core still blocks a line well
    inside it, and a team sees what any of its ships sees (a wingman facing away does not see
    an enemy that its leader, nosed at it, does)."""
    eccentricity = 0.5
    ahead = 0.9 * VISION / (1.0 - eccentricity)  # beyond the circle, inside the ellipse ahead
    assert ahead > VISION
    open_line = duel([(0.0, 0.0), (ahead, 0.0)], headings=[0.0, 0.0], eccentricity=eccentricity)
    assert bool(core.sighted_by_enemy(open_line)[0, 1])
    field = torch.tensor([[[0.5 * ahead], [0.0], [300.0], [160.0], [2.0 * card.LN_SQRT2]]])
    blocked = duel(
        [(0.0, 0.0), (ahead, 0.0)], headings=[0.0, 0.0], fields=field, eccentricity=eccentricity
    )
    assert not bool(core.sighted_by_enemy(blocked)[0, 1])
    # Team 0: the leader at the origin nosed at the enemy, the wingman beside it facing away.
    # Team 1: the enemy ahead, and a far ship nobody sees.
    positions = [(0.0, 0.0), (0.0, 100.0), (ahead, 0.0), (-3.0 * VISION, 0.0)]
    for leader_heading, seen in ((0.0, True), (math.pi, False)):
        state = duel(
            positions, headings=[leader_heading, math.pi, 0.0, 0.0], eccentricity=eccentricity
        )
        assert core.sighted_by_enemy(state)[0, 2:].tolist() == [seen, False]


def test_own_team_always_visible_and_dead_observers_see_nothing():
    """Rules: every ship is always visible to its own team; only living ships observe."""
    near, far = VISION / 3.0, 2.0 * VISION
    state = duel([(0.0, 0.0), (far, 0.0), (near, 0.0), (-far, 0.0)])  # team 0: 0, 1; team 1: 2, 3
    for team in (0, 1):
        obs = observe(FOG, state, team)
        own = state.team[0] == team
        assert (
            obs.visible[0, own].all()
            and obs.key_mask[0, own].all()
            and (obs.since_seen[0, own] == 0).all()
        )
    assert core.sighted_by_enemy(state)[0].tolist() == [True, False, True, False]
    dead = state.replace(alive=torch.tensor([[False, True, True, True]]))
    assert core.sighted_by_enemy(dead)[0].tolist() == [True, False, False, False]


def test_firing_and_hits_reveal_for_that_tick_only_and_boundary_damage_does_not():
    """Rules: a ship is revealed on the tick it fires and on the tick a bullet hits it, for that
    tick only; boundary damage does not reveal. Vision is cut to 1 m so sight reveals nothing."""
    state = duel([(0.0, 0.0), (GUN / 8.0, 0.0)], vision=1.0)  # ship 0 noses at ship 1, in range
    assert (state.seen[:, core.SEEN_KNOWN] == 0).all()
    shoot = torch.tensor([[True, False]])
    state = state.replace(pending=idle(state, shoot))  # the trigger pulled on the tick before
    state, out = step(FOG, state, idle(state), no_noise(state))
    assert state.fired[0].tolist() == [True, False]
    assert state.ticks_since_seen[0, 0] == 0 and state.seen[0, core.SEEN_KNOWN, 0] == 1.0
    hit_tick = None
    for tick in range(1, 6):
        state, out = step(FOG, state, idle(state), no_noise(state))
        assert not state.fired.any()
        assert state.ticks_since_seen[0, 0] == tick  # the shooter's reveal lasted one tick
        if out.damage_taken[0, 1] > 0.0 and hit_tick is None:
            hit_tick = tick
            assert state.ticks_since_seen[0, 1] == 0 and state.seen[0, core.SEEN_KNOWN, 1] == 1.0
        elif hit_tick is not None:
            assert state.ticks_since_seen[0, 1] == tick - hit_tick
    assert hit_tick is not None
    # Boundary damage: ship 1 beyond the edge takes damage and stays hidden.
    state = duel([(0.0, 0.0), (FOG.arena_radius + 200.0, 0.0)], vision=1.0)
    for _ in range(3):
        state, out = step(FOG, state, idle(state), no_noise(state))
        assert out.damage_taken[0, 1] > 0.0
        assert state.seen[0, core.SEEN_KNOWN, 1] == 0.0 and state.ticks_since_seen[0, 1] > 0


def test_last_seen_values_freeze_and_time_since_seen_counts_and_caps():
    """Rules: a hidden enemy that has been seen is its last-seen row; the ticks since seen count
    up from the sighting and stop at v2's cap; since_seen is symlog of the seconds."""
    away = VISION  # moved this far, ship 1 is out of sight
    state = duel([(0.0, 0.0), (0.6 * VISION, 0.0)], headings=[0.0, 0.0])  # in sight, flying apart
    last_x = float(state.x[0, 1])
    for tick in range(1, 4):
        # Move ship 1 out of sight by hand, then step: its row stays at the sighting.
        if tick == 1:
            state = state.replace(x=state.x + torch.tensor([[0.0, away]]))
        state, _ = step(FOG, state, idle(state), no_noise(state))
        obs = observe(FOG, state, 0)
        assert state.ticks_since_seen[0, 1] == tick
        assert float(obs.pos[0, 1, 0]) == last_x and obs.key_mask[0, 1] and not obs.visible[0, 1]
        assert math.isclose(
            float(obs.since_seen[0, 1]), math.log1p(tick * FOG.tick_length), rel_tol=1e-6
        )
        assert float(state.x[0, 1]) > last_x + away  # the true ship has moved on
    capped = dataclasses.replace(
        state, ticks_since_seen=torch.full_like(state.ticks_since_seen, core.TICKS_SINCE_SEEN_CAP)
    )
    capped = core.perceive(capped, torch.zeros_like(capped.exists))
    assert capped.ticks_since_seen[0, 1] == core.TICKS_SINCE_SEEN_CAP


def test_never_seen_enemies_are_masked_and_seen_dead_enemies_drop_out():
    """Rules: a never-seen enemy is masked (and its row is zero); a ship dead for good is masked
    once its death is seen; the enemy alive fraction counts only deaths the team has seen."""
    state = duel([(-VISION, 0.0), (-VISION, 300.0), (VISION, 0.0), (VISION, 300.0)])  # far apart
    obs = observe(FOG, state, 0)
    enemy = state.team == 1
    assert (
        not obs.key_mask[enemy].any()
        and (obs.ship[enemy] == 0).all()
        and not obs.visible[enemy].any()
    )
    assert torch.equal(obs.globals[:, 2], torch.ones(1))
    # Enemy 1 in sight dies: the sighting records the death, the row leaves the mask.
    far = 2.0 * VISION  # ship 1 is a third of the vision range from ship 3, ship 0 out of sight
    state = duel([(0.0, 0.0), (far - VISION / 3.0, 0.0), (-far, 0.0), (far, 0.0)])
    assert state.seen[0, core.SEEN_KNOWN, :2].tolist() == [0.0, 1.0]
    state = state.replace(alive=torch.tensor([[True, False, True, True]]))
    state = core.perceive(state, torch.zeros_like(state.exists))
    view = observe(FOG, state, 1)
    assert not view.key_mask[0, 1] and float(view.globals[0, 2]) == 0.5
    # Enemy 0 of team 1's view dies out of sight: team 1 does not know.
    hidden_death = state.replace(alive=torch.tensor([[False, False, True, True]]))
    assert torch.equal(observe(FOG, hidden_death, 1).globals, view.globals)


def hidden_enemy_scene():
    """5v5 with team 1's ship 7 seen once then hidden, ship 9 never seen, and the others in
    sight; returns the state after a few ticks."""
    positions = [
        (0.0, 0.0),
        (100.0, 300.0),
        (-200.0, 100.0),
        (50.0, -250.0),
        (-100.0, -100.0),
        (700.0, 0.0),
        (650.0, 300.0),
        (900.0, -200.0),
        (600.0, -400.0),
        (VISION + 2000.0, 0.0),  # never in sight
    ]
    state = duel(positions, headings=[0.0] * 5 + [math.pi] * 5)
    away = VISION + 1000.0  # ship 7, seen at 900 m, moved out of sight
    state = state.replace(x=state.x + torch.tensor([[0.0] * 7 + [away, 0.0, 0.0]]))
    for _ in range(3):
        state, _ = step(FOG, state, idle(state), no_noise(state))
    assert state.seen[0, core.SEEN_KNOWN, 7] == 1.0 and state.ticks_since_seen[0, 7] == 3
    assert state.seen[0, core.SEEN_KNOWN, 9] == 0.0
    return state


def perturb_hidden(state, slots):
    """The true state of the given hidden slots changed everywhere it can be (not the record of
    what the enemy saw): motion, store, shield, gun, life and bullets."""
    change = torch.zeros_like(state.exists)
    change[:, slots] = True

    def bump(value, by):
        return torch.where(change, value + by, value)

    bullets = change[..., None].expand_as(state.bullet_state)
    return core.with_velocity(
        state.replace(
            x=bump(state.x, 123.0),
            y=bump(state.y, -77.0),
            speed=bump(state.speed, 40.0),
            heading_cos=torch.where(change, -state.heading_cos, state.heading_cos),
            heading_sin=torch.where(change, -state.heading_sin, state.heading_sin),
            nose_rate=bump(state.nose_rate, 0.3),
            energy=bump(state.energy, -1000.0),
            shield=bump(state.shield, -20.0),
            shield_wait=bump(state.shield_wait, 1.0),
            cooldown=torch.where(change, 1.0, state.cooldown),
            alive=torch.where(change, False, state.alive),
            log_index=torch.where(change, 0.1, state.log_index),
            gradient_x=torch.where(change, 0.1, state.gradient_x),
            gradient_y=torch.where(change, 0.1, state.gradient_y),
            bullet_state=torch.where(bullets, REAL, state.bullet_state),
        )
    )


def test_no_hidden_information_reaches_the_observation():
    """Rule: nothing in a team's observation depends on what it cannot see: changing a hidden
    enemy's true state (even killing it) leaves every field of that team's observation as it was."""
    state = hidden_enemy_scene()
    nothing = torch.zeros_like(state.exists)
    # One more perception (as the end of a step makes) of the true and the changed state.
    state, other = (
        core.perceive(state, nothing),
        core.perceive(perturb_hidden(state, [7, 9]), nothing),
    )
    assert torch.equal(other.seen[..., 5:], state.seen[..., 5:])
    before, after = observe(FOG, state, 0), observe(FOG, other, 0)
    for field in dataclasses.fields(before):
        assert torch.equal(getattr(before, field.name), getattr(after, field.name)), field.name
    # The test can fail: the same change to a ship in sight does change the observation.
    assert not torch.equal(
        observe(FOG, core.perceive(perturb_hidden(state, [5]), nothing), 0).ship, before.ship
    )


def test_scripted_opponent_does_not_cheat():
    """Rule: under fog the scripted opponent reads its team's view: its action does not change
    when a hidden enemy's true state changes (the yardstick at spread 1 and the chaser)."""
    state = hidden_enemy_scene()
    other = perturb_hidden(state, [7, 9])
    config = ScriptedConfig()
    slots = state.x.shape[1]
    persona = personas(config, 1, slots, torch.Generator().manual_seed(1), CPU, 1.0)
    uniforms = draw_uniforms(1, slots, torch.Generator().manual_seed(2), CPU)
    first = act(config, state, 0, persona, uniforms)
    second = act(config, other, 0, persona, uniforms)
    assert torch.equal(first, second)
    assert torch.equal(chaser_act(state, 0), chaser_act(other, 0))
    # With the other seen enemies seen dead, the hidden ship 7 is every ship's nearest scored
    # enemy, though it is truly elsewhere; never-seen ship 9 is not scored.
    seen = state.seen.clone()
    seen[0, core.SEEN_ALIVE, [5, 6, 8]] = 0.0
    lone = dataclasses.replace(other, seen=seen)
    view = core.team_view(lone, 0)
    slot, weight, present = nearest_enemies(view, core.enemy_view(view), 1.0)
    assert (slot[0, :5, 0] == 7).all() and present[0, :5, 0].all()
    assert not present[0, :5, 1:].any()


# --------------------------------------------------------------------------- the tracking circle

TRACK_ECCENTRICITY = 0.5  # an ellipse whose circle differs from it (the card's default is 0)
UNTRACKED = GameConfig(field_count=0, fog_tracking=False)


def tracking_scene():
    """Team 0's ship at the origin nosed east; team 1's ship ahead, inside the sight ellipse
    (seen at once), flying away north-east. Returns the state and the side spot: due north of
    the observer, beyond the ellipse's side reach p but inside the tracking circle p / (1 - e)."""
    circle = VISION / (1.0 - TRACK_ECCENTRICITY)
    ahead = 0.9 * circle
    state = duel([(0.0, 0.0), (ahead, 0.0)], headings=[0.0, 0.0], eccentricity=TRACK_ECCENTRICITY)
    assert state.ticks_since_seen[0, 1] == 0 and state.seen[0, core.SEEN_KNOWN, 1] == 1.0
    return state, 0.5 * (VISION + circle)


def move_target(state, x, y):
    """Ship 1 placed by hand at (x, y), its medium read at the new place."""
    new_x = torch.where(torch.tensor([[False, True]]), x, state.x)
    new_y = torch.where(torch.tensor([[False, True]]), y, state.y)
    log_index, gradient_x, gradient_y = core.medium(state.fields, new_x, new_y)
    return state.replace(
        x=new_x, y=new_y, log_index=log_index, gradient_x=gradient_x, gradient_y=gradient_y
    )


def test_tracking_holds_a_seen_ship_inside_the_circle_and_not_a_never_seen_one():
    """Rules: a ship visible on the tick before stays visible while an observer has it inside the
    tracking circle (the ellipse's forward reach p / (1 - e) all round), outside the ellipse;
    the off switch gives the ellipse alone; a ship never seen is not tracked there."""
    state, side = tracking_scene()
    assert side > VISION  # outside the ellipse beside the observer
    moved = move_target(state, 0.0, side)
    assert not bool(core.sighted_by_enemy(moved)[0, 1])  # the ellipse alone does not see it
    for config, held in ((FOG, True), (UNTRACKED, False)):
        after = moved
        for tick in range(1, 4):
            after, _ = step(config, after, idle(after), no_noise(after))
            assert bool(after.ticks_since_seen[0, 1] == 0) == held, (config.fog_tracking, tick)
    # Never seen: the same spot in a fresh scene (nothing known) is hidden.
    fresh = duel([(0.0, 0.0), (0.0, side)], headings=[0.0, 0.0], eccentricity=TRACK_ECCENTRICITY)
    assert fresh.seen[0, core.SEEN_KNOWN, 1] == 0.0 and fresh.ticks_since_seen[0, 1] == 1
    fresh, _ = step(FOG, fresh, idle(fresh), no_noise(fresh))
    assert fresh.seen[0, core.SEEN_KNOWN, 1] == 0.0 and fresh.ticks_since_seen[0, 1] == 2
    # Beyond the circle: lost even while tracked.
    circle = VISION / (1.0 - TRACK_ECCENTRICITY)
    beyond = move_target(state, 0.0, circle + 50.0)
    beyond, _ = step(FOG, beyond, idle(beyond), no_noise(beyond))
    assert beyond.ticks_since_seen[0, 1] == 1


def test_a_field_core_breaks_tracking_and_then_the_ellipse_is_needed_again():
    """Rules: tracking needs a clear line (a field's core blocks it as it blocks sight); once
    the ship is hidden it is not tracked, so inside the circle but outside the ellipse it stays
    hidden after the core is gone, until it enters the ellipse again."""
    state, side = tracking_scene()
    state = move_target(state, 0.0, side)
    state, _ = step(FOG, state, idle(state), no_noise(state))
    assert state.ticks_since_seen[0, 1] == 0  # tracked
    # A field whose core sits on the line, well away from both ships.
    field = torch.tensor([[[0.0], [0.5 * side], [300.0], [160.0], [2.0 * card.LN_SQRT2]]])
    blocked = state.replace(fields=field)
    blocked, _ = step(FOG, blocked, idle(blocked), no_noise(blocked))
    assert blocked.ticks_since_seen[0, 1] == 1  # lost
    clear = blocked.replace(fields=state.fields)
    for tick in range(2, 5):
        clear, _ = step(FOG, clear, idle(clear), no_noise(clear))
        assert clear.ticks_since_seen[0, 1] == tick  # inside the circle, but no longer tracked
    # Into the ellipse ahead: seen again, then tracked again beside.
    ahead = move_target(clear, 0.9 * VISION / (1.0 - TRACK_ECCENTRICITY), 0.0)
    ahead, _ = step(FOG, ahead, idle(ahead), no_noise(ahead))
    assert ahead.ticks_since_seen[0, 1] == 0
    beside = move_target(ahead, 0.0, side)
    beside, _ = step(FOG, beside, idle(beside), no_noise(beside))
    assert beside.ticks_since_seen[0, 1] == 0


def test_a_shot_reveal_starts_tracking():
    """Rule: every way of becoming visible counts: a ship revealed by its shot, outside the
    observer's ellipse but inside its circle, is tracked on the next tick (with tracking off it
    is hidden again)."""
    circle = VISION / (1.0 - TRACK_ECCENTRICITY)
    side = 0.5 * (VISION + circle)
    for config, held in ((FOG, True), (UNTRACKED, False)):
        state = duel(
            [(0.0, 0.0), (0.0, side)], headings=[0.0, 0.0], eccentricity=TRACK_ECCENTRICITY
        )
        assert state.seen[0, core.SEEN_KNOWN, 1] == 0.0  # hidden, never seen
        state = state.replace(pending=idle(state, torch.tensor([[False, True]])))
        state, _ = step(config, state, idle(state), no_noise(state))
        assert state.fired[0, 1] and state.ticks_since_seen[0, 1] == 0  # revealed by the shot
        state, _ = step(config, state, idle(state), no_noise(state))
        assert not state.fired[0, 1]
        assert bool(state.ticks_since_seen[0, 1] == 0) == held, config.fog_tracking


def random_game_visibility(config, eccentricity, ticks=60):
    """ticks_since_seen (T, E, N) over a game of random actions, 3v3 with fields, every ship's
    vision eccentricity set to `eccentricity`."""
    sizes = torch.tensor([[3, 3]] * 8)
    state = reset(config, 8, sizes, torch.Generator().manual_seed(11), CPU)
    state = core.perceive(
        dataclasses.replace(state, constants=with_eccentricity(state.constants, eccentricity)),
        torch.zeros_like(state.exists),
        config.fog_tracking,
    )
    generator = torch.Generator().manual_seed(12)
    history = []
    for _ in range(ticks):
        action = torch.stack(
            [
                torch.randint(0, n, state.x.shape, generator=generator)
                for n in (STICKS, POWERS, FIRES)
            ],
            -1,
        )
        state, _ = step(config, state, action, draw_noise(config, 8, 6, generator, CPU))
        history.append(state.ticks_since_seen)
    return torch.stack(history)


def test_tracking_at_zero_eccentricity_changes_nothing():
    """Rule: at e = 0 the tracking circle is the ellipse: the sight pairs with every target
    tracked equal the untracked ones on random scenes, and a game's visibility is the same with
    tracking on and off. The comparison can fail: at e = 0.5 the two differ."""
    generator = torch.Generator().manual_seed(7)
    for _ in range(20):
        positions = (torch.rand(6, 2, generator=generator) * 2.0 - 1.0) * 1.5 * VISION
        headings = (torch.rand(6, generator=generator) * 2.0 * math.pi).tolist()
        state = duel(positions.tolist(), headings=headings, eccentricity=0.0)
        everyone = torch.ones_like(state.exists)
        assert torch.equal(core.sight_pairs(state, everyone), core.sight_pairs(state))
    arena = dict(arena_radius=3.0 * VISION, spawn="mixed")
    on, off = GameConfig(**arena), GameConfig(**arena, fog_tracking=False)
    assert torch.equal(random_game_visibility(on, 0.0), random_game_visibility(off, 0.0))
    assert not torch.equal(
        random_game_visibility(on, TRACK_ECCENTRICITY),
        random_game_visibility(off, TRACK_ECCENTRICITY),
    )


def test_tracking_off_is_the_ellipse_and_tracked_means_visible_the_tick_before():
    """Rules: a ship is tracked exactly when it was visible on the tick before (ticks since seen
    0 and known): every ship so marked is held by the circle; the deal's rows (nothing known,
    ticks since seen 0) are not tracked, so the deal sees by the ellipse alone; with tracking off
    perception is the ellipse whatever was seen before."""
    config = GameConfig(spawn="mixed")
    state = reset(config, 8, torch.tensor([[4, 4]] * 8), torch.Generator().manual_seed(2), CPU)
    state = state.replace(constants=with_eccentricity(state.constants, TRACK_ECCENTRICITY))
    nothing = torch.zeros_like(state.exists)
    ellipse = core.sighted_by_enemy(state) & state.exists
    circle = core.sighted_by_enemy(state, torch.ones_like(state.exists)) & state.exists
    assert not torch.equal(ellipse, circle)  # the scene tells the two apart
    # As the deal leaves it before its perception: nothing known, ticks since seen 0.
    seen = torch.zeros_like(state.seen)
    seen[:, core.SEEN_ALIVE] = state.exists.to(torch.float32)
    since = torch.zeros_like(state.ticks_since_seen)
    at_deal = state.replace(seen=seen, ticks_since_seen=since)
    assert not core.tracked_by_enemy(at_deal).any()
    assert torch.equal(core.perceive(at_deal, nothing).ticks_since_seen == 0, ellipse)
    # Everyone seen on the tick before.
    seen[:, core.SEEN_KNOWN] = 1.0
    everyone = state.replace(seen=seen, ticks_since_seen=since)
    assert core.tracked_by_enemy(everyone)[state.exists].all()
    assert torch.equal(core.perceive(everyone, nothing).ticks_since_seen == 0, circle)
    off = core.perceive(everyone, nothing, tracking=False)
    assert torch.equal(off.ticks_since_seen == 0, ellipse)
