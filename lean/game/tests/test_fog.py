"""Fog of war (docs/rewrite/core-decisions.md; docs/questions.md items 2 and 5). Each test names
the rule it guards."""

import dataclasses
import math

import torch

from lean.game import GameConfig, card, core, draw_noise, observe, reset, step
from lean.scripted import ScriptedConfig, act, chaser_act, draw_uniforms, fresh_memory, personas
from lean.scripted.opponent import engage

torch.set_num_threads(1)

FOG = GameConfig(field_count=0)  # the scenes are placed by hand, with no fields unless given
CPU = torch.device("cpu")


def duel(positions, headings=None, vision=None, fields=None):
    """One environment, one ship per entry of positions [(x, y), ...], the first half team 0,
    the rest team 1, nobody seen yet, then one perception at these positions. fields: (1, 5, F)
    as State.fields, or none."""
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
            [torch.randint(0, n, a.x.shape, generator=generator) for n in (7, 3, 2)], -1
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
    """Rule: "in sight" is v2's sight test: within the observer's vision range (1,000 m),
    all round, blocked only by a field's core (core to band sees, core to outside does not,
    a line through a core is blocked, both in one core see each other)."""
    assert (
        abs(reset(FOG, 1, torch.tensor([[1, 1]]), None, CPU).constants["vision_range"] - 1000.0)
        < 1e-3
    )
    assert core.sighted_by_enemy(duel([(0.0, 0.0), (999.0, 0.0)], headings=[math.pi, 0.0])).all()
    assert not core.sighted_by_enemy(duel([(0.0, 0.0), (1001.0, 0.0)])).any()
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


def test_own_team_always_visible_and_dead_observers_see_nothing():
    """Rules: every ship is always visible to its own team; only living ships observe."""
    state = duel(
        [(0.0, 0.0), (3000.0, 0.0), (500.0, 0.0), (-3000.0, 0.0)]
    )  # team 0: 0, 1; team 1: 2, 3
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
    state = duel([(0.0, 0.0), (90.0, 0.0)], vision=1.0)  # ship 0 noses at ship 1
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
    state = duel([(0.0, 0.0), (900.0, 0.0)], headings=[0.0, 0.0])  # in sight, flying apart
    last_x = float(state.x[0, 1])
    for tick in range(1, 4):
        # Move ship 1 out of sight by hand, then step: its row stays at the sighting.
        if tick == 1:
            state = state.replace(x=state.x + torch.tensor([[0.0, 2000.0]]))
        state, _ = step(FOG, state, idle(state), no_noise(state))
        obs = observe(FOG, state, 0)
        assert state.ticks_since_seen[0, 1] == tick
        assert float(obs.pos[0, 1, 0]) == last_x and obs.key_mask[0, 1] and not obs.visible[0, 1]
        assert math.isclose(
            float(obs.since_seen[0, 1]), math.log1p(tick * FOG.tick_length), rel_tol=1e-6
        )
        assert float(state.x[0, 1]) > last_x + 2000.0  # the true ship has moved on
    capped = dataclasses.replace(
        state, ticks_since_seen=torch.full_like(state.ticks_since_seen, core.TICKS_SINCE_SEEN_CAP)
    )
    capped = core.perceive(capped, torch.zeros_like(capped.exists))
    assert capped.ticks_since_seen[0, 1] == core.TICKS_SINCE_SEEN_CAP


def test_never_seen_enemies_are_masked_and_seen_dead_enemies_drop_out():
    """Rules: a never-seen enemy is masked (and its row is zero); a ship dead for good is masked
    once its death is seen; the enemy alive fraction counts only deaths the team has seen."""
    state = duel([(-3000.0, 0.0), (-3000.0, 300.0), (3000.0, 0.0), (3000.0, 300.0)])  # far apart
    obs = observe(FOG, state, 0)
    enemy = state.team == 1
    assert (
        not obs.key_mask[enemy].any()
        and (obs.ship[enemy] == 0).all()
        and not obs.visible[enemy].any()
    )
    assert torch.equal(obs.globals[:, 2], torch.ones(1))
    # Enemy 1 in sight dies: the sighting records the death, the row leaves the mask.
    state = duel(
        [(0.0, 0.0), (2500.0, 0.0), (-3000.0, 0.0), (3000.0, 0.0)]
    )  # team 0: 0, 1; team 1: 2, 3
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
        (3500.0, 0.0),
    ]
    state = duel(positions, headings=[0.0] * 5 + [math.pi] * 5)
    state = state.replace(x=state.x + torch.tensor([[0.0] * 7 + [2500.0, 0.0, 0.0]]))
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

    bullets = change[..., None].expand_as(state.bullet_moves_left)
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
            bullet_moves_left=torch.where(bullets, 5, state.bullet_moves_left),
            bullet_real=torch.where(bullets, True, state.bullet_real),
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
    when a hidden enemy's true state changes (the yardstick with memory and the chaser)."""
    state = hidden_enemy_scene()
    other = perturb_hidden(state, [7, 9])
    config = ScriptedConfig()
    slots = state.x.shape[1]
    persona = personas(config, 1, slots, torch.Generator().manual_seed(1), CPU, 1.0)
    uniforms = draw_uniforms(1, slots, torch.Generator().manual_seed(2), CPU)
    memory = fresh_memory(1, slots, torch.Generator().manual_seed(3), CPU)
    first, first_memory = act(config, state, 0, persona, uniforms, memory)
    second, second_memory = act(config, other, 0, persona, uniforms, memory)
    assert torch.equal(first, second)
    assert torch.equal(first_memory["target"], second_memory["target"])
    assert torch.equal(chaser_act(config, state, 0), chaser_act(config, other, 0))
    # With the other seen enemies seen dead, the hidden ship 7 at its last-seen place is every
    # ship's target, though it is truly elsewhere; never-seen ship 9 is not a target.
    seen = state.seen.clone()
    seen[0, core.SEEN_ALIVE, [5, 6, 8]] = 0.0
    lone = dataclasses.replace(other, seen=seen)
    fight = engage(core.team_view(lone, 0), 0)
    assert (fight.target[0, :5] == 7).all()
    assert torch.allclose(fight.offset_x[0, 0], seen[0, 0, 7] - lone.x[0, 0])
