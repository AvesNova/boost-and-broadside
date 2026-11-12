"""Elimination and fixed map layouts: the win and draw rules (a wipe; at the clock ships
alive, then total shield, then a draw), no respawn, Clover's fixed fields (four level
combinations, diagonal partners equal) and its point-symmetric spawns, the mode's
observation, a saved map's layout, and the scripted agent's set for the mode."""

import dataclasses
import math

import modes_cases as cases
import pytest
import torch

from bnb.contracts import GameConfig
from bnb.contracts.config import FixedField, FixedSpawn, MapLayout
from bnb.contracts.obs import GLOBAL
from bnb.controllers.scripted_defaults import MODE_DEFAULTS, read_real_scale
from bnb.controllers.scripted_parameters import MODES as SCRIPTED_MODES
from bnb.controllers.scripted_parameters import pack
from bnb.game import api
from bnb.game.modes import phase, spawns
from bnb.game.modes.rules import CLOVER_FIELD_DISTANCE, CLOVER_SPAWN_DISTANCE, PRESETS
from bnb.game.state import by_team
from bnb.game.tuning import tuned
from bnb.play.maps import Map, Maps

ELIMINATION = GameConfig(3, 3, 900.0, episode_seconds=6.0, mode="elimination")
CLOVER = PRESETS["clover"]
REAL_SCALE = api.ship_types(ship_set="real-scale")


def _last_tick(rules, state):
    """The state one tick before its clock ends the match."""
    episode = rules.pools[0].episode_ticks
    match = dataclasses.replace(
        state.match, episode_tick=torch.full_like(state.match.episode_tick, episode - 1)
    )
    return dataclasses.replace(state, match=match)


def _ended(alive: list[list[bool]], shield: list[list[float]], clock: bool):
    """The result and done flags of one still tick from these ships (one environment per
    row), at the clock's last tick or mid-match."""
    rules, state = cases.rules_and_state(ELIMINATION, batch=len(alive))
    state = cases.fields_away(state)
    state = cases.edit_ships(
        state, alive=torch.tensor(alive), shield=torch.tensor(shield, dtype=torch.float32)
    )
    if clock:
        state = _last_tick(rules, state)
    _, events = cases.tick(rules, state)
    return events.result.tolist(), events.done.tolist()


def test_a_wipe_ends_the_match_for_the_team_with_ships_left():
    """The mode phase: the last enemy ship's death wins (whatever killed it), as does the
    enemy's last active ship's when benched; one ship left on each side plays on."""
    rules = api.rules_of(ELIMINATION)
    mode_rules = rules.pools[0].mode
    state = cases.rules_and_state(ELIMINATION, batch=3)[1]
    score = torch.tensor([[2, 0], [0, 2], [1, 1]], dtype=torch.int32)
    died = torch.zeros((3, 6), dtype=torch.bool)
    died[0, 5] = died[1, 2] = died[2, 4] = True  # a team 1, team 0, team 1 ship dies
    alive = state.ships.alive & ~died
    mode = dataclasses.replace(state.mode, score=score)
    result = phase.mode_phase(rules, mode_rules, mode, state.ships, alive, died)
    assert result.won.tolist() == [True, True, False]
    assert result.result.tolist() == [1, -1, 0]
    active = torch.tensor([[3, 2], [3, 3], [3, 3]], dtype=torch.int32)
    benched = phase.mode_phase(rules, mode_rules, mode, state.ships, alive, died, active)
    assert benched.won.tolist() == [True, True, False]  # team 1's two active ships are gone


def test_played_out_a_match_won_by_its_rule_leaves_the_loser_with_no_ship():
    """Random play rarely wipes a team that spawns across the map: both teams' areas sit at
    the centre here (a spawn offset of 0), so the fights come quickly."""
    won = 0
    for before, events, _ in cases.play(ELIMINATION, batch=16, ticks=400, seed=4, spawn_offset=0.0):
        living = by_team(before.ships.alive & ~events.died).sum(-1)  # (B, 2) after the tick
        by_rule = events.done & (living == 0).any(-1)
        loser = torch.where(events.result > 0, 1, 0)
        assert (living.gather(1, loser[:, None])[by_rule & (events.result != 0)] == 0).all()
        won += int((by_rule & (events.result != 0)).sum())
    assert won > 0


def test_at_the_clock_more_ships_alive_wins_then_more_shield_then_a_draw():
    alive = [
        [True, True, False, True, False, False],  # 2 against 1
        [True, False, False, True, False, False],  # 1 against 1, team 1 healthier
        [True, True, False, True, True, False],  # 2 against 2, equal shields: a draw
        [True, True, False, True, True, False],  # 2 against 2, team 0 healthier
    ]
    shield = [
        [1.0, 1.0, 9.0, 30.0, 9.0, 9.0],  # the dead's shields do not count
        [5.0, 9.0, 9.0, 6.0, 0.0, 0.0],
        [5.0, 7.0, 0.0, 6.0, 6.0, 0.0],
        [5.0, 7.5, 0.0, 6.0, 6.0, 0.0],
    ]
    result, done = _ended(alive, shield, clock=True)
    assert done == [True] * 4
    assert result == [1, -1, 0, 1]


def test_a_dead_ship_never_respawns_within_the_match():
    """Lively random play: a ship spawns only with its environment's new match, and once
    dead it stays dead until then."""
    deaths = resets = 0
    for before, events, after in cases.play(ELIMINATION, batch=16, ticks=400, seed=2):
        assert not (after.ships.spawned & ~events.done[:, None]).any()
        stays = ~before.ships.alive & ~events.done[:, None]
        assert not after.ships.alive[stays].any()
        deaths += int(events.died.sum())
        resets += int(events.done.sum())
    assert deaths > 0 and resets > 0


def test_the_respawn_wait_is_the_match_length_even_when_tuned():
    rules = api.rules_of(ELIMINATION)
    (pool,) = rules.pools
    assert pool.respawn_ticks == pool.episode_ticks == 180
    assert pool.mode.kills_per_ship == 1
    (longer,) = tuned(rules, {"pool.episode_seconds": 12.0, "pool.respawn_seconds": 1.0}).pools
    assert longer.respawn_ticks == longer.episode_ticks == 360
    with pytest.raises(ValueError, match="no respawn"):
        dataclasses.replace(ELIMINATION, respawn_seconds=1.0)


def test_the_observation_reads_the_mode_and_the_wipe_target():
    rules, state = cases.rules_and_state(ELIMINATION, batch=2)
    _, obs, _ = api.perceive(rules, state)
    globals_ = obs.globals
    assert (globals_[..., GLOBAL["mode_elimination"]] == 1).all()
    assert (globals_[..., GLOBAL["mode_team_deathmatch"]] == 0).all()
    assert (globals_[..., GLOBAL["score_target"]] == 3).all()  # the enemy's ships
    deathmatch = api.rules_of(cases.MODES["team_deathmatch"])
    generator = torch.Generator().manual_seed(1)
    state = api.reset(deathmatch, cases.TYPES, api.uniforms(deathmatch, 2, generator=generator))
    _, obs, _ = api.perceive(deathmatch, state)
    assert (obs.globals[..., GLOBAL["mode_elimination"]] == 0).all()


# ------------------------------------------------------------------ Clover's layout


def _clover(batch: int, seed: int = 0, config: GameConfig = CLOVER):
    rules = api.rules_of(config)
    generator = torch.Generator().manual_seed(seed)
    return rules, api.reset(rules, REAL_SCALE, api.uniforms(rules, batch, generator=generator))


def test_clover_fields_are_fixed_with_four_level_combinations_and_equal_partners():
    rules, state = _clover(256)
    fields = state.fields
    distance = torch.hypot(fields.x, fields.y)
    torch.testing.assert_close(distance, torch.full_like(distance, CLOVER_FIELD_DISTANCE))
    torch.testing.assert_close(fields.radius, torch.full_like(fields.radius, 0.25 * 4933.2))
    # fields 2 and 3 are fields 0 and 1 turned half a turn, with the same level
    torch.testing.assert_close(fields.x[:, 2:], -fields.x[:, :2])
    torch.testing.assert_close(fields.y[:, 2:], -fields.y[:, :2])
    assert torch.equal(fields.level[:, 2:], fields.level[:, :2])
    # field 1 is field 0 turned a quarter turn: a square around the centre
    torch.testing.assert_close(fields.x[:, 1], -fields.y[:, 0], atol=1e-2, rtol=0)
    torch.testing.assert_close(fields.y[:, 1], fields.x[:, 0], atol=1e-2, rtol=0)
    high, low = fields.level[:, 0], fields.level[:, 1]
    assert set(high.tolist()) == {1, 2} and set(low.tolist()) == {-1, -2}
    assert len(set(zip(high.tolist(), low.tolist(), strict=True))) == 4
    # the layout turns by a random angle each match
    bearing = torch.atan2(fields.y[:, 0], fields.x[:, 0])
    assert float(bearing.max() - bearing.min()) > 5.0


def test_clover_spawns_are_point_symmetric_by_the_edge_facing_the_centre():
    rules, state = _clover(256, seed=3)
    ships = state.ships
    x, y = ships.x, ships.y
    # team 1's slot s against team 0's slot s turned half a turn: within both jitters
    gap = torch.hypot(x[:, 2:] + x[:, :2], y[:, 2:] + y[:, :2])
    assert float(gap.max()) <= 2 * 150.0 + 1e-3
    distance = torch.hypot(x, y)
    assert float((distance - CLOVER_SPAWN_DISTANCE).abs().max()) <= 150.0 + 1e-3
    # the nose toward the centre from the unjittered point: within the heading jitter and
    # the angle the position jitter subtends
    to_centre = torch.atan2(-y, -x)
    heading = torch.atan2(ships.heading_sin, ships.heading_cos)
    error = torch.remainder(heading - to_centre + math.pi, 2 * math.pi) - math.pi
    bound = math.radians(5.0) + math.asin(150.0 / (CLOVER_SPAWN_DISTANCE - 150.0))
    assert float(error.abs().max()) <= bound + 1e-4
    # each spawn sits between a high and a low field: its nearest two fields are one of each
    near = torch.hypot(
        x[..., None] - state.fields.x[:, None], y[..., None] - state.fields.y[:, None]
    ).argsort(-1)[..., :2]
    levels = state.fields.level.gather(1, near.flatten(1)).reshape(near.shape)
    assert ((levels[..., 0] > 0) != (levels[..., 1] > 0)).all()


def test_a_layout_without_jitter_or_turn_is_exactly_its_points_and_their_images():
    layout = MapLayout(
        fields=(FixedField(1000.0, 0.0, 300.0, (2,)),),
        spawns=(FixedSpawn(0.0, 2000.0), FixedSpawn(2000.0, 0.0, heading=0.5)),
        rotated=False,
    )
    config = GameConfig(2, 2, 3000.0, mode="team_deathmatch", layout=layout)
    rules, state = _clover(8, config=config)
    assert state.fields.x[0].tolist() == [1000.0, -1000.0]
    assert state.fields.level[0].tolist() == [2, 2]
    ships = state.ships
    torch.testing.assert_close(ships.x[0], torch.tensor([0.0, 2000.0, 0.0, -2000.0]))
    torch.testing.assert_close(ships.y[0], torch.tensor([2000.0, 0.0, -2000.0, 0.0]))
    heading = torch.atan2(ships.heading_sin[0], ships.heading_cos[0])
    torch.testing.assert_close(
        heading, torch.tensor([-math.pi / 2, 0.5, math.pi / 2, 0.5 - math.pi])
    )


@pytest.mark.parametrize(
    "change, message",
    [
        (dict(mode="frontline"), "modes"),
        (dict(ships_per_team=3), "spawn points"),
        (dict(num_fields=6), "fields"),
    ],
)
def test_a_layout_must_fit_its_game(change, message):
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(CLOVER, **change)
    with pytest.raises(ValueError, match="levels"):
        bad = MapLayout(fields=(FixedField(0.0, 0.0, 10.0, (3,)),))
        api.rules_of(GameConfig(2, 2, 900.0, layout=bad))


def test_a_saved_map_keeps_its_layout(tmp_path):
    library = Maps(tmp_path / "user", tmp_path / "project")
    library.save(Map("my-clover", CLOVER, place="user"))
    assert library.find("my-clover").config == CLOVER


# ------------------------------------------------------------------ the scripted agent


def test_the_scripted_agent_starts_elimination_from_the_deathmatch_set():
    """Its own set, copied from real_scale_deathmatch without a search (mode_index's choice
    of it: `test_scripted_scale`)."""
    assert "real_scale_elimination" in SCRIPTED_MODES
    stored = read_real_scale()
    assert (
        stored["real_scale_elimination"]["parameters"]
        == (stored["real_scale_deathmatch"]["parameters"])
    )
    sets = MODE_DEFAULTS.sets
    assert pack(sets["real_scale_elimination"]) == pack(sets["real_scale_deathmatch"])


# ------------------------------------------------------------------ the model's mode input


@pytest.mark.usefixtures("global_generator")
def test_a_new_build_reads_the_elimination_flag_and_an_older_one_does_not():
    """A new run's Yemong (every mode's flag, `TrainConfig`) decides otherwise when the
    elimination flag is cleared; a build of the older record default (two flags) reads an
    elimination match exactly as a match of no mode."""
    import yemong_cases

    from bnb.models.yemong.inputs import MODE_INPUTS, global_inputs
    from bnb.models.yemong.model import YemongConfig
    from bnb.runtime.train import TrainConfig

    assert TrainConfig(GameConfig(5, 10, 2600.0)).yemong.mode_inputs == MODE_INPUTS
    assert YemongConfig().mode_inputs == MODE_INPUTS[:2]
    assert global_inputs(MODE_INPUTS) == global_inputs(MODE_INPUTS[:2]) + 1
    config = GameConfig(5, 10, 2600.0, mode="elimination")
    rules = api.rules_of(config)
    state = api.reset(
        rules, cases.TYPES, api.uniforms(rules, 2, generator=torch.Generator().manual_seed(5))
    )
    _, obs, _ = api.perceive(rules, state)
    view = type(obs)(**{name: part.flatten(0, 1) for name, part in vars(obs).items()})
    cleared = view.globals.clone()
    cleared[:, GLOBAL["mode_elimination"]] = 0.0
    no_mode = dataclasses.replace(view, globals=cleared)
    reset = torch.zeros(view.ships.shape[0], dtype=torch.bool)  # keeps the carry's offset
    for modes, reads in ((MODE_INPUTS, True), (MODE_INPUTS[:2], False)):
        net = yemong_cases.model(mode_inputs=modes)
        carry = net.initial_carry(view.ships.shape[0], 5, 10)
        with torch.no_grad():
            first = net.step(carry, view, reset).policy_logits
            second = net.step(carry, no_mode, reset).policy_logits
        assert torch.equal(first, second) != reads, modes


def test_real_scale_elimination_fits_the_real_scale_training_batch():
    """Its sizes and ship set are the real-scale mix's, so a mix may take it (its share is
    the author's call); its pools then wait no respawn."""
    from bnb.runtime.train import TRAINING_MIXES, mixed

    mix = TRAINING_MIXES["real_scale_mixed"]
    with_elimination = tuple((name, 0.45) for name, _ in mix) + (("real_scale_elimination", 0.1),)
    game, pools = mixed(mix=with_elimination)
    rules = api.rules_of(game, pools, 40)
    games = [pool for pool in rules.pools if pool.mode.name == "elimination"]
    assert games and all(pool.respawn_ticks == pool.episode_ticks == 5400 for pool in games)


SIDES = PRESETS["real_scale_elimination"]


def _side_placement(mode_rules=None, batch=256, seed=11):
    rules = api.rules_of(SIDES, types=REAL_SCALE)
    mode_rules = rules.modes[0] if mode_rules is None else mode_rules
    generator = torch.Generator().manual_seed(seed)
    angle = 2.0 * math.pi * torch.rand(batch, generator=generator)
    radius = torch.full((batch,), SIDES.arena_radius)
    placed = spawns.side_placement(
        rules,
        mode_rules,
        radius,
        torch.rand(batch, mode_rules.uniform_count(rules.num_ships), generator=generator),
        torch.cos(angle),
        torch.sin(angle),
    )
    return rules, mode_rules, angle, placed


def test_elimination_teams_spawn_in_their_own_areas_on_opposite_sides():
    rules, mode_rules, angle, placed = _side_placement()
    offset = rules.spawn_offset * SIDES.arena_radius
    area = mode_rules.side_area_fraction * SIDES.arena_radius
    side = torch.tensor([-1.0, 1.0]).repeat_interleave(rules.ships_per_team)
    centre_x = side * offset * torch.cos(angle)[:, None]
    centre_y = side * offset * torch.sin(angle)[:, None]
    from_centre = torch.hypot(placed.x - centre_x, placed.y - centre_y)
    assert float(from_centre.max()) <= area * (1.0 + 1e-5)
    assert float(from_centre.max()) > 0.9 * area  # the whole area is used
    # along the spawn axis team 0 is on the - side and team 1 on the +, apart by the gap
    along = by_team(placed.x * torch.cos(angle)[:, None] + placed.y * torch.sin(angle)[:, None])
    assert float(along[:, 0].max()) < -(offset - area) + 1e-2
    assert float(along[:, 1].min()) > (offset - area) - 1e-2


def test_elimination_spawns_face_the_enemy_area_within_the_jitter():
    rules, mode_rules, angle, placed = _side_placement()
    side = torch.tensor([-1.0, 1.0]).repeat_interleave(rules.ships_per_team)
    enemy_x = -side * rules.spawn_offset * SIDES.arena_radius * torch.cos(angle)[:, None]
    enemy_y = -side * rules.spawn_offset * SIDES.arena_radius * torch.sin(angle)[:, None]
    to_x, to_y = enemy_x - placed.x, enemy_y - placed.y
    off = torch.atan2(
        placed.heading_cos * to_y - placed.heading_sin * to_x,
        placed.heading_cos * to_x + placed.heading_sin * to_y,
    ).abs()
    assert float(off.max()) <= mode_rules.side_heading_jitter + 1e-4
    assert float(off.max()) > 0.9 * mode_rules.side_heading_jitter


def test_elimination_spawns_near_the_edge_leave_room_to_turn():
    """An area reaching the edge, with every heading allowed: the edge rule turns the
    outward ones the least that keeps the outward cosine at most d / D."""
    rules = api.rules_of(SIDES, types=REAL_SCALE)
    wide = dataclasses.replace(rules.modes[0], side_area_fraction=0.2, side_heading_jitter=math.pi)
    _, _, _, placed = _side_placement(wide)
    distance = torch.hypot(placed.x, placed.y)
    outward = (placed.heading_cos * placed.x + placed.heading_sin * placed.y) / distance
    room = (SIDES.arena_radius - distance) / rules.spawn_turn_distance
    assert bool((outward <= room + 1e-4).all())
    near = room < 1.0
    assert bool(near.any())
    assert float((outward - room)[near].max()) > -1e-2  # the bound is reached


def test_an_elimination_match_spawns_by_side_and_clover_keeps_its_layout():
    rules = api.rules_of(SIDES, types=REAL_SCALE)
    generator = torch.Generator().manual_seed(4)
    state = api.reset(rules, REAL_SCALE, api.uniforms(rules, 8, generator=generator))
    shown = api.display(state)
    x, y = by_team(shown["ships.x"]), by_team(shown["ships.y"])
    # the teams' mean points are about 2 spawn offsets apart, through the centre
    gap = torch.hypot(x[:, 0].mean(-1) - x[:, 1].mean(-1), y[:, 0].mean(-1) - y[:, 1].mean(-1))
    assert float(gap.min()) > 2.0 * (rules.spawn_offset - rules.modes[0].side_area_fraction) * (
        SIDES.arena_radius
    )
