"""Imugi's explore grid (2026-11-15): what its team has seen lately, and the cell it heads
for while it knows no enemy (`imugi_combat.SEARCH` "explore")."""

import dataclasses
import math

import modes_cases as cases
import torch

from bnb.contracts import GameConfig
from bnb.controllers import imugi_combat
from bnb.controllers.imugi_defaults import MODE_DEFAULTS
from bnb.controllers.imugi_memory import (
    EXPLORE_CELLS,
    EXPLORE_SECONDS,
    fresh_carry,
    track_boundary,
    track_seen,
)
from bnb.game import api
from bnb.game.handling.characteristics import characteristics

ELIMINATION = GameConfig(3, 3, 1000.0, episode_seconds=30.0, mode="elimination")


def _view(rules, state, tick: int):
    match = dataclasses.replace(
        state.match, episode_tick=torch.full_like(state.match.episode_tick, tick)
    )
    _, obs, _ = api.perceive(rules, dataclasses.replace(state, match=match))
    return type(obs)(**{name: part[:, 0] for name, part in vars(obs).items()})


def _setup():
    rules = api.rules_of(ELIMINATION)
    _, state = cases.rules_and_state(ELIMINATION, batch=1)
    return rules, state, characteristics(cases.TYPES, rules)


def test_cells_in_view_are_seen_and_the_rest_decay_by_the_time_constant():
    rules, state, measured = _setup()
    reset = torch.zeros(1, dtype=torch.bool)
    first = _view(rules, state, 1)
    carry = track_seen(fresh_carry(1, 3, 3, "cpu"), first, reset, measured, rules.tick_length)
    carry = track_boundary(carry, first, reset, rules.tick_length)
    seen = carry.seen[0]
    assert set(seen.unique().tolist()) <= {0.0, 1.0}
    assert 0 < int(seen.sum()) < seen.numel()  # some cells in view, not all (1 km arena)
    # the ships gone (dead): nothing in view, everything decays one decision's worth
    gone = cases.edit_ships(state, alive=torch.zeros_like(state.ships.alive))
    later = _view(rules, gone, 2)
    after = track_seen(carry, later, reset, measured, rules.tick_length)
    want = seen * math.exp(-rules.tick_length / EXPLORE_SECONDS)
    torch.testing.assert_close(after.seen[0], want)
    # re-reading the observation the carry was made from keeps it
    again = track_seen(carry, first, reset, measured, rules.tick_length)
    torch.testing.assert_close(again.seen, carry.seen)
    # a reset starts from nothing seen
    fresh = track_seen(carry, later, torch.ones(1, dtype=torch.bool), measured, rules.tick_length)
    assert float(fresh.seen.sum()) == 0.0


def test_the_explore_target_is_an_unseen_cell_inside_the_boundary():
    rules, state, measured = _setup()
    obs = _view(rules, state, 1)
    parameters = MODE_DEFAULTS.sets["deathmatch"]
    carry = fresh_carry(1, 3, 3, "cpu")
    seen = torch.zeros_like(carry.seen)
    seen[0, : seen.shape[1] // 2] = 1.0  # the inner half of the grid seen just now
    carry = dataclasses.replace(carry, seen=seen)
    fleet = imugi_combat.context(obs, measured, parameters, carry=carry)
    target_x, target_y = imugi_combat.explore_target(fleet)
    cells = EXPLORE_CELLS * float(fleet.arena_radius[0, 0])
    index = int(torch.cdist(torch.stack([target_x[0], target_y[0]], -1), cells).argmin())
    assert float(seen[0, index]) < imugi_combat.EXPLORE_UNSEEN
    radius = float(torch.hypot(target_x, target_y)[0, 0])
    assert radius < imugi_combat.EXPLORE_INSIDE * float(fleet.boundary_radius[0, 0])


def test_the_explore_search_steers_without_a_target_and_the_old_rule_still_works():
    rules, state, measured = _setup()
    obs = _view(rules, state, 1)
    parameters = MODE_DEFAULTS.sets["deathmatch"]
    fleet = imugi_combat.context(obs, measured, parameters, carry=fresh_carry(1, 3, 3, "cpu"))
    saved = imugi_combat.SEARCH
    try:
        for search in imugi_combat.SEARCH_MODES:
            imugi_combat.SEARCH = search
            x, y, may_hold = imugi_combat.idle_direction(fleet)
            assert x.shape == fleet.x.shape and bool(torch.isfinite(x).all())
            assert may_hold == (search == "centre")
    finally:
        imugi_combat.SEARCH = saved
