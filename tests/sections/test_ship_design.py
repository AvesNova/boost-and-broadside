"""Designing ship types (bnb.game.ship_design, bench/ship_design_search.py): the knobs and
their inverse, the envelope, the instant card at its anchor, the live engine knee, mixed
teams, the optimiser, and (GPU) that a new card plays without recompiling."""

import dataclasses
import math
import pathlib
import sys

import numpy as np
import pytest
import torch

from bnb.game import api, ships
from bnb.game.physics.combat import MEAN_CLOSENESS
from bnb.game.physics.flight import airframe_of, engine_fall
from bnb.game.ship_design import (
    BOUNDS,
    CONTINUOUS,
    ENVELOPE,
    Envelope,
    design_cards,
    envelope_handles,
    handles_of,
    knobs_of,
    set_handles,
)
from bnb.game.ship_types import FIELDS
from bnb.league.matches import Player, team_lineup
from bnb.reference.real_planes import solved_types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # `bench`
from bench import ship_design_search as search  # noqa: E402

GAME = api.PRESETS["real_scale_deathmatch"]


@pytest.fixture(scope="module")
def handles():
    return set_handles()


def test_a_design_with_no_knob_moved_has_its_base_handles(handles):
    rules = api.rules_of(GAME)
    for key in ("yongma", "jujak", "vizia"):
        base = handles[key]
        same = handles_of(base, {}, rules)
        for name, value in base.items():
            assert same[name] == pytest.approx(value, rel=2e-2 if name == "climb_time" else 1e-6)
        assert knobs_of(same) == pytest.approx(knobs_of(base), rel=1e-9)


def test_a_knob_moves_only_its_own_handles_and_those_that_follow(handles):
    rules = api.rules_of(GAME)
    base = handles["vizia"]
    knobs = knobs_of(base)
    faster = handles_of(base, {"fire_rate": 10.0}, rules)
    # the damage per second and the store cost per second of fire stay
    mean_hit = faster["hit_damage"] * MEAN_CLOSENESS  # the handle is a centre hit's
    assert mean_hit * 10.0 == pytest.approx(knobs["damage_per_second"])
    cost = base["ceiling"] / base["shots_per_store"] * base["fire_rate"]
    assert faster["ceiling"] / faster["shots_per_store"] * 10.0 == pytest.approx(cost)
    longer = handles_of(base, {"muzzle_speed": 700.0}, rules)
    assert longer["gun_range"] < base["gun_range"]
    assert longer["bullet_spread"] / 700.0 == pytest.approx(knobs["dispersion"])
    with pytest.raises(ValueError, match="fire rate"):
        handles_of(base, {"fire_rate": 7.0}, rules)


def test_the_envelope_is_the_propeller_fighters_and_holds_its_bounds(handles):
    """The envelope is the 16 real propeller fighters of the reference data (the set's own
    types are no part of it)."""
    assert "me262" not in ENVELOPE and "me163" not in ENVELOPE and len(ENVELOPE) == 16
    reference = envelope_handles()
    envelope = Envelope.of(handles)  # the set's handles hold no envelope type
    assert torch.equal(envelope.mean, Envelope.of().mean)
    # every envelope type is inside the envelope's own bounds (not the author's BOUNDS,
    # which the balancing's damage and the shields may leave)
    for key in ENVELOPE:
        knobs = {k: v for k, v in knobs_of(reference[key]).items() if k not in BOUNDS}
        assert {k: envelope.clamp(knobs_of(reference[key]))[k] for k in knobs} == pytest.approx(
            knobs
        )
    # a jet's speed is outside them, and far from the mean
    jet = knobs_of(solved_types()["me262"].handles)
    assert envelope.clamp(jet)["cruise_speed"] < jet["cruise_speed"]
    assert envelope.distance(jet) > max(envelope.distance(knobs_of(reference[k])) for k in ENVELOPE)
    assert set(CONTINUOUS) | {"fire_rate"} == set(knobs_of(handles["vizia"]))


def test_the_instant_card_of_unmoved_designs_is_their_solved_card(handles):
    rules = api.rules_of(GAME)
    keys = ["yongma", "jujak", "vizia"]
    card = design_cards(rules, keys, [handles[k] for k in keys])
    found = {ship.key: ship for ship in ships.load("real-scale")}
    for index, key in enumerate(keys):
        for name in FIELDS:
            assert float(getattr(card, name)[index]) == pytest.approx(
                found[key].card[name], rel=1e-5
            )


def test_the_engine_knee_scale_moves_the_knee_and_one_is_the_rules_knee():
    rules = api.rules_of(GAME)
    types = api.ship_types(ship_set=GAME.ship_set)
    kinds = torch.zeros(3, dtype=torch.int32)
    store = types.max_energy[0] * torch.tensor([0.4, 0.6, 0.8])
    plain = engine_fall(rules.flight, airframe_of(types, kinds), store)
    assert plain[0] == 0.0 and plain[1] > 0.0
    raised = dataclasses.replace(types, engine_knee_scale=types.engine_knee_scale * 1.4)
    fall = engine_fall(rules.flight, airframe_of(raised, kinds), store)
    assert fall[1] == 0.0 and 0.0 < fall[2] < plain[2]  # the knee at 0.7 of the store


def test_a_player_may_fly_a_mixed_team():
    rules = api.rules_of(GAME)
    assert team_lineup(Player("a", ship_type=3), rules) == (3,) * 5
    assert team_lineup(Player("b", ship_type=(1, 2, 1, 2, 1)), rules) == (1, 2, 1, 2, 1)
    with pytest.raises(ValueError, match="lineup"):
        team_lineup(Player("c", ship_type=(1, 2)), rules)


def test_the_optimiser_finds_a_quadratic_minimum():
    target = np.array([0.7, -0.4, 1.2, 0.1])
    cma = search.CMA(4, 10, 0.5, seed=3)
    for _ in range(80):
        samples = cma.ask()
        cma.tell(samples, [float(((s - target) ** 2).sum()) for s in samples])
    assert np.abs(cma.mean - target).max() < 1e-2
    again = search.CMA(4, 10, 0.5, seed=3, state=cma.state())
    assert np.allclose(again.mean, cma.mean) and again.generation == cma.generation


def test_the_band_loss_is_zero_inside_and_grows_outside():
    band = (0.6, 0.7)
    assert search.band_loss(0.65, 128, band) == 0.0
    assert search.band_loss(0.5, 128, band) == pytest.approx(math.log(1.5) ** 2)
    assert search.band_loss(0.0, 128, band) > search.band_loss(0.3, 128, band)


@pytest.mark.gpu
@pytest.mark.slow
def test_a_new_card_and_knee_play_without_recompiling():
    """The arena's card (every field, the engine knee and the coast exchange scales among
    them) is live: a second card plays with no new compiled graph, and it is the card that
    is played (shields no gun can break: no kills)."""
    from torch._dynamo.utils import counters

    from bnb.league.matches import Arena, ArenaConfig
    from bnb.runtime.opponents import Rung

    device = torch.device("cuda")
    base = api.ship_types(device, ship_set=GAME.ship_set)
    two = type(base)(**{k: v[:2].clone() for k, v in vars(base).items()})  # Yongma, Jujak
    players = [Player(f"slot {i}", rung=Rung(1.0), ship_type=i) for i in range(2)]
    arena = Arena(players, GAME, device, ArenaConfig(blocks=1, block_environments=8), card=two)
    first = arena.play([(0, 1)])
    graphs = counters["stats"]["unique_graphs"]
    changed = dataclasses.replace(
        two,
        engine_knee_scale=two.engine_knee_scale * 1.6,
        coast_exchange_scale=two.coast_exchange_scale * 3.0,
        max_shield=two.max_shield * 0 + 1e9,
        fire_rate=torch.tensor([10.0, 15.0], device=device),
    )
    arena.set_card(changed)
    second = arena.play([(0, 1)])
    assert counters["stats"]["unique_graphs"] == graphs
    assert float(arena.rollout.carry["types"].engine_knee_scale[0]) == pytest.approx(1.6)
    assert int(first.sum()) == 8 and int(second.sum()) == 8
    assert float(arena.statistics()[..., 0].sum()) == 0.0  # no kills through 1e9 shields


def test_the_nash_of_a_cycle_is_uniform_and_a_dominated_type_gets_none():
    rps = np.array([[0.0, 0.2, -0.2], [-0.2, 0.0, 0.2], [0.2, -0.2, 0.0]])
    assert search.nash(rps) == pytest.approx([1 / 3] * 3, abs=0.02)
    # a fourth type that loses to every other
    dominated = np.zeros((4, 4))
    dominated[:3, :3] = rps
    dominated[3, :3], dominated[:3, 3] = -0.1, 0.1
    assert search.nash(dominated)[3] < 0.02


def test_a_player_brings_its_own_doctrine_only_to_an_arena_with_one():
    from bnb.controllers.scripted_parameters import ScriptedParameters
    from bnb.league.matches import scripted_play
    from bnb.runtime.opponents import Rung

    own = ScriptedParameters(aggression=0.5)
    play = scripted_play(Player("a", rung=Rung(1.0), ship_type=0, parameters=own))
    assert play.parameters is own
    assert scripted_play(Player("b", rung=Rung(1.0))).parameters is None


@pytest.mark.gpu
@pytest.mark.slow
def test_a_slots_doctrine_is_live_and_compiles_nothing():
    """In an arena built with a doctrine, a player's own parameter set is written into its
    rows of the table: the next batch plays it with no new compiled graph."""
    from torch._dynamo.utils import counters

    from bnb.controllers.scripted_defaults import MODE_DEFAULTS
    from bnb.controllers.scripted_parameters import pack
    from bnb.league.matches import Arena, ArenaConfig
    from bnb.runtime.opponents import Rung

    device = torch.device("cuda")
    doctrine = MODE_DEFAULTS.sets["real_scale_deathmatch"]
    players = [Player(f"slot {i}", rung=Rung(1.0), ship_type=0) for i in range(2)]
    arena = Arena(
        players, GAME, device, ArenaConfig(blocks=1, block_environments=8), doctrine=doctrine
    )
    arena.play([(0, 1)])
    graphs = counters["stats"]["unique_graphs"]
    timid = dataclasses.replace(doctrine, aggression=-5.0)
    arena.players[1] = dataclasses.replace(players[1], parameters=timid)
    arena.play([(0, 1)])
    assert counters["stats"]["unique_graphs"] == graphs
    table = arena.seat.doctrine_table
    rows = arena.seat._slot_rows(1)
    assert torch.allclose(table[rows][0].cpu(), torch.tensor(pack(timid), dtype=torch.float32))
    assert torch.allclose(table[arena.seat._slot_rows(0)][0].cpu(), torch.tensor(pack(doctrine)))


def test_the_curl_shortfall_is_zero_for_a_strong_cycle_and_counts_a_flat_one():
    spec = dict(cycle=["a", "b", "c"], band=[0.6, 0.7])
    assert search.curl_shortfall([0.65, 0.65, 0.65], 1024, spec) == 0.0
    flat = search.curl_shortfall([0.5, 0.5, 0.5], 1024, spec)
    assert flat == pytest.approx((3 * math.log(1.5)) ** 2)
    assert search.curl_shortfall([0.5] * 3, 1024, dict(edges=[], band=[0.6, 0.7])) == 0.0
