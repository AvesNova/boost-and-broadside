"""Designing ship types (bnb.game.ship_design, bench/ship_design_search.py): the knobs and
their inverse, the envelope, the instant card at its anchor, the live engine knee, mixed
teams, the optimiser, and (GPU) that a new card plays without recompiling."""

import dataclasses
import json
import math
import pathlib
import sys

import numpy as np
import pytest
import torch

from bnb.game import api, ships
from bnb.game.physics.flight import airframe_of, engine_fall
from bnb.game.ship_design import (
    CONTINUOUS,
    ENVELOPE,
    NARROWED,
    Envelope,
    design_cards,
    handles_of,
    knobs_of,
    set_handles,
)
from bnb.game.ship_types import FIELDS
from bnb.league.matches import Player, team_lineup

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # `bench`
from bench import ship_design_search as search  # noqa: E402

GAME = api.PRESETS["real_scale_deathmatch"]


@pytest.fixture(scope="module")
def handles():
    return set_handles()


def test_a_design_with_no_knob_moved_has_its_base_handles(handles):
    rules = api.rules_of(GAME)
    for key in ("p51d", "fw190-a8", "albatross"):
        base = handles[key]
        same = handles_of(base, {}, rules)
        for name, value in base.items():
            assert same[name] == pytest.approx(value, rel=2e-2 if name == "climb_time" else 1e-9)
        assert knobs_of(same) == pytest.approx(knobs_of(base), rel=1e-9)


def test_a_knob_moves_only_its_own_handles_and_those_that_follow(handles):
    rules = api.rules_of(GAME)
    base = handles["p51d"]
    knobs = knobs_of(base)
    faster = handles_of(base, {"fire_rate": 10.0}, rules)
    # the damage per second and the store cost per second of fire stay
    assert faster["hit_damage"] * 10.0 == pytest.approx(knobs["damage_per_second"])
    cost = base["ceiling"] / base["shots_per_store"] * base["fire_rate"]
    assert faster["ceiling"] / faster["shots_per_store"] * 10.0 == pytest.approx(cost)
    longer = handles_of(base, {"muzzle_speed": 700.0}, rules)
    assert longer["gun_range"] < base["gun_range"]
    assert longer["bullet_spread"] / 700.0 == pytest.approx(knobs["dispersion"])
    with pytest.raises(ValueError, match="fire rate"):
        handles_of(base, {"fire_rate": 7.0}, rules)


def test_the_envelope_is_the_propeller_fighters_and_holds_its_bounds(handles):
    assert "me262" not in ENVELOPE and "me163" not in ENVELOPE and len(ENVELOPE) == 16
    envelope = Envelope.of(handles)
    for key in ENVELOPE:  # every envelope type is inside its own bounds but the narrowed ones
        knobs = knobs_of(handles[key])
        clamped = envelope.clamp(knobs)
        for name in CONTINUOUS:
            low, high = NARROWED.get(name, (0.0, math.inf))
            if low <= knobs[name] <= high:
                assert clamped[name] == pytest.approx(knobs[name])
    # the steep knobs stay in their narrowed band: the Bf 109K's 520 m/s cannon is lifted
    assert (
        envelope.clamp(knobs_of(handles["bf109-k4"]))["muzzle_speed"] == NARROWED["muzzle_speed"][0]
    )
    assert (
        envelope.clamp({**knobs_of(handles["p51d"]), "max_shield": 200.0})["max_shield"]
        == NARROWED["max_shield"][1]
    )
    # a jet's speed is outside them, and far from the mean
    jet = knobs_of(handles["me262"])
    assert envelope.clamp(jet)["cruise_speed"] < jet["cruise_speed"]
    assert envelope.distance(jet) > max(envelope.distance(knobs_of(handles[k])) for k in ENVELOPE)
    assert set(CONTINUOUS) | {"fire_rate"} == set(knobs_of(handles["p51d"]))


def test_the_instant_card_of_unmoved_designs_is_their_solved_card(handles):
    rules = api.rules_of(GAME)
    keys = ["p51d", "spitfire-ix", "p47d"]
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
    two = type(base)(**{k: v[:2].clone() for k, v in vars(base).items()})  # P-51D, Heavy
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


# ---------------------------------------------------------------- doctrine, trim, roles


def _space(**extra):
    """A three-design space on the La-7 (the seed's shape), with extra spec fields."""
    spec = dict(
        preset="real_scale_deathmatch",
        band=[0.6, 0.7],
        realism_radius=3.5,
        realism_weight=1.0,
        distinct=1.0,
        distinct_weight=1.0,
        designs=[
            dict(name=name, base="la7", knobs=["max_shield", "sustained_turn_rate"])
            for name in ("A", "B", "C")
        ],
        cycle=["A", "B", "C"],
    )
    spec |= extra
    return search.setup(spec)


def test_a_doctrine_dimension_reads_and_writes_a_field_or_its_part():
    from bnb.controllers.scripted_defaults import MODE_DEFAULTS

    preset = MODE_DEFAULTS.sets["real_scale_deathmatch"]
    assert search.doctrine_value(preset, "energy_fight_strength") == 0.0
    centre = search.doctrine_value(preset, "shoot_in_gun_range.centre")
    moved = search.with_doctrine(
        preset, {"shoot_in_gun_range.centre": 2 * centre, "energy_fight_strength": 1.7}
    )
    assert moved.shoot_in_gun_range.centre == pytest.approx(2 * centre)
    assert moved.shoot_in_gun_range.rising == preset.shoot_in_gun_range.rising
    assert moved.energy_fight_strength == 1.0  # held within its additive bounds
    assert moved.aim_in_target_size == preset.aim_in_target_size  # the rest the preset's
    with pytest.raises(ValueError, match="no such"):
        search.with_doctrine(preset, {"bravery": 1.0})


def test_the_search_coordinates_move_knobs_then_doctrine():
    space = _space()
    space.designs[0]["doctrine"] = ["energy_fight_strength", "extend_in_gun_range.centre"]
    space.designs[0]["doctrine_start"] = {"energy_fight_strength": 0.5}
    space.__post_init__()
    assert space.size == 3 * 2 + 2
    x = np.zeros(space.size)
    assert space.knobs(x)[0] == pytest.approx(space.base(0))
    start = space.doctrine(x)
    assert start[0]["energy_fight_strength"] == 0.5 and start[1] == {}
    x[6], x[7] = 1.0, 1.0  # one step of each: a quarter of the bounds, a quarter of the log
    moved = space.doctrine(x)[0]
    assert moved["energy_fight_strength"] == pytest.approx(0.5 + search.DOCTRINE_STEP)
    ratio = moved["extend_in_gun_range.centre"] / start[0]["extend_in_gun_range.centre"]
    assert ratio == pytest.approx(math.exp(search.DOCTRINE_STEP))
    x[7] = 100.0  # a multiplicative dimension stays within DOCTRINE_RANGE of its start
    ratio = (
        space.doctrine(x)[0]["extend_in_gun_range.centre"] / start[0]["extend_in_gun_range.centre"]
    )
    assert ratio == pytest.approx(search.DOCTRINE_RANGE)
    assert space.parameters(moved).energy_fight_strength == pytest.approx(0.75)


def test_the_edge_ratings_leave_the_cycle_and_the_trim_takes_out_the_rest():
    edges = [(0, 1), (1, 2), (2, 0)]
    curl, ratings = 0.9, np.array([0.4, -0.1, -0.3])
    logits = [curl / 3 + ratings[a] - ratings[b] for a, b in edges]
    assert search.edge_ratings(edges, logits, 3) == pytest.approx(ratings)
    space = _space(trim=dict(knob="muzzle_speed", slope=1400.0))
    knobs = space.knobs(np.zeros(space.size))
    scores = [1 / (1 + math.exp(-v)) for v in logits]
    trim = search.predicted_trim(space, knobs, edges, scores, 10**6)
    # every edge is the cyclic part, a third of the curl
    assert trim["scores"] == pytest.approx([1 / (1 + math.exp(-curl / 3))] * 3, abs=1e-4)
    elo = ratings * search.ELO
    for factor, one in zip(trim["factors"], elo, strict=True):
        assert factor == pytest.approx(2 ** (-one / 1400.0))
    # a trim beyond the knob's band leaves the rest in the edges
    far = [1 / (1 + math.exp(-(v * 8))) for v in logits]
    clipped = search.predicted_trim(space, knobs, edges, far, 10**6)
    assert max(k["muzzle_speed"] for k in clipped["knobs"]) <= NARROWED["muzzle_speed"][1]
    assert min(clipped["scores"]) < 0.45 or max(clipped["scores"]) > 0.75


def test_the_trim_slope_is_the_specs_or_a_case_studys(tmp_path):
    assert search.trim_slope(dict(trim=dict(knob="muzzle_speed", slope=1300.0))) == 1300.0
    report = pathlib.Path(__file__).resolve().parents[2] / (
        "docs/explanation/game/case-studies/guns-la7/report.json"
    )
    relative = report.relative_to(pathlib.Path(__file__).resolve().parents[2])
    slope = search.trim_slope(dict(trim=dict(knob="muzzle_speed", slope_from=str(relative))))
    assert 1000.0 < slope < 2000.0


def test_with_a_trim_the_loss_judges_the_trimmed_edges():
    space = _space(trim=dict(knob="muzzle_speed", slope=1400.0), curl_weight=1.0)
    knobs = space.knobs(np.zeros(space.size))
    edges = [(0, 1), (1, 2), (2, 0)]
    lopsided = [0.80, 0.55, 0.50]  # curl 2.0: a cycle of 0.66 once trimmed
    row = search.scored(space, knobs, edges, lopsided, 10**6)
    assert row["edge_loss"] == pytest.approx(0.0, abs=1e-6)
    assert all(0.6 < p < 0.7 for p in row["trimmed_scores"])
    plain = search.scored(_space(curl_weight=1.0), knobs, edges, lopsided, 10**6)
    assert plain["edge_loss"] > 0.0 and "trimmed_scores" not in plain


def test_the_roles_ask_each_design_to_lead_on_its_strong_axis_and_trail_on_its_weak():
    axes = {
        "armour": {"max_shield": 1},
        "agility": {"sustained_turn_rate": 1, "doctrine:energy_fight_strength": -1},
    }
    designs = [
        dict(name="A", base="la7", strong="armour", weak="agility"),
        dict(name="B", base="la7", strong="agility", weak="armour"),
        dict(name="C", base="la7"),
    ]
    space = _space(axes=axes, designs=designs, role_margin=0.5)
    assert space.designs[0]["knobs"] == ["max_shield", "sustained_turn_rate"]
    assert space.designs[0]["doctrine"] == ["energy_fight_strength"]
    x = np.zeros(space.size)
    knobs, doctrine = space.knobs(x), space.doctrine(x)
    assert space.roles(knobs, doctrine) == pytest.approx(4 * 0.5**2)  # all equal: all short
    x[0], x[1] = 2.0, -2.0  # A: more shield, less turn
    x[2], x[3] = -2.0, 2.0  # B: the reverse
    knobs, doctrine = space.knobs(x), space.doctrine(x)
    assert space.roles(knobs, doctrine) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="not in axes"):
        _space(axes=axes, designs=[dict(name="A", base="la7", strong="reach")] * 3)


def test_a_trim_candidate_is_the_round_of_the_smallest_spread(tmp_path):
    rows = [
        dict(label=f"trim:{k}", source="9:0", card="set", spread=s, knobs=[], doctrine=[])
        for k, s in enumerate((80.0, 12.0, 30.0))
    ]
    (tmp_path / "trim.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert search.candidate_row(tmp_path, "trim")["label"] == "trim:1"
    assert search.candidate_row(tmp_path, "trim:2")["spread"] == 30.0
    search_rows = [
        dict(generation=g, index=0, mean=True, loss=loss, knobs=[{}])
        for g, loss in enumerate((3.0, 1.0, 2.0))
    ]
    (tmp_path / "search.jsonl").write_text("".join(json.dumps(r) + "\n" for r in search_rows))
    best = search.candidate_row(tmp_path, None)
    assert search.label_of(best) == "1:0" and best["doctrine"] == [{}]  # an older search's row


def _check_rows(card, scores):
    names = ("A", "B", "C")
    rows = []
    for (a, b), score in zip(((0, 1), (0, 2), (1, 2)), scores, strict=True):
        rows.append(
            dict(
                kind="pair",
                first=names[a],
                second=names[b],
                games=1024,
                score=score,
                candidate="5:0",
                card=card,
            )
        )
    return rows


def test_the_verdict_gates_only_the_specs_preset_and_accepts_only_exact_robust_cards(tmp_path):
    spec = dict(_space().spec)
    cycle = [0.65, 1 - 0.65, 0.65]  # A > B, C > A, B > C
    verdict = search.verdict_of(
        spec, tmp_path, _check_rows("design-x", cycle), "real_scale_deathmatch"
    )
    assert verdict["gate"] and verdict["exact"] and verdict["pass"] and verdict["directions_kept"]
    assert not verdict["accepted"]  # no robustness test yet
    robust = dict(candidate="5:0", card="design-x", pass_=None, **{"pass": True})
    (tmp_path / "robust-summary-real_scale_deathmatch.json").write_text(json.dumps(robust))
    verdict = search.verdict_of(
        spec, tmp_path, _check_rows("design-x", cycle), "real_scale_deathmatch"
    )
    assert verdict["accepted"]
    estimated = search.verdict_of(
        spec, tmp_path, _check_rows("design-x+estimate", cycle), "real_scale_deathmatch"
    )
    assert estimated["pass"] and not estimated["exact"] and not estimated["accepted"]
    frontline = search.verdict_of(spec, tmp_path, _check_rows("design-x", cycle), search.FRONTLINE)
    assert frontline["gate"] is False and frontline["pass"] is None and "accepted" not in frontline


def test_the_robust_verdict_wants_every_curl_positive_and_the_median_strong(tmp_path):
    spec = dict(cycle=["A", "B", "C"], band=[0.6, 0.7])
    names = ["A", "B", "C"]

    def draw(k, edge):  # every edge of the cycle at `edge`
        return [
            dict(draw=k, sigma=0.1, first="A", second="B", games=256, score=edge),
            dict(draw=k, sigma=0.1, first="B", second="C", games=256, score=edge),
            dict(draw=k, sigma=0.1, first="A", second="C", games=256, score=1 - edge),
        ]

    strong = [r for k, e in enumerate((0.6, 0.62, 0.58)) for r in draw(k, e)]
    assert search.robust_verdict(spec, tmp_path, strong, names)["pass"] is True
    flipped = [r for k, e in enumerate((0.6, 0.62, 0.45)) for r in draw(k, e)]
    assert search.robust_verdict(spec, tmp_path, flipped, names)["pass"] is False
    weak = [r for k, e in enumerate((0.53, 0.54, 0.52)) for r in draw(k, e)]
    assert search.robust_verdict(spec, tmp_path, weak, names)["pass"] is False
    wider = [dict(r, sigma=0.25) for r in strong]
    assert search.robust_verdict(spec, tmp_path, wider, names)["pass"] is None  # not the gate


def test_the_hit_radius_is_a_knob():
    rules = api.rules_of(GAME)
    base = set_handles()["la7"]
    bigger = handles_of(base, {"hit_radius": 2 * base["hit_radius"]}, rules)
    assert bigger["hit_radius"] == pytest.approx(2 * base["hit_radius"])
    assert knobs_of(bigger)["hit_radius"] == pytest.approx(2 * base["hit_radius"])
