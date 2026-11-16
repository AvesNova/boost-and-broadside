"""The balance measurement (`bench.balance_sweep`) on the CPU uncompiled, at tiny sizes: the
switches (no fog, no fields, seeded spawns), the pairs' card table, the kill share and a
short sweep of a candidate and a pair."""

import dataclasses
import json
import math

import torch

from bench.balance_sweep import (
    BALANCE,
    CONDITIONS,
    as_pair,
    balance_loop,
    balance_summary,
    card_table,
    fog_of,
    mode_parameters,
    ratio,
    sweep,
)
from bench.game_loop import FILLER, Group, base_card
from bnb.game import api

SHORT = dataclasses.replace(BALANCE, seconds=1.0)


def small_loop(condition="deathmatch-5v5", environments=4, variants=2):
    return balance_loop(CONDITIONS[condition], environments, variants, SHORT, "cpu", False)


def test_the_balance_settings_fog_by_mode_and_turn_fields_off():
    state, _, _, _ = small_loop("elimination-5v5").start(
        [Group(0, "energy v turn", FILLER, FILLER, 4)]
    )
    assert bool(state.match.fogged.all())
    loop = small_loop()
    state, _, _, _ = loop.start([Group(0, "energy v turn", FILLER, FILLER, 4)])
    assert not bool(state.match.fogged.any())
    assert loop.game.num_fields == 0 and not bool(state.fields.present.any())


def test_the_fog_switch_turns_fog_off():
    settings = dataclasses.replace(SHORT, fog=False)
    loop = balance_loop(CONDITIONS["deathmatch-5v5"], 4, 2, settings, "cpu", False)
    state, observation, _, _ = loop.start([Group(0, "energy v turn", FILLER, FILLER, 4)])
    assert not bool(state.match.fogged.any())
    from bnb.contracts.obs import SHIP

    alive = observation.ships[..., SHIP["alive"]] > 0.5
    assert bool((observation.ships[..., SHIP["visible"]] > 0.5)[alive].all())


def test_deathmatch_measures_300_seconds():
    loop = balance_loop(CONDITIONS["deathmatch-5v5"], 2, 1, BALANCE, "cpu", False)
    assert loop.ticks * loop.rules.tick_length == 300.0


def test_the_game_defaults_keep_fog_and_fields():
    """The switches are the measurement's: the presets keep their fields."""
    assert api.PRESETS["real_scale_deathmatch"].num_fields > 0
    assert BALANCE.fog is None and BALANCE.fields is False


def test_fog_follows_the_mode():
    """Deathmatch is measured without fog, elimination and Frontline with it."""
    for name, condition in CONDITIONS.items():
        assert fog_of(condition, BALANCE) == (condition.mode != "deathmatch"), name
    assert fog_of(CONDITIONS["deathmatch-5v5"], dataclasses.replace(BALANCE, fog=True))


def test_seeded_spawns_repeat_by_game_index():
    loop = small_loop()

    def spawns(seed, first_game=0):
        loop.spawn_seed = seed
        group = Group(0, "energy v turn", FILLER, FILLER, 4, first_game=first_game)
        shown = api.display(loop.start([group])[0])
        return torch.stack([shown["ships.x"], shown["ships.y"]], -1)

    first = spawns(3)
    assert torch.equal(first, spawns(3))
    assert not torch.allclose(first, spawns(4))
    assert torch.equal(spawns(3, first_game=2)[:2], first[2:])


def test_a_pair_table_holds_any_card_against_any_card():
    table = card_table(
        [dict(ship="jujak", factors={"max_shield": 2.0}, doctrine={})],
        [dict(ship="jujak", factors={}, doctrine={})],
    )
    base = base_card("cpu")  # rows yongma, jujak
    assert math.isclose(float(table.max_shield[0]), 2.0 * float(base.max_shield[1]), rel_tol=1e-6)
    assert float(table.max_shield[1]) == float(base.max_shield[1])
    pair = as_pair({"name": "p", "a": {"ship": "jujak"}})
    assert pair["a"]["ship"] == "jujak" and pair["b"]["ship"] == "jujak" and not pair["mixed"]
    candidate = as_pair({"name": "c", "card": {"turn": {"max_shield": 0.5}}})
    assert candidate["a"]["ship"] == "yongma" and candidate["b"]["factors"] == {"max_shield": 0.5}


def test_the_kill_share_pools_the_kills():
    out = dict(
        match_kills=torch.tensor([[3.0, 1.0], [0.0, 0.0], [1.0, 3.0], [2.0, 0.0]]),
        match_deaths=torch.tensor([[1.0, 3.0], [1.0, 0.0], [3.0, 1.0], [0.0, 2.0]]),
        match_ticks=torch.full((4,), 1800.0),
    )
    row = balance_summary(out, 1.0 / 30.0)
    assert math.isclose(row["kill_share_a"][0], 6.0 / 10.0)
    assert math.isclose(row["point_share_a"][0], 6.0 / 11.0)
    assert row["kills_per_minute"] == [1.5, 1.0]  # 6 and 4 kills over 4 minutes
    assert math.isclose(row["match_seconds"], 60.0)
    by_minute = torch.zeros((4, 3, 2))
    by_minute[0, 0] = torch.tensor([1.0, 0.0])  # minute 0: A 1, B 0
    by_minute[0, 2] = torch.tensor([0.0, 1.0])  # minute 2: B 1
    shares = balance_summary(out | dict(kills_by_minute=by_minute), 1.0 / 30.0)
    assert [s for s, _ in shares["kill_share_by_minute"]] == [1.0, 1.0, 0.5]
    value, error = ratio(torch.tensor([1.0, 1.0]), torch.tensor([2.0, 2.0]))
    assert value == 0.5 and error == 0.0


def test_a_short_sweep_writes_a_line_per_entry(tmp_path, global_generator):
    loop = small_loop(environments=4, variants=2)
    out = tmp_path / "out.jsonl"
    entries = [
        {"name": "candidate", "card": {"energy": {"max_shield": 0.9}}},
        {"name": "pair", "a": {"ship": "jujak"}, "b": {"ship": "yongma"}},
    ]
    parameters = mode_parameters(CONDITIONS["deathmatch-5v5"])
    sweep(loop, entries, out, 2, 2, 0, 2, "deathmatch-5v5", parameters, log=lambda line: None)
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert [line["name"] for line in lines] == ["candidate", "pair"]
    assert set(lines[0]["jobs"]) == {"energy v turn", "mixed", "mixed reversed"}
    assert set(lines[1]["jobs"]) == {"energy v turn"}
    job = lines[0]["jobs"]["energy v turn"]
    assert job["games"] == 2 and 0.0 <= job["kill_share_a"][0] <= 1.0
    assert lines[0]["fog"] is False and lines[0]["fields"] is False


def test_the_static_no_fog_path_sees_what_the_flag_sees():
    """`perceive(fog=False)` skips the sight test; on unfogged games (with fields, so the
    sight test has work) it gives the flag's observation."""
    game = api.PRESETS["real_scale_deathmatch"]
    rules = api.rules_of(game)
    types = api.ship_types(ship_set=game.ship_set)
    generator = torch.Generator().manual_seed(0)
    uniforms = api.uniforms(rules, 4, generator=generator)
    state = api.reset(rules, types, uniforms, torch.zeros(4, dtype=torch.bool))
    _, flagged, _ = api.perceive(rules, state)
    _, skipped, _ = api.perceive(rules, state, fog=False)
    assert torch.equal(flagged.ships, skipped.ships)
