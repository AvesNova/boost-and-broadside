"""The static Elo-curve design (`bnb.league.unified.static_design`, `static_target`) and the
engine's streaming window, on the CPU."""

import json
import math
import pathlib
import random
import types

import numpy

from bnb.league.matches import Player
from bnb.league.ratings import SCALE
from bnb.league.unified import static_design
from bnb.league.unified.cells import Cell, Condition, Outcome
from bnb.league.unified.engine import BucketEngine
from bnb.league.unified.planner import Planner
from bnb.league.unified.static_target import StaticEloTarget, static_elo

EL = "real_scale_elimination"


def chain_players(bc: int, rl: int, run: str = "dry") -> list[Player]:
    players = [Player("scripted")]
    for stage, count in (("bc", bc), ("rl", rl)):
        for step in range(1, count + 1):
            path = pathlib.Path(
                f"/runs/{run}-{stage}/policies/step_{step * 1000}/exponential_average"
            )
            players.append(Player(f"{stage}_{step}", policy=path))
    return players


def test_prior_pairs_and_order_follow_the_chain():
    found, other = static_design.checkpoints(chain_players(3, 3))
    assert other == ["scripted"]
    chains = static_design.chains_of(found)
    order = static_design.design_order(chains)
    assert order == ["bc_1", "bc_2", "bc_3", "rl_1", "rl_2", "rl_3"]
    prior = static_design.prior_ratings(chains)
    assert prior["rl_1"] == 0.0 and prior["bc_3"] == -100.0 and prior["rl_3"] == 200.0
    pairs = static_design.candidate_pairs(chains, "scripted")
    assert ("bc_3", "rl_1") in pairs and ("bc_2", "rl_1") in pairs  # across the transition
    assert ("bc_1", "bc_3") in pairs and ("bc_1", "rl_1") not in pairs
    assert ("scripted", "rl_1") in pairs


def test_a_optimal_meets_every_goal_connected_and_existing_games_reduce_it():
    found, _ = static_design.checkpoints(chain_players(6, 6))
    chains = static_design.chains_of(found)
    names = ["scripted"] + static_design.design_order(chains)
    index = {n: i for i, n in enumerate(names)}
    prior = static_design.prior_ratings(chains)
    theta = numpy.array([0.0] + [prior[n] / SCALE for n in names[1:]])
    pairs = [(index[a], index[b]) for a, b in static_design.candidate_pairs(chains, "scripted")]
    design = static_design.a_optimal(len(names), pairs, theta, 60.0)
    assert design.rank == len(names) - 1
    assert design.stderr[1:].max() <= 60.0 * (1 + 1e-5) and design.games > 0
    # the same design's games already played: nothing more is needed
    first = numpy.array([a for a, _ in pairs])
    second = numpy.array([b for _, b in pairs])
    held = static_design.precision(
        len(names), first, second, design.counts * static_design.pair_weights(theta, first, second)
    )
    again = static_design.a_optimal(len(names), pairs, theta, 60.0, existing=held)
    assert again.games == 0


def test_streaming_schedule_plays_every_cell_with_both_players_seated():
    order = [f"p{i}" for i in range(20)]
    cells = {(order[i], order[j]): 5 for i in range(20) for j in range(i + 1, min(i + 3, 20))}
    cells.update({("scripted", name): 3 for name in order[::4]})
    steps = static_design.streaming_schedule(order, cells, 5)
    played = sum(step["games"] for step in steps)
    assert played == sum(cells.values())
    for step in steps:
        assert len(step["seated"]) <= 5


def simulate(target: StaticEloTarget, truth: dict[str, float], seed: int = 0) -> Planner:
    planner = Planner([target])
    rng = random.Random(seed)
    for _ in range(100_000):
        chosen = planner.choose(5, EL)
        if chosen is None:
            break
        cell = chosen[0]
        p = 1.0 / (1.0 + math.exp(-(truth[cell.first] - truth[cell.second]) / SCALE))
        planner.finished(cell, Outcome(1.0 if rng.random() < p else 0.0, seconds=30.0))
    return planner


def test_two_stages_then_done_with_every_error_reported():
    players = chain_players(4, 6)
    target = static_elo(
        "curves/elimination",
        Condition(EL, 5, 5),
        [p.name for p in players],
        players,
        80.0,
        "scripted",
    )
    truth = {"scripted": 0.0} | {n: 60.0 * (i - 4) for i, n in enumerate(target.order)}
    planner = simulate(target, truth)
    assert target.stage == 3 and target.done() and planner.done()
    assert set(target.designs) == {1, 2}
    report = target.report()
    assert report["design"]["kind"] == "static" and report["stopped"] in ("met", "design")
    assert planner.choose(5, EL) is None  # no overtime, no filler: the counts are the design
    # every played pair is a candidate (no per-game planning beyond the design)
    played = {(a, b) for a, b in zip(*numpy.nonzero(numpy.triu(target.games)), strict=True)}
    assert played <= set(target.pairs) | {(b, a) for a, b in target.pairs}


def test_prior_games_reduce_the_stage_one_counts():
    players = chain_players(3, 4)
    names = [p.name for p in players]

    def fresh():
        return static_elo("c", Condition(EL, 5, 5), names, players, 80.0, "scripted")

    empty = fresh()
    empty.design()
    loaded = fresh()
    for _ in range(200):
        loaded.record(Cell(Condition(EL, 5, 5), "rl_1", "rl_2"), Outcome(0.5), pending=False)
    loaded.design()
    assert loaded.designs[1].games < empty.designs[1].games


def test_a_champion_of_a_snapshots_weights_is_not_a_second_player():
    players = chain_players(2, 3)
    players.append(Player("champion", policy=players[-1].policy))
    target = static_elo(
        "c", Condition(EL, 5, 5), [p.name for p in players], players, 80.0, "scripted"
    )
    assert "champion" not in target.players


def test_stream_window_unloads_done_residents_and_loads_the_next_in_order():
    swapped = []

    class Seat:
        lanes = [[1, 2]]

        def swap(self, run, lane, index):
            swapped.append((lane, index))
            self.lanes[run][lane] = index

    condition = Condition(EL, 5, 5)
    values = {Cell(condition, "b", "c"): 1.0, Cell(condition, "scripted", "d"): 1.0}
    engine = types.SimpleNamespace(
        bucket=5,
        planner=types.SimpleNamespace(values=lambda overtime=False: values, buckets=(5, 10, 20)),
        index={"scripted": 0, "a": 1, "b": 2, "c": 3, "d": 4},
        network=[False, True, True, True, True],
        group_of=[None, 0, 0, 0, 0],
        runs=[(0, [1, 2], 100)],
        seat=Seat(),
        capacity_of={1: 100, 2: 100},
        leaving=set(),
        in_use=[0, 2, 0, 0, 0],
        swaps=0,
        stream_rank={0: 9, 1: 0, 2: 1, 3: 2, 4: 3},
    )
    engine._working = types.MethodType(BucketEngine._working, engine)
    BucketEngine._stream(engine)
    assert engine.leaving == {1}  # "a" has nothing left: it drains
    assert swapped == []  # no lane free yet
    engine.in_use[1] = 0
    BucketEngine._stream(engine)
    # "a"'s lane is emptied, then seats "c" (next in order with games), before "d"
    assert swapped == [(0, None), (0, 3)] and 3 in engine.capacity_of


def test_spec_static_curves_load_with_the_anchor_and_the_chain(tmp_path):
    from bnb.league.unified.run import load_spec, stream_order

    lines = ["[players]"]
    for stage in ("bc", "rl"):
        for step in (1000, 2000, 3000):
            where = tmp_path / f"dry-{stage}/policies/step_{step}/exponential_average"
            where.mkdir(parents=True)
            record = {"files": {"weights.safetensors": f"{stage}{step}"}}
            (where / "checkpoint.json").write_text(json.dumps(record))
            lines.append(f'{stage}_{step} = "{where}"')
    lines.append(f'champion = "{where}"')
    names = ["scripted"] + [f"{s}_{n}" for s in ("bc", "rl") for n in (1000, 2000, 3000)]
    lines += ["", "[[elo]]", f'preset = "{EL}"', 'design = "static"', "reach = 3", "pilot = 2.0"]
    lines.append(f"players = {names + ['champion']}".replace("'", '"'))
    (tmp_path / "spec.toml").write_text("\n".join(lines) + "\n")
    spec = load_spec(tmp_path / "spec.toml")
    (target,) = spec.targets
    assert isinstance(target, StaticEloTarget) and target.pilot == 2.0
    assert target.players == names  # the champion: the same weights as rl_3000
    pairs = {(target.players[a], target.players[b]) for a, b in target.pairs}
    assert ("bc_1000", "rl_1000") in pairs and ("bc_1000", "rl_2000") not in pairs  # reach 3
    assert stream_order(Planner(spec.targets)) == names[1:]


def test_fixed_windows_overlap_and_each_pair_in_its_first_window():
    order = [f"c{i}" for i in range(20)]
    held = static_design.windows(order, 8, 4)
    assert [w[0] for w in held] == ["c0", "c4", "c8", "c12"] and held[-1][-1] == "c19"
    pairs = [("c5", "c7"), ("c6", "c9"), ("scripted", "c13"), ("c17", "c19")]
    where = static_design.assign_windows(held, pairs, {"scripted"})
    assert where == {("c5", "c7"): 0, ("c6", "c9"): 1, ("scripted", "c13"): 2, ("c17", "c19"): 3}
    try:
        static_design.assign_windows(held, [("c0", "c9")], {"scripted"})
    except ValueError:
        pass
    else:
        raise AssertionError("a pair no window holds must raise")


def test_next_window_seats_in_the_spare_lanes_while_the_leavers_drain():
    swapped = []

    class Seat:
        lanes = [[1, 2, 3, None]]  # a window of 3 and one spare lane (overlap 2)

        def swap(self, run, lane, index):
            swapped.append((lane, index))
            self.lanes[run][lane] = index

    condition = Condition(EL, 5, 5)
    values = {Cell(condition, "c", "d"): 1.0}
    names = ["scripted", "a", "b", "c", "d"]
    engine = types.SimpleNamespace(
        bucket=5,
        planner=types.SimpleNamespace(
            values=lambda overtime=False: values, buckets=(5, 10, 20), targets=[]
        ),
        index={n: i for i, n in enumerate(names)},
        network=[False, True, True, True, True],
        window_sets=[["a", "b", "c"], ["b", "c", "d"]],
        window=0,
        window_stages=(),
        window_tail=None,
        window_closed=False,
        seat=Seat(),
        runs=[(0, [1, 2, 3, None], 100)],
        capacity_of={1: 100, 2: 100, 3: 100},
        leaving=set(),
        in_use=[0, 2, 0, 0, 0],  # "a" still plays its last game
        stream_rank={1: 0, 2: 1, 3: 2, 4: 3},
        swaps=0,
        environments=0,
        current=[],
        next=[],
        log=None,
    )
    for name in (
        "_stages",
        "_window_cells",
        "_window_of",
        "_in_window",
        "_next_window",
        "_window_advance",
    ):
        setattr(engine, name, types.MethodType(getattr(BucketEngine, name), engine))
    for name in ("_window_record", "_window_opened", "_say", "_set_next", "_write"):
        setattr(engine, name, lambda *a, **k: None)
    engine.drain = lambda: 0
    BucketEngine._window_step(engine)
    # "d" comes in at once in the spare lane while "a" (dropped) drains
    assert swapped == [(3, 4)] and engine.window == 1 and engine.leaving == {1}
    engine.in_use[1] = 0
    BucketEngine._window_step(engine)
    assert swapped[-1] == (0, None) and 1 not in engine.capacity_of and not engine.leaving


def test_spec_windows_must_overlap_by_the_reach(tmp_path):
    from bnb.league.unified.run import load_spec

    lines = ["[players]"]
    for step in (1000, 2000, 3000, 4000):
        where = tmp_path / f"dry-rl/policies/step_{step}/exponential_average"
        where.mkdir(parents=True)
        record = {"files": {"weights.safetensors": f"rl{step}"}}
        (where / "checkpoint.json").write_text(json.dumps(record))
        lines.append(f'rl_{step} = "{where}"')
    names = ["scripted"] + [f"rl_{n}" for n in (1000, 2000, 3000, 4000)]
    body = ["", "[[elo]]", f'preset = "{EL}"', 'design = "static"', "reach = 2"]
    body.append(f"players = {names}".replace("'", '"'))
    for overlap, fits in ((1, False), (2, True)):
        head = [f"windows = {{size = 3, overlap = {overlap}}}"]
        (tmp_path / "spec.toml").write_text("\n".join(head + lines + body) + "\n")
        try:
            spec = load_spec(tmp_path / "spec.toml")
        except ValueError:
            assert not fits
        else:
            assert fits and spec.windows == (3, overlap, None)


def test_a_new_stage_rechooses_the_parked_next_games():
    from bnb.league.unified.engine import Slot

    condition = Condition(EL, 5, 5)
    target = types.SimpleNamespace(stage=1, name="t", designs={})
    chosen, written = [], []
    real = Slot(Cell(condition, "a", "scripted"), 0)
    engine = types.SimpleNamespace(
        planner=types.SimpleNamespace(targets=[target]),
        seen_stages=(1,),
        environments=3,
        next=[Slot(None, 0, parked=True), real, None],
        log=None,
    )
    engine._stages = types.MethodType(BucketEngine._stages, engine)
    engine._say = lambda text: None
    engine.drain = lambda: 0
    engine._choose = lambda e: chosen.append(e) or real
    engine._set_next = lambda e, slot: engine.next.__setitem__(e, slot)
    engine._write = lambda environments, slots: written.extend(environments)
    assert BucketEngine._stage_step(engine) == 0 and not chosen  # no new stage: nothing
    target.stage = 2
    BucketEngine._stage_step(engine)
    assert chosen == [0, 2] and written == [0, 2] and engine.seen_stages == (2,)
