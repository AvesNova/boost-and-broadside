"""The scripted agent's parameter search (`bench/scripted_search.py`), its parts on the CPU:
the bounds, the children, the stop rule, the resume and the mixed opponent seat. The search
itself runs on the GPU (the section notes record its run)."""

import json
import math
import pathlib
import sys

import pytest
import torch
from scripted_cases import CHARACTERISTICS, absent, observation, ship, value

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # `bench` is a top-level package of the repository
    sys.path.insert(0, str(ROOT))

from bench import scripted_search as search  # noqa: E402
from bnb.contracts import Obs  # noqa: E402
from bnb.controllers.scripted import Chaser, Scripted  # noqa: E402
from bnb.controllers.scripted_parameters import (  # noqa: E402
    NAMES,
    SCALE_FLOORS,
    Logistic,
    ScriptedParameters,
    pack,
)

# The mixed seat's test draws from torch's default generator: seeded and restored.
pytestmark = pytest.mark.usefixtures("global_generator")
DEFAULTS = ScriptedParameters()


def _logistics(parameters: ScriptedParameters) -> dict[str, Logistic]:
    return {name: getattr(parameters, name) for name in SCALE_FLOORS}


def test_children_stay_in_bounds_and_keep_every_direction():
    generator = torch.Generator().manual_seed(0)
    for _ in range(200):
        child = search.perturb(DEFAULTS, 1.0, generator)  # a wild step
        for name, shape in _logistics(child).items():
            floor = SCALE_FLOORS[name]
            assert floor <= shape.scale <= search.SCALE_CEILING * floor + 1e-9, name
            assert shape.centre > 0.0 and shape.rising == getattr(DEFAULTS, name).rising
        assert child.recovery_shield_fraction <= 0.9
        assert child.combat_radius_in_gun_range >= child.shoot_in_gun_range.centre
        ramp = child.boundary_steer_ramp_in_margin
        assert ramp.low < ramp.high and (ramp.at_low, ramp.at_high) == (0.0, 1.0)
    narrow = ScriptedParameters(turn_in_stall_angle=Logistic(0.1, 1e-3, True))
    assert search.clamp(narrow).turn_in_stall_angle.scale == SCALE_FLOORS["turn_in_stall_angle"]


def test_a_crossover_takes_each_field_from_a_parent():
    generator = torch.Generator().manual_seed(1)
    other = search.perturb(DEFAULTS, 0.5, generator)
    child = search.crossover(DEFAULTS, other, generator)
    for name in NAMES:
        assert getattr(child, name) in (getattr(DEFAULTS, name), getattr(other, name)), name
    assert child != DEFAULTS and child != other  # with 22 fields, both parents contribute


def test_the_next_generation_keeps_the_top_first():
    generator = torch.Generator().manual_seed(2)
    ranked = [search.perturb(DEFAULTS, 0.3, generator) for _ in range(6)]
    population = search.next_generation(ranked, 10, 3, 0.1, generator)
    assert len(population) == 10 and population[:3] == ranked[:3]
    assert all(child not in ranked for child in population[3:])


def test_the_search_stops_when_the_top_plateaus():
    assert not search.plateaued([0.0, 50.0, 80.0], 3, 10.0)
    assert not search.plateaued([0.0, 50.0, 80.0, 95.0], 3, 10.0)
    assert search.plateaued([0.0, 50.0, 80.0, 85.0], 3, 10.0) is False  # 85 > 0 + 10
    assert search.plateaued([100.0, 50.0, 80.0, 105.0], 3, 10.0)
    assert not search.plateaued([100.0, 50.0, 80.0, 115.0], 3, 10.0)


def test_a_search_resumes_from_its_log(tmp_path):
    generator = torch.Generator().manual_seed(3)
    candidates = [search.perturb(DEFAULTS, 0.3, generator) for _ in range(3)]
    log = tmp_path / "search.jsonl"
    with open(log, "w") as out:
        for generation, ratings in enumerate(([5.0, 9.0, 1.0], [2.0, 30.0, 40.0])):
            for index, (parameters, rating) in enumerate(zip(candidates, ratings, strict=True)):
                row = dict(generation=generation, index=index, rating=rating)
                out.write(json.dumps(dict(row, parameters=pack(parameters))) + "\n")
    start, ranked, tops, champion = search.resume(log)
    assert start == 2 and tops == [9.0, 40.0] and champion is None
    for expected, got in zip((candidates[2], candidates[1], candidates[0]), ranked, strict=True):
        assert pack(got) == pytest.approx(pack(expected), rel=1e-6)


def test_the_mixed_seat_plays_each_rows_controller():
    gun_range = value("gun_range", 0)
    target = ship(x=0.5 * gun_range, team_relative=1)
    one = observation([ship(), absent(team_relative=0)], [target, absent()])
    views = 6
    obs = Obs(
        ships=one.ships.expand(views, -1, -1),
        fields=None,
        globals=one.globals.expand(views, -1),
        zones=one.zones.expand(views, -1, -1),
    )
    choice = torch.tensor([0, 1, 0, 1, 1, 0])
    uniform = Scripted(CHARACTERISTICS, 2, temperature=math.inf)
    mixed = search.Mixed([uniform, Chaser(CHARACTERISTICS, 2)], choice)
    assert set(mixed.reads) == {"ships", "zones", "globals"}
    torch.manual_seed(0)
    action = mixed.step(
        mixed.init_carry(views, obs.ships.device), obs, None, torch.arange(obs.ships.shape[0])
    )[0].packed()
    chased = (
        Chaser(CHARACTERISTICS, 2)
        .step(None, obs, None, torch.arange(obs.ships.shape[0]))[0]
        .packed()
    )
    assert torch.equal(action[choice == 1], chased[choice == 1])


def test_mixed_lineups_draw_from_the_pool_and_mirror_the_teams():
    generator = torch.Generator().manual_seed(4)
    lineup = search.mixed_lineups(4000, 5, (2, 7, 11), generator)
    assert lineup.shape == (4000, 10) and lineup.dtype == torch.int32
    assert torch.equal(lineup[:, :5], lineup[:, 5:])
    counts = torch.bincount(lineup.flatten().long(), minlength=12)
    assert set(torch.nonzero(counts).flatten().tolist()) == {2, 7, 11}
    assert counts[[2, 7, 11]].min() > 0.3 * lineup.numel() / 3  # about even
    assert (lineup[:, :5] != lineup[:, :1]).any(1).float().mean() > 0.9  # mixed teams


def test_the_type_pool_names_types_and_holds_some_out():
    names = list(search.api.type_names("real-scale"))
    assert search.type_pool("real-scale", None, None) == tuple(range(len(names)))
    pool = search.type_pool("real-scale", None, "Yongma,vizia")
    assert len(pool) == len(names) - 2 and names.index("vizia") not in pool and 0 not in pool
    assert search.type_pool("real-scale", "vizia,Yongma", None) == (names.index("vizia"), 0)
    with pytest.raises(SystemExit):
        search.type_pool("real-scale", "Sopwith Camel", None)


def _champion_rows(generation, scores, champion):
    return [
        dict(
            generation=generation,
            index=index,
            rating=100.0 * index,
            scores={search.CHAMPION: [score]},
            champion=champion,
            parameters=pack(DEFAULTS),
        )
        for index, score in enumerate(scores)
    ]


def test_the_top_candidate_is_promoted_once_the_median_beats_the_champion(tmp_path):
    assert search.promoted(_champion_rows(3, [0.5, 0.6, 0.7], "defaults"), 0.75) is None
    assert search.promoted(_champion_rows(3, [0.7, 0.8, 0.9], "defaults"), 0.75) == "3:2"
    assert search.promoted(_champion_rows(3, [0.7, 0.8, 0.9], "defaults"), None) is None
    log = tmp_path / "search.jsonl"
    rows = _champion_rows(0, [0.4, 0.5, 0.6], "defaults") + _champion_rows(
        1, [0.7, 0.8, 0.9], "defaults"
    )
    log.write_text("".join(json.dumps(row) + "\n" for row in rows))
    start, _, tops, champion = search.resume(log, "median", 0.75)
    assert start == 2 and champion == "1:2" and tops == []  # promoted: the history restarts
    start, _, tops, champion = search.resume(log, "median", 0.95)
    assert champion == "defaults" and tops == [100.0, 100.0]
    assert pack(search.champion_of("1:2", "real_scale_deathmatch", log)) == pack(DEFAULTS)
    assert (
        search.champion_of("defaults", "real_scale_deathmatch", log)
        == (search.MODE_DEFAULTS.sets["real_scale_deathmatch"])
    )
