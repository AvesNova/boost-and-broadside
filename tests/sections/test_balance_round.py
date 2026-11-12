"""The real-scale balancing's round script (`bench/balance_round.py`) and the real-scale
sets' file (`bnb.controllers.scripted_defaults`): the parts that run on the CPU."""

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # `bench`
from bench import balance_round  # noqa: E402
from bnb.controllers import scripted_defaults  # noqa: E402
from bnb.controllers.scripted_parameters import (  # noqa: E402
    REAL_SCALE_MODES,
    ScriptedParameters,
    pack,
)


def test_a_search_summary_picks_the_best_survivor_of_the_last_generation(tmp_path):
    log = tmp_path / "search.jsonl"
    rows = []
    for generation in range(2):
        for index in range(10):
            rating = 10.0 * index if generation else 0.0
            rows.append(
                dict(
                    generation=generation,
                    index=index,
                    rating=rating,
                    stderr=30.0,
                    scores={"champion": [0.4 + 0.01 * index]},
                    presets=["real_scale_deathmatch"],
                    champion="defaults",
                )
            )
    log.write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary = balance_round.summary_of(log)
    assert summary["generations"] == 2
    assert summary["candidate"] == f"1:{balance_round.KEEP - 1}"  # not the better child 9
    last = summary["per_generation"][-1]
    assert last["top"] == 90.0 and last["median"] == 40.0
    assert last["median_score"] == pytest.approx(0.45)  # the upper middle of ten
    assert balance_round.generations_done(log) == 2
    assert balance_round.generations_done(tmp_path / "absent.jsonl") == 0


def test_the_real_scale_sets_round_trip_through_their_file(tmp_path):
    path = tmp_path / "sets.json"
    path.write_text(scripted_defaults.REAL_SCALE.read_text())
    stored = scripted_defaults.read_real_scale(path)
    assert set(stored) == set(REAL_SCALE_MODES)
    for mode, entry in stored.items():
        parameters = scripted_defaults.from_named(entry["parameters"])
        assert pack(parameters) == pack(scripted_defaults.MODE_DEFAULTS.sets[mode])
    scripted_defaults.write_real_scale("real_scale_frontline", ScriptedParameters(), "a test", path)
    again = scripted_defaults.read_real_scale(path)
    assert again["real_scale_frontline"]["source"] == "a test"
    assert again["real_scale_deathmatch"] == stored["real_scale_deathmatch"]
    written = scripted_defaults.from_named(again["real_scale_frontline"]["parameters"])
    assert pack(written) == pytest.approx(pack(ScriptedParameters()), rel=5e-3)


def _round(tmp_path, monkeypatch, shield_spearman, elo_of):
    """A round's tournament outputs in a temporary docs directory: every type rated by
    `elo_of(name)` (+- 20), the shield correlated `shield_spearman` with the Elo."""
    from bnb.game import api, real_scale

    monkeypatch.setattr(balance_round, "DOCS", tmp_path / "docs")
    monkeypatch.setattr(real_scale, "BALANCE_FILE", tmp_path / "balance.json")
    docs, _ = balance_round.paths(1)
    docs.mkdir(parents=True)
    rated = [
        dict(type=name, elo=elo_of(name), stderr=20.0) for name in api.type_names("real-scale")
    ]
    (docs / "tournament.json").write_text(json.dumps(dict(ratings=rated)))
    correlations = {"max_shield": {"spearman": shield_spearman, "pearson": shield_spearman}}
    (docs / "analysis.json").write_text(json.dumps(dict(correlations=correlations)))
    return real_scale


def test_the_rule_lowers_the_shield_step_and_lifts_an_invented_outlier(tmp_path, monkeypatch):
    real_scale = _round(
        tmp_path,
        monkeypatch,
        0.55,
        lambda name: {"Firecracker": 400.0, "P-47D": 300.0}.get(name, 1000.0),
    )
    proposal = balance_round.propose(1)
    moves = real_scale.balance_moves()
    assert moves["_rules"]["shield_per_rating"]["value"] == pytest.approx(0.15)
    assert moves["_rules"]["shield_per_rating"]["round"] == 2
    from bnb.game import ships

    shield = {ship.key: ship.handles["max_shield"] for ship in ships.load("real-scale")}
    assert moves["firecracker"]["max_shield"]["value"] == min(shield["firecracker"] + 20.0, 160.0)
    assert "p47d" not in moves  # a researched type is never moved on its own
    assert not proposal["stop"]


def test_the_rule_stops_when_nothing_fires(tmp_path, monkeypatch):
    real_scale = _round(tmp_path, monkeypatch, 0.2, lambda name: 1000.0)
    proposal = balance_round.propose(1)
    assert proposal["stop"] and proposal["moves"] == []
    assert real_scale.balance_moves() == {}
