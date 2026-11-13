"""The real-scale balancing's round script (`bench/balance_round.py`: its search summary)
and the real-scale sets' file (`bnb.controllers.scripted_defaults`): the parts that run on
the CPU. The balance rule's moves were of the researched types, which left the game on
2026-11-13 (their moves: docs/explanation/game/real-scale-balance/moves.json)."""

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
