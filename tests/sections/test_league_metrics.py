"""The league's metrics (league spec 7): per mode the live gauge and each entry's rating,
sampling share and score; the pool; games per update by opponent kind."""

import math

import league_cases as cases
import pytest

from bnb.league.league import ENTRY_KINDS

pytestmark = pytest.mark.usefixtures("global_generator")


@pytest.fixture(scope="module")
def rows(tmp_path_factory):
    import torch

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        trainer, league = cases.league(tmp_path_factory.mktemp("metrics"))
        out = []
        for _ in range(8):  # the league's side of an update, without learning
            trainer.collect()
            league.collected(trainer)
            trainer.ppo.environment_steps += trainer.steps_per_update
            out.append(league.updated(trainer))
        return out, league


def test_each_mode_reports_the_live_gauge_and_every_entry(rows):
    out, league = rows
    for row in out:
        table = row["pool/entries"]
        entries = [dict(zip(table["columns"], values, strict=True)) for values in table["rows"]]
        for mode in league.modes:
            assert math.isfinite(row[f"rating/live/{mode}"])  # the last finite value
            assert f"rating/live_stderr/{mode}" in row
            assert 0.0 <= row[f"pool/sampling_entropy/{mode}"] <= 1.0
            shares = [entry["sampling_share"] for entry in entries if entry["mode"] == mode]
            assert math.isclose(sum(shares), 1.0, abs_tol=1e-12)
            named = {entry["entry"] for entry in entries if entry["mode"] == mode}
            assert named <= set(league.entries)


def test_the_pool_and_the_games_by_kind_are_reported(rows):
    out, league = rows
    last = out[-1]
    assert last["pool/size"] == len(league.entries)
    assert last["pool/snapshots"] == len(league.snapshots) > 0
    assert last["pool/draining"] in (0, 1)
    shares = [
        row[f"pool/game_share/{kind}"]
        for row in out
        for kind in ENTRY_KINDS
        if f"pool/game_share/{kind}" in row
    ]
    assert shares
    for row in out:
        present = [row[name] for name in row if name.startswith("pool/game_share/")]
        assert not present or math.isclose(sum(present), 1.0, abs_tol=1e-12)
    scores = [value for row in out for key, value in row.items() if key.startswith("pool/score/")]
    assert scores and all(0.0 <= score <= 1.0 for score in scores)
