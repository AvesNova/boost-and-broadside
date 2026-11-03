"""The league's metrics (league spec 7): per mode the live gauge and each entry's rating,
sampling share and score; the pool; games per update by opponent kind."""

import math

import league_cases as cases
import pytest

pytestmark = pytest.mark.usefixtures("global_generator")


@pytest.fixture(scope="module")
def rows(tmp_path_factory):
    import torch

    with torch.random.fork_rng():
        torch.manual_seed(0)
        trainer, league = cases.league(tmp_path_factory.mktemp("metrics"))
        out = []
        for _ in range(8):  # the league's side of an update, without learning
            trainer.collect()
            league.collected(trainer)
            trainer.ppo.environment_steps += trainer.steps_per_update
            out.append(league.updated(trainer)["league"])
        return out, league


def test_each_mode_reports_the_live_gauge_and_every_entry(rows):
    out, league = rows
    for row in out:
        for mode in league.modes:
            assert math.isfinite(row[f"live_elo/{mode}"])  # the last finite value
            assert f"live_elo_stderr/{mode}" in row
            shares = [
                row[f"sampling_share/{mode}/{name}"]
                for name in league.entries
                if f"sampling_share/{mode}/{name}" in row
            ]
            assert math.isclose(sum(shares), 1.0, abs_tol=1e-12)
            assert all(
                f"rating/{mode}/{name}" in row
                for name in league.entries
                if f"sampling_share/{mode}/{name}" in row
            )


def test_the_pool_and_the_games_by_kind_are_reported(rows):
    out, league = rows
    last = out[-1]
    assert last["pool/entries"] == len(league.entries)
    assert last["pool/snapshots"] == len(league.snapshots) > 0
    assert len(last["pool/slots"]) == league.config.slots
    assert (
        sum(row[f"games/{kind}"] for row in out for kind in ("checkpoint", "average", "ladder")) > 0
    )
    scores = [value for row in out for key, value in row.items() if key.startswith("score/")]
    assert scores and all(0.0 <= score <= 1.0 for score in scores)
