"""Tests for the post-training calibration mode's pure logic."""

import json
from pathlib import Path

import numpy as np
import pytest

from boost_and_broadside.artifacts import (
    ArtifactIncomplete,
    ArtifactRecipe,
    ArtifactStore,
    Invocation,
    load_artifact,
)
from boost_and_broadside.config import EloCalibrateConfig, EnvConfig, ShipConfig
from boost_and_broadside.config.defaults import MODEL_CONFIG
from boost_and_broadside.modes.elo_calibrate import (
    SCRIPTED_ANCHOR_ELO,
    calibrate_live_curve,
    run_elo_calibrate_mode,
)
from boost_and_broadside.train.rl.bradley_terry import win_probability


def _record(update: int, step: int, counts: dict, live: float = 0.0, avg: float = 0.0) -> dict:
    return {"update": update, "global_step": step, "live": live, "avg": avg, "counts": counts}


class TestCalibrateLiveCurve:
    def test_recovers_the_rating_that_explains_the_record(self):
        """The live policy's rating comes from its own results, not from whatever
        the in-training filter had drifted to."""
        opponents = {"random": 0.0, "ckpt_1": 500.0}
        games = 4_000
        truth = 300.0
        counts = {}
        for label, rating in opponents.items():
            probability = win_probability(truth, rating)
            counts[label] = [probability * games, (1 - probability) * games, 0]
        curve = calibrate_live_curve([_record(1, 100, counts, live=-999.0)], opponents)
        assert curve[0]["live_calibrated"] == pytest.approx(truth, abs=5.0)
        assert curve[0]["live_elo"] == -999.0  # untouched, for comparison

    def test_ignores_opponents_with_no_calibrated_rating(self):
        counts = {"random": [90, 10, 0], "mystery_agent": [50, 50, 0]}
        curve = calibrate_live_curve([_record(1, 100, counts)], {"random": 0.0})
        assert curve[0]["games"] == 100

    def test_avg_is_excluded_from_the_live_fit(self):
        """The averaged policy moves every update, so it cannot serve as a fixed
        reference for the live policy that update."""
        counts = {"random": [90, 10, 0], "avg": [500, 500, 0]}
        curve = calibrate_live_curve([_record(1, 100, counts)], {"random": 0.0, "avg": 700.0})
        assert curve[0]["games"] == 100, "avg games must not enter the live fit"

    def test_avg_is_rated_against_the_calibrated_live_policy(self):
        """Second stage: avg's only opponent is live, so it is placed once live is."""
        counts = {"random": [990, 10, 0], "avg": [500, 500, 0]}
        curve = calibrate_live_curve([_record(1, 100, counts)], {"random": 0.0})
        # An even record against live puts avg at live's rating.
        assert curve[0]["avg_calibrated"] == pytest.approx(curve[0]["live_calibrated"], abs=1.0)

    def test_avg_record_is_inverted_from_the_live_perspective(self):
        """Counts are stored as live's wins/losses; avg beating live must raise
        avg above live, not sink it."""
        counts = {"random": [990, 10, 0], "avg": [100, 900, 0]}  # live loses to avg
        curve = calibrate_live_curve([_record(1, 100, counts)], {"random": 0.0})
        assert curve[0]["avg_calibrated"] > curve[0]["live_calibrated"]

    def test_updates_without_usable_counts_are_dropped(self):
        history = [_record(1, 10, {}), _record(2, 20, {"random": [5, 5, 0]})]
        curve = calibrate_live_curve(history, {"random": 0.0})
        assert [point["update"] for point in curve] == [2]

    def test_ties_do_not_enter_the_fit(self):
        """Draws are excluded from the likelihood, so adding them to a record
        must not move the rating it implies."""
        ratings = {"random": 0.0}
        without = calibrate_live_curve([_record(1, 10, {"random": [800, 200, 0]})], ratings)
        with_ties = calibrate_live_curve([_record(1, 10, {"random": [800, 200, 500]})], ratings)
        assert without[0]["live_calibrated"] == pytest.approx(with_ties[0]["live_calibrated"])

    def test_drawing_every_game_matches_the_opponents_rating(self):
        """Drawing every game against a known opponent is exactly the evidence
        that you are that opponent's equal, and half-win scoring says so. This is
        the case decisive-only cannot answer at all: with no decisive games it
        has nothing to fit, and reports the rating as undefined."""
        opponent = {"stockfish": 3_600.0}
        counts = {"stockfish": [0, 0, 1_000]}
        half = calibrate_live_curve([_record(1, 10, counts)], opponent, "half_win")
        assert half[0]["live_calibrated"] == pytest.approx(3_600.0, abs=1.0)

        decisive = calibrate_live_curve([_record(1, 10, counts)], opponent, "decisive")
        assert np.isnan(decisive[0]["live_calibrated"])

    def test_half_win_extracts_more_information_from_a_draw_heavy_record(self):
        """Draws carry information about parity. Dropping them throws it away,
        which is most costly exactly where draws are common."""
        ratings = {"random": 0.0}
        record = _record(1, 10, {"random": [2794, 10, 1120]})
        half = calibrate_live_curve([record], ratings, "half_win")[0]
        decisive = calibrate_live_curve([record], ratings, "decisive")[0]
        assert half["live_stderr"] < decisive["live_stderr"] / 4.0

    def test_half_win_mode_counts_draws_as_half(self):
        """Under the half-win convention a drawn game is half a win and half a
        loss, so a record of pure draws sits level with its opponent."""
        curve = calibrate_live_curve(
            [_record(1, 10, {"random": [0, 0, 400]})], {"random": 0.0}, tie_mode="half_win"
        )
        assert curve[0]["live_calibrated"] == pytest.approx(0.0, abs=1.0)

    def test_half_win_and_decisive_disagree_exactly_where_draws_are(self):
        """A tie-free record is identical under both conventions; adding draws
        pulls the half-win rating toward the opponent and leaves the other put."""
        ratings = {"random": 0.0}
        clean = {"random": [900, 100, 0]}
        drawn = {"random": [900, 100, 1000]}
        decisive_clean = calibrate_live_curve([_record(1, 10, clean)], ratings)[0]
        decisive_drawn = calibrate_live_curve([_record(1, 10, drawn)], ratings)[0]
        half_clean = calibrate_live_curve([_record(1, 10, clean)], ratings, "half_win")[0]
        half_drawn = calibrate_live_curve([_record(1, 10, drawn)], ratings, "half_win")[0]

        assert decisive_clean["live_calibrated"] == pytest.approx(decisive_drawn["live_calibrated"])
        assert half_clean["live_calibrated"] == pytest.approx(decisive_clean["live_calibrated"])
        assert 0.0 < half_drawn["live_calibrated"] < half_clean["live_calibrated"]

    def test_a_clean_sweep_is_reported_as_unbounded(self):
        curve = calibrate_live_curve([_record(1, 10, {"random": [100, 0, 0]})], {"random": 0.0})
        assert not np.isfinite(curve[0]["live_calibrated"])
        assert not np.isfinite(curve[0]["live_stderr"])


def _stored_result(reference: str) -> dict:
    """A minimal persisted calibration: random < scripted < ckpt_100 by wins."""
    labels = ["random", "scripted", "ckpt_100"]
    wins = [
        [0.0, 10.0, 5.0],
        [190.0, 0.0, 60.0],
        [195.0, 140.0, 0.0],
    ]
    ties = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    return {
        "run": "test-run",
        "players": [
            {"label": "random", "live_elo": 0.0, "global_step": 0},
            {"label": "scripted", "live_elo": None, "global_step": None},
            {"label": "ckpt_100", "live_elo": 1500.0, "global_step": 100},
        ],
        "player_labels": labels,
        "wins_matrix": wins,
        "ties_matrix": ties,
        "batches": [
            {
                "batch": 1,
                "games": 305,
                "cumulative_games": 305,
                "max_stderr": 8.0,
                "mean_stderr": 5.0,
                "seconds": 1.0,
                "ratings": [0.0, 0.0, 0.0],
            }
        ],
        "target_stderr": 10.0,
        "converged": True,
        "reference": reference,
        "tie_mode": "half_win",
        "tie_mode_alt": "decisive",
    }


def _store(tmp_path) -> ArtifactStore:
    return ArtifactStore(
        checkpoint_root=tmp_path / "checkpoints",
        standalone_root=tmp_path / "artifacts",
        invocation=Invocation(argv=("bnb", "elo-calibrate"), command="elo-calibrate"),
    )


def _source_measurement(tmp_path, reference: str, *, complete: bool = True) -> Path:
    """A calibration artifact for the refit path to read, finished unless asked otherwise."""

    run_dir = tmp_path / "checkpoints" / "test-run"
    run_dir.mkdir(parents=True)
    (run_dir / "elo_history.jsonl").write_text(
        json.dumps(
            {
                "update": 1,
                "global_step": 100,
                "live": 1200.0,
                "avg": None,
                "counts": {"scripted": [60, 40, 0]},
            }
        )
        + "\n"
    )
    store = _store(tmp_path)
    artifact = store.create(
        ArtifactRecipe(
            artifact_type="elo-calibration",
            result_schema_version=1,
            subjects={"run": "test-run"},
            parameters={"tie_mode": "half_win"},
        ),
        store.run_owner("test-run"),
    )
    artifact.write_json(_stored_result(reference))
    if complete:
        artifact.complete()
    return artifact.path


def _run_refit(tmp_path, reference: str = "scripted") -> dict:
    source = _source_measurement(tmp_path, reference)
    return run_elo_calibrate_mode(
        run_spec=None,
        from_artifact=source,
        ship_config=ShipConfig(),
        device="cpu",
        config=EloCalibrateConfig(num_envs=4, target_stderr=10.0, max_batches=1),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        store=_store(tmp_path),
    )


class TestRefit:
    def test_refit_pins_scripted_to_the_anchor_rating(self, tmp_path):
        """Refit plays nothing; it refits stored matrices and reports on the
        scripted-anchored scale."""
        result = _run_refit(tmp_path)
        by_label = {p["label"]: p for p in result["players"]}
        assert by_label["scripted"]["calibrated_elo"] == pytest.approx(SCRIPTED_ANCHOR_ELO)
        assert by_label["scripted"]["calibrated_elo_alt"] == pytest.approx(SCRIPTED_ANCHOR_ELO)
        assert result["anchor"] == "scripted"
        assert result["anchor_elo"] == SCRIPTED_ANCHOR_ELO

    def test_refit_orders_players_by_their_stored_record(self, tmp_path):
        result = _run_refit(tmp_path)
        by_label = {p["label"]: p["calibrated_elo"] for p in result["players"]}
        assert by_label["random"] < SCRIPTED_ANCHOR_ELO < by_label["ckpt_100"]

    def test_refit_recovers_the_live_curve_from_history(self, tmp_path):
        """A 60/40 record against scripted places the live policy above it."""
        result = _run_refit(tmp_path)
        assert result["curve"][0]["live_calibrated"] > SCRIPTED_ANCHOR_ELO

    def test_refit_writes_a_new_artifact_naming_the_one_it_derives_from(self, tmp_path):
        source = _source_measurement(tmp_path, "scripted")
        run_elo_calibrate_mode(
            run_spec=None,
            from_artifact=source,
            ship_config=ShipConfig(),
            device="cpu",
            config=EloCalibrateConfig(num_envs=4, target_stderr=10.0, max_batches=1),
            checkpoint_dir=str(tmp_path / "checkpoints"),
            store=_store(tmp_path),
        )
        written = sorted(
            (tmp_path / "checkpoints" / "test-run" / "artifacts" / "elo-calibration").iterdir()
        )
        assert len(written) == 2  # the source measurement plus its refit
        refit = load_artifact(next(path for path in written if path != source))
        stored = refit.read_json()
        assert stored["anchor"] == "scripted"
        assert stored["wins_matrix"] == _stored_result("scripted")["wins_matrix"]
        assert refit.manifest["recipe"]["sources"]["measurement"]["artifact_id"] == source.name
        assert refit.manifest["recipe"]["parameters"]["refit"] is True

    def test_refitting_an_unfinished_measurement_is_refused(self, tmp_path):
        """A sweep that never completed holds part of the games it was asked for;
        refitting it would report that fraction as a measurement."""
        source = _source_measurement(tmp_path, "scripted", complete=False)

        with pytest.raises(ArtifactIncomplete, match="cannot be cited"):
            run_elo_calibrate_mode(
                run_spec=None,
                from_artifact=source,
                ship_config=ShipConfig(),
                device="cpu",
                config=EloCalibrateConfig(num_envs=4, target_stderr=10.0, max_batches=1),
                checkpoint_dir=str(tmp_path / "checkpoints"),
                store=_store(tmp_path),
            )

    def test_rating_gaps_are_invariant_to_the_stored_reference_gauge(self, tmp_path):
        """The reference only sets the error gauge; reported gaps must not move
        beyond the shrinkage of prior_games, whose virtual games attach to
        whichever player is the gauge."""
        gaps = []
        for index, reference in enumerate(("scripted", "ckpt_100")):
            result = _run_refit(tmp_path / f"gauge_{index}", reference)
            by_label = {p["label"]: p["calibrated_elo"] for p in result["players"]}
            gaps.append(by_label["ckpt_100"] - by_label["random"])
        assert gaps[0] == pytest.approx(gaps[1], abs=2.0)


def test_arbitrary_agent_calibration_has_no_run_and_owns_its_artifact(tmp_path) -> None:
    result = run_elo_calibrate_mode(
        run_spec=None,
        agent_specs=["random", "scripted"],
        ship_config=ShipConfig(),
        model_config=MODEL_CONFIG,
        env_config=EnvConfig(num_ships=8, max_bullets=4, max_episode_steps=1),
        device="cpu",
        config=EloCalibrateConfig(num_envs=2, target_stderr=1_000_000, max_batches=1),
        store=_store(tmp_path),
    )
    assert result["run"] is None
    assert result["agent_specs"] == ["random", "scripted"]
    assert result["anchor"] == "scripted"

    written = list((tmp_path / "artifacts" / "elo-calibration").glob("*/result.json"))
    assert len(written) == 1
    artifact = load_artifact(written[0].parent)
    assert artifact.manifest["owner"] == {"kind": "standalone", "run": None}
    assert artifact.has("chart_history.jsonl") and artifact.has("chart_summary.json")


class TestBuildPlayersField:
    def test_reference_rungs_sit_between_the_endpoints(self, tmp_path):
        """Rungs are sorted, labeled canonically, and never duplicate the
        random/scripted endpoint players."""
        from boost_and_broadside.modes.elo_calibrate import build_players

        players = build_players(
            tmp_path,
            {"entries": []},
            None,
            ShipConfig(),
            8,
            "cpu",
            reference_probabilities=(0.5, 0.2),
        )
        assert [p.label for p in players] == [
            "random",
            "scripted",
            "semi_scripted_0p2",
            "semi_scripted_0p5",
        ]
        assert all(p.agent.kind == "semi_random" for p in players[2:])

    def test_endpoint_probability_is_rejected(self, tmp_path):
        from boost_and_broadside.modes.elo_calibrate import build_players

        with pytest.raises(AssertionError):
            build_players(
                tmp_path,
                {"entries": []},
                None,
                ShipConfig(),
                8,
                "cpu",
                reference_probabilities=(0.0,),
            )

    def test_rung_labels_match_the_semi_random_mode(self):
        from boost_and_broadside.agents.semi_random_scripted import semi_random_label

        assert semi_random_label(0.0) == "random"
        assert semi_random_label(1.0) == "scripted"
        assert semi_random_label(0.95) == "semi_scripted_0p95"
