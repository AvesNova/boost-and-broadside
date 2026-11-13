"""The play shell's scripted doctrines (`bnb.play.sides.DOCTRINES`): r9m2 by default, r9 as
an option, chosen per team on the command line and in the fleets combo; r9m2's file holds
maneuvers-2's champion2 doctrine and the training's stored doctrine is left alone. And the
frame-rate readout's numbers (`bnb.play.frame_rate`)."""

import json
import pathlib
import typing

import pytest
import tyro

from bnb.controllers import scripted_doctrine
from bnb.play.app import Options
from bnb.play.frame_rate import FrameRate
from bnb.play.interface.fleets import SCRIPTED_CHOICES, controller_label_of_setup, scripted_label
from bnb.play.interface.shortcuts import SHORTCUTS
from bnb.play.loop import SideSetup
from bnb.play.sides import DEFAULT_DOCTRINE, DOCTRINES

ROOT = pathlib.Path(__file__).parents[2]
CHAMPION = ROOT / "docs/rewrite/sections/maneuvers-2/champion2.json"


def test_r9m3_is_the_default_and_r9_the_stored_training_doctrine():
    assert DEFAULT_DOCTRINE == "r9m3" and SideSetup().doctrine == "r9m3"
    assert DOCTRINES["r9"] == scripted_doctrine.DOCTRINES
    assert set(typing.get_args(Options.__annotations__["doctrine0"])) == set(DOCTRINES)
    assert set(typing.get_args(Options.__annotations__["doctrine1"])) == set(DOCTRINES)


@pytest.mark.skipif(not CHAMPION.exists(), reason="maneuvers-2's champion2.json is not here")
def test_r9m2s_file_is_maneuvers_2s_champion2():
    champion = json.loads(CHAMPION.read_text())
    stored = scripted_doctrine.stored(DOCTRINES["r9m2"])["real-scale"]
    for key in ("yongma", "jujak"):
        assert stored[key] == scripted_doctrine.from_named(champion[key])
    # r9m2 differs from r9 in the maneuver elements: the duel extension is on
    r9 = scripted_doctrine.stored(DOCTRINES["r9"])["real-scale"]
    assert stored["yongma"].duel_extend == 1.0 and r9["yongma"].duel_extend == 0.0


def test_r9m3_is_r9m2_with_the_lock_aware_goal():
    """r9m3 (the play shell's default since 2026-11-13) is r9m2 with lock_press 1 on both
    types and every other field r9m2's; the wave hold stays off."""
    import dataclasses

    r9m2 = scripted_doctrine.stored(DOCTRINES["r9m2"])["real-scale"]
    r9m3 = scripted_doctrine.stored(DOCTRINES["r9m3"])["real-scale"]
    for key in ("yongma", "jujak"):
        assert r9m3[key].lock_press == 1.0 and r9m3[key].wave_hold == 0.0
        assert dataclasses.replace(r9m3[key], lock_press=0.0) == r9m2[key]


def test_both_files_pack_to_the_full_table_width_for_the_real_scale_set():
    for path in DOCTRINES.values():
        rows = scripted_doctrine.rows_by_count(path)
        assert rows and all(
            len(row) == scripted_doctrine.WIDTH for table in rows.values() for row in table
        )


def test_the_command_line_chooses_each_teams_doctrine():
    options = tyro.cli(Options, args=["--doctrine1", "r9"])
    sides = options.setup().sides
    assert [side.doctrine for side in sides] == ["r9m3", "r9"]
    assert all(side.controller == "scripted_agent" for side in sides)
    with pytest.raises(SystemExit):
        tyro.cli(Options, args=["--doctrine0", "r10"])


def test_the_fleets_combo_offers_the_chaser_and_each_doctrine():
    labels = [label for _, label in SCRIPTED_CHOICES]
    assert labels == ["Chaser", "Scripted agent r9m3", "Scripted agent r9m2", "Scripted agent r9"]
    assert scripted_label(SideSetup("scripted_agent", doctrine="r9")) == "Scripted agent r9"
    assert scripted_label(SideSetup("scripted")) == "Chaser"
    assert scripted_label(SideSetup("agent", "runs/x")) is None
    assert controller_label_of_setup(SideSetup()) == "scripted agent r9m3"


def test_the_frame_rate_is_smoothed_and_keeps_the_last_seconds_longest_frame():
    rate = FrameRate()
    assert rate.per_second == 0.0
    rate.add(0.0)  # skipped
    for _ in range(120):
        rate.add(1.0 / 60.0)
    assert rate.per_second == pytest.approx(60.0)
    assert rate.worst_seconds == pytest.approx(1.0 / 60.0)
    rate.add(0.25)  # a stall: the smoothed rate drops by its share of the time, the max shows it
    assert 5.0 < rate.per_second < 30.0
    for _ in range(60):
        rate.add(1.0 / 60.0)
    assert rate.worst_seconds == pytest.approx(0.25)
    assert rate.text().startswith(f"{rate.per_second:.0f} fps") and "max 250 ms" in rate.text()
    for _ in range(80):
        rate.add(1.0 / 60.0)
    assert rate.worst_seconds == pytest.approx(1.0 / 60.0)  # the stall's second has passed


def test_f4_toggles_the_frame_rate():
    assert any(method == "toggle_fps" for _, _, method, _ in SHORTCUTS)
    assert Options().fps
