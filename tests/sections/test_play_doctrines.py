"""The play shell's scripted doctrine (`bnb.play.sides.DOCTRINES`): Baekho, the one scripted
doctrine (the stored real-scale doctrine, with the Frontline lock-aware goal), in the fleets
combo and on both teams by default. And the frame-rate readout's numbers
(`bnb.play.frame_rate`)."""

import pytest
import tyro

from bnb.controllers import scripted_doctrine
from bnb.play.app import Options
from bnb.play.frame_rate import FrameRate
from bnb.play.interface.fleets import SCRIPTED_CHOICES, controller_label_of_setup, scripted_label
from bnb.play.interface.shortcuts import SHORTCUTS
from bnb.play.loop import SideSetup
from bnb.play.sides import DEFAULT_DOCTRINE, DOCTRINES


def test_baekho_is_the_one_doctrine_and_the_stored_one():
    assert DOCTRINES == {"baekho": scripted_doctrine.DOCTRINES}
    assert DEFAULT_DOCTRINE == "baekho" and SideSetup().doctrine == "baekho"


def test_baekho_flies_the_lock_aware_goal_and_the_maneuver_elements():
    """Baekho: the game loop's r9 with maneuvers-2's elements (the duel one-circle sense, the
    crowd extension, the team-gated duel extension) and lock_press 1, for both types."""
    stored = scripted_doctrine.stored()["real-scale"]
    for key in ("yongma", "jujak"):
        doctrine = stored[key]
        assert doctrine.lock_press == 1.0
        assert doctrine.sense == 1.0 and doctrine.sense_one_circle == 1.0
        assert doctrine.crowd_sense == 1.0 and doctrine.duel_extend == 1.0


def test_the_stored_file_packs_to_the_full_table_width_for_the_real_scale_set():
    rows = scripted_doctrine.rows_by_count(DOCTRINES["baekho"])
    assert rows and all(
        len(row) == scripted_doctrine.WIDTH for table in rows.values() for row in table
    )


def test_both_teams_fly_baekho_from_the_command_line():
    sides = tyro.cli(Options, args=[]).setup().sides
    assert [side.doctrine for side in sides] == ["baekho", "baekho"]
    assert all(side.controller == "scripted_agent" for side in sides)


def test_the_fleets_combo_offers_the_chaser_and_baekho():
    labels = [label for _, label in SCRIPTED_CHOICES]
    assert labels == ["Chaser", "Scripted agent Baekho"]
    assert scripted_label(SideSetup("scripted_agent")) == "Scripted agent Baekho"
    assert scripted_label(SideSetup("scripted")) == "Chaser"
    assert scripted_label(SideSetup("agent", "runs/x")) is None
    assert controller_label_of_setup(SideSetup()) == "scripted agent baekho"


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
