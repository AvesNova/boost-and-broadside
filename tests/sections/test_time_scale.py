"""The tuning panel's time scale: the game k times faster at fixed map lengths, by the
units of every rule setting and card field (`bnb.game.handling.similarity`), energies with
gravity; the values held to whole ticks or Hz snap."""

import dataclasses

import pytest
import torch

from bnb.contracts import GameConfig
from bnb.game import api
from bnb.game.handling.similarity import CARD_UNITS, HANDLE_UNITS, RULE_UNITS, scale_handles
from bnb.play.loop import TIME_SCALES, MatchLoop, MatchSetup
from bnb.play.tuning import Tuner, Tuning, base_rules, time_scaled_card, time_scaled_settings

RULES = api.rules_of(api.PRESETS["frontline"])


def _values(rules):
    return {s.key: api.tuning.value(rules, s.key) for s in api.tuning.settings_for(rules)}


def test_every_setting_handle_and_card_field_has_units():
    assert {s.key for s in api.tuning.SETTINGS} <= set(RULE_UNITS)
    assert set(HANDLE_UNITS) == set(api.HANDLE_NAMES)
    assert set(CARD_UNITS) == set(api.default_card())


def test_settings_scale_by_their_units_and_times_snap():
    values = _values(RULES)
    faster = time_scaled_settings(values, 2.0, RULES.tick_length)
    assert faster["flight.gravity"] == 4.0 * values["flight.gravity"]
    assert faster["rules.spawn_speed"] == 2.0 * values["rules.spawn_speed"]
    assert faster["flight.nose_frequency_floor"] == 2.0 * values["flight.nose_frequency_floor"]
    assert faster["pool.episode_seconds"] == 150.0
    assert faster["mode.capture_seconds"] == 5.0
    assert faster["pool.arena_radius"] == values["pool.arena_radius"]  # lengths fixed
    assert faster["rules.bullet_drag"] == values["rules.bullet_drag"]  # per length: fixed
    snapped = time_scaled_settings(values, 0.7, RULES.tick_length)
    ticks = snapped["pool.respawn_seconds"] / RULES.tick_length
    assert ticks == pytest.approx(round(ticks))


def test_the_card_scales_by_its_units_with_energies_and_snaps_the_fire_rate():
    types = api.ship_types()
    faster = time_scaled_card(types, RULES, 2.0)
    assert torch.allclose(faster.engine_power, 8.0 * types.engine_power)
    assert torch.allclose(faster.max_energy, 4.0 * types.max_energy)  # the height E/g kept
    assert torch.allclose(faster.spawn_energy, 4.0 * types.spawn_energy)
    assert torch.allclose(faster.bullet_energy_cost, 4.0 * types.bullet_energy_cost)
    assert torch.allclose(faster.max_acceleration, 4.0 * types.max_acceleration)
    assert torch.allclose(faster.coast_exchange_scale, 4.0 * types.coast_exchange_scale)
    assert torch.allclose(faster.muzzle_speed, 2.0 * types.muzzle_speed)
    assert torch.allclose(faster.max_lift_coefficient, types.max_lift_coefficient, rtol=1e-6)  # 1/m
    assert faster.fire_rate.tolist() == [30.0, 15.0]  # 20 Hz is not a whole number of ticks
    assert torch.allclose(
        faster.shield_recharge_delay_scale, 0.5 * types.shield_recharge_delay_scale
    )
    assert faster.bullet_life.tolist() == pytest.approx([0.2, 0.2])  # 6 ticks
    slower = time_scaled_card(types, RULES, 0.5)
    assert slower.fire_rate.tolist() == [6.0, 5.0]  # 7.5 and 5 Hz, held to whole Hz at 30 Hz
    snapped = time_scaled_card(types, RULES, 0.7)
    ticks = snapped.bullet_life / RULES.tick_length
    assert torch.allclose(ticks, ticks.round()) and round(float(ticks[0])) == round(12 / 0.7)


@pytest.mark.slow
def test_a_time_scaled_game_flies_the_scaled_handles():
    """Measured, the scaled card under the scaled rules flies every handle scaled by its
    units (the tick stays, so within its discretisation)."""
    factor = 2.0
    card = api.ship_types(dtype=torch.float64)
    values = time_scaled_settings(_values(RULES), factor, RULES.tick_length)
    rules = api.tuning.tuned(RULES, values, bounded=False)
    handles = api.measure_handles(card, RULES)
    scaled = api.measure_handles(time_scaled_card(card, RULES, factor), rules)
    expected = scale_handles(handles, 1.0, 1.0 / factor)
    for name in api.FLOWN_HANDLES:
        assert torch.allclose(getattr(scaled, name), getattr(expected, name), rtol=0.03), name
    assert torch.allclose(scaled.ceiling, factor**2 * handles.ceiling)
    assert torch.allclose(scaled.gun_range, handles.gun_range, rtol=1e-4)  # 6 ticks exactly


def test_playback_runs_from_a_quarter_to_eight_times():
    assert TIME_SCALES[0] == 0.25 and TIME_SCALES[1] == 0.5 and TIME_SCALES[-1] == 8.0


def test_the_time_scale_scales_the_live_game_and_keeps_the_design():
    """Faster by 4: the match plays gravity x 16 and stores x 16; the tuner's handles and
    rules stay the design's (nothing re-measured) and the tuning names the factor."""
    setup = MatchSetup(GameConfig(2, 4, 900.0, mode="frontline"))
    loop = MatchLoop(setup, compiled=False)
    tuner = Tuner(loop, base_rules(setup.config, setup.lineup))
    try:
        assert tuner.ready()
        cruise = tuner.targets.cruise_speed.clone()
        gravity, store = loop.rules.flight.gravity, loop.types.max_energy.clone()
        done = tuner.time_scale(4.0)
        assert loop.rules.flight.gravity == 16.0 * gravity
        assert torch.allclose(loop.types.max_energy, 16.0 * store)
        assert "fire rate" in done  # 15 Hz x 4 is past the tick rate
        assert tuner.ready() and torch.equal(tuner.targets.cruise_speed, cruise)
        assert tuner.tuning.rules.get("flight.gravity", gravity) == gravity
        assert tuner.tuning.time_factor == 4.0
        tuner.time_scale(1.0)  # back to x1: the game's own numbers
        assert loop.rules.flight.gravity == gravity and not tuner.tuning
    finally:
        tuner.close()


def test_faster_by_sets_the_factor_relative_to_the_design():
    """Faster by k sets x k against the design, whatever the factor was: 2 then 2 is x2
    (not x4), 3 is x3, 1 is the design's own speed and numbers again."""
    setup = MatchSetup(GameConfig(2, 4, 900.0, mode="frontline"))
    loop = MatchLoop(setup, compiled=False)
    tuner = Tuner(loop, base_rules(setup.config, setup.lineup))
    try:
        gravity = loop.rules.flight.gravity
        tuner.time_scale(2.0)
        assert "already" in tuner.time_scale(2.0)
        assert tuner.time_factor == 2.0 and loop.rules.flight.gravity == 4.0 * gravity
        tuner.time_scale(3.0)
        assert tuner.time_factor == 3.0 and loop.rules.flight.gravity == pytest.approx(
            9.0 * gravity
        )
        tuner.time_scale(0.5)
        assert tuner.time_factor == 0.5 and loop.rules.flight.gravity == 0.25 * gravity
        tuner.time_scale(1.0)
        assert tuner.time_factor == 1.0 and loop.rules.flight.gravity == gravity
        assert not tuner.tuning
    finally:
        tuner.close()


def test_a_tuning_plays_its_design_k_times_faster():
    """A tuning holds the design (x1) and the factor; the match plays the scaled rules and
    card, past the sliders' ends where the scaling carries them."""
    setup = MatchSetup(
        GameConfig(2, 4, 900.0, mode="frontline"),
        tuning=Tuning(rules={"flight.gravity": 12.0}, time_factor=3.0),
    )
    loop = MatchLoop(setup, compiled=False)
    assert loop.rules.flight.gravity == pytest.approx(9.0 * 12.0)
    card = api.ship_types()
    assert torch.allclose(loop.types.engine_power, 27.0 * card.engine_power)
    assert torch.allclose(loop.types.muzzle_speed, 3.0 * card.muzzle_speed)
    settings, types = Tuning().played(base_rules(setup.config, None))
    assert settings == {} and types is None  # x1 and untuned: the game's own


def test_under_a_time_scale_the_tuner_keeps_the_design_and_saves_the_same_files(
    tmp_path, monkeypatch
):
    """The tuner works on the design at x1: a time scale changes what the match plays, not
    the values the panel shows or a save writes (the ship files and the tuning are the same
    as at x1, here written and compared byte for byte), and an edit under x k sets the x1
    value. A new match from the tuning plays it at the same factor, its tuner showing the
    same values."""
    from bnb.game import ships
    from bnb.play.tuning_panel import TuningPanel

    setup = MatchSetup(GameConfig(2, 4, 900.0, mode="frontline"))
    loop = MatchLoop(setup, compiled=False)
    tuner = Tuner(loop, base_rules(setup.config, setup.lineup))
    panel = TuningPanel()
    try:
        assert tuner.ready()

        def written(directory):
            """The ship files Save writes now, as text by name (into `directory`)."""
            files = panel._current(tuner)
            monkeypatch.setattr(ships, "SHIPS", directory)
            ships.write(files)
            monkeypatch.undo()
            return {path.name: path.read_text() for path in directory.iterdir()}

        at_one, values, tuning = written(tmp_path / "x1"), dict(tuner.values), tuner.tuning
        gravity, store = loop.rules.flight.gravity, loop.types.max_energy.clone()
        done = tuner.time_scale(2.0)
        assert "x2" in done and "fire rate" in done  # 15 Hz x 2 is past the tick rate
        assert loop.rules.flight.gravity == 4.0 * gravity
        assert torch.allclose(loop.types.max_energy, 4.0 * store)
        assert tuner.ready() and tuner.values == values  # nothing to re-measure
        assert written(tmp_path / "x2") == at_one
        assert tuner.tuning == dataclasses.replace(tuning, time_factor=2.0)

        tuner.set_setting("flight.gravity", 2.0 * gravity)  # the x1 value
        assert tuner.values["flight.gravity"] == 2.0 * gravity
        assert loop.rules.flight.gravity == pytest.approx(8.0 * gravity)
        assert tuner.tuning.rules["flight.gravity"] == 2.0 * gravity

        again = MatchLoop(dataclasses.replace(setup, tuning=tuner.tuning), compiled=False)
        assert again.rules.flight.gravity == pytest.approx(8.0 * gravity)
        assert torch.equal(again.types.fire_rate, loop.types.fire_rate)
        follower = Tuner(again, base_rules(setup.config, setup.lineup))
        try:
            assert follower.time_factor == 2.0 and follower.values == tuner.values
            assert follower.tuning == tuner.tuning
        finally:
            follower.close()
    finally:
        tuner.close()


@pytest.mark.parametrize("factor", [1.0, 2.0, 0.5])
def test_the_clocks_count_the_seconds_played_at_x1_playback(factor):
    """Every time readout counts ticks times the tick (1/30 s): the seconds a player sees
    pass at x1 playback. Under a time scale k the tick stays, so 30 ticks are 1 s on the
    clock however fast the game runs, and the match length and the respawn wait are the
    design's (x1, what the panel shows) over k. Playback x p runs p times as many ticks per
    second of the display, so the clock runs p times faster than the wall clock."""
    config = api.PRESETS["frontline"]
    loop = MatchLoop(MatchSetup(config, tuning=Tuning(time_factor=factor)), compiled=False)
    design = api.rules_of(config).pools[0]
    for _ in range(30):
        loop.step()
    frame = loop.current
    assert frame.clock == pytest.approx(1.0)  # the top bar's clock
    assert frame.episode_length == pytest.approx(design.episode_ticks / 30.0 / factor)
    respawn = loop.rules.pools[0].respawn_ticks * loop.tick_length  # "respawns in" at a death
    assert respawn == pytest.approx(design.respawn_ticks / 30.0 / factor)
    assert loop.ticks * loop.tick_length == pytest.approx(1.0)  # "last shot ... ago"'s now
    for playback in (0.5, 2.0, 8.0):
        loop.time_scale = playback
        before = loop.current.clock
        for _ in range(60):  # one second of the display at 60 frames a second
            loop.advance(1.0 / 60.0)
        assert loop.current.clock - before == pytest.approx(playback, abs=loop.tick_length)
