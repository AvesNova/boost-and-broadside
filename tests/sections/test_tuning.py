"""Live tuning (`bnb.game.tuning`, `Realtime.retune`, `bnb.play.tuning`): the settings round
trip, a running match takes new rules and a new card from the next tick (never
recompiling) and refuses new sizes, presets round trip, and the panel shows the ship
files' handles without measuring them."""

import math

import pytest
import torch

from bnb.contracts import GameConfig
from bnb.game import api
from bnb.play.interface.layout import FLEETS_WIDTH, Layout
from bnb.play.loop import MatchLoop, MatchSetup
from bnb.play.tuning import (
    Presets,
    Tuning,
    base_rules,
    columns,
    file_handles,
    fire_rates,
    snap_handle,
    with_columns,
)
from bnb.play.tuning_panel import (
    from_log_axis,
    is_off,
    log_axis,
    placement,
    short,
    to_log_axis,
)
from bnb.runtime.realtime import Realtime

FRONTLINE = GameConfig(2, 4, 900.0, episode_seconds=20.0, mode="frontline")
DEATHMATCH = GameConfig(2, 4, 900.0, episode_seconds=20.0)


# ------------------------------------------------------------------ the settings


@pytest.mark.parametrize("config", [FRONTLINE, DEATHMATCH], ids=["frontline", "deathmatch"])
def test_every_setting_round_trips(config):
    """Setting each of a game's settings to a value inside its bounds gives rules that
    report that value for it and differ from the game's in it alone."""
    base = api.rules_of(config)
    for setting in api.tuning.settings_for(base):
        target = setting.low + 0.37 * (setting.high - setting.low)
        target = api.tuning.snap(setting, target, base.tick_length)
        if target == api.tuning.value(base, setting.key):
            target = api.tuning.snap(setting, setting.high, base.tick_length)
        rules = api.tuning.tuned(base, {setting.key: target})
        assert api.tuning.value(rules, setting.key) == pytest.approx(target), setting.key
        assert api.tuning.changed(rules, base) == pytest.approx({setting.key: target})


def test_a_modes_settings_belong_to_its_games_only():
    keys = {s.key for s in api.tuning.settings_for(api.rules_of(DEATHMATCH))}
    assert "mode.zone_radius" not in keys and "mode.kills_per_ship" in keys
    keys = {s.key for s in api.tuning.settings_for(api.rules_of(FRONTLINE))}
    assert "mode.zone_radius" in keys and "mode.kills_per_ship" not in keys


def test_times_hold_to_whole_ticks():
    base = api.rules_of(FRONTLINE)
    rules = api.tuning.tuned(base, {"pool.respawn_seconds": 3.01, "rules.bullet_life": 0.51})
    assert rules.pools[0].respawn_ticks == 90  # 3.0 s at 30 Hz
    assert rules.bullet_moves == 15  # 0.5 s
    with pytest.raises(KeyError):
        api.tuning.tuned(base, {"rules.no_such_setting": 1.0})


# ------------------------------------------------------------------ a running match


def test_new_rules_and_card_apply_from_the_next_tick():
    """The zone ring takes the new radius at once (the state is refit), a lower maximum
    shield caps every shield, and the tick after reads both."""
    realtime = Realtime(FRONTLINE, compiled=False)
    actions = torch.zeros(FRONTLINE.num_ships, 3, dtype=torch.int32)
    for _ in range(3):
        realtime.step(actions)
    map_scale = FRONTLINE.arena_radius / api.MAP_SCALE_RADIUS
    realtime.retune(rules=api.tuning.tuned(realtime.rules, {"mode.zone_radius": 600.0}))
    radius = api.display(realtime.state)["mode.zone_radius"]
    torch.testing.assert_close(radius, torch.full_like(radius, 600.0 * map_scale))
    realtime.retune(types=api.ship_types(max_shield=(20.0, 30.0)))
    shield = api.display(realtime.state)["ships.shield"]
    assert shield.max() <= 30.0
    realtime.step(actions)
    assert api.display(realtime.state)["ships.shield"].max() <= 30.0
    torch.testing.assert_close(api.display(realtime.state)["mode.zone_radius"], radius)


def test_a_retune_keeps_the_sizes():
    realtime = Realtime(FRONTLINE, compiled=False)
    with pytest.raises(ValueError, match="sizes"):
        realtime.retune(rules=api.rules_of(GameConfig(3, 4, 900.0, mode="frontline")))
    with pytest.raises(ValueError, match="card"):
        realtime.retune(types=api.ship_types(max_shield=torch.ones(3)))


@pytest.mark.slow
def test_a_retune_never_recompiles():
    """The compiled tick takes the rules' live values as tensors: new values run on the same
    compiled tick, with no new graph, and match a match started with them."""
    from torch._dynamo.utils import counters

    realtime = Realtime(DEATHMATCH, compiled=True)
    actions = torch.zeros(DEATHMATCH.num_ships, 3, dtype=torch.int32)
    realtime.step(actions)
    graphs = counters["stats"]["unique_graphs"]
    changes = {"flight.gravity": 33.0, "rules.bullet_life": 0.3, "pool.respawn_seconds": 2.0}
    realtime.retune(rules=api.tuning.tuned(realtime.rules, changes))
    for _ in range(5):
        realtime.step(actions)
    assert realtime.uses_compiled
    assert counters["stats"]["unique_graphs"] == graphs
    assert realtime.rules.flight.gravity == 33.0 and realtime.rules.bullet_moves == 9


def test_a_match_starts_from_its_tuning():
    tuning = Tuning(rules={"mode.zone_radius": 450.0}, card={"hit_damage": (20.0, 25.0)})
    loop = MatchLoop(MatchSetup(FRONTLINE, tuning=tuning), compiled=False)
    assert loop.rules.pools[0].mode.zone_radius == 450.0
    assert loop.types.hit_damage.tolist() == [20.0, 25.0]
    assert Tuning.of(loop.rules, loop.types, base_rules(FRONTLINE, None)) == tuning


# ------------------------------------------------------------------ handles and presets


def test_handles_hold_to_what_the_card_can_keep():
    tick = 1.0 / 30.0
    assert fire_rates(tick) == [1.0, 2.0, 3.0, 5.0, 6.0, 10.0, 15.0, 30.0]
    assert snap_handle("fire_rate", 14.0, tick) == 15.0
    assert snap_handle("fire_rate", 8.0, tick) == 10.0
    assert snap_handle("shield_recharge_delay", 6.01, tick) == pytest.approx(6.0)
    assert snap_handle("cruise_speed", 101.3, tick) == 101.3


def test_bullet_slots_follow_the_card_and_a_tunable_match_has_room():
    """Training's rules hold what the card needs (6 for the default card: 12 moves, 2 ticks
    apart); a play-shell match holds a shot every tick for the longest bullet life."""
    types = api.ship_types()
    assert api.rules_of(DEATHMATCH, types=types).bullet_slots == 6
    fast = api.ship_types(fire_rate=(30.0, 10.0))
    assert api.rules_of(DEATHMATCH, types=fast).bullet_slots == 12
    assert api.tuning.most_bullet_slots(1.0 / 30.0) == 45
    assert Realtime(DEATHMATCH, compiled=False).rules.bullet_slots == 45


def test_a_bullet_life_changes_the_gun_range_at_once():
    """The card is kept: the gun range handle follows the bullet life in closed form."""
    rules = api.rules_of(DEATHMATCH)
    longer = api.tuning.tuned(rules, {"rules.bullet_life": 0.8})
    cruise = torch.tensor([100.0, 100.0])
    near, far = (
        api.gun_range(rules, api.ship_types(), cruise),
        api.gun_range(longer, api.ship_types(), cruise),
    )
    assert (far > 1.5 * near).all()


@pytest.mark.parametrize("low, high", [(1.7e-4, 2.8e-3), (-96.0, -6.0), (0.03, 0.7)])
def test_a_handle_slider_runs_from_low_to_high_at_any_scale(low, high):
    """The panel's logarithmic slider (a linear slider over the log of the magnitude) puts
    tiny values (the zero-lift drag, ~7e-4) inside it, not at an end, and runs from low to
    high for negative values (the zoom acceleration) too."""
    start, end = log_axis(low, high)
    assert from_log_axis(start, low) == pytest.approx(low)
    assert from_log_axis(end, low) == pytest.approx(high)
    middle = -math.sqrt(low * high) if low < 0 else math.sqrt(low * high)
    assert to_log_axis(middle, low) == pytest.approx(0.5 * (start + end))
    assert from_log_axis(to_log_axis(middle, low), low) == pytest.approx(middle)


def test_the_panels_numbers_hold_at_most_four_significant_digits():
    """The number boxes write 4 significant digits, and numbers from 10000 up in full rather
    than with an exponent."""
    assert [short(v) for v in (31.6228, 0.00069612, -23.994, 8.0, 10000.0, 12345.6)] == [
        "31.62",
        "0.0006961",
        "-23.99",
        "8",
        "10000",
        "12346",
    ]


def test_a_flown_handle_is_marked_only_past_half_a_percent_off():
    assert not is_off(100.4, 100.0) and is_off(100.6, 100.0)
    assert not is_off(-23.99, -24.0) and is_off(-24.2, -24.0)
    assert is_off(1e-3, 0.0) and not is_off(0.0, 0.0)


@pytest.mark.parametrize(
    "width, height, scale", [(1280, 720, 1.0), (1280, 720, 1.5), (1920, 1080, 2.0), (360, 300, 1.0)]
)
def test_the_tuning_panel_opens_inside_the_display(width, height, scale):
    """The panel opens at the right, below the top bar, about 400 px wide per interface
    scale and no wider or taller than the display leaves room for."""
    x, y, panel_width, panel_height = placement(width, height, scale)
    assert x >= 0.0 and y > 0.0
    assert x + panel_width < width and y + panel_height < height
    assert panel_width == pytest.approx(min(400.0 * scale, width - 16.0 * scale))


def test_presets_round_trip(tmp_path):
    presets = Presets(tmp_path)
    tuning = Tuning(rules={"rules.bullet_life": 0.5}, card={"hit_damage": (20.0, 25.0)})
    path = presets.save("fast-guns", tuning, tuning.types())
    assert presets.names() == ["fast-guns"]
    assert presets.load("fast-guns") == tuning
    with pytest.raises(ValueError, match="lowercase"):
        presets.save("Fast Guns", tuning, tuning.types())
    assert set(__import__("json").loads(path.read_text())["ship_card"]) == set(api.default_card())


# ------------------------------------------------------------------ the ship files


@pytest.fixture
def ship_files(tmp_path, monkeypatch):
    """A copy of the ship files to change; the game's defaults are read back from the real
    ones afterwards."""
    import shutil

    from bnb.game import ships

    copy = tmp_path / "ships"
    shutil.copytree(ships.SHIPS, copy)
    monkeypatch.setattr(ships, "SHIPS", copy)
    yield copy
    monkeypatch.undo()
    api.reload_ships()


def test_the_default_card_is_the_ship_files():
    from bnb.game import ships

    files = ships.load()
    assert api.type_names() == tuple(ship.name for ship in files) == ("Fighter", "Heavy")
    for name, value in vars(api.ship_types(dtype=torch.float64)).items():
        assert value.tolist() == [ship.card[name] for ship in files], name
    assert set(files[0].handles) == set(api.HANDLE_NAMES)


def test_ship_files_round_trip_and_a_new_type_joins_the_card(ship_files):
    from bnb.game import ships

    files = ships.load()
    scout = ships.Ship(
        "scout", "Scout", dict(files[0].handles), dict(files[0].card, hit_damage=7.0)
    )
    ships.write([*files, scout])
    api.reload_ships()
    assert api.type_names() == ("Fighter", "Heavy", "Scout")
    assert api.ship_types().hit_damage.tolist() == [12.0, 16.0, 7.0]
    ships.write(files[:1])
    api.reload_ships()
    assert api.type_names() == ("Fighter",) and not (ship_files / "heavy.toml").exists()


def test_the_panel_takes_each_types_handles_from_its_ship_file():
    """The handles shown before anything is measured: each type's file's, in float64; a
    type past the files (or one missing a handle) takes a stand-in and is not complete."""
    from bnb.game import ships

    files = ships.load("real-scale")
    handles, complete = file_handles(files, len(files))
    assert all(complete) and handles.cruise_speed.dtype == torch.float64
    for kind, ship in enumerate(files):
        for name in api.HANDLE_NAMES:
            assert float(getattr(handles, name)[kind]) == ship.handles[name], (ship.key, name)
    handles, complete = file_handles(files[:2], 3)
    assert complete == [True, True, False]
    assert float(handles.ceiling[2]) == files[0].handles["ceiling"]


def test_a_cards_columns_are_taken_and_put_back_by_type():
    types = api.ship_types(ship_set="real-scale")
    part = columns(types, [3, 1])
    assert part.max_energy.tolist() == [types.max_energy[3].item(), types.max_energy[1].item()]
    doubled = api.ShipTypes(**{n: 2.0 * v for n, v in vars(part).items()})
    changed = with_columns(types, [3, 1], doubled)
    assert changed.max_energy[3] == 2.0 * types.max_energy[3]
    assert changed.max_energy[0] == types.max_energy[0]
    assert changed.max_energy.dtype == types.max_energy.dtype
    assert types.max_energy[3] != changed.max_energy[3]  # the original is untouched


@pytest.mark.parametrize("scale", [1.0, 1.5])
def test_the_fleets_card_widens_for_long_ship_names(scale):
    assert Layout(1920, 1080, scale).fleets.width == FLEETS_WIDTH * scale
    wide = Layout(1920, 1080, scale, fleets_width=380.0)
    assert (
        wide.fleets.width == 380.0 * scale and wide.fleets.x == Layout(1920, 1080).fleets.x * scale
    )


def test_a_measurement_stops_when_another_type_is_wanted_first(monkeypatch):
    """The measuring worker drops a type's measurement within CHECK_EVERY ticks once the
    panel shows another unmeasured type, or the rules change, so that one goes first."""
    from bnb.play import tuning

    rules = api.rules_of(FRONTLINE)
    card = tuning.columns(api.ship_types(), [0])
    assert tuning._measure_job([0, 0, 1], None, 0, 0, rules, card)[-1] is None  # at once
    versions = [0, 0, -1]

    def step(*arguments):  # the panel shows type 1 while type 0 flies
        versions[tuning.FIRST] = 1
        return api.flight_tick(*arguments)

    monkeypatch.setattr(tuning, "_STEP", step)
    monkeypatch.setattr(tuning, "CHECK_EVERY", 1)
    assert tuning._measure_job(versions, None, 0, 0, rules, card)[-1] is None
    versions[:] = [0, 1, -1]  # the rules changed since the job was submitted (version 0)
    assert tuning._measure_job(versions, None, 0, 0, rules, card)[-1] is None
