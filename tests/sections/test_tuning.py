"""Live tuning (`bnb.game.tuning`, `Realtime.retune`, `bnb.play.tuning`): the settings round
trip, a running match takes new rules and a new card from the next tick (never
recompiling) and refuses new sizes, and presets round trip."""

import pytest
import torch

from bnb.contracts import GameConfig
from bnb.game import api
from bnb.play.loop import MatchLoop, MatchSetup
from bnb.play.tuning import Presets, Tuning, base_rules, bullet_slots_needed, snap_handle
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
    assert snap_handle("fire_rate", 14.0, tick) == pytest.approx(15.0)  # 2 ticks apart
    assert snap_handle("fire_rate", 8.0, tick) == pytest.approx(7.5)  # 4 ticks
    assert snap_handle("shield_recharge_delay", 6.01, tick) == pytest.approx(6.0)
    assert snap_handle("cruise_speed", 101.3, tick) == 101.3
    rules = api.rules_of(FRONTLINE)
    assert bullet_slots_needed(rules, [15.0, 10.0]) == 6  # 12 moves, 2 ticks apart
    assert bullet_slots_needed(rules, [30.0, 10.0]) == 12


def test_presets_round_trip(tmp_path):
    presets = Presets(tmp_path)
    tuning = Tuning(rules={"rules.bullet_life": 0.5}, card={"hit_damage": (20.0, 25.0)})
    path = presets.save("fast-guns", tuning, tuning.types())
    assert presets.names() == ["fast-guns"]
    assert presets.load("fast-guns") == tuning
    with pytest.raises(ValueError, match="lowercase"):
        presets.save("Fast Guns", tuning, tuning.types())
    assert set(__import__("json").loads(path.read_text())["ship_card"]) == set(api.default_card())
