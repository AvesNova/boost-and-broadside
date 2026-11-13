"""Frontline's capture lock and respawn waves, and team deathmatch's respawn wait (the
author's rules of 2026-11-13, game version 8): the lock holds only the team that lost the
zone, for exactly its length, which grows as the square root of the zone spacing; a dead
Frontline ship respawns at its team's first wave after its minimum wait; team deathmatch
waits a fixed 8 s at real scale; the observation shows the locks to both teams and each
team only its own wave clock."""

import dataclasses
import math

import modes_cases as cases
import pytest
import torch

from bnb import game
from bnb.contracts import GameConfig
from bnb.contracts.obs import GLOBAL, ZONE
from bnb.game import api
from bnb.game.handling.similarity import RULE_UNITS
from bnb.game.modes import frontline
from bnb.game.modes.observation import mode_globals, zone_rows
from bnb.game.modes.phase import mode_phase
from bnb.game.modes.rules import PRESETS
from bnb.game.state import replace

TYPES = api.ship_types()
LOCK = 10  # ticks, the lock the mode-phase tests set


def _frontline(seed: int = 0):
    rules = api.rules_of(GameConfig(2, 4, 2600.0, mode="frontline"))
    state = api.reset(
        rules, TYPES, api.uniforms(rules, 1, generator=torch.Generator().manual_seed(seed))
    )
    return rules, state


def _centre(mode, role: int) -> tuple[torch.Tensor, torch.Tensor]:
    index = (mode.role == role).int().argmax(-1, keepdim=True)
    return mode.zone_x.gather(1, index)[:, 0], mode.zone_y.gather(1, index)[:, 0]


def _positions(rules, mode, placed: dict[int, int]):
    """(1, N) x, y: ship -> the zone role it sits on; every other ship parked."""
    x = torch.zeros(1, rules.num_ships)  # the ring's centre, outside every zone
    y = torch.zeros(1, rules.num_ships)
    for ship, role in placed.items():
        x[:, ship], y[:, ship] = _centre(mode, role)
    return x, y


def _captured_setup():
    """2v2 at front 0, team 0's ship 0 on team 1's defense zone with its meter a hair from
    full, the lock set to LOCK ticks: one mode phase captures it."""
    rules, state = _frontline()
    mode = state.mode
    spacing = float(frontline.adjacent_distance(mode.zone_x, mode.zone_y)[0])
    per_root_km = LOCK * rules.tick_length / math.sqrt(spacing / 1000.0)
    mode_rules = dataclasses.replace(rules.modes[0], lock_seconds_per_root_km=per_root_km)
    target = int((mode.role[0] == frontline.TEAM1_DEFENSE).int().argmax())
    mode = replace(mode, progress=mode.progress.index_fill(1, torch.tensor([target]), 1.0 - 1e-4))
    x, y = _positions(rules, mode, {0: frontline.TEAM1_DEFENSE})
    ships = replace(state.ships, x=x, y=y)
    alive = torch.ones(1, rules.num_ships, dtype=torch.bool)
    died = torch.zeros_like(alive)
    phase = mode_phase(rules, mode_rules, mode, ships, alive, died)
    assert int(phase.mode.front[0]) == 1, "the capture should complete"
    return rules, mode_rules, state.ships, phase.mode, target


# ------------------------------------------------------------------ the capture lock


def test_a_capture_locks_only_the_team_that_lost_the_zone():
    _, _, _, mode, target = _captured_setup()
    want = torch.zeros_like(mode.lock)
    want[0, 1, target] = LOCK  # team 1 lost it
    assert torch.equal(mode.lock, want)


def test_the_lock_holds_the_losing_team_for_exactly_its_length_and_not_the_winner():
    rules, mode_rules, ships, mode, target = _captured_setup()
    # The captured zone is now team 0's defense, which team 1 attacks: team 1's ship 2
    # sits on it; team 0's ship 0 attacks team 1's new defense zone.
    assert int(mode.role[0, target]) == frontline.TEAM0_DEFENSE
    other = int((mode.role[0] == frontline.TEAM1_DEFENSE).int().argmax())
    x, y = _positions(rules, mode, {0: frontline.TEAM1_DEFENSE, 2: frontline.TEAM0_DEFENSE})
    ships = replace(ships, x=x, y=y)
    alive = torch.ones(1, rules.num_ships, dtype=torch.bool)
    died = torch.zeros_like(alive)
    for tick in range(LOCK):
        assert int(mode.lock[0, 1, target]) == LOCK - tick
        mode = mode_phase(rules, mode_rules, mode, ships, alive, died).mode
        assert float(mode.progress[0, target]) == 0.0, f"tick {tick + 1} of the lock"
        assert float(mode.progress[0, other]) > 0.0, "the winning team is not held"
    assert int(mode.lock.abs().sum()) == 0
    mode = mode_phase(rules, mode_rules, mode, ships, alive, died).mode
    assert float(mode.progress[0, target]) > 0.0, "the lock is over"


def test_a_defending_majority_still_moves_a_locked_zones_meter():
    rules, mode_rules, ships, mode, target = _captured_setup()
    mode = replace(mode, progress=mode.progress.index_fill(1, torch.tensor([target]), 0.5))
    x, y = _positions(rules, mode, {0: frontline.TEAM0_DEFENSE})  # team 0 defends it
    ships = replace(ships, x=x, y=y)
    alive = torch.ones(1, rules.num_ships, dtype=torch.bool)
    after = mode_phase(rules, mode_rules, mode, ships, alive, torch.zeros_like(alive)).mode
    assert float(after.progress[0, target]) < 0.5


def test_the_lock_grows_as_the_square_root_of_the_zone_spacing():
    tick = 1.0 / 30.0
    angle = torch.arange(5) * 2.0 * math.pi / 5.0
    lengths = []
    for radius in (1000.0, 4000.0):
        x, y = radius * torch.cos(angle)[None], radius * torch.sin(angle)[None]
        lengths.append(int(frontline.lock_ticks(19.0, x, y, tick)[0]))
    assert abs(lengths[1] / lengths[0] - 2.0) < 2e-3  # 4x the spacing, 2x the lock


@pytest.mark.parametrize(
    "preset, seconds",
    [
        ("real_scale_frontline", 45.0),
        ("real_scale_frontline_10v10_density", 54.0),
        ("real_scale_frontline_20v20_density", 64.0),
    ],
)
def test_the_default_lock_is_the_decided_length_at_each_fleet_size(preset, seconds):
    rules, state = cases.rules_and_state(PRESETS[preset], batch=1)
    mode_rules = rules.modes[0]
    assert mode_rules.lock_seconds_per_root_km == 19.0
    length = frontline.lock_ticks(19.0, state.mode.zone_x, state.mode.zone_y, rules.tick_length)
    assert abs(float(length[0]) * rules.tick_length - seconds) < 0.5


# ------------------------------------------------------------------ respawn waves


def _wave_rules():
    """3v3 Frontline with no fields: a 1 s respawn minimum and 2 s waves (30 and 60 ticks)."""
    rules = api.rules_of(GameConfig(3, 0, 2600.0, mode="frontline"))
    return api.tuning.tuned(rules, {"pool.respawn_seconds": 1.0, "mode.wave_seconds": 2.0})


def test_a_dead_ship_respawns_at_its_teams_first_wave_after_the_minimum():
    rules = _wave_rules()
    assert rules.pools[0].respawn_ticks == 30
    state = api.reset(
        rules, TYPES, api.uniforms(rules, 2, generator=torch.Generator().manual_seed(3))
    )
    alive, wait = state.ships.alive.clone(), state.ships.respawn_wait.clone()
    alive[:, 4], wait[:, 4] = False, 30  # team 1's ship, dead this tick
    state = cases.edit_ships(state, alive=alive, respawn_wait=wait)
    # environment 0: team 1's wave lands at tick 10, before the minimum, then at 70;
    # environment 1: at tick 45, after it. Team 0's clock is far off and does not matter.
    state = cases.edit_mode(
        state, wave_wait=torch.tensor([[500, 10], [500, 45]], dtype=torch.int32)
    )
    spawned_at = [None, None]
    for tick in range(1, 80):
        state, events = cases.tick(rules, state, seed=tick)
        assert not events.done.any()
        for environment in range(2):
            if events.spawned[environment, 4]:
                assert spawned_at[environment] is None
                spawned_at[environment] = tick
        if tick == 1:
            _, obs, _ = api.perceive(rules, state)
            seconds = obs.globals[:, 1, GLOBAL["next_wave_own"]]
            torch.testing.assert_close(seconds, torch.tensor([9.0, 44.0]) * rules.tick_length)
    assert spawned_at == [70, 45]


def test_waves_land_every_interval_from_a_random_phase_per_team():
    rules = _wave_rules()
    state = api.reset(
        rules, TYPES, api.uniforms(rules, 256, generator=torch.Generator().manual_seed(4))
    )
    first = state.mode.wave_wait
    assert int(first.min()) >= 1 and int(first.max()) <= 60
    assert (first[:, 0] != first[:, 1]).float().mean() > 0.9  # each team its own draw
    assert first.float().std() > 10.0
    wave_wait, landed = frontline.waves_after(torch.tensor([[1, 2]]), torch.tensor([[60]]))
    assert torch.equal(landed, torch.tensor([[True, False]]))
    assert torch.equal(wave_wait, torch.tensor([[60, 1]]))


def test_frontline_respawns_at_real_scale_take_the_decided_minimum_and_interval():
    rules = api.rules_of(PRESETS["real_scale_frontline"])
    assert rules.pools[0].respawn_ticks * rules.tick_length == pytest.approx(8.0)
    assert rules.modes[0].wave_seconds == 20.0
    assert api.rules_of(PRESETS["frontline"]).modes[0].wave_seconds == 10.0  # the half clock


# ------------------------------------------------------------------ team deathmatch


def test_team_deathmatch_waits_a_fixed_8_s_at_real_scale_with_no_waves():
    config = dataclasses.replace(PRESETS["real_scale_deathmatch"], ships_per_team=2, num_fields=0)
    rules = api.rules_of(config)
    assert rules.pools[0].respawn_ticks == 240  # 8 s
    assert api.rules_of(PRESETS["deathmatch"]).pools[0].respawn_ticks == 120  # the half clock
    types = api.ship_types(ship_set="real-scale")
    uniforms = api.uniforms(rules, 1, generator=torch.Generator().manual_seed(5))
    state = api.reset(rules, types, uniforms)
    alive, wait = state.ships.alive.clone(), state.ships.respawn_wait.clone()
    alive[:, 3], wait[:, 3] = False, 240
    state = cases.edit_ships(state, alive=alive, respawn_wait=wait)
    for tick in range(1, 241):
        state, events = cases.tick(rules, state, seed=tick, types=types)
        assert bool(events.spawned[0, 3]) == (tick == 240), tick
    assert int(state.mode.wave_wait.abs().sum()) == 0


# ------------------------------------------------------------------ the observation


def test_both_views_see_the_locks_by_team_relative_rows():
    rules, mode_rules, _, mode, target = _captured_setup()
    rows = zone_rows(mode_rules, mode, rules.tick_length)  # (1, 2, Z, columns)
    seconds = LOCK * rules.tick_length
    mirrored = (-target) % frontline.ZONES  # team 1's row of the zone
    want_own = torch.zeros(1, 2, frontline.ZONES)
    want_enemy = torch.zeros(1, 2, frontline.ZONES)
    want_enemy[0, 0, target] = seconds  # team 0 sees the enemy held
    want_own[0, 1, mirrored] = seconds  # team 1 sees itself held
    torch.testing.assert_close(rows[..., ZONE["lock_own"]], want_own)
    torch.testing.assert_close(rows[..., ZONE["lock_enemy"]], want_enemy)


def test_each_view_sees_only_its_own_wave_clock():
    rules = _wave_rules()
    state = api.reset(
        rules, TYPES, api.uniforms(rules, 1, generator=torch.Generator().manual_seed(6))
    )
    alive = state.ships.alive.clone()
    alive[:, 4] = False
    state = cases.edit_ships(
        state, alive=alive, respawn_wait=torch.full_like(state.ships.respawn_wait, 7)
    )
    early = cases.edit_mode(state, wave_wait=torch.tensor([[40, 3]], dtype=torch.int32))
    late = cases.edit_mode(state, wave_wait=torch.tensor([[40, 57]], dtype=torch.int32))
    _, seen_early, _ = api.perceive(rules, early)
    _, seen_late, _ = api.perceive(rules, late)
    for part in ("ships", "fields", "globals", "zones"):  # team 0's view: nothing changes
        assert torch.equal(getattr(seen_early, part)[:, 0], getattr(seen_late, part)[:, 0]), part
    wave = GLOBAL["next_wave_own"]
    assert float(seen_early.globals[0, 0, wave]) == pytest.approx(40 * rules.tick_length)
    assert float(seen_early.globals[0, 1, wave]) == pytest.approx(3 * rules.tick_length)
    assert float(seen_late.globals[0, 1, wave]) == pytest.approx(57 * rules.tick_length)
    others = [index for index in range(seen_early.globals.shape[-1]) if index != wave]
    assert torch.equal(seen_early.globals[:, 1, others], seen_late.globals[:, 1, others])
    globals_ = mode_globals(rules, rules.modes[0], early.mode)
    assert float(globals_[0, 1, -1]) == pytest.approx(3 * rules.tick_length)


def test_modes_without_waves_show_no_timers():
    rules, state = cases.rules_and_state(GameConfig(2, 0, 2600.0), batch=1)
    _, obs, _ = api.perceive(rules, state)
    assert float(obs.globals[..., GLOBAL["next_wave_own"]].abs().sum()) == 0.0
    assert float(obs.zones[..., ZONE["lock_own"]].abs().sum()) == 0.0


# ------------------------------------------------------------------ tuning and version


def test_the_new_rules_are_live_settings_in_their_mode_groups():
    frontline_rules = api.rules_of(PRESETS["real_scale_frontline"])
    deathmatch_rules = api.rules_of(PRESETS["real_scale_deathmatch"])
    keys = {setting.key: setting for setting in api.tuning.settings_for(frontline_rules)}
    for key in ("mode.lock_seconds_per_root_km", "mode.wave_seconds"):
        assert keys[key].group == api.tuning.FRONTLINE_GROUP
        assert key in RULE_UNITS
    respawn = api.tuning.BY_KEY["pool.respawn_seconds"]
    assert api.tuning.shown(respawn, frontline_rules)[0] == api.tuning.FRONTLINE_GROUP
    assert api.tuning.shown(respawn, deathmatch_rules)[0] == api.tuning.TEAM_DEATHMATCH_GROUP
    tuned = api.tuning.tuned(
        frontline_rules,
        {
            "mode.lock_seconds_per_root_km": 30.0,
            "mode.wave_seconds": 15.0,
            "pool.respawn_seconds": 5.0,
        },
    )
    assert tuned.modes[0].lock_seconds_per_root_km == 30.0 and tuned.modes[0].wave_seconds == 15.0
    # live values: the compiled tick is keyed on the static rules, which do not change
    assert api.tuning.static_of(tuned) == api.tuning.static_of(frontline_rules)


def test_the_game_version_names_the_lock_and_the_waves():
    assert game.VERSION >= 8 and "capture lock" in game.__doc__
