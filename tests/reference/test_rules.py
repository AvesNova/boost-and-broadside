"""Constructed situations with known answers: firing, bullets, hits, damage, respawn,
episode end, visibility (SPEC 5-10). Fields are moved out of play unless a test needs one."""

import math

import numpy as np
import pytest

from helpers import H, PRESETS, actions, aspect, far_fields, first_move, fresh, map_uniform_count, place, step

COST = PRESETS["squad"]["bullet_cost"]


def duel(arm):
    st = far_fields(fresh(arm, "duel"))
    place(st, 0, 0.0, 0.0, 0.0)
    place(st, 1, 700.0, 700.0, math.pi)
    st["E"][:] = 4000.0
    return st


# ------------------------------------------------------------------ firing and bullets

def test_firing_costs_energy_and_starts_a_bullet(arm):
    st = duel(arm)
    fired, _, _ = step(arm, "duel", st, actions(1, 2, shoot=1))
    calm, _, _ = step(arm, "duel", st, actions(1, 2, shoot=0))
    assert fired["fired"][0, 0] and not calm["fired"][0, 0]
    assert abs((calm["E"][0, 0] - fired["E"][0, 0]) - COST) < 0.02
    assert fired["ttl"][0, 0, 0] == 11 and fired["cursor"][0, 0] == 1 and fired["gun_wait"][0, 0] == 1
    # First move starts at the tick-start position, at muzzle + ship speed along the nose,
    # less half a tick of drag.
    muzzle = PRESETS["squad"]["muzzle_speed"]
    assert abs(fired["bx"][0, 0, 0] - first_move(100.0 + muzzle)) < 0.05
    assert abs(fired["by"][0, 0, 0]) < 1e-3


def test_fire_interval_and_energy_floor(arm):
    st = duel(arm)
    pattern = []
    for _ in range(6):
        st, _, _ = step(arm, "duel", st, actions(1, 2, shoot=1))
        pattern.append(bool(st["fired"][0, 0]))
    assert pattern == [True, False, True, False, True, False]
    st = duel(arm)
    st["E"][0, 0] = COST - 1.0
    st, _, _ = step(arm, "duel", st, actions(1, 2, shoot=1))
    assert not st["fired"][0, 0]


def test_bullet_lives_twelve_moves(arm):
    st = duel(arm)
    st, _, _ = step(arm, "duel", st, actions(1, 2, shoot=1))
    alive_after = []
    for _ in range(12):
        alive_after.append(int(st["ttl"][0, 0, 0]))
        st, _, _ = step(arm, "duel", st, actions(1, 2))
    assert alive_after == list(range(11, -1, -1))
    assert st["ttl"][0, 0, 0] == 0


# ------------------------------------------------------------------ hits

def crossing(arm, preset="duel", target=1, ahead=9.0, shield=100.0):
    """Shooter 0 at the origin heading +x; `target` `ahead` metres in front, crossing +y."""
    st = far_fields(fresh(arm, preset))
    N = st["x"].shape[1]
    for i in range(N):
        place(st, i, 2000.0 + 50.0 * i, -2000.0, 0.0)  # everyone else far away
    place(st, 0, 0.0, 0.0, 0.0)
    place(st, target, ahead, 0.0, math.pi / 2)
    st["shield"][:] = 100.0
    st["shield"][0, target] = shield
    st["E"][:] = 4000.0
    act = actions(1, N)
    act[0, 0, 2] = 1
    return st, act


def test_relative_sweep_catches_a_pass_through(arm):
    """Both endpoints are more than 8 m apart; only the swept relative motion meets."""
    st, act = crossing(arm)
    new, ev, _ = step(arm, "duel", st, act)
    assert ev["damage"][0, 0, 1] > 0, "a bullet passing through the target within one tick must hit"
    assert new["ttl"][0, 0, 0] == 0
    want = 12.0 * aspect(math.pi / 2)
    assert abs(ev["damage"][0, 0, 1] - want) < 0.1, (ev["damage"][0, 0, 1], want)
    assert abs(new["shield"][0, 1] - (100.0 - ev["damage"][0, 0, 1])) < 1e-3
    assert abs(new["shield_wait"][0, 1] - 6.0) < 1e-6


def test_first_contact_only(arm):
    st, act = crossing(arm, "squad", target=5, ahead=9.0)
    place(st, 6, 15.0, 0.0, math.pi / 2)
    _, ev, _ = step(arm, "squad", st, act)
    assert ev["damage"][0, 0, 5] > 0 and ev["damage"][0, 0, 6] == 0


def test_friendly_fire_hits_allies(arm):
    """A bullet is a bullet: ship 1 is on the shooter's team and is hit all the same."""
    st, act = crossing(arm, "squad", target=1, ahead=9.0)
    new, ev, _ = step(arm, "squad", st, act)
    assert ev["damage"][0, 0, 1] > 0 and new["ttl"][0, 0, 0] == 0


def test_an_ally_kill_credits_the_shooter(arm):
    st, act = crossing(arm, "squad", target=1, ahead=9.0, shield=0.0)
    _, ev, _ = step(arm, "squad", st, act)
    assert ev["died"][0, 1] and ev["killer"][0, 1] == 0


def test_a_bullet_never_hits_its_shooter(arm):
    """The bullet starts on the shooter (zero miss distance at tau = 0) and still flies on."""
    st, act = crossing(arm, "squad", target=1, ahead=9.0)
    place(st, 1, 2000.0, 2000.0, 0.0)  # nobody in front
    st["w"][0, 0] = 1e-3  # the shooter barely moves, so it stays inside its own bullet's start
    new, ev, _ = step(arm, "squad", st, act)
    assert ev["damage"].sum() == 0 and new["ttl"][0, 0, 0] == 11


def test_head_on_hits_do_the_head_on_fraction(arm):
    st, act = crossing(arm, ahead=9.0)
    place(st, 1, 9.0, 0.0, math.pi)  # facing the shooter
    st["w"][0, 1] = 1e-3
    _, ev, _ = step(arm, "duel", st, act)
    assert abs(ev["damage"][0, 0, 1] - 12.0 * PRESETS["duel"]["head_on_fraction"]) < 0.05


def test_exposed_core_dies_and_credits_the_shooter(arm):
    st, act = crossing(arm, shield=0.0)
    new, ev, _ = step(arm, "duel", st, act)
    assert ev["died"][0, 1] and ev["killer"][0, 1] == 0
    assert not new["alive"][0, 1] and new["respawn"][0, 1] == 240
    assert abs(new["w"][0, 1] - 1e-6) < 1e-9 and new["r"][0, 1] == 0.0


def test_a_shield_absorbs_the_hit_that_empties_it(arm):
    st, act = crossing(arm, shield=5.0)
    new, ev, _ = step(arm, "duel", st, act)
    assert not ev["died"][0, 1] and new["alive"][0, 1] and new["shield"][0, 1] == 0.0


# ------------------------------------------------------------------ shields, boundary

def test_shield_recharge_after_the_delay(arm):
    st = duel(arm)
    st["shield"][0, 0] = 50.0
    st["shield_wait"][0, 0] = 0.01
    st, _, _ = step(arm, "duel", st, actions(1, 2))
    assert abs(st["shield"][0, 0] - (50.0 + 12.0 * (H - 0.01))) < 1e-4
    assert st["shield_wait"][0, 0] == 0.0
    st, _, _ = step(arm, "duel", st, actions(1, 2))
    assert abs(st["shield"][0, 0] - (50.0 + 12.0 * (2 * H - 0.01))) < 1e-4


def test_boundary_damage_grows_with_depth(arm):
    st = duel(arm)
    R = PRESETS["duel"]["arena_radius"]
    place(st, 0, R + 100.0, 0.0, 0.0)
    new, ev, _ = step(arm, "duel", st, actions(1, 2))
    depth = math.hypot(new["x"][0, 0], new["y"][0, 0]) - R
    assert abs(ev["boundary_damage"][0, 0] - (5.0 + 0.05 * depth) * H) < 1e-4
    assert ev["killer"][0, 0] == -1
    st["shield"][0, 0] = 0.0
    new, ev, _ = step(arm, "duel", st, actions(1, 2))
    assert ev["died"][0, 0] and ev["killer"][0, 0] == -1


# ------------------------------------------------------------------ respawn, reset

def tick_uniforms(preset, rotation=(0.0, 0.0), keys=None):
    """(1, U) tick uniforms with a chosen formation rotation per team and keys per ship."""
    p = PRESETS[preset]
    N, m = p["num_ships"], map_uniform_count(preset)
    u = np.zeros((1, m + 2 + N), np.float32)
    u[0, m:m + 2] = rotation
    if keys is not None:
        u[0, m + 2:] = keys
    return u


def test_respawn_places_the_ship_on_the_shuffled_rotated_formation(arm):
    """Squad, team 1 keys (0.9, 0.1, 0.5, 0.7, 0.3): slot ranks 4, 0, 2, 3, 1. Ships 6 and 8
    (slots 1 and 3) respawn together at phases 0/5 + 0.25 and 3/5 + 0.25."""
    p = PRESETS["squad"]
    st = far_fields(fresh(arm, "squad"))
    for i in range(10):
        place(st, i, -1500.0 + 300.0 * i, 0.0, 0.0)
    for i in (6, 8):
        st["alive"][0, i] = False
        st["respawn"][0, i] = 1
    keys = [0.0] * 5 + [0.9, 0.1, 0.5, 0.7, 0.3]
    uni = tick_uniforms("squad", rotation=(0.6, 0.25), keys=keys)
    spread = np.zeros((1, 10, 2), np.float32)
    new, _, _ = arm.step("squad", st, actions(1, 10), spread, uni)
    ax, ay = st["axis_x"][0], st["axis_y"][0]
    rho = p["spawn_radius"] * p["arena_radius"] / 2600.0
    cx, cy = 0.8 * p["arena_radius"] * ax, 0.8 * p["arena_radius"] * ay
    for ship, rank in ((6, 0), (8, 3)):
        phase = rank / 5 + 0.25
        assert new["alive"][0, ship] and new["spawned"][0, ship]
        assert abs(new["x"][0, ship] - (cx + rho * math.cos(2 * math.pi * phase))) < 1e-2
        assert abs(new["y"][0, ship] - (cy + rho * math.sin(2 * math.pi * phase))) < 1e-2
        assert abs(new["c"][0, ship] + ax) < 1e-5 and abs(new["s"][0, ship] + ay) < 1e-5
    # Ship 6 is team 1's slot 1: a heavy (full heavy shield and spawn energy).
    assert new["shield"][0, 6] == 160.0 and new["E"][0, 6] == 5600.0 and new["gun_wait"][0, 6] == 0
    assert not new["spawned"][0, 5]


def test_reset_formation_is_evenly_spread_per_team(arm):
    rng = np.random.default_rng(5)
    from helpers import reset_uniform_count
    st = arm.reset("squad", rng.random((8, reset_uniform_count("squad"))).astype(np.float32))
    for team in (0, 1):
        x, y = st["x"][:, 5 * team:5 * team + 5], st["y"][:, 5 * team:5 * team + 5]
        ang = np.sort(np.mod(np.arctan2(y - y.mean(1, keepdims=True), x - x.mean(1, keepdims=True)), 2 * np.pi), 1)
        gaps = np.diff(np.concatenate([ang, ang[:, :1] + 2 * np.pi], 1), axis=1)
        assert np.allclose(gaps, 2 * np.pi / 5, atol=1e-3), gaps


@pytest.mark.parametrize("preset", ["duel", "squad", "battle"])
def test_maps_are_point_symmetric(arm, preset):
    rng = np.random.default_rng(6)
    from helpers import reset_uniform_count
    st = arm.reset(preset, rng.random((4, reset_uniform_count(preset))).astype(np.float32))
    M = PRESETS[preset]["num_fields"]
    P = M // 2
    for k, sign in (("fx", -1.0), ("fy", -1.0), ("fr", 1.0), ("fw", 1.0), ("flevel", 1.0)):
        np.testing.assert_allclose(st[k][:, P:2 * P], sign * st[k][:, :P], rtol=1e-6, atol=1e-3)
    if M % 2:
        assert np.allclose(st["fx"][:, -1], 0.0) and np.allclose(st["fy"][:, -1], 0.0)


def test_spawned_ships_are_revealed(arm):
    """Ship 0 sits on its own spawn, ~1920 m from the enemy spawn: only the reveal shows ship 1."""
    st = duel(arm)
    R = PRESETS["duel"]["arena_radius"]
    ax, ay = float(st["axis_x"][0]), float(st["axis_y"][0])
    place(st, 0, -0.8 * R * ax, -0.8 * R * ay, math.atan2(-ay, -ax), speed=1e-3)
    st["alive"][0, 1] = False
    st["respawn"][0, 1] = 1
    st, _, obs = step(arm, "duel", st, actions(1, 2))
    assert st["spawned"][0, 1]
    assert obs["ships"][0, 0, 1, 12] == 1.0, "a ship that spawns this tick is visible to both teams"
    st, _, obs = step(arm, "duel", st, actions(1, 2))
    assert obs["ships"][0, 0, 1, 12] == 0.0, "the reveal lasts only the tick of the spawn"


def test_dead_ships_wait_and_stay_put(arm):
    st = duel(arm)
    st["alive"][0, 1] = False
    st["respawn"][0, 1] = 100
    new, _, _ = step(arm, "duel", st, actions(1, 2, power=1, stick=6))
    assert new["respawn"][0, 1] == 99 and not new["alive"][0, 1]
    assert new["x"][0, 1] == st["x"][0, 1] and new["y"][0, 1] == st["y"][0, 1]


def test_episode_end_resets(arm):
    st = duel(arm)
    st, _, _ = step(arm, "duel", st, actions(1, 2, shoot=1))  # a bullet in flight
    st["t"][0] = PRESETS["duel"]["episode_ticks"] - 1
    new, ev, obs = step(arm, "duel", st, actions(1, 2))
    assert ev["done"][0] and new["t"][0] == 0
    assert new["spawned"].all() and new["alive"].all() and (new["ttl"] == 0).all() and (new["cursor"] == 0).all()
    assert obs["globals"][0, 0] == 0.0


# ------------------------------------------------------------------ visibility

def vis_state(arm):
    st = far_fields(fresh(arm, "duel"))
    place(st, 0, -300.0, 0.0, 0.0)
    place(st, 1, 300.0, 0.0, math.pi)
    st["fired"][:] = False
    st["spawned"][:] = False
    return st


def core_at(st, x, y, radius):
    """Field 0 becomes a core of the given radius (band 40 m) at (x, y)."""
    st["fx"][0, 0], st["fy"][0, 0], st["fr"][0, 0], st["fw"][0, 0] = x, y, radius + 20.0, 40.0


def seen(arm, st, team=0, ship=1):
    return bool(arm.observe("duel", st)["ships"][0, team, ship, 12])


def test_sight_lines_and_cores(arm):
    st = vis_state(arm)
    assert seen(arm, st)
    core_at(st, 0.0, 0.0, 100.0)
    assert not seen(arm, st), "a core between the ships blocks sight"
    core_at(st, 0.0, 150.0, 100.0)
    assert seen(arm, st), "a core beside the line does not"
    core_at(st, 300.0, 0.0, 100.0)
    assert not seen(arm, st), "a target inside a core is hidden from outside"
    place(st, 0, 250.0, 30.0, 0.0)
    assert seen(arm, st), "two ships inside the same core see each other"


def test_a_field_without_a_flat_interior_has_no_core(arm):
    st = vis_state(arm)
    st["fx"][0, 0], st["fy"][0, 0], st["fr"][0, 0], st["fw"][0, 0] = 0.0, 0.0, 15.0, 40.0
    assert seen(arm, st), "radius below half the band: the core radius is max(R - b/2, 0) = 0"


def test_vision_range_and_reveals(arm):
    st = vis_state(arm)
    place(st, 1, -300.0 + 1030.0, 0.0, math.pi)
    assert not seen(arm, st), "beyond 1024 m"
    st["fired"][0, 1] = True
    assert seen(arm, st), "firing reveals the shooter"
    st["fired"][0, 1] = False
    st["spawned"][0, 1] = True
    assert seen(arm, st), "spawning reveals the ship"
    st["spawned"][0, 1] = False
    place(st, 1, -300.0 + 1000.0, 0.0, math.pi)
    assert seen(arm, st)
    st["alive"][0, 1] = False
    obs = arm.observe("duel", st)["ships"]
    assert obs[0, 0, 1, 12] == 0.0 and obs[0, 1, 1, 12] == 1.0, "dead enemies are hidden, own ships never are"


def test_hidden_rows_are_masked(arm):
    st = vis_state(arm)
    place(st, 1, 2000.0, 0.0, math.pi)
    st["respawn"][0, 1] = 0
    row = arm.observe("duel", st)["ships"][0, 0, 1]
    shown = {11, 12, 13, 15, 16}  # alive, visible, ally, respawn, type
    assert all(row[i] == 0.0 for i in range(row.size) if i not in shown)
    assert row[11] == 1.0 and row[12] == 0.0 and row[13] == 0.0


