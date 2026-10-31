"""Constructed situations with known answers: firing, bullets, hits, damage, respawn,
episode end, visibility (SPEC 5-10). Fields are moved out of play unless a test needs one."""

import math

import numpy as np

from helpers import H, PRESETS, actions, aspect, far_fields, fresh, place, step

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
    # First move starts at the tick-start position, at muzzle + ship speed along the nose.
    assert abs(fired["bx"][0, 0, 0] - (100.0 + 465.0) * H) < 0.05
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


def test_allied_bullets_pass_through(arm):
    st, act = crossing(arm, "squad", target=1, ahead=9.0)  # ship 1 is on team 0
    new, ev, _ = step(arm, "squad", st, act)
    assert ev["damage"].sum() == 0 and new["ttl"][0, 0, 0] == 11


def test_head_on_hits_do_a_tenth(arm):
    st, act = crossing(arm, ahead=9.0)
    place(st, 1, 9.0, 0.0, math.pi)  # facing the shooter
    st["w"][0, 1] = 1e-3
    _, ev, _ = step(arm, "duel", st, act)
    assert abs(ev["damage"][0, 0, 1] - 1.2) < 0.05


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

def test_respawn_places_the_ship_at_its_spawn(arm):
    st = duel(arm)
    st["alive"][0, 1] = False
    st["respawn"][0, 1] = 1
    st["t"][0] = 77
    new, _, obs = step(arm, "duel", st, actions(1, 2))
    p = PRESETS["duel"]
    assert new["alive"][0, 1] and new["spawned"][0, 1] and not new["spawned"][0, 0]
    ax, ay = st["axis_x"][0], st["axis_y"][0]
    phase = (1 * 0.61803398875 + 77 * 0.38196601125) % 1.0
    rho = p["spawn_radius"] * p["arena_radius"] / 2600.0
    cx, cy = 0.8 * p["arena_radius"] * ax, 0.8 * p["arena_radius"] * ay
    assert abs(new["x"][0, 1] - (cx + rho * math.cos(2 * math.pi * phase))) < 1e-2
    assert abs(new["y"][0, 1] - (cy + rho * math.sin(2 * math.pi * phase))) < 1e-2
    assert abs(new["c"][0, 1] + ax) < 1e-5 and abs(new["s"][0, 1] + ay) < 1e-5
    assert new["shield"][0, 1] == 100.0 and new["E"][0, 1] == 4000.0 and new["gun_wait"][0, 1] == 0


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


