"""Modes (modes spec 8): team deathmatch's score, win and spawns, grouped spawns,
Frontline's map, protection and an end-to-end capture, the objective and shield events,
the mode's observation, and the tick compiled."""

import dataclasses
import math
import re

import modes_cases as cases
import pytest
import torch

from bnb.contracts import REWARD_LEVELS, GameConfig
from bnb.contracts.obs import GLOBAL, SHIP, ZONE
from bnb.game import api
from bnb.game.modes import frontline, spawns
from bnb.game.modes.rules import PRESETS
from bnb.game.physics.medium import medium
from bnb.game.physics.spawning import TickUniforms, formation_rank
from bnb.game.state import by_team, replace
from bnb.learn import rewards
from bnb.runtime.train import TRAINING_MIXES

TYPES = cases.TYPES
REAL_SCALE = api.ship_types(ship_set="real-scale")

# ------------------------------------------------------------------ configuration


def test_modes_take_their_clocks_and_presets_build():
    def game(**config):
        (pool,) = api.rules_of(GameConfig(5, 10, 2600.0, **config)).pools
        return pool

    assert game().episode_ticks == 9000
    assert game().respawn_ticks == 120  # 4 s (8 s at real scale)
    assert game(mode="frontline").respawn_ticks == 120  # the minimum, then the next wave
    assert game(mode="sandbox").episode_ticks == 1800
    assert game(episode_seconds=2.0).episode_ticks == 60
    for name, config in PRESETS.items():
        rules = api.rules_of(config)
        # the training presets are 5v5; the fleet-size ones (`<mode>_<n>v<n>_<arena|density>`)
        # are as named, and the arena and Clover are 2v2s
        fleet = re.fullmatch(r"\w+_(\d+)v\1_(arena|density)", name)
        size = int(fleet.group(1)) if fleet else 2 if name in ("arena", "clover") else 5
        assert rules.num_ships == 2 * size, name
    assert api.rules_of(PRESETS["deathmatch_wide"]).pools[0].mode.spawn_groups == (2, 2, 1)
    with pytest.raises(ValueError):
        GameConfig(5, 10, 2600.0, spawn_groups=(2, 2))  # must sum to the team's size
    with pytest.raises(ValueError):
        GameConfig(5, 10, 2600.0, mode="frontline", spawn_groups=(5,))
    with pytest.raises(ValueError):
        GameConfig(5, 10, 2600.0, mode="capture_the_flag")


def test_tick_uniform_counts_are_exact_functions_of_the_sizes():
    N, M, K = 10, 10, 8
    game = 4 * M + 1 + 2 + N
    assert api.rules_of(GameConfig(5, M, 2600.0, mode="sandbox")).tick_uniform_count == game
    assert api.rules_of(GameConfig(5, M, 2600.0)).tick_uniform_count == game + N * (2 * K + 1)
    grouped = api.rules_of(GameConfig(5, M, 2600.0, spawn_groups=(2, 2, 1)))
    assert grouped.tick_uniform_count == game + 2 * K * 3 + N * (2 * K + 2)
    assert (
        api.rules_of(GameConfig(5, M, 2600.0, mode="frontline")).tick_uniform_count
        == game + 4 + 2 * N
    )


def test_the_arena_is_a_2v2_dogfight_with_no_fields_or_zones():
    """The arena preset: team deathmatch of a Yongma and a Jujak a team, real-scale,
    on a small map with no fields and no zones, a killed ship respawning after one tick."""
    game = PRESETS["arena"]
    assert (game.mode, game.ships_per_team, game.num_fields) == ("team_deathmatch", 2, 0)
    assert game.ship_set == "real-scale" and game.arena_radius == 1950.0
    lineup = api.preset_lineup("arena")
    names = api.type_names("real-scale")
    assert [names[kind] for kind in lineup] == ["Yongma", "Jujak"]
    assert api.preset_lineup("real_scale_deathmatch") is None
    assert "arena" not in {name for mix in TRAINING_MIXES.values() for name, _ in mix}
    rules = dataclasses.replace(api.rules_of(game), team_types=lineup)
    (pool,) = rules.pools
    assert pool.respawn_ticks == 1 and rules.num_fields == 0
    state = api.reset(rules, REAL_SCALE, api.uniforms(rules, 4, generator=torch.Generator()))
    assert state.fields.x.shape == (4, 0)
    assert not state.mode.zone_radius.any()  # no zone in team deathmatch
    assert torch.equal(state.ships.ship_type[0], torch.tensor(lineup * 2, dtype=torch.int32))
    log_index, gradient_x, gradient_y = medium(state.fields, state.ships.x, state.ships.y)
    assert not (log_index.any() or gradient_x.any() or gradient_y.any())  # a uniform medium


def test_an_arena_ship_killed_spawns_again_on_the_next_tick():
    """A ship that dies (here at the boundary, its shield already broken) is back on the
    next tick: the arena's respawn wait is one tick."""
    rules = api.rules_of(PRESETS["arena"])
    state = api.reset(rules, REAL_SCALE, api.uniforms(rules, 1, generator=torch.Generator()))
    far = torch.tensor([[10.0 * rules.pools[0].arena_radius, 0.0, 0.0, 0.0]])
    state = cases.edit_ships(
        state,
        x=state.ships.x + far,
        shield=torch.where(far > 0.0, 0.0, state.ships.shield),
    )
    state, events = cases.tick(rules, state, types=REAL_SCALE)
    assert events.died[0, 0] and not state.ships.alive[0, 0]
    assert state.ships.respawn_wait[0, 0] == 1
    state, events = cases.tick(rules, state, types=REAL_SCALE)
    assert state.ships.alive[0, 0] and state.ships.spawned[0, 0]
    assert state.ships.x[0, 0].abs() <= rules.pools[0].arena_radius


# ------------------------------------------------------------------ team deathmatch


def _boundary_kill(state, ships):
    """The state with these ships (sim indices) alive, unshielded and far outside the
    arena, so the boundary kills them this tick."""
    alive = state.ships.alive.clone()
    shield, x = state.ships.shield.clone(), state.ships.x.clone()
    for ship in ships:
        alive[:, ship], shield[:, ship], x[:, ship] = True, 0.0, 1e5
    return cases.edit_ships(state, alive=alive, shield=shield, x=x)


@pytest.mark.parametrize(
    "score, dying, result",
    [
        ((24, 20), (3,), 1),  # team 0 reaches 25 by a team-1 death
        ((20, 24), (0,), -1),
        ((24, 24), (0, 3), 0),  # both reach it on the same tick: a draw
    ],
)
def test_deathmatch_first_to_the_target_wins(score, dying, result):
    config = GameConfig(3, 2, 1200.0, kills_per_ship=int(25 / 3) + 1)  # target 27
    rules, state = cases.rules_and_state(config, batch=1)
    target = rules.pools[0].mode.kills_per_ship * rules.ships_per_team
    shift = target - 25
    state = cases.edit_mode(
        state, score=torch.tensor([[score[0] + shift, score[1] + shift]], dtype=torch.int32)
    )
    after, events = cases.tick(rules, _boundary_kill(state, dying))
    assert events.done.item() and events.result.item() == result
    assert after.mode.score.sum() == 0 and after.ships.spawned.all()  # a new match


def test_deathmatch_one_short_plays_on():
    rules, state = cases.rules_and_state(GameConfig(3, 2, 1200.0), batch=1)
    target = 5 * 3
    state = cases.edit_mode(state, score=torch.tensor([[target - 2, 0]], dtype=torch.int32))
    after, events = cases.tick(rules, _boundary_kill(state, (3,)))
    assert not events.done.item() and events.result.item() == 0
    assert after.mode.score.tolist() == [[target - 1, 0]]


@pytest.mark.parametrize("score, result", [((3, 5), -1), ((5, 3), 1), ((4, 4), 0)])
def test_deathmatch_clock_goes_to_the_higher_score(score, result):
    rules, state = cases.rules_and_state(GameConfig(3, 2, 1200.0), batch=1)
    state = cases.edit_mode(state, score=torch.tensor([score], dtype=torch.int32))
    state = replace(
        state,
        match=replace(
            state.match,
            episode_tick=torch.tensor([rules.pools[0].episode_ticks - 1], dtype=torch.int32),
        ),
    )
    _, events = cases.tick(rules, state)
    assert events.done.item() and events.result.item() == result


def test_every_death_scores_for_the_other_team():
    """Enemy kills, friendly fire and the boundary all score, and only on the other side."""
    causes = {"enemy": 0, "ally": 0, "boundary": 0}
    S = 3
    for before, events, after in cases.play(cases.MODES["team_deathmatch"], batch=32, ticks=400):
        live = ~events.done
        deaths = by_team(events.died.int()).sum(-1)
        gained = after.mode.score - before.mode.score
        assert torch.equal(gained[live], deaths.flip(-1)[live])
        killer, died = events.killer, events.died
        team = torch.arange(2 * S) // S
        killer_team = torch.where(killer >= 0, killer // S, -1)
        causes["boundary"] += int((died & (killer < 0)).sum())
        causes["ally"] += int((died & (killer_team == team)).sum())
        causes["enemy"] += int((died & (killer >= 0) & (killer_team != team)).sum())
    assert all(count > 0 for count in causes.values()), causes


def _placement(rules, uniforms, x, y, alive, reset, keys):
    """Team deathmatch placement in a one-game batch."""
    (pool,) = rules.pools
    arena_radius = torch.full((x.shape[0],), pool.arena_radius)
    return spawns.deathmatch_placement(
        rules, pool.mode, arena_radius, uniforms, x, y, alive, reset, keys
    )


def _uniform_layout(rules, uniforms):
    """The team-deathmatch candidates as the spec draws them: (x, y) (B, N, K), heading."""
    K, B, N = rules.pools[0].mode.spawn_candidates, uniforms.shape[0], rules.num_ships
    per_ship = TickUniforms(rules, uniforms).mode.reshape(B, N, -1)
    distance = rules.pools[0].arena_radius * per_ship[..., :K].sqrt()
    angle = 2.0 * math.pi * per_ship[..., K : 2 * K]
    return distance * angle.cos(), distance * angle.sin(), 2.0 * math.pi * per_ship[..., 2 * K]


def test_deathmatch_respawn_takes_the_candidate_farthest_from_living_enemies():
    rules = api.rules_of(GameConfig(4, 2, 2600.0))
    (pool,) = rules.pools
    generator = torch.Generator().manual_seed(3)
    B, N = 256, rules.num_ships
    uniforms = api.uniforms(rules, B, generator=generator)
    x = (torch.rand(B, N, generator=generator) - 0.5) * 4000
    y = (torch.rand(B, N, generator=generator) - 0.5) * 4000
    alive = torch.rand(B, N, generator=generator) < 0.6
    keys = TickUniforms(rules, uniforms).keys
    place = _placement(
        rules,
        TickUniforms(rules, uniforms).mode,
        x,
        y,
        alive,
        torch.zeros(B, dtype=torch.bool),
        keys,
    )
    cx, cy, heading = _uniform_layout(rules, uniforms)
    team = torch.arange(N) // rules.ships_per_team
    for b in range(B):
        for i in range(N):
            enemies = alive[b] & (team != team[i])
            if enemies.any():
                d = (
                    (
                        (cx[b, i, :, None] - x[b, enemies]) ** 2
                        + (cy[b, i, :, None] - y[b, enemies]) ** 2
                    )
                    .min(-1)
                    .values
                )
                k = int(d.argmax())
            else:
                k = 0
            assert math.isclose(place.x[b, i], cx[b, i, k], abs_tol=1e-3)
            assert math.isclose(place.y[b, i], cy[b, i, k], abs_tol=1e-3)
    # Game v4: the heading is the uniform draw spread over the directions that leave room to
    # turn away from the edge (modes spec 4), not the plain uniform angle; written out here from
    # the spec (angle off the outward normal, at least acos(d / D) within D of the edge).
    distance = place.x.hypot(place.y)
    room = ((pool.arena_radius - distance) / rules.spawn_turn_distance).clamp(0.0, 1.0)
    least = room.acos()
    drawn = math.pi * (2.0 * heading / (2.0 * math.pi) - 1.0)
    off_normal = torch.copysign(least + drawn.abs() * (1.0 - least / math.pi), drawn)
    angle = torch.atan2(place.y, place.x) + off_normal
    torch.testing.assert_close(place.heading_cos, angle.cos(), atol=1e-4, rtol=0)
    torch.testing.assert_close(place.heading_sin, angle.sin(), atol=1e-4, rtol=0)


def test_deathmatch_reset_is_one_symmetric_pass():
    """Each ship picks among its candidates by distance to the enemies' candidate 0, so
    swapping the two teams' draws swaps their places."""
    rules = api.rules_of(GameConfig(3, 2, 2600.0))
    generator = torch.Generator().manual_seed(4)
    B, N, S = 64, rules.num_ships, rules.ships_per_team
    uniforms = api.uniforms(rules, B, generator=generator)
    tick = TickUniforms(rules, uniforms)
    zeros = torch.zeros(B, N)
    reset = torch.ones(B, dtype=torch.bool)
    place = _placement(rules, tick.mode, zeros, zeros, zeros.bool(), reset, tick.keys)
    swapped = tick.mode.reshape(B, 2, S, -1).flip(1).reshape(B, -1)
    other = _placement(rules, swapped, zeros, zeros, zeros.bool(), reset, tick.keys)
    torch.testing.assert_close(other.x, place.x.roll(S, 1))
    torch.testing.assert_close(other.y, place.y.roll(S, 1))
    # and the choice is the farthest from the enemies' provisional points
    cx, cy, _ = _uniform_layout(rules, uniforms)
    team = torch.arange(N) // S
    enemy = team[:, None] != team
    d = (cx[..., :, None] - cx[:, None, None, :, 0]) ** 2 + (
        cy[..., :, None] - cy[:, None, None, :, 0]
    ) ** 2
    nearest = torch.where(enemy[None, :, None, :], d, math.inf).amin(-1)
    pick = nearest.argmax(-1, keepdim=True)
    torch.testing.assert_close(place.x, cx.gather(-1, pick)[..., 0])


def test_deathmatch_candidates_are_uniform_over_the_arena():
    """With no living enemy a respawn takes candidate 0: uniform over the disc."""
    rules = api.rules_of(GameConfig(5, 2, 2000.0))
    B = 4096
    uniforms = api.uniforms(rules, B, generator=torch.Generator().manual_seed(5))
    tick = TickUniforms(rules, uniforms)
    zeros = torch.zeros(B, rules.num_ships)
    place = _placement(
        rules, tick.mode, zeros, zeros, zeros.bool(), torch.zeros(B, dtype=torch.bool), tick.keys
    )
    radius = torch.hypot(place.x, place.y) / rules.pools[0].arena_radius
    assert radius.max() <= 1.0
    area = (radius**2).flatten()  # uniform on [0, 1] for a uniform disc
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        assert abs(float((area < q).float().mean()) - q) < 0.01
    angle = torch.atan2(place.y, place.x).flatten()
    assert abs(float(angle.cos().mean())) < 0.02 and abs(float(angle.sin().mean())) < 0.02


# ------------------------------------------------------------------ grouped spawns


def _group_centres(rules, uniforms):
    """Centre 0 uniform in 0.85 R_a; centre g the best of K draws by the nearest earlier."""
    G, K = len(rules.pools[0].mode.spawn_groups), rules.pools[0].mode.spawn_candidates
    drawn = TickUniforms(rules, uniforms).mode[:, : 2 * K * G].reshape(-1, G, K, 2)
    distance = 0.85 * rules.pools[0].arena_radius * drawn[..., 0].sqrt()
    angle = 2.0 * math.pi * drawn[..., 1]
    x, y = distance * angle.cos(), distance * angle.sin()
    centres = [torch.stack([x[:, 0, 0], y[:, 0, 0]], -1)]
    for g in range(1, G):
        earlier = torch.stack(centres, 1)  # (B, g, 2)
        candidates = torch.stack([x[:, g], y[:, g]], -1)  # (B, K, 2)
        nearest = torch.cdist(candidates, earlier).amin(-1)
        centres.append(candidates[torch.arange(len(x)), nearest.argmax(-1)])
    return torch.stack(centres, 1)


def test_grouped_reset_spreads_the_groups_and_fills_them_by_rank():
    config = GameConfig(5, 4, 8200.0, spawn_groups=(2, 2, 1))
    rules = api.rules_of(config)
    B = 128
    uniforms = api.uniforms(rules, B, generator=torch.Generator().manual_seed(6))
    state = api.reset(rules, TYPES, uniforms)
    centres = _group_centres(rules, uniforms)
    position = torch.stack([state.ships.x, state.ships.y], -1)
    distance = torch.cdist(position, centres)  # (B, N, G)
    group = distance.argmin(-1)
    assert (distance.amin(-1) <= rules.pools[0].mode.group_radius + 1e-3).all()
    for team in range(2):
        counts = torch.stack(
            [(group[:, team * 5 : (team + 1) * 5] == g).sum(-1) for g in range(3)], -1
        )
        assert (counts == torch.tensor([2, 2, 1])).all()
    # spread: farther apart than independent uniform centres would be
    spread = torch.pdist(centres[0]).min()
    gaps = torch.stack([torch.pdist(c).min() for c in centres])
    independent = torch.rand(B, 3, generator=torch.Generator().manual_seed(7))
    radius = 0.85 * rules.pools[0].arena_radius * independent.sqrt()
    angle = 2 * math.pi * torch.rand(B, 3, generator=torch.Generator().manual_seed(8))
    plain = torch.stack([radius * angle.cos(), radius * angle.sin()], -1)
    plain_gaps = torch.stack([torch.pdist(c).min() for c in plain])
    assert gaps.mean() > 1.5 * plain_gaps.mean() and spread > 0


def test_grouped_reset_picks_by_the_enemies_of_its_own_group():
    """Brute force on a small arena, where groups overlap: each ship takes the candidate
    (in the 300 m disc around its group's centre, clipped into the disc R_a - 50 m)
    farthest from the provisional points of the enemies in its own group."""
    rules = api.rules_of(GameConfig(4, 2, 600.0, spawn_groups=(2, 1, 1)))
    B, N, S, K = 128, rules.num_ships, rules.ships_per_team, rules.pools[0].mode.spawn_candidates
    uniforms = api.uniforms(rules, B, generator=torch.Generator().manual_seed(18))
    tick = TickUniforms(rules, uniforms)
    zeros = torch.zeros(B, N)
    place = _placement(
        rules, tick.mode, zeros, zeros, zeros.bool(), torch.ones(B, dtype=torch.bool), tick.keys
    )
    centres = _group_centres(rules, uniforms)
    rank = formation_rank(rules, tick.keys)
    group = (rank >= 2).long() + (rank >= 3).long()
    per_ship = tick.mode[:, 2 * K * 3 :].reshape(B, N, -1)
    distance = 300.0 * per_ship[..., :K].sqrt()
    angle = 2.0 * math.pi * per_ship[..., K : 2 * K]
    centre = centres[torch.arange(B)[:, None], group]  # (B, N, 2)
    x = centre[..., :1] + distance * angle.cos()
    y = centre[..., 1:] + distance * angle.sin()
    reach = torch.hypot(x, y)
    assert (reach > 550.0).any(), "the case should clip some candidates"
    clip = torch.where(reach > 550.0, 550.0 / reach, 1.0)
    x, y = x * clip, y * clip
    team = torch.arange(N) // S
    rival = (team[:, None] != team) & (group[:, :, None] == group[:, None, :])
    d = (x[..., :, None] - x[:, None, None, :, 0]) ** 2 + (
        y[..., :, None] - y[:, None, None, :, 0]
    ) ** 2
    nearest = torch.where(rival[:, :, None, :], d, math.inf).amin(-1)
    pick = nearest.argmax(-1, keepdim=True)
    torch.testing.assert_close(place.x, x.gather(-1, pick)[..., 0])
    torch.testing.assert_close(place.y, y.gather(-1, pick)[..., 0])


def test_grouped_respawns_land_near_a_living_anchor_inside_the_arena():
    """A grouped respawn lands 250-600 m from a living ship, or on the edge of the disc
    R_a - 50 m when that ring reaches past it (it is clipped there); every grouped spawn is
    inside that disc."""
    config = cases.MODES["grouped"]
    edge = config.arena_radius - 50.0
    landed = clipped = 0
    for _, events, after in cases.play(config, ticks=300, seed=2):
        spawned = after.ships.spawned
        reach = torch.hypot(after.ships.x, after.ships.y)
        assert (reach[spawned] <= edge + 1e-2).all()
        respawned = spawned & ~events.done[:, None]
        if not respawned.any():
            continue
        position = torch.stack([after.ships.x, after.ships.y], -1)
        distance = torch.cdist(position, position)  # (B, N, N)
        anchors = (after.ships.alive & ~after.ships.spawned)[:, None, :]
        within = anchors & (distance >= 250.0 - 1e-2) & (distance <= 600.0 + 1e-2)
        for b, i in respawned.nonzero().tolist():
            if anchors[b, 0].any():
                on_edge = abs(float(reach[b, i]) - edge) < 1e-2
                assert within[b, i].any() or on_edge, (b, i)
                landed += 1
                clipped += on_edge
    assert landed > 20 and clipped > 0, (landed, clipped)


@pytest.mark.parametrize("mode", list(cases.MODES))
def test_spawn_layout_does_not_reveal_the_seat(mode):
    """Every mode: a view's own ships sit the same way whichever seat it has."""
    config = cases.MODES[mode]
    rules = api.rules_of(
        GameConfig(3, 4, 2600.0, mode=config.mode, spawn_groups=config.spawn_groups)
    )
    state = api.reset(
        rules, TYPES, api.uniforms(rules, 2048, generator=torch.Generator().manual_seed(9))
    )
    _, obs, _ = api.perceive(rules, state)
    S = rules.ships_per_team
    own = obs.ships[..., :S, :]
    dx = own[..., 0, SHIP["x"]] - own[..., SHIP["x"]].mean(-1)
    dy = own[..., 0, SHIP["y"]] - own[..., SHIP["y"]].mean(-1)
    angle = torch.atan2(dy, dx)
    spread = 1.0 - torch.hypot(angle.cos().mean(0), angle.sin().mean(0))
    assert (spread > 0.5).all(), spread
    # where a view's centre of mass sits, and how far its ships are from the map centre
    for column in ("x", "y"):
        mean = own[..., SHIP[column]].mean((0, 2))
        assert (mean.abs() < 0.1 * rules.pools[0].arena_radius).all(), (column, mean)
    reach = torch.hypot(own[..., SHIP["x"]], own[..., SHIP["y"]]).mean((0, 2))
    assert abs(float(reach[0] - reach[1])) < 0.05 * rules.pools[0].arena_radius


@pytest.mark.parametrize("mode", list(cases.MODES))
def test_mode_rows_do_not_reveal_the_seat(mode):
    """Every mode: at a match's start both views read the same globals and the same zone
    rows (row by row: roles, progress, direction, distances), and the ring's handedness
    from either view is either way, half the time. Through play, each view's zone roles
    by row are those of its own front."""
    config = cases.MODES[mode]
    rules = cases.lively(config)
    state = api.reset(
        rules,
        cases.LIVELY_TYPES,
        api.uniforms(rules, 1024, generator=torch.Generator().manual_seed(19)),
    )
    _, obs, _ = api.perceive(rules, state)
    # every global but the own team's next respawn wave, each team's own draw
    same = [GLOBAL[name] for name in GLOBAL if name != "next_wave_own"]
    assert torch.equal(obs.globals[:, 0, same], obs.globals[:, 1, same])
    zones = obs.zones
    shown = [
        ZONE[name]
        for name in (
            "radius",
            "role",
            "progress",
            "direction",
            "offensive_distance",
            "defensive_distance",
            "lock_own",
            "lock_enemy",
        )
    ]
    assert torch.equal(zones[:, 0][..., shown], zones[:, 1][..., shown])
    assert torch.equal(
        zones[..., ZONE["present"]] == 1, torch.full(zones.shape[:-1], config.mode == "frontline")
    )
    if config.mode == "frontline":
        assert torch.equal(zones[0, :, :, ZONE["role"]], torch.tensor([[2.0, 0, 1, 3, 4]] * 2))
        angle = torch.atan2(zones[..., ZONE["y"]], zones[..., ZONE["x"]])
        step = torch.remainder(angle[..., 1] - angle[..., 0] + math.pi, 2 * math.pi) - math.pi
        counterclockwise = (step > 0).float().mean(0)
        assert ((counterclockwise > 0.4) & (counterclockwise < 0.6)).all(), counterclockwise
    for _, _, after in cases.play(config, batch=8, ticks=120, seed=19):
        _, obs, _ = api.perceive(rules, after)
        if config.mode == "frontline":
            for view, sign in ((0, 1), (1, -1)):
                want = frontline.roles_from_front(sign * after.mode.front).float()
                assert torch.equal(obs.zones[:, view, :, ZONE["role"]], want)
                assert torch.equal(
                    obs.globals[:, view, GLOBAL["front"]], (sign * after.mode.front).float()
                )


# ------------------------------------------------------------------ Frontline

FRONTLINE = GameConfig(3, 7, 2600.0, mode="frontline")


def _reflect(x, y, angle):
    c, s = torch.cos(2 * angle), torch.sin(2 * angle)
    return c * x + s * y, s * x - c * y


def test_frontline_map_is_a_mirror_image_about_the_neutral_axis():
    rules, state = cases.rules_and_state(FRONTLINE, batch=256, seed=10)
    mode, fields = state.mode, state.fields
    axis = mode.ring_rotation[:, None]
    P = rules.num_fields // 2
    rx, ry = _reflect(fields.x[:, :P], fields.y[:, :P], axis)
    torch.testing.assert_close(fields.x[:, P : 2 * P], rx, atol=2e-3, rtol=0)
    torch.testing.assert_close(fields.y[:, P : 2 * P], ry, atol=2e-3, rtol=0)
    torch.testing.assert_close(fields.radius[:, P : 2 * P], fields.radius[:, :P])
    assert torch.equal(fields.level[:, P : 2 * P], fields.level[:, :P])
    last = torch.stack([fields.x[:, -1], fields.y[:, -1]], -1)  # odd M: on the axis
    direction = torch.stack([axis[:, 0].cos(), axis[:, 0].sin()], -1)
    across = last[:, 0] * direction[:, 1] - last[:, 1] * direction[:, 0]
    assert across.abs().max() < 2e-3
    # zones: zone k mirrors zone -k, with the mirror role; zone 0 is neutral on the axis
    zx, zy = _reflect(mode.zone_x, mode.zone_y, axis)
    mirror = (-torch.arange(5)) % 5
    torch.testing.assert_close(zx, mode.zone_x[:, mirror], atol=2e-3, rtol=0)
    torch.testing.assert_close(zy, mode.zone_y[:, mirror], atol=2e-3, rtol=0)
    assert (mode.role[:, 0] == frontline.NEUTRAL).all()
    assert torch.equal(mode.role[:, mirror], frontline.TEAM1_SPAWN - mode.role)
    distance = torch.hypot(mode.zone_x, mode.zone_y)
    torch.testing.assert_close(distance, torch.full_like(distance, 0.615 * 2600.0))
    assert torch.allclose(mode.zone_radius, torch.tensor(330.0))


def test_frontline_draws_both_handednesses_and_any_rotation():
    _, state = cases.rules_and_state(FRONTLINE, batch=256, seed=11)
    mode = state.mode
    handed = mode.handedness
    assert 0.35 < float((handed > 0).float().mean()) < 0.65
    step = torch.atan2(mode.zone_y[:, 1], mode.zone_x[:, 1]) - torch.atan2(
        mode.zone_y[:, 0], mode.zone_x[:, 0]
    )
    step = torch.remainder(step + math.pi, 2 * math.pi) - math.pi
    torch.testing.assert_close(step, handed * 2 * math.pi / 5, atol=1e-4, rtol=0)
    rotation = mode.ring_rotation
    assert abs(float(rotation.cos().mean())) < 0.15 and abs(float(rotation.sin().mean())) < 0.15


def _zone_centre(state, role):
    """(B, 2) the centre of the zone with this role (team 0's frame)."""
    mine = state.mode.role == role
    return torch.stack([(state.mode.zone_x * mine).sum(-1), (state.mode.zone_y * mine).sum(-1)], -1)


def _shot_at(state, target: int, shooter: int):
    """The state with one of the shooter's bullets 10 m behind the target, flying through
    it; a uniform medium."""
    state = cases.fields_away(state)
    bullets = state.bullets
    x, y = bullets.x.clone(), bullets.y.clone()
    moves = bullets.moves_left.clone()
    x[:, shooter, 0] = state.ships.x[:, target] - 10.0
    y[:, shooter, 0] = state.ships.y[:, target]
    moves[:, shooter, 0] = 5
    direction_x, direction_y = torch.zeros_like(x), torch.zeros_like(x)
    direction_x[:, shooter, 0] = 1.0
    zero = torch.zeros_like(x)
    bullets = replace(
        bullets,
        x=x,
        y=y,
        direction_x=direction_x,
        direction_y=direction_y,
        proper_speed=torch.full_like(x, 500.0),
        moves_left=moves,
        log_refractive_index=zero,
        log_index_gradient_x=zero,
        log_index_gradient_y=zero,
    )
    zeros = torch.zeros_like(state.ships.x)
    state = cases.edit_ships(
        state, log_refractive_index=zeros, log_index_gradient_x=zeros, log_index_gradient_y=zeros
    )
    return replace(state, bullets=bullets)


@pytest.mark.parametrize(
    "role, protected",
    [(frontline.TEAM0_SPAWN, True), (frontline.NEUTRAL, False), (frontline.TEAM1_SPAWN, False)],
)
def test_frontline_spawn_zone_protects_its_own_team(role, protected):
    rules, state = cases.rules_and_state(FRONTLINE, batch=2, seed=12)
    centre = _zone_centre(state, role)
    x, y = state.ships.x.clone(), state.ships.y.clone()
    x[:, 0], y[:, 0] = centre[:, 0], centre[:, 1]
    state = _shot_at(cases.edit_ships(state, x=x, y=y), target=0, shooter=3)
    after, events = cases.tick(rules, state)
    assert (after.bullets.moves_left[:, 3, 0] == 0).all(), "the bullet is spent either way"
    if protected:
        assert (events.damage == 0).all() and (events.shield_lost[:, 0] == 0).all()
    else:
        assert (events.damage[:, 3, 0] > 0).all() and (events.shield_lost[:, 0] > 0).all()


def _hold(state, attackers: int, team: int = 0):
    """The attacking team's first `attackers` ships parked in the zone it attacks (the
    rest at the neutral zone), the other team's in its own spawn zone."""
    S = state.ships.x.shape[1] // 2
    target = _zone_centre(state, frontline.TEAM1_DEFENSE if team == 0 else frontline.TEAM0_DEFENSE)
    home = _zone_centre(state, frontline.TEAM1_SPAWN if team == 0 else frontline.TEAM0_SPAWN)
    x, y = state.ships.x.clone(), state.ships.y.clone()
    away = _zone_centre(state, frontline.NEUTRAL)
    attacking, holding = (0, S) if team == 0 else (S, 0)
    for i in range(S):
        spot = target if i < attackers else away
        x[:, attacking + i], y[:, attacking + i] = spot[:, 0] + 20.0 * i, spot[:, 1]
        x[:, holding + i], y[:, holding + i] = home[:, 0] + 20.0 * i, home[:, 1]
    return cases.edit_ships(state, x=x, y=y)


def test_frontline_capture_end_to_end_through_the_game():
    """Three captures ahead win. Each capture moves the front one step and rotates the
    roles; the objective events are zero-sum every tick and sum to the captures."""
    rules, state = cases.rules_and_state(FRONTLINE, batch=1, seed=13)
    state = cases.fields_away(state)
    objective = torch.zeros(rules.num_ships)
    fronts, ticks = [], 0
    while True:
        roles = state.mode.role.clone()
        state, events = cases.tick(rules, _hold(state, attackers=3), seed=ticks)
        ticks += 1
        for amount in (events.objective_progress, events.objective):
            assert abs(float(amount.sum())) < 1e-5
        objective += events.objective[0]
        if events.done.item():
            break
        if state.mode.front.item() != (fronts[-1] if fronts else 0):
            fronts.append(state.mode.front.item())
            assert torch.equal(state.mode.role, frontline.roles_from_front(state.mode.front))
            assert not torch.equal(state.mode.role, roles)
            assert (state.mode.progress == 0).all()
        assert ticks < 2000
    assert fronts == [1, 2] and events.result.item() == 1
    # lead 3: H_3 = 11/6, so 10 s / H_3 = 5.45 s, 164 ticks per capture
    assert abs(ticks - 3 * math.ceil(10.0 / (11.0 / 6.0) * 30)) <= 3
    torch.testing.assert_close(objective[:3].sum(), torch.tensor(3.0))
    torch.testing.assert_close(objective[3:].sum(), torch.tensor(-3.0))
    assert state.mode.front.item() == 0 and (state.mode.progress == 0).all()  # a new match


def _outcome(events, num_ships: int) -> torch.Tensor:
    """The outcome each ship is paid on this tick, in match results (over its weight)."""
    team = torch.arange(num_ships) // (num_ships // 2)
    config = rewards.RewardConfig()
    levels, _ = rewards.reward_step(config, rewards.blank_ledger(1, num_ships), events, team)
    return levels[0, :, REWARD_LEVELS.index("outcome")] / config.outcome


@pytest.mark.parametrize("captures, team, result", [(1, 0, 1), (0, 0, 0), (1, 1, -1)])
def test_frontline_timeout_goes_to_the_front(captures, team, result):
    """At the clock the sign of the front decides: a team-1 capture ahead wins for team 1,
    and the outcome pays its ships +1 and team 0's -1."""
    rules, state = cases.rules_and_state(FRONTLINE, batch=1, seed=14)
    state = cases.fields_away(state)
    ticks = 0
    while abs(state.mode.front.item()) < captures:
        state, _ = cases.tick(rules, _hold(state, attackers=3, team=team), seed=ticks)
        ticks += 1
    assert state.mode.front.item() == captures * (1 - 2 * team)
    state = replace(
        state,
        match=replace(
            state.match,
            episode_tick=torch.tensor([rules.pools[0].episode_ticks - 1], dtype=torch.int32),
        ),
    )
    _, events = cases.tick(rules, _hold(state, attackers=0))
    assert events.done.item() and events.result.item() == result
    paid = torch.tensor([float(result)] * 3 + [float(-result)] * 3)
    torch.testing.assert_close(_outcome(events, rules.num_ships), paid)


def test_frontline_team1_wins_by_capture():
    """Three team-1 captures: the front reaches -3, team 1 wins, and the outcome pays it."""
    rules, state = cases.rules_and_state(FRONTLINE, batch=1, seed=21)
    state = cases.fields_away(state)
    fronts, ticks = [], 0
    while True:
        state, events = cases.tick(rules, _hold(state, attackers=3, team=1), seed=ticks)
        ticks += 1
        if events.done.item():
            break
        if state.mode.front.item() != (fronts[-1] if fronts else 0):
            fronts.append(state.mode.front.item())
        assert ticks < 2000
    assert fronts == [-1, -2] and events.result.item() == -1
    torch.testing.assert_close(
        _outcome(events, rules.num_ships), torch.tensor([-1.0] * 3 + [1.0] * 3)
    )


def _check_frontline_spawn(state, ships: slice, team: int):
    """The ships sit in their team's spawn zone of `state`, nose toward the zone it attacks."""
    home, attack = (
        (frontline.TEAM0_SPAWN, frontline.TEAM1_DEFENSE)
        if team == 0
        else (frontline.TEAM1_SPAWN, frontline.TEAM0_DEFENSE)
    )
    position = torch.stack([state.ships.x[:, ships], state.ships.y[:, ships]], -1)
    distance = (position - _zone_centre(state, home)[:, None]).norm(dim=-1)
    assert (distance <= 0.35 * 330.0 + 1e-3).all()
    to = _zone_centre(state, attack)[:, None] - position
    heading = torch.stack(
        [state.ships.heading_cos[:, ships], state.ships.heading_sin[:, ships]], -1
    )
    torch.testing.assert_close(heading, to / to.norm(dim=-1, keepdim=True))


def test_frontline_reset_spawns_in_the_spawn_zones_facing_the_attack():
    _, state = cases.rules_and_state(GameConfig(3, 2, 2600.0, mode="frontline"), batch=64, seed=15)
    _check_frontline_spawn(state, slice(0, 3), 0)
    _check_frontline_spawn(state, slice(3, 6), 1)


def test_frontline_respawn_on_a_capture_tick_uses_the_new_spawn_zone():
    """A team-1 ship whose wait ends on the tick team 0 captures spawns in team 1's spawn
    zone after the roles rotate, not before."""
    rules, state = cases.rules_and_state(FRONTLINE, batch=1, seed=22)
    state = cases.fields_away(state)
    capture_seconds = rules.pools[0].mode.capture_seconds
    rate = frontline.harmonic(torch.tensor(3)) * rules.tick_length / capture_seconds
    ticks = 0
    while True:
        attacked = state.mode.role == frontline.TEAM1_DEFENSE
        if (state.mode.progress[attacked] + rate >= 1.0).all():
            break  # the next tick captures
        state, _ = cases.tick(rules, _hold(state, attackers=3), seed=ticks)
        ticks += 1
    old_home = _zone_centre(state, frontline.TEAM1_SPAWN)
    alive, wait = state.ships.alive.clone(), state.ships.respawn_wait.clone()
    alive[:, 3], wait[:, 3] = False, 1
    state = cases.edit_ships(_hold(state, attackers=3), alive=alive, respawn_wait=wait)
    # team 1's respawn wave lands on this tick (team 0's does not: it has no dead ship)
    state = replace(state, mode=replace(state.mode, wave_wait=torch.tensor([[5, 1]])))
    after, events = cases.tick(rules, state, seed=ticks)
    assert after.mode.front.item() == 1 and not events.done.item()
    assert after.ships.spawned[0, 3] and after.ships.spawned.sum() == 1
    new_home = _zone_centre(after, frontline.TEAM1_SPAWN)
    assert float((new_home - old_home).norm()) > 2 * 0.35 * 330.0
    _check_frontline_spawn(after, slice(3, 4), 1)


def test_deathmatch_respawn_keeps_away_from_the_enemies_alive_after_this_ticks_deaths():
    """Through the game: ship 0 respawns on the tick enemy ship 3 dies at the arena's edge.
    Its candidates are scored against the surviving enemy (ship 2) only."""
    rules, state = cases.rules_and_state(GameConfig(2, 1, 2600.0), batch=64, seed=23)
    state = cases.fields_away(state)
    B, R = 64, rules.pools[0].arena_radius
    ships = state.ships
    alive, wait, shield = ships.alive.clone(), ships.respawn_wait.clone(), ships.shield.clone()
    x, y = ships.x.clone(), ships.y.clone()
    alive[:, 0], wait[:, 0] = False, 1  # due this tick
    x[:, 2], y[:, 2] = -0.5 * R, 0.0  # the survivor
    x[:, 3], y[:, 3], shield[:, 3] = R + 20.0, 0.0, 0.0  # dies to the boundary this tick
    state = cases.edit_ships(state, alive=alive, respawn_wait=wait, shield=shield, x=x, y=y)
    uniforms = api.uniforms(rules, B, generator=torch.Generator().manual_seed(24))
    after, events = api.step(
        rules,
        TYPES,
        state,
        cases.still(B, rules.num_ships),
        torch.zeros(B, rules.num_ships, 2),
        uniforms,
    )
    assert events.died[:, 3].all() and after.ships.spawned[:, 0].all()
    cx, cy, _ = _uniform_layout(rules, uniforms)

    def farthest(enemies):
        d = torch.stack(
            [
                (cx[:, 0] - after.ships.x[:, e, None]) ** 2
                + (cy[:, 0] - after.ships.y[:, e, None]) ** 2
                for e in enemies
            ]
        ).amin(0)
        pick = d.argmax(-1, keepdim=True)
        return cx[:, 0].gather(-1, pick)[:, 0]

    torch.testing.assert_close(after.ships.x[:, 0], farthest([2]))
    assert (farthest([2, 3]) != farthest([2])).any(), "the case should tell the rules apart"


# ------------------------------------------------------------------ events


@pytest.mark.parametrize("mode", list(cases.MODES))
def test_shield_events_match_the_shield_changes(mode):
    """shield = shield0 - lost + restored for every ship alive at tick start that did not
    spawn; nothing is lost or restored by the dead; deaths, respawns (none in elimination)
    and resets in play."""
    seen = dict(lost=0, restored=0, deaths=0, respawns=0, resets=0)
    # Since game v5 a recharge needs the delay's length of unbroken quiet flight after a hit,
    # which only the calm ships (a quarter of the environments) give and few of them survive
    # unhit; a batch of 128 sees it in every mode (16 saw none in team deathmatch).
    for before, events, after in cases.play(cases.MODES[mode], batch=128, ticks=250, seed=1):
        alive0 = before.ships.alive
        kept = alive0 & ~after.ships.spawned
        expected = before.ships.shield - events.shield_lost + events.shield_restored
        torch.testing.assert_close(after.ships.shield[kept], expected[kept], atol=1e-4, rtol=0)
        assert (events.shield_lost[~alive0] == 0).all()
        assert (events.shield_restored[~alive0] == 0).all()
        hurt = (events.damage.sum(1) + events.boundary_damage) > 0
        assert (events.shield_restored[hurt] == 0).all() and (events.shield_lost[~hurt] == 0).all()
        assert (events.shield_lost >= 0).all() and (events.shield_restored >= 0).all()
        seen["lost"] += int((events.shield_lost > 0).sum())
        seen["restored"] += int((events.shield_restored > 0).sum())
        seen["deaths"] += int(events.died.sum())
        seen["respawns"] += int((after.ships.spawned & ~events.done[:, None]).sum())
        seen["resets"] += int(events.done.sum())
    if mode == "elimination":
        assert seen.pop("respawns") == 0, seen
    assert all(count > 0 for count in seen.values()), seen


@pytest.mark.parametrize("mode", list(cases.MODES))
def test_objective_events_are_zero_sum_and_results_only_at_the_end(mode):
    moved = 0
    for _, events, _ in cases.play(cases.MODES[mode], ticks=250, seed=3):
        for amount in (events.objective_progress, events.objective):
            assert (amount.sum(-1).abs() <= 1e-6 * (1.0 + amount.abs().sum(-1))).all()
        assert (events.result[~events.done] == 0).all()
        moved += int((events.objective_progress != 0).any())
    if mode == "frontline":
        assert moved > 0, "random play should move a meter now and then"
    else:
        assert moved == 0


# ------------------------------------------------------------------ observation

MIRRORED = (-torch.arange(5)) % 5  # team 1's row z shows zone (-z) mod 5


def _direction(rules, state, team: int):
    """Team 0's and team 1's direction column for the zone team 0 attacks."""
    _, obs, _ = api.perceive(rules, state)
    attacked = int((state.mode.role[0] == frontline.TEAM1_DEFENSE).nonzero())
    row = (-attacked) % 5 if team == 1 else attacked
    return float(obs.zones[0, team, row, ZONE["direction"]])


def test_frontline_zone_rows_are_team_relative():
    rules, state = cases.rules_and_state(FRONTLINE, batch=8, seed=16)
    state = cases.fields_away(state)
    for t in range(40):
        state, _ = cases.tick(rules, _hold(state, attackers=2), seed=t)
    _, obs, _ = api.perceive(rules, state)
    zones = obs.zones
    assert zones.shape == (8, 2, 5, len(ZONE))
    assert (zones[..., ZONE["present"]] == 1).all()
    torch.testing.assert_close(zones[:, 0, :, ZONE["x"]], state.mode.zone_x)
    torch.testing.assert_close(zones[:, 1, :, ZONE["x"]], state.mode.zone_x[:, MIRRORED])
    role = zones[..., ZONE["role"]]
    assert torch.equal(role[:, 1], 4.0 - role[:, 0][:, MIRRORED])
    # the meter rises on the zone team 0 attacks: toward team 0's capture, team 1's loss
    attacked = state.mode.role == frontline.TEAM1_DEFENSE
    assert (zones[:, 0, :, ZONE["direction"]][attacked] == 1).all()
    assert (zones[:, 1, :, ZONE["direction"]][attacked[:, MIRRORED]] == -1).all()
    assert (zones[:, 0, :, ZONE["progress"]][attacked] > 0).all()
    offensive, defensive = frontline.terminal_distances(state.mode.front, 3)
    assert torch.equal(zones[:, 0, :, ZONE["offensive_distance"]], offensive.float())
    assert torch.equal(zones[:, 1, :, ZONE["offensive_distance"]], defensive[:, MIRRORED].float())
    globals_ = obs.globals
    assert (globals_[..., GLOBAL["mode_frontline"]] == 1).all()
    assert (globals_[..., GLOBAL["front_threshold"]] == 3).all()
    assert torch.equal(globals_[:, 0, GLOBAL["front"]], -globals_[:, 1, GLOBAL["front"]])


def test_zone_direction_is_the_meters_movement_for_the_viewer():
    """+1 toward the viewer's capture, -1 toward the enemy's, 0 still: a meter rising on
    the zone team 0 attacks, the same meter falling, and an empty meter the defenders
    hold (the majority pushes, the meter cannot move)."""
    rules, state = cases.rules_and_state(FRONTLINE, batch=1, seed=20)
    state = cases.fields_away(state)
    state, _ = cases.tick(rules, _hold(state, attackers=0))  # team 1 holds nothing; empty
    defended = _hold(state, attackers=0, team=1)  # team 1's ships elsewhere: no majority
    S = 3
    x, y = defended.ships.x.clone(), defended.ships.y.clone()
    centre = _zone_centre(defended, frontline.TEAM1_DEFENSE)
    x[:, S], y[:, S] = centre[:, 0], centre[:, 1]  # one defender in its empty zone
    still, _ = cases.tick(rules, cases.edit_ships(defended, x=x, y=y))
    assert _direction(rules, still, 0) == 0 and _direction(rules, still, 1) == 0
    for t in range(5):
        state, _ = cases.tick(rules, _hold(state, attackers=3), seed=t)
    assert _direction(rules, state, 0) == 1 and _direction(rules, state, 1) == -1
    x, y = state.ships.x.clone(), state.ships.y.clone()
    away = _zone_centre(state, frontline.NEUTRAL)
    x[:, :S], y[:, :S] = away[:, 0, None], away[:, 1, None]
    x[:, S], y[:, S] = centre[:, 0], centre[:, 1]
    falling, _ = cases.tick(rules, cases.edit_ships(state, x=x, y=y))
    attacked = falling.mode.role == frontline.TEAM1_DEFENSE
    assert (falling.mode.progress[attacked] < state.mode.progress[attacked]).all()
    assert _direction(rules, falling, 0) == -1 and _direction(rules, falling, 1) == 1


def test_deathmatch_globals_carry_the_scores_by_team_relative():
    rules, state = cases.rules_and_state(GameConfig(5, 2, 2600.0), batch=2)
    state = cases.edit_mode(state, score=torch.tensor([[3, 7], [12, 0]], dtype=torch.int32))
    _, obs, _ = api.perceive(rules, state)
    g = obs.globals
    assert g[0, 0, GLOBAL["score_own"]] == 3 and g[0, 0, GLOBAL["score_enemy"]] == 7
    assert g[0, 1, GLOBAL["score_own"]] == 7 and g[0, 1, GLOBAL["score_enemy"]] == 3
    assert (g[..., GLOBAL["score_target"]] == 25).all()
    assert (g[..., GLOBAL["mode_team_deathmatch"]] == 1).all()
    assert (g[..., GLOBAL["mode_frontline"]] == 0).all() and (g[..., GLOBAL["front"]] == 0).all()
    assert obs.zones.shape == (2, 2, 5, len(ZONE)) and not obs.zones.any()  # absent slots


# ------------------------------------------------------------------ compiled


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["frontline", "grouped", "sandbox", "elimination"])
def test_compiled_tick_matches_eager_in_every_mode(mode):
    """One tick plus perception from mid-game, as test_game_section's for the default
    mode: floats to fusion rounding, flags, counters and indices exactly."""
    config = cases.MODES[mode]
    rules = cases.lively(config)
    generator = torch.Generator().manual_seed(17)
    state = None
    for before, _, after in cases.play(config, batch=4, ticks=40, seed=17):
        state = after
    B, N = state.ships.x.shape
    inputs = (
        cases.random_actions(B, N, generator),
        api.spread(rules, B, generator=generator),
        api.uniforms(rules, B, generator=generator),
    )

    def step(state, action, spread, uniforms):
        state, events = api.step(rules, cases.LIVELY_TYPES, state, action, spread, uniforms)
        return api.perceive(rules, state), events

    want = step(state, *inputs)
    got = torch.compile(step, fullgraph=True, dynamic=False)(state, *inputs)
    for a, b in zip(
        torch.utils._pytree.tree_leaves(got), torch.utils._pytree.tree_leaves(want), strict=True
    ):
        if b.dtype.is_floating_point:
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-5)
        else:
            assert torch.equal(a, b)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("mode", ["frontline", "grouped"])
def test_eager_cuda_graph_capture_in_every_mode(mode):
    """The whole tick, mode phase and placements included, captures eagerly (no host
    copies) and replays with fresh draws."""
    rules = api.rules_of(cases.MODES[mode])
    device = torch.device("cuda")
    card = api.ship_types(device)
    state = api.reset(rules, card, api.uniforms(rules, 16, device))
    action = cases.random_actions(16, rules.num_ships, torch.Generator().manual_seed(0))
    action = type(action)(*(a.to(device) for a in (action.power, action.stick, action.shoot)))
    spread, uniforms = api.spread(rules, 16, device), api.uniforms(rules, 16, device)

    def tick():
        spread.copy_(api.spread(rules, 16, device))
        uniforms.copy_(api.uniforms(rules, 16, device))
        new, _ = api.step(rules, card, state, action, spread, uniforms)
        torch.utils._pytree.tree_map(lambda d, s: d.copy_(s), state, new)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        tick()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        tick()
    ticks = []
    for _ in range(3):
        graph.replay()
        ticks.append(state.match.episode_tick.clone())
    assert not torch.equal(ticks[0], ticks[1])


def test_frontline_plays_with_no_fields():
    """A frontline map of no fields (the scripted agent's measurements play them): the
    mirror's odd field reads no uniform, so a match resets and steps."""
    config = dataclasses.replace(PRESETS["frontline"], num_fields=0)
    rules = api.rules_of(config)
    generator = torch.Generator().manual_seed(0)
    state = api.reset(rules, api.ship_types(), api.uniforms(rules, 2, generator=generator))
    assert state.fields.x.shape == (2, 0)
    assert bool(torch.isfinite(state.mode.zone_x).all())
