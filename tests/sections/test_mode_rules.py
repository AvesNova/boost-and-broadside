"""Mode rules the reference fixtures and the seat test did not reach (modes spec 4-7): the
front seen from team 1 once it has moved, the dead and the capture meters, spawn-zone
protection by each ship's own team, and team deathmatch's spawn counting. Mutation-sweep
items 7 to 12."""

import math

import pytest
import test_seat_symmetry as seat
import torch

from bnb.contracts import GameConfig
from bnb.contracts.obs import GLOBAL, ZONE
from bnb.game import api
from bnb.game.modes import frontline, spawns
from bnb.game.modes.observation import zone_rows
from bnb.game.modes.phase import mode_phase, protected
from bnb.game.modes.rules import PRESETS
from bnb.game.physics.spawning import formation_rank
from bnb.game.state import replace

TYPES = api.ship_types()


def _frontline(ships_per_team: int = 2, batch: int = 1, seed: int = 0):
    rules = api.rules_of(GameConfig(ships_per_team, 4, 2600.0, mode="frontline"))
    generator = torch.Generator().manual_seed(seed)
    state = api.reset(rules, TYPES, api.uniforms(rules, batch, generator=generator))
    return rules, rules.modes[0], state


def _moved_front(state, front: torch.Tensor):
    return replace(
        state, mode=replace(state.mode, front=front, role=frontline.roles_from_front(front))
    )


def _zone_centre(mode, role: int):
    """(B,) x, y of each environment's zone with this role."""
    index = (mode.role == role).int().argmax(-1, keepdim=True)
    return mode.zone_x.gather(1, index)[:, 0], mode.zone_y.gather(1, index)[:, 0]


def _outside_every_zone(mode):
    """(B,) x, y of a point inside the arena and outside every zone: the neutral zone's
    centre pushed out past it, away from the ring's centre, by half the ring's gap."""
    x, y = _zone_centre(mode, frontline.NEUTRAL)
    distance = torch.sqrt(x * x + y * y)
    scale = (distance - 2.0 * mode.zone_radius.max(-1).values) / distance
    point_x, point_y = x * scale, y * scale
    inside = frontline.zone_membership(
        torch.stack([point_x, point_y], -1)[:, None],
        torch.stack([mode.zone_x, mode.zone_y], -1),
        mode.zone_radius,
    )
    assert not bool(inside.any()), "the parking point should be outside every zone"
    return point_x, point_y


def _place(ships, x, y):
    return replace(ships, x=x, y=y)


# ------------------------------------------------------------------ item 7: the front


@pytest.mark.parametrize("front", [-2, -1, 1, 2])
def test_team_one_sees_a_moved_front_negated(front):
    rules, _, state = _frontline()
    state = _moved_front(state, torch.tensor([front], dtype=torch.int32))
    _, observation, _ = api.perceive(rules, state)
    shown = observation.globals[0, :, GLOBAL["front"]]
    assert shown.tolist() == [float(front), float(-front)]


def test_a_tick_with_a_moved_front_and_its_mirror_agree():
    """The seat-symmetry mirror test from states whose front has moved (its own melee
    states and chaser games keep the front at 0 within their horizon): the views' front,
    the zones' roles and the captures agree with the mirror."""
    rules = api.rules_of(PRESETS["frontline"])
    generator = torch.Generator().manual_seed(11)
    failures = set()
    for _ in range(2):
        state = seat._melee(rules, 32, generator)
        front = torch.randint(-2, 3, (32,), generator=generator, dtype=torch.int32)
        assert bool((front != 0).any())
        state = _moved_front(state, front)
        seat._assert_mirrored(rules, state, *seat._random_tick(rules, 32, generator), failures)
    assert not failures, sorted(failures)


# ------------------------------------------------------------------ items 8 and 9: meters


def _attack_setup(attacker_alive: bool, progress: float):
    """Front 0: team 0's ship 0 sits on the zone team 0 attacks (team 1's defense), every
    other ship outside every zone; that zone's meter at `progress`."""
    rules, mode_rules, state = _frontline()
    mode = state.mode
    target = int((mode.role[0] == frontline.TEAM1_DEFENSE).int().argmax())
    zone_x, zone_y = _zone_centre(mode, frontline.TEAM1_DEFENSE)
    park_x, park_y = _outside_every_zone(mode)
    N = rules.num_ships
    x = park_x[:, None].expand(1, N).clone()
    y = park_y[:, None].expand(1, N).clone()
    x[:, 0], y[:, 0] = zone_x, zone_y
    ships = _place(state.ships, x, y)
    alive = torch.ones(1, N, dtype=torch.bool)
    alive[0, 0] = attacker_alive
    mode = replace(mode, progress=mode.progress.index_fill(1, torch.tensor([target]), progress))
    died = torch.zeros(1, N, dtype=torch.bool)
    return rules, mode_rules, mode, ships, alive, died, target


def test_a_dead_ship_inside_a_zone_does_not_count_toward_its_capture():
    rules, mode_rules, mode, ships, alive, died, target = _attack_setup(False, 0.0)
    phase = mode_phase(rules, mode_rules, mode, ships, alive, died)
    assert float(phase.mode.progress[0, target]) == 0.0
    assert int(phase.mode.movement.abs().sum()) == 0
    # the same ship alive does push the meter
    alive = torch.ones_like(alive)
    phase = mode_phase(rules, mode_rules, mode, ships, alive, died)
    assert float(phase.mode.progress[0, target]) > 0.0


def test_every_zones_observed_movement_is_zero_on_the_tick_a_capture_completes():
    rules, mode_rules, mode, ships, alive, died, _ = _attack_setup(True, 1.0 - 1e-4)
    phase = mode_phase(rules, mode_rules, mode, ships, alive, died)
    assert int(phase.mode.front[0]) == 1, "the capture should complete this tick"
    assert int(phase.mode.movement.abs().sum()) == 0
    rows = zone_rows(mode_rules, phase.mode)  # (B, 2, Z, columns)
    assert torch.all(rows[..., ZONE["direction"]] == 0.0)


# ------------------------------------------------------------------ item 10: protection


@pytest.mark.parametrize("own", [True, False])
def test_spawn_protection_is_by_each_ships_own_team(own):
    """Three ships a team: every ship of each team on its own team's spawn zone is
    protected, and none on the enemy's."""
    rules, mode_rules, state = _frontline(ships_per_team=3)
    mode, S = state.mode, rules.ships_per_team
    roles = (frontline.TEAM0_SPAWN, frontline.TEAM1_SPAWN)
    if not own:
        roles = roles[::-1]
    centres = [_zone_centre(mode, role) for role in roles]
    x = torch.cat([centres[team][0][:, None].expand(1, S) for team in (0, 1)], 1)
    y = torch.cat([centres[team][1][:, None].expand(1, S) for team in (0, 1)], 1)
    shielded = protected(rules, mode_rules, mode, _place(state.ships, x, y))
    assert bool(shielded.all()) if own else not bool(shielded.any())


# ------------------------------------------------------------------ items 11 and 12: spawns


def _deathmatch_placement(
    monkeypatch, reset: bool, alive: torch.Tensor, arena_radius: float = 8200.0, seed: int = 4
):
    """deathmatch_placement with the wide preset's groups (2, 2, 1), recording what it
    compared: (rules, mode_rules, inputs, placement, recorded candidates and targets)."""
    game = PRESETS["deathmatch_wide"]
    rules = api.rules_of(game)
    mode_rules = rules.modes[0]
    B, N = alive.shape
    generator = torch.Generator().manual_seed(seed)
    radius = torch.full((B,), arena_radius)
    uniforms = torch.rand(B, mode_rules.uniform_count(N), generator=generator)
    keys = torch.rand(B, N, generator=generator)
    angle = 2.0 * math.pi * torch.rand(B, N, generator=generator)
    distance = 3000.0 * torch.rand(B, N, generator=generator)
    x, y = distance * torch.cos(angle), distance * torch.sin(angle)
    recorded = {}
    original = spawns._farthest_from_enemies

    def recording(candidate_x, candidate_y, target_x, target_y, counts=None):
        recorded.update(x=candidate_x, y=candidate_y, target_x=target_x, target_y=target_y)
        return original(candidate_x, candidate_y, target_x, target_y, counts)

    monkeypatch.setattr(spawns, "_farthest_from_enemies", recording)
    reset_flags = torch.full((B,), reset)
    placement = spawns.deathmatch_placement(
        rules, mode_rules, radius, uniforms, x, y, alive, reset_flags, keys
    )
    inputs = dict(radius=radius, uniforms=uniforms, keys=keys)
    return rules, mode_rules, inputs, placement, recorded


def _expected_choice(rules, mode_rules, keys, recorded, same_group_only: bool):
    """Each ship's candidate farthest from the nearest counted enemy target (ties to the
    lower candidate), by brute force over the recorded candidates and targets."""
    S, N = rules.ships_per_team, rules.num_ships
    group = spawns._group_of_rank(mode_rules.spawn_groups, formation_rank(rules, keys))
    team = api.ship_team(S)
    enemy = team[:, None] != team[None, :]  # (N, N)
    counted = enemy & (group[:, :, None] == group[:, None, :]) if same_group_only else enemy
    dx = recorded["x"][..., None] - recorded["target_x"][:, None, None, :]  # (B, N, C, N)
    dy = recorded["y"][..., None] - recorded["target_y"][:, None, None, :]
    squared = torch.where(
        counted[:, :, None, :].expand(-1, N, recorded["x"].shape[-1], N)
        if counted.dim() == 3
        else counted[None, :, None, :],
        dx * dx + dy * dy,
        math.inf,
    ).amin(-1)
    pick = squared.argmax(-1, keepdim=True)
    return recorded["x"].gather(-1, pick)[..., 0], recorded["y"].gather(-1, pick)[..., 0]


@pytest.mark.parametrize("reset", [False, True])
def test_spawns_count_every_enemy_on_a_respawn_and_same_group_rivals_on_a_reset(monkeypatch, reset):
    B, N = 64, 10
    generator = torch.Generator().manual_seed(9)
    alive = torch.rand(B, N, generator=generator) < 0.6
    # A small arena on a reset, so the groups' discs overlap and another group's enemy can
    # be nearer than the ship's own group's rivals.
    rules, mode_rules, inputs, placement, recorded = _deathmatch_placement(
        monkeypatch, reset, alive, arena_radius=300.0 if reset else 8200.0
    )
    want_x, want_y = _expected_choice(
        rules, mode_rules, inputs["keys"], recorded, same_group_only=reset
    )
    assert torch.equal(placement.x, want_x) and torch.equal(placement.y, want_y)
    # Not vacuous: counting the other way picks other points.
    other_x, _ = _expected_choice(
        rules, mode_rules, inputs["keys"], recorded, same_group_only=not reset
    )
    assert not torch.equal(other_x, want_x)


def test_a_match_with_no_living_ship_respawns_at_the_uniform_fallback_points(monkeypatch):
    """Every ship dead at stale positions: a respawn has no anchor, so every candidate is
    the uniform rule's point in the arena, not a point around a stale position."""
    B, N = 16, 10
    alive = torch.zeros(B, N, dtype=torch.bool)
    rules, mode_rules, inputs, placement, recorded = _deathmatch_placement(
        monkeypatch, False, alive
    )
    C = mode_rules.spawn_candidates
    per_environment = 2 * C * len(mode_rules.spawn_groups)
    per_ship = inputs["uniforms"][:, per_environment:].reshape(B, N, -1)
    radius = inputs["radius"][:, None, None]
    uniform_x, uniform_y = spawns._disc(radius, per_ship[..., :C], per_ship[..., C : 2 * C])
    want_x, want_y = spawns._clip_to_disc(
        uniform_x, uniform_y, (radius - mode_rules.spawn_margin).clamp_min(0.0)
    )
    torch.testing.assert_close(recorded["x"], want_x, rtol=0, atol=0)
    torch.testing.assert_close(recorded["y"], want_y, rtol=0, atol=0)
    chosen = (placement.x[..., None] == want_x) & (placement.y[..., None] == want_y)
    assert bool(chosen.any(-1).all())
