"""A random spawn heading leaves room to turn away from the edge (modes spec 4,
`bnb.game.modes.spawns._heading`): within `spawn_turn_distance` D of the edge its outward
cosine is at most d / D, at the edge none points outward, from D inward it is the plain
uniform heading, and it changes continuously with d."""

import math

import torch

from bnb.game import api
from bnb.game.modes import spawns
from bnb.game.modes.rules import PRESETS

ARENA = PRESETS["arena"]
RULES = api.rules_of(ARENA)
RADIUS = ARENA.arena_radius
TURN = RULES.spawn_turn_distance


def _headings(distance_from_edge: torch.Tensor, bearing: torch.Tensor, uniform: torch.Tensor):
    """Headings for points at these distances from the edge and bearings from the centre
    (each (B, N)), and the points' outward cosines."""
    radius = torch.full((distance_from_edge.shape[0],), RADIUS)
    distance = RADIUS - distance_from_edge
    x, y = distance * torch.cos(bearing), distance * torch.sin(bearing)
    heading_cos, heading_sin = spawns._heading(RULES, radius, x, y, uniform)
    outward = heading_cos * torch.cos(bearing) + heading_sin * torch.sin(bearing)
    return heading_cos, heading_sin, outward


def _draws(shape, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return (
        2.0 * math.pi * torch.rand(shape, generator=generator),
        torch.rand(shape, generator=generator),
    )


def test_the_set_gives_the_turn_distance_in_its_own_lengths():
    assert api.rules_of(PRESETS["deathmatch"]).spawn_turn_distance == 75.0
    assert TURN == 436.5  # 75 m times the real-scale set's LENGTH, 5.82


def test_at_the_edge_no_spawn_heading_points_outward():
    bearing, uniform = _draws((64, 512))
    _, _, outward = _headings(torch.zeros_like(bearing), bearing, uniform)
    assert float(outward.max()) <= 1e-5


def test_near_the_edge_the_outward_cosine_is_at_most_the_distance_over_the_turn_distance():
    bearing, uniform = _draws((64, 512), seed=1)
    distance = TURN * torch.rand(bearing.shape, generator=torch.Generator().manual_seed(2))
    _, _, outward = _headings(distance, bearing, uniform)
    assert bool((outward <= distance / TURN + 1e-5).all())
    # Not vacuous: the allowed arc is used up to its end.
    assert float((outward - distance / TURN).max()) > -1e-2


def test_from_the_turn_distance_inward_the_heading_is_the_plain_uniform_one():
    bearing, uniform = _draws((64, 512), seed=3)
    distance = TURN + (RADIUS - TURN) * torch.rand(
        bearing.shape, generator=torch.Generator().manual_seed(8)
    )
    heading_cos, heading_sin, outward = _headings(distance, bearing, uniform)
    off_normal = math.pi * (2.0 * uniform - 1.0)  # the draw's own angle off the normal
    torch.testing.assert_close(outward, torch.cos(off_normal), atol=1e-5, rtol=0)
    # The headings are uniform: each of 8 sectors holds its share.
    angle = torch.atan2(heading_sin, heading_cos)
    counts = torch.histc(angle, bins=8, min=-math.pi, max=math.pi)
    assert float((counts / angle.numel() - 1 / 8).abs().max()) < 0.01


def test_near_the_edge_the_heading_is_spread_evenly_over_the_allowed_arc():
    distance = torch.full((1, 200_000), 0.5 * TURN)
    bearing, uniform = _draws(distance.shape, seed=4)
    heading_cos, heading_sin, outward = _headings(distance, bearing, uniform)
    least = math.acos(0.5)
    off_normal = torch.acos(outward.clamp(-1.0, 1.0))  # in [least, pi]
    counts = torch.histc(off_normal, bins=8, min=least, max=math.pi)
    assert float((counts / off_normal.numel() - 1 / 8).abs().max()) < 0.01


def test_the_heading_is_continuous_in_the_distance_from_the_edge():
    steps = 4001
    distance = torch.linspace(0.0, 2.0 * TURN, steps)[None].expand(64, -1)
    bearing, uniform = _draws((64, 1), seed=5)
    heading_cos, heading_sin, _ = _headings(
        distance, bearing.expand(-1, steps), uniform.expand(-1, steps)
    )
    turned = torch.atan2(
        heading_cos[:, :-1] * heading_sin[:, 1:] - heading_sin[:, :-1] * heading_cos[:, 1:],
        heading_cos[:, :-1] * heading_cos[:, 1:] + heading_sin[:, :-1] * heading_sin[:, 1:],
    ).abs()
    step = 2.0 * TURN / (steps - 1)
    # The least angle off the normal, acos(d / D), turns by at most step / (D sqrt(1 -
    # 0.9^2)) per step below 0.9 D, and by about sqrt(2 step / D) at D, where its slope is
    # infinite but it is still continuous; a jump would be a sizable fraction of a radian.
    below = distance[0, 1:] < 0.9 * TURN
    assert float(turned[:, below].max()) < 1.01 * step / (TURN * math.sqrt(1.0 - 0.81))
    assert float(turned.max()) < 1.5 * math.sqrt(2.0 * step / TURN)
    assert float(turned[:, distance[0, 1:] > 1.01 * TURN].max()) < 1e-5  # unchanged beyond D


def test_the_arenas_spawns_leave_room_to_turn():
    """End to end through team deathmatch's placement on the arena preset."""
    mode_rules = RULES.modes[0]
    B, N = 256, RULES.num_ships
    generator = torch.Generator().manual_seed(6)
    radius = torch.full((B,), RADIUS)
    placement = spawns.deathmatch_placement(
        RULES,
        mode_rules,
        radius,
        torch.rand(B, mode_rules.uniform_count(N), generator=generator),
        torch.zeros(B, N),
        torch.zeros(B, N),
        torch.rand(B, N, generator=generator) < 0.5,
        torch.zeros(B, dtype=torch.bool),
        torch.rand(B, N, generator=generator),
    )
    distance = torch.sqrt(placement.x**2 + placement.y**2)
    outward = (placement.heading_cos * placement.x + placement.heading_sin * placement.y) / distance
    assert bool((outward <= (RADIUS - distance) / TURN + 1e-5).all())
    assert bool(((RADIUS - distance) < TURN).any())  # some spawns are near the edge


def test_clovers_fixed_spawns_face_inward_whatever_their_jitter():
    """A fixed layout's spawns keep their own headings: Clover's face the centre from at
    least 0.875 R_a - 150 m out, well inside the turn distance of the edge's reach."""
    rules = api.rules_of(PRESETS["clover"])
    mode_rules = rules.modes[0]
    B, N = 512, rules.num_ships
    generator = torch.Generator().manual_seed(7)
    angle = 2.0 * math.pi * torch.rand(B, generator=generator)
    placement = spawns.layout_placement(
        rules,
        mode_rules,
        torch.rand(B, mode_rules.uniform_count(N), generator=generator),
        torch.cos(angle),
        torch.sin(angle),
    )
    distance = torch.sqrt(placement.x**2 + placement.y**2)
    outward = (placement.heading_cos * placement.x + placement.heading_sin * placement.y) / distance
    assert float(outward.max()) < -0.9
    assert float((PRESETS["clover"].arena_radius - distance).min()) > rules.spawn_turn_distance
