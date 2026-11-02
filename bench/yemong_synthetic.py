"""Synthetic observation segments for the Yemong speed benchmark (from the blind race):
plausible ranges, fog with persistent sight, rare spawns, resets at chosen steps. No
physics."""

import math

import torch

from bnb.contracts.obs import (
    FIELD_COLUMNS,
    GLOBAL,
    GLOBAL_COLUMNS,
    SHIP,
    SHIP_COLUMNS,
    SHIP_HIDDEN_ZERO,
    ZONE_COLUMNS,
    Obs,
)


def random_segment(
    steps: int,
    views: int,
    ships: int = 10,
    fields: int = 10,
    zones: int = 5,
    device=None,
    dtype=torch.float32,
    seed: int = 0,
    fog_share: float = 0.5,
    reset_steps=(),
    arena_radius: float = 2600.0,
):
    """Returns (obs with leaves (L, R, ...), reset (L, R) bool). Views [0, fog_share R) are
    fogged; every view resets at each step in reset_steps."""
    generator = torch.Generator().manual_seed(seed)

    def uniform(*shape, low=0.0, high=1.0):
        return low + (high - low) * torch.rand(*shape, generator=generator, dtype=torch.float64)

    own = ships // 2
    shape = (steps, views, ships)
    radius = arena_radius * 0.8 * torch.sqrt(uniform(views, ships))
    angle = uniform(views, ships, high=2 * math.pi)
    start = torch.stack([radius * torch.cos(angle), radius * torch.sin(angle)], -1)
    velocity = uniform(*shape, 2, low=-150, high=150)
    position = start + torch.cumsum(velocity, 0) * (1 / 30)
    heading = uniform(*shape, high=2 * math.pi) - math.pi
    columns = torch.zeros(*shape, len(SHIP_COLUMNS), dtype=torch.float64)

    def put(name, value):
        columns[..., SHIP[name]] = value

    put("x", position[..., 0])
    put("y", position[..., 1])
    put("heading_cos", torch.cos(heading))
    put("heading_sin", torch.sin(heading))
    put("velocity_x", velocity[..., 0])
    put("velocity_y", velocity[..., 1])
    put("proper_speed", velocity.norm(dim=-1))
    put("angle_of_attack", uniform(*shape, low=-0.5, high=0.5))
    put("nose_rate", uniform(*shape, low=-4, high=4))
    put("energy", uniform(*shape, high=1000))
    put("shield", uniform(*shape, high=100))
    put("shield_recharge_wait", uniform(*shape, high=6))
    put("gun_cooldown", torch.randint(0, 4, shape, generator=generator).double() / 30)
    put("log_refractive_index", uniform(*shape, low=-0.6, high=0.6))
    put("log_index_gradient_x", uniform(*shape, low=-0.05, high=0.05))
    put("log_index_gradient_y", uniform(*shape, low=-0.05, high=0.05))

    fogged = torch.arange(views) < round(fog_share * views)
    sight = torch.zeros(shape, dtype=torch.bool)
    seen = torch.rand(views, ships, generator=generator) < 0.5
    for t in range(steps):
        flip = torch.rand(views, ships, generator=generator) < 0.1
        seen = seen ^ flip
        sight[t] = seen
    sight[..., :own] = True
    sight = sight | ~fogged.view(1, -1, 1)
    spawned = torch.rand(shape, generator=generator) < 0.02
    alive = torch.rand(shape, generator=generator) > 0.05
    sight = sight | spawned  # a ship is seen at its spawn
    columns[..., :SHIP_HIDDEN_ZERO] *= sight.unsqueeze(-1)
    put("alive", alive.double())
    put("visible", sight.double())
    put("fired", (torch.rand(shape, generator=generator) < 0.2).double())
    put("respawn_wait", torch.where(alive, 0.0, uniform(*shape, high=8)))
    put(
        "ship_type",
        torch.randint(0, 2, (1, views, ships), generator=generator).double().expand(shape),
    )
    put("team_relative", (torch.arange(ships) >= own).double().expand(shape))
    put("time_since_seen", uniform(*shape, high=10) * ~sight)
    put("spawned", spawned.double())

    field_angle = uniform(views, fields, high=2 * math.pi)
    field_radius = arena_radius * 0.7 * torch.sqrt(uniform(views, fields))
    field = torch.zeros(steps, views, fields, len(FIELD_COLUMNS), dtype=torch.float64)
    field[..., 0] = field_radius * torch.cos(field_angle)
    field[..., 1] = field_radius * torch.sin(field_angle)
    field[..., 2] = uniform(views, fields, low=100, high=500)
    field[..., 3] = uniform(views, fields, low=20, high=60)
    field[..., 4] = uniform(views, fields, low=0.5, high=2.0)

    reset = torch.zeros(steps, views, dtype=torch.bool)
    for t in reset_steps:
        reset[t] = True
    globals_ = torch.zeros(steps, views, len(GLOBAL_COLUMNS), dtype=torch.float64)
    globals_[..., GLOBAL["episode_progress"]] = (torch.arange(steps) / steps).view(-1, 1)
    globals_[..., GLOBAL["arena_radius"]] = arena_radius
    globals_[..., GLOBAL["fogged"]] = fogged.double()
    globals_[..., GLOBAL["reset"]] = reset.double()
    zone = torch.zeros(steps, views, zones, len(ZONE_COLUMNS), dtype=torch.float64)

    obs = Obs(
        ships=columns.to(device=device, dtype=dtype),
        fields=field.to(device=device, dtype=dtype),
        globals=globals_.to(device=device, dtype=dtype),
        zones=zone.to(device=device, dtype=dtype),
    )
    return obs, reset.to(device)


def time_slice(obs: Obs, t: int) -> Obs:
    return Obs(
        *(
            None if leaf is None else leaf[t]
            for leaf in (obs.ships, obs.fields, obs.globals, obs.zones)
        )
    )
