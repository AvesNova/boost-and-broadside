"""The game state: typed dataclasses of tensors, registered as pytrees.

Opaque outside `bnb.game`: the runtime copies, clones and moves it as a pytree and never
reads a field. Ship order is team-major (ship `n = team * S + slot`), so a (B, N, ...)
tensor viewed as (B, T, S, ...) is indexed by team for free.
"""

import dataclasses

import torch

from bnb.contracts.tree import contract

from .rules import Rules


@contract
class Ships:
    """(B, N). Floats: position (m), nose (c, s), proper speed w, angle of attack alpha with
    its (ca, sa), nose rate r, energy E, the cached medium (logn, gx, gy), shield and
    shield_wait (s). Flags: alive, fired and spawned this tick. Ints: respawn wait (ticks),
    gun_wait (ticks), the next bullet slot (cursor) and ship_type."""

    x: torch.Tensor
    y: torch.Tensor
    c: torch.Tensor
    s: torch.Tensor
    w: torch.Tensor
    alpha: torch.Tensor
    ca: torch.Tensor
    sa: torch.Tensor
    r: torch.Tensor
    E: torch.Tensor
    logn: torch.Tensor
    gx: torch.Tensor
    gy: torch.Tensor
    shield: torch.Tensor
    shield_wait: torch.Tensor
    alive: torch.Tensor
    fired: torch.Tensor
    spawned: torch.Tensor
    respawn: torch.Tensor
    gun_wait: torch.Tensor
    cursor: torch.Tensor
    ship_type: torch.Tensor


@contract
class Bullets:
    """(B, N, K), by owner and slot: position, unit direction (dx, dy), proper speed w, the
    cached medium, and ttl (moves left, 0 = inactive)."""

    x: torch.Tensor
    y: torch.Tensor
    dx: torch.Tensor
    dy: torch.Tensor
    w: torch.Tensor
    logn: torch.Tensor
    gx: torch.Tensor
    gy: torch.Tensor
    ttl: torch.Tensor


@contract
class Fields:
    """(B, M): centre, radius, band width (m) and level in {-2, -1, 1, 2}."""

    x: torch.Tensor
    y: torch.Tensor
    radius: torch.Tensor
    width: torch.Tensor
    level: torch.Tensor


@contract
class Match:
    """t (B,) ticks into the episode; the spawn axis (axis_x, axis_y) (B,); last_seen
    (B, T, N) ticks since team T last saw ship n (0 while visible)."""

    t: torch.Tensor
    axis_x: torch.Tensor
    axis_y: torch.Tensor
    last_seen: torch.Tensor


@contract
class State:
    ships: Ships
    bullets: Bullets
    fields: Fields
    match: Match

    def __getitem__(self, key: str) -> torch.Tensor:
        """Read-only access by the race's flat key names (CANONICAL), for tests and
        debugging; game code uses the typed fields."""
        part, name = CANONICAL[key]
        return getattr(getattr(self, part), name)


@contract
class Origin:
    """Tick-start positions the hit test sweeps from: ships (B, N), bullets after firing
    (B, N, K). Passed from `move` to `resolve`."""

    x: torch.Tensor
    y: torch.Tensor
    bx: torch.Tensor
    by: torch.Tensor


replace = dataclasses.replace

FLOAT, BOOL, INT = torch.float32, torch.bool, torch.int32
SHIP_DTYPES = {f.name: FLOAT for f in dataclasses.fields(Ships)} | dict(
    alive=BOOL, fired=BOOL, spawned=BOOL, respawn=INT, gun_wait=INT, cursor=INT, ship_type=INT)
BULLET_DTYPES = {f.name: FLOAT for f in dataclasses.fields(Bullets)} | dict(ttl=INT)
FIELD_DTYPES = {f.name: FLOAT for f in dataclasses.fields(Fields)} | dict(level=INT)


def blank(rules: Rules, batch: int, device=None) -> State:
    """Every value zero and every flag false, in the canonical shapes and dtypes."""
    B, N, K, M, T = batch, rules.num_ships, rules.slots, rules.num_fields, rules.num_teams

    def zeros(dtypes: dict, shape: tuple) -> dict:
        return {k: torch.zeros(shape, dtype=d, device=device) for k, d in dtypes.items()}

    return State(ships=Ships(**zeros(SHIP_DTYPES, (B, N))),
                 bullets=Bullets(**zeros(BULLET_DTYPES, (B, N, K))),
                 fields=Fields(**zeros(FIELD_DTYPES, (B, M))),
                 match=Match(t=torch.zeros(B, dtype=INT, device=device),
                             axis_x=torch.zeros(B, device=device),
                             axis_y=torch.zeros(B, device=device),
                             last_seen=torch.zeros((B, T, N), dtype=INT, device=device)))


def where(mask: torch.Tensor, new, old):
    """`new` where mask, `old` elsewhere, field by field for two dataclasses of one type.
    mask broadcasts from the left over each field's trailing axes."""
    out = {}
    for f in dataclasses.fields(old):
        a, b = getattr(new, f.name), getattr(old, f.name)
        m = mask.reshape(mask.shape + (1,) * (b.dim() - mask.dim()))
        out[f.name] = torch.where(m, a, b)
    return type(old)(**out)


def apply(fn, obj):
    """A dataclass of the same type with `fn` applied to every field."""
    return type(obj)(**{f.name: fn(getattr(obj, f.name)) for f in dataclasses.fields(obj)})


def by_team(v: torch.Tensor, teams: int = 2) -> torch.Tensor:
    """(B, N, ...) per-ship values as (B, T, S, ...)."""
    return v.reshape(v.shape[0], teams, v.shape[1] // teams, *v.shape[2:])


# The race's flat key names (game spec 3.1, 6, 4), for tests and debugging only.
CANONICAL = (
    {k: ("ships", k) for k in SHIP_DTYPES}
    | {"b" + k: ("bullets", k) for k in BULLET_DTYPES if k != "ttl"} | {"ttl": ("bullets", "ttl")}
    | {"fx": ("fields", "x"), "fy": ("fields", "y"), "fr": ("fields", "radius"),
       "fw": ("fields", "width"), "flevel": ("fields", "level")}
    | {k: ("match", k) for k in ("t", "axis_x", "axis_y", "last_seen")}
)


def flat(st: State) -> dict[str, torch.Tensor]:
    """The state under the race's flat key names."""
    return {k: getattr(getattr(st, part), name) for k, (part, name) in CANONICAL.items()}


def unflat(values: dict[str, torch.Tensor]) -> State:
    """Inverse of `flat`."""
    parts: dict[str, dict] = {"ships": {}, "bullets": {}, "fields": {}, "match": {}}
    for k, (part, name) in CANONICAL.items():
        parts[part][name] = values[k]
    return State(ships=Ships(**parts["ships"]), bullets=Bullets(**parts["bullets"]),
                 fields=Fields(**parts["fields"]), match=Match(**parts["match"]))
