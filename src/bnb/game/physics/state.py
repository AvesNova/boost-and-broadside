"""The state: a dict of tensors in the canonical layout of INTERFACE.md.

Ships 0 ... N/2-1 are team 0 and the rest team 1, so a (B, N, ...) tensor viewed as
(B, 2, N/2, ...) is indexed by team, and flipping that team axis lines every team up with
its enemy. Every pairwise rule (hits, sight, the chaser) is written that way.
"""

import torch

from .constants import Preset

SHIP_FLOATS = ("x", "y", "c", "s", "w", "alpha", "ca", "sa", "r", "E", "logn", "gx", "gy", "shield", "shield_wait")
SHIP_FLAGS = ("alive", "fired", "spawned")
SHIP_INTS = ("respawn", "gun_wait", "cursor", "ship_type")
BULLET_FLOATS = ("bx", "by", "bdx", "bdy", "bw", "blogn", "bgx", "bgy")
FIELD_FLOATS = ("fx", "fy", "fr", "fw")
FIELD_KEYS = FIELD_FLOATS + ("flevel",)
ENV_FLOATS = ("axis_x", "axis_y")

SHIP_KEYS = SHIP_FLOATS + SHIP_FLAGS + SHIP_INTS
BULLET_KEYS = BULLET_FLOATS + ("ttl",)


def blank_state(p: Preset, B: int, device) -> dict:
    """All zeros (every flag false) in the canonical shapes and dtypes."""
    N, K, M = p.num_ships, p.slots, p.num_fields

    def z(shape, dtype):
        return torch.zeros(shape, dtype=dtype, device=device)

    st = {k: z((B, N), torch.float32) for k in SHIP_FLOATS}
    st.update({k: z((B, N), torch.bool) for k in SHIP_FLAGS})
    st.update({k: z((B, N), torch.int32) for k in SHIP_INTS})
    st.update({k: z((B, N, K), torch.float32) for k in BULLET_FLOATS})
    st["ttl"] = z((B, N, K), torch.int32)
    st.update({k: z((B, M), torch.float32) for k in FIELD_FLOATS})
    st["flevel"] = z((B, M), torch.int32)
    st["t"] = z((B,), torch.int32)
    st.update({k: z((B,), torch.float32) for k in ENV_FLOATS})
    return st


def by_team(v: torch.Tensor) -> torch.Tensor:
    """(B, N, ...) per-ship values as (B, 2, N/2 * ...): one row per team."""
    return v.reshape(v.shape[0], 2, -1)


def enemy(v: torch.Tensor) -> torch.Tensor:
    """(B, N, ...) per-ship values as seen by each team's enemy: row T holds team 1 - T."""
    return by_team(v).flip(1)


def per_ship(v: torch.Tensor) -> torch.Tensor:
    """Inverse of by_team for per-ship values: (B, 2, N/2) to (B, N)."""
    return v.reshape(v.shape[0], -1)


def fields_of(st: dict) -> dict:
    return {k: st[k] for k in FIELD_KEYS}


def select(mask: torch.Tensor, new: dict, old: dict, keys) -> dict:
    """new[k] where mask, old[k] elsewhere; mask broadcasts from the left over trailing axes."""
    out = {}
    for k in keys:
        m = mask.reshape(mask.shape + (1,) * (old[k].dim() - mask.dim()))
        out[k] = torch.where(m, new[k], old[k])
    return out
