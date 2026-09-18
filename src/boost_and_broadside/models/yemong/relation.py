"""Shared pairwise relational bias on spatial attention scores.

Rotary encoding puts relative geometry into the score as a *fixed sinusoidal
basis* of displacement, in the world frame. Two things it structurally cannot
say are the two this adds:

* **a monotone sense of range.** ``cos(w * dx)`` summed over a base-2 frequency
  ladder is not "how far away", and a network that wants a smooth near/far
  preference has to synthesize one from the ladder inside a bilinear form.
* **anything in the query's own frame.** The rotation's attitude block compares
  *headings* (``cos(k(theta_i - theta_j))``); it says nothing about the angle
  between the query's nose and the direction to the key. That bearing is exactly
  the quantity the behaviour-cloning turn-head diagnostics identify as the
  limiting one -- the scripted teacher's steering is ``angle()`` of a sum of
  ego-frame vectors, and the trunk represents that sum to only two significant
  figures.

So the relation function is six shared scalars per ordered pair, mapped to one
bias per head by a per-sublayer linear map:

    0  proximity              exp(-|d|^2 / 2r^2)
    1  proximity * forward    proximity-weighted cosine of the ego bearing
    2  proximity * lateral    proximity-weighted sine of the ego bearing
    3  forward                ego bearing cosine, unweighted
    4  lateral                ego bearing sine, unweighted
    5  proximity * closing    compressed range rate, weighted by proximity

``d`` is the minimum-image displacement from query to key, and "ego" means
rotated into the query ship's heading frame. Proximity-weighted and unweighted
copies of the bearing are both present because they answer different questions:
the weighted pair vanishes for distant pairs (*who nearby is in front of me*),
the raw pair does not (*which way is that, wherever it is*).

Deliberately **not** an edge MLP. One linear map per sublayer is 6*H weights --
12 at two heads -- shared by every pair, so there is no fleet-size-dependent
parameter anywhere and the mechanism is permutation equivariant by construction:
permuting tokens permutes the rows and columns of the relation tensor and
nothing else.

The weights are zero-initialised, so a run that enables this starts from exactly
the function it would have had without it.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.train.rl.features import PRESENCE_RADIUS, symmetric_logarithm

#: Number of shared pairwise scalars. Fixed by :func:`relation_features`.
RELATION_FEATURES = 6

#: Range scale of the proximity kernel, in world pixels. The same bullet-travel
#: length the presence features use; one physical "engagement radius" constant
#: rather than two that could drift apart.
RELATION_RADIUS = PRESENCE_RADIUS

#: Divisor applied to the range rate before ``symlog``. Ship speeds run to about
#: 180 px/s, so a closing pair is O(1) after this and the compression only bites
#: on the tail.
RELATION_SPEED_SCALE = 100.0


def relation_features(
    position: torch.Tensor,
    attitude: torch.Tensor,
    velocity: torch.Tensor,
    world_size: tuple[float, float],
    radius: float = RELATION_RADIUS,
) -> torch.Tensor:
    """Shared pairwise relational scalars for one batch of tokens.

    Computed once per forward pass and reused by every spatial sublayer, which
    each apply their own linear map to it — the geometry is a property of the
    observation, not of the layer.

    Args:
        position: (B, T, 2) world x/y.
        attitude: (B, T, 2) Cartesian heading. Heading-less tokens carry
            ``(0, 0)``; their ego frame degenerates to the world frame, which is
            the same convention the rotary encoding uses.
        velocity: (B, T, 2) world velocity.
        world_size: (width, height) of the toroid.
        radius: Proximity kernel scale in pixels.

    Returns:
        (B, T, T, RELATION_FEATURES) — entry [b, i, j] describes key ``j`` as
        seen from query ``i``. Not symmetric: the ego frame belongs to ``i``.
    """

    width, height = world_size
    delta_x = position[:, None, :, 0] - position[:, :, None, 0]  # key minus query
    delta_y = position[:, None, :, 1] - position[:, :, None, 1]
    delta_x = (delta_x + width / 2.0) % width - width / 2.0
    delta_y = (delta_y + height / 2.0) % height - height / 2.0

    squared = delta_x * delta_x + delta_y * delta_y
    proximity = torch.exp(-squared / (2.0 * radius * radius))
    distance = torch.sqrt(squared.clamp(min=1e-12))

    # Rotate the displacement into the query's heading frame. A heading-less
    # query has attitude (0, 0); normalising it would be 0/0, so it falls back to
    # the world frame's +x axis, which is what an unrotated token means.
    heading = attitude[:, :, None, :]
    heading_norm = heading.norm(dim=-1, keepdim=True)
    unit_heading = torch.where(
        heading_norm > 1e-6,
        heading / heading_norm.clamp(min=1e-6),
        torch.tensor([1.0, 0.0], device=position.device, dtype=position.dtype),
    )
    heading_x, heading_y = unit_heading[..., 0], unit_heading[..., 1]
    forward = (delta_x * heading_x + delta_y * heading_y) / distance
    lateral = (delta_y * heading_x - delta_x * heading_y) / distance

    relative_vx = velocity[:, None, :, 0] - velocity[:, :, None, 0]
    relative_vy = velocity[:, None, :, 1] - velocity[:, :, None, 1]
    range_rate = (relative_vx * delta_x + relative_vy * delta_y) / distance
    closing = symmetric_logarithm(range_rate / RELATION_SPEED_SCALE)

    return torch.stack(
        (
            proximity,
            proximity * forward,
            proximity * lateral,
            forward,
            lateral,
            proximity * closing,
        ),
        dim=-1,
    )


class RelationalBias(nn.Module):
    """One spatial sublayer's map from shared pairwise scalars to per-head bias.

    Zero-initialised: enabling the mechanism must not, on its own, change what
    the network computes, so a controlled comparison starts from the same
    function.
    """

    def __init__(self, n_heads: int) -> None:
        super().__init__()
        self.project = nn.Linear(RELATION_FEATURES, n_heads, bias=False)
        nn.init.zeros_(self.project.weight)

    def forward(self, relation: torch.Tensor) -> torch.Tensor:
        """(B, T, T, F) pairwise scalars → (B, H, T_query, T_key) additive bias."""

        return self.project(relation).permute(0, 3, 1, 2)


def relation_inputs_from_observation(obs, world_size: tuple[float, float]) -> torch.Tensor:
    """Build the pairwise scalars from an observation's geometry channels."""

    from boost_and_broadside.env.observation import ObsKey

    return relation_features(
        obs[ObsKey.POS].float(),
        obs[ObsKey.ATT].float(),
        obs[ObsKey.VEL].float(),
        world_size,
    )


def world_size_of(ship_config: ShipConfig) -> tuple[float, float]:
    """The toroid the relation function wraps on, as plain floats."""

    return (float(ship_config.world_size[0]), float(ship_config.world_size[1]))
