"""Composable feature pipeline for observation encoding.

Each Feature bundles:
  - Accessor:  extracts raw channels from YemongObservation
  - Transform: encodes raw values into network-ready representation (input path)

FeatureCoordinator integrates a list of Features into
``get_input_vector(obs)`` — a flat encoded observation for the encoder MLP.
Physical-state auxiliary prediction (the next-state head) is a separate path;
see ``train/rl/physical_belief.py``.
"""

import math
from abc import ABC, abstractmethod
from enum import StrEnum

import torch
import torch.nn.functional as F

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.env.observation import (
    NUM_GAME_MODES,
    BulletObsKey,
    ObjectType,
    ObsKey,
    YemongObservation,
)
from boost_and_broadside.train.rl.checkpoint_schema import (
    ATTITUDE_FOURIER_FREQUENCIES,
    position_fourier_frequencies,
)
from boost_and_broadside.train.rl.physical_belief import (
    CORRELATION_COLUMNS,
    LOG_SIGMA_MAX,
    PHYSICAL_UNCERTAINTY_DIM,
)


def _uncertainty_input_scales() -> list[float]:
    """Per-column divisors for the thirteen uncertainty channels.

    The log sigmas divide by their own clamp bound, so the encoder sees the full
    range of what the head can say as ``[-1, 1]``. The correlation latents are
    left alone: they feed a ``tanh`` whose interesting range is a few units
    wide, and scaling them down by the sigma bound would flatten it.
    """

    scales = [LOG_SIGMA_MAX] * PHYSICAL_UNCERTAINTY_DIM
    for column in CORRELATION_COLUMNS:
        scales[column] = 1.0
    return scales


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


def symmetric_logarithm(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def phase_shift_circle(
    sc: torch.Tensor,
    delta: torch.Tensor,
    cosine_first: bool = False,
) -> torch.Tensor:
    """Rotate a (sin,cos) or (cos,sin) pair by scalar phase shifts.

    sc:    (..., 2) — unit circle pair
    delta: (...,)   — phase shifts in radians
    Returns (..., 2) rotated pair.
    """
    cd, sd = delta.cos(), delta.sin()
    if cosine_first:
        c, s = sc[..., 0], sc[..., 1]
        return torch.stack([c * cd - s * sd, s * cd + c * sd], dim=-1)
    else:
        s, c = sc[..., 0], sc[..., 1]
        return torch.stack([s * cd + c * sd, c * cd - s * sd], dim=-1)


# ---------------------------------------------------------------------------
# Accessor
# ---------------------------------------------------------------------------


class Accessor:
    """Reads specific channels from an YemongObservation tensor."""

    def __init__(
        self,
        key: ObsKey,
        channels: list[int] | None = None,
        absent_width: int | None = None,
    ):
        self.key = key
        self.channels = channels
        # Width to synthesise when the channel is absent entirely, for the one
        # channel whose width only the feature layout knows. Resolved by
        # ``build_standard_coordinator`` after the predictors are known, because
        # nothing outside this module can derive it -- see the note there.
        self.absent_width = absent_width
        # A Python list index makes advanced indexing build the index tensor on
        # the host and copy it over, which drains the CUDA queue on every read.
        # Every channel list this pipeline uses is a contiguous run, so it is
        # expressible as a slice: same values, a view instead of a gather, and
        # no host synchronization. Non-contiguous lists keep the list form.
        self._channel_slice = _contiguous_slice(channels)

    def get(self, obs: YemongObservation) -> torch.Tensor:
        try:
            val = obs[self.key]
        except KeyError:
            # Compact ship-only dict fixtures predate map metadata. Serialized
            # checkpoints do not use this compatibility path and are schema
            # gated; these defaults only preserve direct in-process callers.
            team_id = obs[ObsKey.TEAM_ID]
            if self.key == ObsKey.OBJECT_TYPE:
                val = torch.where(
                    team_id == 2,
                    torch.full_like(team_id, int(ObjectType.FIELD)),
                    torch.full_like(team_id, int(ObjectType.SHIP)),
                )
            elif self.key == ObsKey.ZONE_ROLE:
                val = torch.full_like(team_id, 5)
            elif self.key in {ObsKey.VISIBLE, ObsKey.BELIEF_VALID}:
                val = obs[ObsKey.ALIVE]
            elif self.key in {ObsKey.TIME_SINCE_OBSERVATION, ObsKey.SHIELD_DELAY}:
                val = torch.zeros((*team_id.shape, 1), dtype=torch.float32, device=team_id.device)
            elif self.key == ObsKey.BELIEF_UNCERTAINTY:
                # Only a BeliefTracker fills this; a caller without one has
                # forecast nothing, so nothing is in doubt.
                if self.absent_width is None:
                    raise ValueError(
                        "belief_uncertainty accessor has no absent_width; it must be "
                        "PHYSICAL_UNCERTAINTY_DIM"
                    )
                val = torch.zeros(
                    (*team_id.shape, self.absent_width),
                    dtype=torch.float32,
                    device=team_id.device,
                )
            elif self.key in {
                ObsKey.CAPTURE_PROGRESS,
                ObsKey.CAPTURE_DIRECTION,
                ObsKey.ZONE_OFFENSIVE_DISTANCE,
                ObsKey.ZONE_DEFENSIVE_DISTANCE,
                ObsKey.FRONT_POSITION,
                ObsKey.FRONT_WIN_THRESHOLD,
                ObsKey.TIME_REMAINING,
            }:
                val = torch.zeros((*team_id.shape, 1), dtype=torch.float32, device=team_id.device)
            elif self.key == ObsKey.GAME_MODE:
                val = torch.zeros(
                    (*team_id.shape, NUM_GAME_MODES), dtype=torch.float32, device=team_id.device
                )
            else:
                raise
        return self._select(val)

    def _select(self, val: torch.Tensor) -> torch.Tensor:
        """Narrow ``val`` to this accessor's channels without a host round trip."""

        if self._channel_slice is not None:
            return val[..., self._channel_slice]
        if self.channels is not None:
            return val[..., self.channels]
        return val


def _contiguous_slice(channels: list[int] | None) -> slice | None:
    """The equivalent slice for an ascending, step-one channel list, else None."""

    if not channels:
        return None
    if any(b - a != 1 for a, b in zip(channels, channels[1:])):
        return None
    return slice(channels[0], channels[-1] + 1)


# ---------------------------------------------------------------------------
# Transforms (pure tensor → tensor, shape-preserving or expanding)
# ---------------------------------------------------------------------------


class Transform(ABC):
    @abstractmethod
    def out_dim(self, in_dim: int) -> int: ...

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


class Identity(Transform):
    """Pass-through; ensures at least 3D."""

    def out_dim(self, in_dim: int) -> int:
        return in_dim

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x.unsqueeze(-1).float()
        return x.float()


class OneHot(Transform):
    """Integer scalar channel → one-hot vector."""

    def __init__(self, n: int):
        self.n = n

    def out_dim(self, in_dim: int) -> int:
        return self.n

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.long()
        if x.dim() > 2 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        return F.one_hot(x, self.n).float()


class Normalize(Transform):
    """Divide by a scale factor."""

    def __init__(self, scales: float | list[float]):
        self.scales = scales
        self._s_tensor = None

    def out_dim(self, in_dim: int) -> int:
        return in_dim

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.scales, list):
            if (
                self._s_tensor is None
                or self._s_tensor.device != x.device
                or self._s_tensor.dtype != x.dtype
            ):
                self._s_tensor = torch.tensor(self.scales, device=x.device, dtype=x.dtype)
            return x.float() / self._s_tensor
        return x.float() / self.scales


class Symlog(Transform):
    def out_dim(self, in_dim: int) -> int:
        return in_dim

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return symmetric_logarithm(x.float())


class Fourier(Transform):
    """Base-2 power frequency Fourier expansion.

    Input: (..., C) — C scalar channels
    Output: (..., C * 2 * n_freqs) — interleaved [sin_k, cos_k] per channel per freq
    """

    def __init__(self, n_freqs: int, periods: float | list[float]):
        self.n_freqs = n_freqs
        self.periods = periods

    def _frequencies(self, period: float, like: torch.Tensor) -> torch.Tensor:
        """``(2*pi / period) * 2**k`` on ``like``'s device and dtype.

        Built inline on every call, which is deliberate in both directions.

        Not from a host-side list: ``torch.tensor([...], device=cuda)`` is a
        synchronizing copy, and this runs on every encoder forward.

        And not cached either. A device tensor first created inside a
        CUDA-graph capture belongs to that graph's private memory pool, and the
        next replay overwrites it -- so a cache hit on a later call hands back a
        tensor whose storage has been reused, which torch catches as "accessing
        tensor output of CUDAGraphs that has been overwritten by a subsequent
        run". Caching here silently broke `--compile reduce-overhead` and
        `max-autotune` for every policy.

        ``base2_frequencies`` stays the written definition and
        ``tests/models/test_spatial_geometry.py`` pins this against it, so the
        encoder and the rotary encoding cannot drift apart.
        """

        exponents = torch.arange(self.n_freqs, device=like.device, dtype=like.dtype)
        return (2.0 * math.pi / period) * (2.0**exponents)

    def out_dim(self, in_dim: int) -> int:
        return in_dim * 2 * self.n_freqs

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        ps = (
            [self.periods] * x.shape[-1] if isinstance(self.periods, (float, int)) else self.periods
        )
        results = []
        for i, period in enumerate(ps):
            xi = x[..., i]
            args = xi.unsqueeze(-1) * self._frequencies(float(period), x)
            results.append(torch.sin(args))
            results.append(torch.cos(args))
        return torch.cat(results, dim=-1)


class SymlogVelocity(Transform):
    """Map 2D velocity to (vx_norm, vy_norm) where ‖output‖ = symlog(speed).

    Avoids direction discontinuity at zero speed by encoding direction and
    magnitude together. At zero speed, output is (0, 0). Smoothly handles
    direction reversal since the entire vector passes through zero continuously.
    """

    def out_dim(self, in_dim: int) -> int:
        return 2

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        speed = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)
        direction = x / speed
        symlog_speed = symmetric_logarithm(speed)
        return direction * symlog_speed


# ---------------------------------------------------------------------------
# Feature
# ---------------------------------------------------------------------------


class FeatureScope(StrEnum):
    """Which entity types a feature actually carries information for.

    Ship and field tokens share one dense observation layout, so channels that
    only apply to one type are zero-filled for the other. The scope makes that
    explicit, letting the split encoder give each type a first projection over
    just its own channels instead of a mostly-zero shared vector.
    """

    SHARED = "shared"  # meaningful for both ships and fields
    SHIP = "ship"  # zero-filled on field tokens
    FIELD = "field"  # zero-filled on ship tokens
    ZONE = "zone"
    GLOBAL = "global"  # the global/game token


class Feature:
    def __init__(
        self,
        name: str,
        accessor: Accessor,
        input_encoder: Transform,
        scope: FeatureScope = FeatureScope.SHARED,
    ):
        self.name = name
        self.accessor = accessor
        self.input_encoder = input_encoder
        self.scope = scope

    def get_input(self, obs: YemongObservation) -> torch.Tensor:
        return self.input_encoder(self.accessor.get(obs))

    def input_dimension(self, dummy: YemongObservation) -> int:
        """Encoded width this feature contributes to the input vector.

        A method rather than an expression in the coordinator so a feature whose
        value is computed from several observation channels can state its own
        width instead of having one inferred from a single accessor.
        """
        raw = self.accessor.get(dummy)
        in_channels = raw.shape[-1] if raw.dim() > 2 else 1
        return self.input_encoder.out_dim(in_channels)


# ---------------------------------------------------------------------------
# Local presence (ally / enemy density)
# ---------------------------------------------------------------------------

# Radius of the presence kernel, in world pixels.
#
# A physical length rather than a fraction of the map, because that is what makes
# it mean the same thing at every fleet size: "how crowded is my 500 px
# neighbourhood" transfers from 5v5 to 50v50, while "how crowded is my
# map-sixteenth" does not. 500 px is one bullet's travel -- ``bullet_speed`` 500
# px/s for a ``bullet_lifetime`` of 1 s -- so the kernel's half-weight contour sits
# at roughly the distance from which a ship can be shot.
#
# Narrowed from 500 px on September 30 2026, to the Frontline zone radius
# (``benchmarks/presence_density_study.py``,
# docs/internal/presence-radius-sep2026.json).
#
# 500 px was justified as putting the kernel's half-weight contour "at roughly
# the distance from which a ship can be shot", taking that distance as
# ``bullet_speed * bullet_lifetime``. That ignores the bullet's quadratic drag:
# simulated with the real constants a bullet travels 420.6 px, not 500, so the
# half-weight contour at 1.177 * 500 = 589 px sat beyond anything a ship can
# reach with.
#
# 330 px is the zone radius exactly, so the feature reads "how contested is my
# zone", and its half-weight contour at 389 px lands on the scripted agent's
# combat radius of 393 and just inside the bullet's 421. The scripted shoot
# ramp runs from certain at 200 px to stopped at 500 px, and this sits in the
# middle of that band rather than past the end of it.
#
# Measured at 5v5 the enemy channel carries median 0.99 with a 10th-to-90th
# spread of 1.53, and separates 5v5 from 50v50 on one map by x1.67 (against
# 1.24 / 1.66 / x1.98 at 500 px). There is a floor not far below: at 150 px the
# cross-scale response *inverts*, 50v50 reading less crowded than 5v5, which is
# the feature failing rather than merely weakening.
PRESENCE_RADIUS = 330.0
# Divisor applied after ``log1p``. 1.0 -- the compression alone already lands the
# feature in a usable range (5v5 medians 1.40 ally / 1.04 enemy, 50v50 on the same
# map 3.44 / 3.16), so there is nothing left for a scale factor to fix.
#
# ``log1p`` rather than a bounded ``s / (s + k)``, decided on the same scenes. Both
# compress; only one keeps the crowded end legible. At 50 ships a side on the
# training map, the upper half of the population (median to 99th percentile) spans
# 0.38 ally / 0.52 enemy under log1p -- about as much range as the whole 5v5 median
# -- against 0.02 / 0.03 under saturation, which is to say the saturating form
# tells a swarmed ship and a very swarmed ship apart to two decimal places of a
# quantity whose units are nothing in particular.
PRESENCE_SCALE = 1.0


def local_presence(
    position: torch.Tensor,
    team_id: torch.Tensor,
    source: torch.Tensor,
    world_size: tuple[float, float],
    radius: float = PRESENCE_RADIUS,
    scale: float = PRESENCE_SCALE,
) -> torch.Tensor:
    """Smooth, self-excluding ally and enemy presence around every ship token.

    Softmax attention returns proportions, which is exactly the invariant that
    survives a change in fleet size -- and exactly why it cannot report *how
    many*. "Outnumbered two to one" reads the same at any scale; "three enemies
    within weapons range" does not, and nothing else in the observation says it.
    These two scalars are that missing quantity, in the one form that transfers.

    The aggregate is a Gaussian kernel over toroidal distance, summed over every
    contributing ship and then compressed with ``log1p``:

        presence = log1p( sum_j exp(-|d_ij|^2 / (2 r^2)) ) / scale

    Each property is load-bearing:

    * a *sum* over all ships (not a top-k, not a nearest-N) is permutation
      invariant and has no fleet-size-dependent shape;
    * a *smooth* kernel means a ship drifting across the radius moves the feature
      continuously, where a hard count would step;
    * *toroidal* distance means the seam is not a wall;
    * ``log1p`` keeps the value finite and well-scaled as crowding grows without
      flattening the high end the way a bounded ``s/(s+k)`` saturation does -- at
      the fleet sizes this has to span, 10 and 30 neighbours must not read the
      same. It is the difference between a count and a *sense of crowding*, which
      is the semantics wanted here.

    Args:
        position:  (..., T, 2) world x/y for every token.
        team_id:   (..., T) 0/1 for ships, 2 for neutral map objects.
        source:    (..., T) bool — tokens allowed to contribute presence. Pass the
            same mask attention keys on, so a ship never counts a neighbour it is
            not allowed to see.
        world_size: (width, height) of the toroid.
        radius:    Kernel radius in pixels.
        scale:     Divisor applied after ``log1p``.

    Returns:
        (..., T, 2) — [ally, enemy] presence. Rows for non-ship tokens are zero:
        presence is a property of a ship's neighbourhood, and a zone does not
        have one.
    """

    width, height = world_size
    delta_x = position[..., :, None, 0] - position[..., None, :, 0]
    delta_y = position[..., :, None, 1] - position[..., None, :, 1]
    # Minimum image on the torus, matching env.frontline.toroidal_displacement.
    delta_x = (delta_x + width / 2.0) % width - width / 2.0
    delta_y = (delta_y + height / 2.0) % height - height / 2.0
    weight = torch.exp(-(delta_x * delta_x + delta_y * delta_y) / (2.0 * radius * radius))

    contributes = source.unsqueeze(-2)  # (..., 1, T) — over the *source* axis
    same_team = team_id.unsqueeze(-1) == team_id.unsqueeze(-2)  # (..., T, T)
    identity = torch.eye(weight.shape[-1], dtype=torch.bool, device=weight.device)

    ally = (weight * (contributes & same_team & ~identity)).sum(dim=-1)
    enemy = (weight * (contributes & ~same_team)).sum(dim=-1)
    presence = torch.log1p(torch.stack((ally, enemy), dim=-1)) / scale
    # Only ships have a neighbourhood; ``source`` already restricts who counts,
    # this restricts who is counted *for*.
    is_ship = (team_id < 2).unsqueeze(-1)
    return presence * is_ship


class LocalPresenceFeature(Feature):
    """Ally/enemy presence, computed from several observation channels at once.

    A plain ``Feature`` reads one channel through one ``Accessor``; this one needs
    positions, team identities and the belief-validity mask together, so it
    overrides the input path and declares its own width.
    """

    def __init__(self, ship_config: ShipConfig, radius: float = PRESENCE_RADIUS):
        super().__init__(
            name="local_presence",
            accessor=Accessor(ObsKey.POS),
            input_encoder=Identity(),
            scope=FeatureScope.SHIP,
        )
        self.world_size = tuple(float(side) for side in ship_config.world_size)
        self.radius = radius

    def input_dimension(self, dummy: YemongObservation) -> int:
        return 2  # ally, enemy

    def get_input(self, obs: YemongObservation) -> torch.Tensor:
        team_id = obs[ObsKey.TEAM_ID]
        # The same mask spatial attention keys on. A remembered-but-hidden enemy
        # is a token the policy is allowed to reason about, so it contributes;
        # a never-seen one is not, and does not.
        source = obs[ObsKey.BELIEF_VALID].bool() & (team_id < 2)
        return local_presence(
            obs[ObsKey.POS].float(),
            team_id,
            source,
            self.world_size,
            radius=self.radius,
        )


# ---------------------------------------------------------------------------
# FeatureCoordinator
# ---------------------------------------------------------------------------


class FeatureCoordinator:
    """Integrates a list of Features into one input vector for the encoder."""

    def __init__(self, features: list[Feature], dummy_obs: YemongObservation | None = None):
        self.features = features
        # Bullet features read a different observation axis, so their coordinator
        # supplies its own probe rather than the ship/field one.
        self._dummy_override = dummy_obs
        self._init_dims()

    def _init_dims(self) -> None:
        dummy = self._dummy_obs()
        self.total_input_dimension = sum(f.input_dimension(dummy) for f in self.features)

    def _dummy_obs(self) -> YemongObservation:
        from boost_and_broadside.env.observation import ObsKey, YemongObservation

        if self._dummy_override is not None:
            return self._dummy_override

        return YemongObservation(
            data={
                ObsKey.POS: torch.zeros((1, 1, 2)),
                ObsKey.VEL: torch.zeros((1, 1, 2)),
                ObsKey.ATT: torch.zeros((1, 1, 2)),
                ObsKey.ANG_VEL: torch.zeros((1, 1, 1)),
                ObsKey.SHIELD_DELAY: torch.zeros((1, 1, 1)),
                ObsKey.HEALTH: torch.zeros((1, 1, 1)),
                ObsKey.POWER: torch.zeros((1, 1, 1)),
                ObsKey.COOLDOWN: torch.zeros((1, 1, 1)),
                ObsKey.TEAM_ID: torch.zeros((1, 1), dtype=torch.long),
                ObsKey.ALIVE: torch.zeros((1, 1), dtype=torch.bool),
                ObsKey.VISIBLE: torch.zeros((1, 1), dtype=torch.bool),
                ObsKey.BELIEF_VALID: torch.zeros((1, 1), dtype=torch.bool),
                ObsKey.TIME_SINCE_OBSERVATION: torch.zeros((1, 1, 1)),
                ObsKey.RADIUS: torch.zeros((1, 1, 1)),
                ObsKey.PREVIOUS_ACTION: torch.zeros((1, 1, 42), dtype=torch.float32),
                ObsKey.LOCAL_LOG_INDEX: torch.zeros((1, 1, 1)),
                ObsKey.LOCAL_INDEX_GRADIENT: torch.zeros((1, 1, 2)),
                ObsKey.FIELD_TRANSITION_WIDTH: torch.zeros((1, 1, 1)),
                ObsKey.FIELD_TARGET_LOG_INDEX: torch.zeros((1, 1, 1)),
            }
        )

    def get_input_vector(self, obs: YemongObservation) -> torch.Tensor:
        return torch.cat([f.get_input(obs) for f in self.features], dim=-1)

    def get_scoped_input_vector(
        self, obs: YemongObservation, scope: "FeatureScope"
    ) -> torch.Tensor:
        """Encode only shared channels plus those belonging to ``scope``.

        Used by the split encoder so a field token's first projection never sees
        the ship-only channels that are hard zeros for it, and vice versa.
        """
        parts = [
            f.get_input(obs)
            for f in self.features
            if f.scope is FeatureScope.SHARED or f.scope is scope
        ]
        return torch.cat(parts, dim=-1)

    def scoped_input_dimension(self, scope: "FeatureScope") -> int:
        """Width of ``get_scoped_input_vector`` for the given entity type."""
        dummy = self._dummy_obs()
        total = 0
        for f in self.features:
            if f.scope is not FeatureScope.SHARED and f.scope is not scope:
                continue
            total += f.input_dimension(dummy)
        return total


# ---------------------------------------------------------------------------
# Standard coordinator factory
# ---------------------------------------------------------------------------


def build_standard_coordinator(
    ship_config: ShipConfig, *, local_presence: bool = False
) -> FeatureCoordinator:
    """Standard feature pipeline matching the current game's physics.

    Physical-state auxiliary prediction (position, velocity, attitude, health,
    power, cooldown, local index, and their uncertainty) is the next-state
    head's job, not this pipeline's — see ``train/rl/physical_belief.py`` and
    ``PHYSICAL_MEAN_NAMES``. This coordinator only encodes the input vector.
    """
    world_w, world_h = ship_config.world_size

    features = [
        # Ten harmonics at the Frontline world (65536 px down to 128 px) are
        # deliberately unequal: four of those periods exceed the 5v5 playable
        # diameter, so their encodings barely vary, while the finest wraps 41
        # times across it. See docs/training.md.
        Feature(
            name="position_x",
            accessor=Accessor(ObsKey.POS, channels=[0]),
            input_encoder=Fourier(n_freqs=position_fourier_frequencies(world_w), periods=world_w),
        ),
        Feature(
            name="position_y",
            accessor=Accessor(ObsKey.POS, channels=[1]),
            input_encoder=Fourier(n_freqs=position_fourier_frequencies(world_h), periods=world_h),
        ),
        # SymlogVelocity encodes (vx, vy) → direction * symlog(speed). The 2D
        # encoding has no angle discontinuity near zero speed.
        Feature(
            name="velocity",
            accessor=Accessor(ObsKey.VEL),
            input_encoder=SymlogVelocity(),
            scope=FeatureScope.SHIP,
        ),
        # Attitude: position's treatment on the heading circle. Four harmonics
        # over 2*pi.
        Feature(
            name="attitude",
            accessor=Accessor(ObsKey.ATT),
            input_encoder=AttitudeFourier(),
            scope=FeatureScope.SHIP,
        ),
        Feature(
            name="angular_velocity",
            accessor=Accessor(ObsKey.ANG_VEL),
            input_encoder=Symlog(),
            scope=FeatureScope.SHIP,
        ),
        Feature(
            name="shield_delay",
            accessor=Accessor(ObsKey.SHIELD_DELAY),
            input_encoder=Symlog(),
            scope=FeatureScope.SHIP,
        ),
        # Resources are plain bounded scalars, normalised to [0, 1].
        Feature(
            name="health",
            accessor=Accessor(ObsKey.HEALTH),
            input_encoder=Normalize(scales=ship_config.max_health),
        ),
        Feature(
            name="power",
            accessor=Accessor(ObsKey.POWER),
            input_encoder=Normalize(scales=ship_config.max_power),
            scope=FeatureScope.SHIP,
        ),
        Feature(
            name="cooldown",
            accessor=Accessor(ObsKey.COOLDOWN),
            input_encoder=Normalize(scales=ship_config.firing_cooldown),
            scope=FeatureScope.SHIP,
        ),
        # Categoricals and static
        Feature("team_id", Accessor(ObsKey.TEAM_ID), OneHot(3)),
        Feature("alive", Accessor(ObsKey.ALIVE), Identity()),
        Feature("visible", Accessor(ObsKey.VISIBLE), Identity(), scope=FeatureScope.SHIP),
        Feature(
            "time_since_observation",
            Accessor(ObsKey.TIME_SINCE_OBSERVATION),
            Symlog(),
            scope=FeatureScope.SHIP,
        ),
        Feature("object_type", Accessor(ObsKey.OBJECT_TYPE), OneHot(4)),
        Feature("zone_role", Accessor(ObsKey.ZONE_ROLE), OneHot(6)),
        Feature(
            "pending_action",
            Accessor(ObsKey.PREVIOUS_ACTION),
            Identity(),
            scope=FeatureScope.SHIP,
        ),
        Feature(
            "radius",
            Accessor(ObsKey.RADIUS),
            Normalize(0.5 * min(ship_config.world_size)),
        ),
        # Field material features are numeric physical quantities. Ship slots are
        # zero for field-only channels; field slots are zero for ship-local index.
        Feature(
            "field_transition_width",
            Accessor(ObsKey.FIELD_TRANSITION_WIDTH),
            Normalize(ship_config.field_transition_width_max),
            scope=FeatureScope.FIELD,
        ),
        Feature(
            "field_target_log_index",
            Accessor(ObsKey.FIELD_TARGET_LOG_INDEX),
            Identity(),
            scope=FeatureScope.FIELD,
        ),
        Feature(
            "capture_progress",
            Accessor(ObsKey.CAPTURE_PROGRESS),
            Identity(),
            scope=FeatureScope.ZONE,
        ),
        Feature(
            "capture_direction",
            Accessor(ObsKey.CAPTURE_DIRECTION),
            Identity(),
            scope=FeatureScope.ZONE,
        ),
        Feature(
            "zone_offensive_distance",
            Accessor(ObsKey.ZONE_OFFENSIVE_DISTANCE),
            Symlog(),
            scope=FeatureScope.ZONE,
        ),
        Feature(
            "zone_defensive_distance",
            Accessor(ObsKey.ZONE_DEFENSIVE_DISTANCE),
            Symlog(),
            scope=FeatureScope.ZONE,
        ),
        Feature(
            "front_position",
            Accessor(ObsKey.FRONT_POSITION),
            Symlog(),
            scope=FeatureScope.GLOBAL,
        ),
        Feature(
            "front_win_threshold",
            Accessor(ObsKey.FRONT_WIN_THRESHOLD),
            Symlog(),
            scope=FeatureScope.GLOBAL,
        ),
        # How uncertain the belief is, as the next-state head's own thirteen
        # terms: a log sigma per channel plus one correlation latent for position
        # and one for velocity. Already in log/unconstrained form, so the input
        # encoding is a plain division that lands the clamped range in [-1, 1] --
        # a symlog of a log would compress twice and flatten the difference
        # between a ship in sight and one unseen for a minute, which is the whole
        # signal. The correlation latents keep their own scale: they are not
        # logs, and dividing them by the sigma bound would shrink a saturating
        # tanh input to noise.
        #
        # ``time_since_observation`` says only how long it has been; this says
        # what that cost, which is the quantity a policy needs to decide whether
        # to act on a remembered position or go and look.
        Feature(
            "belief_uncertainty",
            Accessor(ObsKey.BELIEF_UNCERTAINTY, absent_width=PHYSICAL_UNCERTAINTY_DIM),
            Normalize(scales=_uncertainty_input_scales()),
            scope=FeatureScope.SHIP,
        ),
        Feature(
            "time_remaining",
            Accessor(ObsKey.TIME_REMAINING),
            Identity(),
            scope=FeatureScope.GLOBAL,
        ),
        Feature(
            "game_mode",
            Accessor(ObsKey.GAME_MODE),
            Identity(),
            scope=FeatureScope.GLOBAL,
        ),
        Feature(
            name="local_log_index",
            accessor=Accessor(ObsKey.LOCAL_LOG_INDEX),
            input_encoder=Identity(),
            scope=FeatureScope.SHIP,
        ),
        # grad(n) at the ship, already normalised in observation_from_state. A
        # deterministic function of position given the static map.
        Feature(
            name="local_index_gradient",
            accessor=Accessor(ObsKey.LOCAL_INDEX_GRADIENT),
            input_encoder=Identity(),
            scope=FeatureScope.SHIP,
        ),
    ]

    if local_presence:
        features.append(LocalPresenceFeature(ship_config))

    return FeatureCoordinator(features)


# ---------------------------------------------------------------------------
# Bullet coordinator
# ---------------------------------------------------------------------------


class BulletAccessor(Accessor):
    """Reads a channel from the bullet axis instead of the entity-token axis."""

    def get(self, obs: YemongObservation) -> torch.Tensor:
        assert obs.bullets is not None, "observation carries no bullet channels"
        return self._select(obs.bullets[self.key])


def build_bullet_coordinator(ship_config: ShipConfig) -> FeatureCoordinator:
    """Feature pipeline for key/value-only bullet tokens.

    Position and velocity use the *same* encodings as ships. This is required,
    not stylistic: a ship's query and a bullet's key meet in a bilinear form, and
    ``q.k`` only reduces to a function of their displacement when both sides
    expand position on one shared Fourier basis. Mismatched frequencies leave
    cross terms that never combine into relative geometry, and the ship could not
    compute "how far away is that bullet" at all.

    Damage and lifetime are plain normalised scalars rather than the quarter-wave
    encoding ships use for bounded resources: bullets are never predicted.

    Shooter identity is carried as a team one-hot and never as an index over
    ships — a per-ship one-hot would fix N in the weights and destroy zero-shot
    transfer to other fleet sizes.
    """
    world_w, world_h = ship_config.world_size

    features = [
        Feature(
            name="bullet_position_x",
            accessor=BulletAccessor(BulletObsKey.POS, channels=[0]),
            input_encoder=Fourier(n_freqs=position_fourier_frequencies(world_w), periods=world_w),
        ),
        Feature(
            name="bullet_position_y",
            accessor=BulletAccessor(BulletObsKey.POS, channels=[1]),
            input_encoder=Fourier(n_freqs=position_fourier_frequencies(world_h), periods=world_h),
        ),
        Feature(
            name="bullet_velocity",
            accessor=BulletAccessor(BulletObsKey.VEL),
            input_encoder=SymlogVelocity(),
        ),
        Feature(
            name="bullet_lifetime",
            accessor=BulletAccessor(BulletObsKey.LIFETIME),
            input_encoder=Identity(),
        ),
        Feature(
            name="bullet_local_log_index",
            accessor=BulletAccessor(BulletObsKey.LOCAL_LOG_INDEX),
            input_encoder=Identity(),
        ),
        Feature(
            name="bullet_local_index_gradient",
            accessor=BulletAccessor(BulletObsKey.LOCAL_INDEX_GRADIENT),
            input_encoder=Identity(),
        ),
        Feature(
            name="bullet_team_id",
            accessor=BulletAccessor(BulletObsKey.TEAM_ID),
            input_encoder=OneHot(2),
        ),
        Feature(
            name="bullet_active",
            accessor=BulletAccessor(BulletObsKey.ACTIVE),
            input_encoder=Identity(),
        ),
    ]
    return FeatureCoordinator(features, dummy_obs=_dummy_bullet_obs())


def _dummy_bullet_obs() -> YemongObservation:
    """Minimal observation used to derive bullet channel widths."""
    return YemongObservation(
        data={},
        bullets={
            BulletObsKey.POS: torch.zeros((1, 1, 2)),
            BulletObsKey.VEL: torch.zeros((1, 1, 2)),
            BulletObsKey.LIFETIME: torch.zeros((1, 1, 1)),
            BulletObsKey.LOCAL_LOG_INDEX: torch.zeros((1, 1, 1)),
            BulletObsKey.LOCAL_INDEX_GRADIENT: torch.zeros((1, 1, 2)),
            BulletObsKey.TEAM_ID: torch.zeros((1, 1), dtype=torch.long),
            BulletObsKey.ACTIVE: torch.zeros((1, 1), dtype=torch.bool),
        },
    )


class AttitudeFourier(Fourier):
    """Encode a Cartesian heading as its phase's Fourier expansion."""

    def __init__(self):
        super().__init__(n_freqs=ATTITUDE_FOURIER_FREQUENCIES, periods=2.0 * math.pi)

    def out_dim(self, in_dim):
        return 2 * self.n_freqs

    def __call__(self, x):
        return super().__call__(torch.atan2(x[..., 1:2], x[..., 0:1]))
