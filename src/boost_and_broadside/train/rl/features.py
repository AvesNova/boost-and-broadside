"""Composable feature pipeline for observation encoding and aux prediction.

Each Feature bundles:
  - Accessor:  extracts raw channels from YemongObservation
  - Transform: encodes raw values into network-ready representation (input path)
  - Transform: encodes raw values into target space (aux prediction path)
  - Predictor: defines label computation and how predictions update the target

FeatureCoordinator integrates a list of Features into:
  - get_input_vector(obs)  → flat encoded observation for the encoder MLP
  - get_target_vector(obs) → flat target representation for aux loss
  - compute_labels(curr, next) → ground-truth labels (deltas or absolutes)
  - apply_all_predictions(curr, preds) → apply predicted updates to targets
"""

import dataclasses
import math
from abc import ABC, abstractmethod
from enum import StrEnum

import torch
import torch.nn.functional as F

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.env.observation import BulletObsKey, ObjectType, ObsKey, YemongObservation
from boost_and_broadside.train.rl.checkpoint_schema import (
    ATTITUDE_FOURIER_FREQUENCIES,
    position_fourier_frequencies,
)

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

    def __init__(self, key: ObsKey, channels: list[int] | None = None):
        self.key = key
        self.channels = channels
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
            elif self.key in {
                ObsKey.CAPTURE_PROGRESS,
                ObsKey.CAPTURE_DIRECTION,
                ObsKey.ZONE_OFFENSIVE_DISTANCE,
                ObsKey.ZONE_DEFENSIVE_DISTANCE,
                ObsKey.FRONT_POSITION,
                ObsKey.FRONT_WIN_THRESHOLD,
                ObsKey.TIME_REMAINING,
                ObsKey.GAME_MODE,
            }:
                val = torch.zeros((*team_id.shape, 1), dtype=torch.float32, device=team_id.device)
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

    def invert(self, x: torch.Tensor) -> torch.Tensor:
        """Map target-encoded values back to raw physical space.

        Only defined for transforms used as a Feature's target encoder on the
        aux-prediction path; transforms that never need inversion inherit this
        fail-fast default rather than a silently-wrong stub.
        """
        raise NotImplementedError(f"{type(self).__name__} does not define an inverse")


class Identity(Transform):
    """Pass-through; ensures at least 3D."""

    def out_dim(self, in_dim: int) -> int:
        return in_dim

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x.unsqueeze(-1).float()
        return x.float()

    def invert(self, x: torch.Tensor) -> torch.Tensor:
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

    def invert(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.scales, list):
            if (
                self._s_tensor is None
                or self._s_tensor.device != x.device
                or self._s_tensor.dtype != x.dtype
            ):
                self._s_tensor = torch.tensor(self.scales, device=x.device, dtype=x.dtype)
            return x.float() * self._s_tensor
        return x.float() * self.scales


class Symlog(Transform):
    def out_dim(self, in_dim: int) -> int:
        return in_dim

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return symmetric_logarithm(x.float())

    def invert(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        return torch.sign(x) * torch.expm1(x.abs())


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

    def invert(self, x: torch.Tensor) -> torch.Tensor:
        """Recover the raw channels from a single-frequency (sin, cos) encoding.

        Only the ``n_freqs == 1`` case is invertible in closed form (one phase
        per channel); higher-frequency inputs are the encoder-only path and never
        decoded, so requesting their inverse fails fast.
        """
        if self.n_freqs != 1:
            raise NotImplementedError("Fourier.invert supports only n_freqs=1 encodings")
        x = x.float()
        num_channels = x.shape[-1] // 2  # (..., 2C) laid out [sin_c, cos_c] per channel
        ps = (
            [self.periods] * num_channels
            if isinstance(self.periods, (float, int))
            else self.periods
        )
        outs = []
        for c in range(num_channels):
            angle = torch.atan2(x[..., 2 * c], x[..., 2 * c + 1]) % (2.0 * math.pi)
            outs.append(angle * ps[c] / (2.0 * math.pi))
        return torch.stack(outs, dim=-1)


class UnitCircle(Transform):
    """Map a [0, scale] scalar to a quarter-wave (sin, cos) pair."""

    def __init__(self, scales: float = 1.0):
        self.scales = scales

    def out_dim(self, in_dim: int) -> int:
        return in_dim * 2

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        norm = (x / self.scales).clamp(0.0, 1.0)
        angle = (math.pi / 2.0) * norm
        return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1).flatten(-2)

    def invert(self, x: torch.Tensor) -> torch.Tensor:
        """Recover the raw [0, scale] scalar(s) from the (sin, cos) quarter-wave."""
        x = x.float()
        num_channels = x.shape[-1] // 2  # (..., 2C) laid out [sin_c, cos_c] per channel
        outs = []
        for c in range(num_channels):
            angle = torch.atan2(x[..., 2 * c], x[..., 2 * c + 1]).clamp(0.0, math.pi / 2.0)
            outs.append(angle / (math.pi / 2.0) * self.scales)
        return torch.stack(outs, dim=-1)


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

    def invert(self, x: torch.Tensor) -> torch.Tensor:
        """Recover raw (vx, vy) from direction * symlog(speed)."""
        x = x.float()
        symlog_speed = torch.norm(x, dim=-1, keepdim=True)
        direction = x / symlog_speed.clamp(min=1e-8)
        speed = torch.expm1(symlog_speed.clamp(min=0.0))
        return direction * speed


# ---------------------------------------------------------------------------
# Predictors (define label computation and prediction application)
# ---------------------------------------------------------------------------


_LOG_TWO_PI = math.log(2.0 * math.pi)


class Predictor(ABC):
    @abstractmethod
    def target_dim(self, in_channels: int) -> int: ...

    @abstractmethod
    def prediction_dim(self, in_channels: int) -> int: ...

    #: What the head's uncertainty output means for this predictor, or None for
    #: a predictor that reports none and keeps a plain squared error.
    uncertainty_kind: str | None = None

    def uncertainty_dim(self, in_channels: int) -> int:
        """Uncertainty outputs this predictor wants, or 0 for a plain squared error.

        A predictor that reports one makes its loss a Gaussian negative log
        likelihood instead, which is what removes ``label_scale`` from the
        objective: ``(y - mu)^2 / sigma^2`` is invariant to how the label is
        scaled, so a feature whose labels are a thousand times too large learns
        a correspondingly larger sigma rather than dominating the sum.

        It also weights the gradient on the mean by ``1 / sigma^2``, which is
        what this model needs: a long-hidden token's label is mostly belief
        error nobody could predict, so the head learns a wide sigma there and
        the signal concentrates on the tokens whose labels are real dynamics.
        """

        return 0

    @abstractmethod
    def compute_labels(self, curr: torch.Tensor, next_: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def apply_prediction(self, curr: torch.Tensor, pred: torch.Tensor) -> torch.Tensor: ...


class AbsolutePredictor(Predictor):
    """Predict next state directly (absolute, no delta)."""

    uncertainty_kind = "gaussian"

    def target_dim(self, in_channels: int) -> int:
        return in_channels

    def prediction_dim(self, in_channels: int) -> int:
        return in_channels

    def uncertainty_dim(self, in_channels: int) -> int:
        return in_channels

    def compute_labels(self, curr: torch.Tensor, next_: torch.Tensor) -> torch.Tensor:
        return next_

    def apply_prediction(self, curr: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        return pred


class AdditivePredictor(Predictor):
    """Predict delta: next − curr in target space."""

    uncertainty_kind = "gaussian"

    def target_dim(self, in_channels: int) -> int:
        return in_channels

    def prediction_dim(self, in_channels: int) -> int:
        return in_channels

    def uncertainty_dim(self, in_channels: int) -> int:
        return in_channels

    def compute_labels(self, curr: torch.Tensor, next_: torch.Tensor) -> torch.Tensor:
        return next_ - curr

    def apply_prediction(self, curr: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        return curr + pred


class UnitCirclePredictor(Predictor):
    """Predict a phase shift for a 2-channel (sin,cos) or (cos,sin) unit circle.

    Label: scalar phase delta wrapped to [-π, π].
    Application: rotation — preserves unit norm exactly.

    Its uncertainty is a von Mises concentration, not a variance: the quantity
    lives on a circle, and a Gaussian over an angle has no idea that -pi and pi
    are the same place. Concentration plays sigma's role inversely -- large kappa
    is a tight belief -- and the likelihood becomes the Gaussian one in the limit,
    with kappa standing in for 1/sigma^2.

    Unlike the unbounded channels, a circular one has no scale ambiguity to
    remove: an angle is already measured in radians against a fixed 2*pi period.
    So the point here is the geometry, not scale invariance -- which is why the
    loss has to undo ``label_scale`` before taking a cosine of anything.
    """

    uncertainty_kind = "von_mises"

    def __init__(self, cosine_first: bool = False):
        self.cosine_first = cosine_first

    def uncertainty_dim(self, in_channels: int) -> int:
        return 1

    def target_dim(self, in_channels: int) -> int:
        return 2

    def prediction_dim(self, in_channels: int) -> int:
        return 1

    def compute_labels(self, curr: torch.Tensor, next_: torch.Tensor) -> torch.Tensor:
        if self.cosine_first:
            curr_angle = torch.atan2(curr[..., 1], curr[..., 0])
            next_angle = torch.atan2(next_[..., 1], next_[..., 0])
        else:
            curr_angle = torch.atan2(curr[..., 0], curr[..., 1])
            next_angle = torch.atan2(next_[..., 0], next_[..., 1])
        delta = (next_angle - curr_angle + math.pi) % (2.0 * math.pi) - math.pi
        return delta.unsqueeze(-1)

    def apply_prediction(self, curr: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        return phase_shift_circle(curr, pred.squeeze(-1), self.cosine_first)


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
    BOUNDARY = "boundary"


class Feature:
    def __init__(
        self,
        name: str,
        accessor: Accessor,
        input_encoder: Transform,
        target_encoder: Transform,
        predictor: Predictor | None = None,
        label_scale: float | tuple[float, ...] = 1.0,
        scope: FeatureScope = FeatureScope.SHARED,
    ):
        self.name = name
        self.accessor = accessor
        self.input_encoder = input_encoder
        self.target_encoder = target_encoder
        self.predictor = predictor
        self.label_scale = label_scale
        self.scope = scope

    def get_input(self, obs: YemongObservation) -> torch.Tensor:
        return self.input_encoder(self.accessor.get(obs))

    def get_target(self, obs: YemongObservation) -> torch.Tensor:
        return self.target_encoder(self.accessor.get(obs))

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
# Measured against 250 and 1000 px on real Frontline scenes
# (``benchmarks/presence_density_study.py``, artifacts/benchmarks/presence_density.json):
# at 250 px the 5v5 enemy channel is dead, median 0.18 with most ships reading
# zero; at 1000 px the 5v5 ally channel's 10th-to-90th percentile spread collapses
# from 1.51 to 0.84 because every ship reads crowded. 500 px is the setting where
# both channels carry a distribution at 5v5 and still separate at 50v50.
PRESENCE_RADIUS = 500.0
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
    overrides the input path and declares its own width. It has no target
    encoding and no predictor: it is a deterministic function of channels the
    auxiliary head already predicts, so predicting it again would supervise the
    same information twice under an invented label scale.
    """

    def __init__(self, ship_config: ShipConfig, radius: float = PRESENCE_RADIUS):
        super().__init__(
            name="local_presence",
            accessor=Accessor(ObsKey.POS),
            input_encoder=Identity(),
            target_encoder=Identity(),
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


@dataclasses.dataclass(frozen=True)
class _PredictorSpec:
    """Cached per-feature layout for a predictor feature.

    Computed once in ``FeatureCoordinator._init_dims`` so every downstream method
    reads offsets/dimensions from one source of truth instead of re-deriving them
    from a fresh dummy observation on each call.
    """

    name: str
    predictor: Predictor
    target_encoder: Transform
    t_dim: int  # target-space width
    p_dim: int  # prediction-space width
    t_offset: int  # start of this feature's slice in the target vector
    p_offset: int  # start of this feature's slice in the prediction vector
    label_scale: tuple[float, ...]  # per-prediction-dim scale, length == p_dim
    # Log-variance outputs, laid out in a block *after* every mean. Keeping the
    # means contiguous and first is what lets ``apply_prediction`` and every
    # rollout consumer go on slicing by ``p_offset`` against a widened head
    # without knowing uncertainty exists.
    u_dim: int  # 0 for a predictor that does not report uncertainty
    u_offset: int  # start of this feature's slice in the uncertainty block


class FeatureCoordinator:
    """Integrates a list of Features into cohesive input/target vectors."""

    def __init__(self, features: list[Feature], dummy_obs: YemongObservation | None = None):
        self.features = features
        # Bullet features read a different observation axis, so their coordinator
        # supplies its own probe rather than the ship/field one.
        self._dummy_override = dummy_obs
        self._init_dims()

    def _init_dims(self) -> None:
        dummy = self._dummy_obs()
        self.total_input_dimension = 0
        self.total_target_dimension = 0
        self.total_prediction_dimension = 0
        self.total_uncertainty_dimension = 0
        # One cached spec per predictor feature — the single source of truth for
        # every per-feature offset/dimension lookup below.
        self._predictor_specs: list[_PredictorSpec] = []
        # Lazily-built label-scale tensor, cached per device (see label_scale_vector).
        self._label_scale_cache: torch.Tensor | None = None
        self._uncertainty_cache: tuple[torch.Tensor, torch.Tensor] | None = None

        t_offset = 0
        p_offset = 0
        u_offset = 0
        for f in self.features:
            self.total_input_dimension += f.input_dimension(dummy)

            if f.predictor:
                t_dim = f.get_target(dummy).shape[-1]
                p_dim = f.predictor.prediction_dim(t_dim)
                u_dim = f.predictor.uncertainty_dim(t_dim)
                if isinstance(f.label_scale, (list, tuple)):
                    label_scale = tuple(float(s) for s in f.label_scale)
                else:
                    label_scale = (float(f.label_scale),) * p_dim
                self._predictor_specs.append(
                    _PredictorSpec(
                        name=f.name,
                        predictor=f.predictor,
                        target_encoder=f.target_encoder,
                        t_dim=t_dim,
                        p_dim=p_dim,
                        t_offset=t_offset,
                        p_offset=p_offset,
                        label_scale=label_scale,
                        u_dim=u_dim,
                        u_offset=u_offset,
                    )
                )
                self.total_target_dimension += t_dim
                self.total_prediction_dimension += p_dim
                self.total_uncertainty_dimension += u_dim
                t_offset += t_dim
                p_offset += p_dim
                u_offset += u_dim

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
                ObsKey.PREVIOUS_ACTION: torch.zeros((1, 1, 3), dtype=torch.long),
                ObsKey.LOCAL_LOG_INDEX: torch.zeros((1, 1, 1)),
                ObsKey.LOCAL_INDEX_GRADIENT: torch.zeros((1, 1, 2)),
                ObsKey.FIELD_TRANSITION_WIDTH: torch.zeros((1, 1, 1)),
                ObsKey.FIELD_TARGET_LOG_INDEX: torch.zeros((1, 1, 1)),
            }
        )

    # ------------------------------------------------------------------
    # Forward paths
    # ------------------------------------------------------------------

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

    def get_target_vector(self, obs: YemongObservation) -> torch.Tensor:
        parts = [f.get_target(obs) for f in self.features if f.predictor]
        if not parts:
            return obs.pos.new_zeros((*obs.pos.shape[:-1], 0))
        return torch.cat(parts, dim=-1)

    def target_slices(self) -> dict[str, slice]:
        """Map predicted feature names to their slices in the target vector."""
        return {
            spec.name: slice(spec.t_offset, spec.t_offset + spec.t_dim)
            for spec in self._predictor_specs
        }

    def decode_targets(self, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        """Invert each predictor feature's target encoding back to raw physical space.

        Args:
            targets: (..., total_target_dimension) — absolute target-space values,
                e.g. the output of ``apply_scaled_predictions``.

        Returns:
            Feature-name → raw tensor (..., raw_channels), one entry per predictor
            feature, with each feature's own target Transform performing the inverse.
        """
        return {
            spec.name: spec.target_encoder.invert(
                targets[..., spec.t_offset : spec.t_offset + spec.t_dim]
            )
            for spec in self._predictor_specs
        }

    # ------------------------------------------------------------------
    # Aux loss label computation
    # ------------------------------------------------------------------

    def label_scale_vector(self, device: torch.device) -> torch.Tensor:
        """Per-prediction-dim scale factors (1/std of raw labels).

        The vector is constant per coordinator, so it is built once and cached
        per device (mirroring Normalize's own scale-tensor caching).
        """
        cache = self._label_scale_cache
        if cache is None or cache.device != torch.device(device):
            scales = [s for spec in self._predictor_specs for s in spec.label_scale]
            cache = torch.tensor(scales, device=device, dtype=torch.float32)
            self._label_scale_cache = cache
        return cache

    def compute_labels(
        self, curr_targets: torch.Tensor, next_targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute prediction labels from curr/next target vectors, scaled to O(1).

        Both curr_targets and next_targets come from get_target_vector() and
        have the same per-feature layout: t_dim channels per feature.
        Labels are multiplied by label_scale so the network predicts O(1) values.
        """
        results = []
        for spec in self._predictor_specs:
            sl = slice(spec.t_offset, spec.t_offset + spec.t_dim)
            curr_slice = curr_targets[..., sl]
            next_slice = next_targets[..., sl]
            results.append(spec.predictor.compute_labels(curr_slice, next_slice))

        labels = torch.cat(results, dim=-1)
        return labels * self.label_scale_vector(labels.device)

    def _uncertainty_layout(self, device: torch.device) -> tuple[torch.Tensor, ...]:
        """Per-prediction-dim masks and the gather into the uncertainty block.

        ``gaussian`` and ``von_mises`` are disjoint masks over prediction dims;
        ``gather[i]`` is dim i's index in the uncertainty block (0 and unused
        where it has none). Cached per device the way ``label_scale_vector`` is
        -- this sits in the per-micro-batch loss path.
        """

        cached = self._uncertainty_cache
        if cached is not None and cached[0].device == device:
            return cached
        P = self.total_prediction_dimension
        gaussian = torch.zeros(P, dtype=torch.bool)
        von_mises = torch.zeros(P, dtype=torch.bool)
        gather = torch.zeros(P, dtype=torch.long)
        for spec in self._predictor_specs:
            if not spec.u_dim:
                continue
            mask = gaussian if spec.predictor.uncertainty_kind == "gaussian" else von_mises
            for offset in range(spec.p_dim):
                mask[spec.p_offset + offset] = True
                # A von Mises predictor reports one concentration for its single
                # phase output, so the two blocks line up elementwise either way.
                gather[spec.p_offset + offset] = spec.u_offset + min(offset, spec.u_dim - 1)
        cached = tuple(t.to(device) for t in (gaussian, von_mises, gather))
        self._uncertainty_cache = cached
        return cached

    def prediction_loss(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Per-prediction-dimension loss, given the head's full output.

        ``predictions`` is ``[means | uncertainties]``: the first
        ``total_prediction_dimension`` channels are means, the rest the
        uncertainty block, in log space and clamped by the head.

        Three forms, chosen per predictor, so channels convert one at a time:

        * Gaussian, for unbounded channels --
          ``0.5 * ((y - mu)^2 / sigma^2 + log sigma^2 + log 2*pi)``. Scale-free
          in the label, which is what takes ``label_scale`` out of the objective.
        * von Mises, for circular channels --
          ``kappa * (1 - cos(d)) + log I0e(kappa) + log 2*pi``, where ``d`` is
          the angular residual in radians. Written through ``i0e`` because
          ``log I0`` overflows for a confident belief.
        * plain squared error, for anything not yet converted.

        Both likelihoods carry their normalising constant, which cancels out of
        every gradient but makes the per-channel series comparable: they are
        then nats, and a channel costing more of them is genuinely harder to
        predict than one costing fewer.

        Both likelihoods weight the gradient on the mean by their precision, so
        a token whose label is mostly unpredictable belief error earns a wide
        spread and stops dominating the sum. Both are unbounded below as that
        spread shrinks, which is why the head clamps -- nothing here can recover
        from a sigma of zero.

        Returns:
            (..., total_prediction_dimension) elementwise loss.
        """

        mean = predictions[..., : self.total_prediction_dimension]
        sq_err = (mean - labels).pow(2)
        if not self.total_uncertainty_dimension:
            return sq_err

        gaussian_mask, von_mises_mask, gather = self._uncertainty_layout(predictions.device)
        log_uncertainty = predictions[..., self.total_prediction_dimension :].index_select(
            -1, gather
        )

        out = sq_err
        if gaussian_mask.any():
            nll = 0.5 * (
                sq_err * torch.exp(-log_uncertainty) + log_uncertainty + _LOG_TWO_PI
            )
            out = torch.where(gaussian_mask, nll, out)
        if von_mises_mask.any():
            # Undo label_scale first: it conditions the mean, and a cosine of a
            # residual multiplied by 177 would be measuring nothing.
            residual = (mean - labels) / self.label_scale_vector(predictions.device)
            kappa = torch.exp(log_uncertainty)
            nll = (
                kappa * (1.0 - torch.cos(residual))
                + torch.log(torch.special.i0e(kappa))
                + _LOG_TWO_PI
            )
            out = torch.where(von_mises_mask, nll, out)
        return out

    def apply_all_predictions(
        self, curr_targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.Tensor:
        results = []
        for spec in self._predictor_specs:
            t_slice = curr_targets[..., spec.t_offset : spec.t_offset + spec.t_dim]
            p_slice = predictions[..., spec.p_offset : spec.p_offset + spec.p_dim]
            results.append(spec.predictor.apply_prediction(t_slice, p_slice))

        return torch.cat(results, dim=-1)

    def apply_scaled_predictions(
        self, curr_targets: torch.Tensor, scaled_predictions: torch.Tensor
    ) -> torch.Tensor:
        """Unscale network outputs then apply to curr_targets.

        The network predicts in scaled space (labels * label_scale). Dividing by
        label_scale recovers the raw delta/absolute before calling apply_all_predictions.

        Takes the mean block only, so a head that also emits log variances can be
        handed straight to every rollout consumer without any of them knowing.
        """
        scaled_predictions = scaled_predictions[..., : self.total_prediction_dimension]
        scale = self.label_scale_vector(scaled_predictions.device)
        predictions = scaled_predictions / scale
        return self.apply_all_predictions(curr_targets, predictions)

    # ------------------------------------------------------------------
    # Loss weights and feature names
    # ------------------------------------------------------------------

    def get_loss_weights(self, device: torch.device) -> torch.Tensor:
        """Return uniform per-prediction weights (all 1.0).

        Importance weighting is deferred; relative scaling is handled by label_scale
        in compute_labels so that all predictions are already O(1).
        """
        return torch.ones(self.total_prediction_dimension, device=device, dtype=torch.float32)

    def get_feature_names(self) -> list[str]:
        names = []
        for spec in self._predictor_specs:
            names.extend(f"{spec.name}_{i}" for i in range(spec.p_dim))
        return names


# ---------------------------------------------------------------------------
# Standard coordinator factory
# ---------------------------------------------------------------------------


def build_standard_coordinator(
    ship_config: ShipConfig, *, local_presence: bool = False
) -> FeatureCoordinator:
    """Standard feature pipeline matching the current game's physics.

    Prediction layout (10 dims total):
      pos_x phase delta (1) | pos_y phase delta (1) | vel Δ(vx_norm, vy_norm) (2)
      att phase delta (1)   | ang_vel absolute (1)
      health phase delta (1) | power phase delta (1) | cooldown phase delta (1)
      local encoded log-index delta (1)

    label_scale values are 1/std(raw label) estimates so all scaled labels are O(1).
    These are rough estimates derived from old calibration weights and will tighten
    with dedicated measurement after training.
    """
    world_w, world_h = ship_config.world_size

    features = [
        # Position: Fourier input (rich freq) + unit-circle target (phase prediction)
        Feature(
            name="position_x",
            accessor=Accessor(ObsKey.POS, channels=[0]),
            input_encoder=Fourier(n_freqs=position_fourier_frequencies(world_w), periods=world_w),
            target_encoder=Fourier(n_freqs=1, periods=world_w),
            predictor=UnitCirclePredictor(cosine_first=False),  # Fourier gives (sin, cos)
            label_scale=177.4,
        ),
        Feature(
            name="position_y",
            accessor=Accessor(ObsKey.POS, channels=[1]),
            input_encoder=Fourier(n_freqs=position_fourier_frequencies(world_h), periods=world_h),
            target_encoder=Fourier(n_freqs=1, periods=world_h),
            predictor=UnitCirclePredictor(cosine_first=False),
            label_scale=177.4,
        ),
        # Velocity: SymlogVelocity encodes (vx, vy) → direction * symlog(speed).
        # AdditivePredictor on this 2D space avoids the angle discontinuity near
        # zero speed that plagued the old (Δphase, Δsymlog_speed) decomposition.
        Feature(
            name="velocity",
            accessor=Accessor(ObsKey.VEL),
            input_encoder=SymlogVelocity(),
            target_encoder=SymlogVelocity(),
            predictor=AdditivePredictor(),
            label_scale=(20.0, 20.0),
            scope=FeatureScope.SHIP,
        ),
        # Attitude: Fourier input, raw (cos,sin) target — phase prediction
        # wrapper produces (cos θ, sin θ) — cosine_first=True
        Feature(
            name="attitude",
            accessor=Accessor(ObsKey.ATT),
            input_encoder=AttitudeFourier(),
            target_encoder=Identity(),
            predictor=UnitCirclePredictor(cosine_first=True),
            label_scale=1.5,
            scope=FeatureScope.SHIP,
        ),
        # Angular velocity: symlog scalar, absolute prediction
        Feature(
            name="angular_velocity",
            accessor=Accessor(ObsKey.ANG_VEL),
            input_encoder=Symlog(),
            target_encoder=Symlog(),
            predictor=AbsolutePredictor(),
            label_scale=0.447,
            scope=FeatureScope.SHIP,
        ),
        Feature(
            name="shield_delay",
            accessor=Accessor(ObsKey.SHIELD_DELAY),
            input_encoder=Symlog(),
            target_encoder=Symlog(),
            predictor=AbsolutePredictor(),
            label_scale=1.0,
            scope=FeatureScope.SHIP,
        ),
        # Resources predict as plain bounded scalars, normalised to [0, 1].
        # They are not circular quantities: ``UnitCircle`` maps them onto a
        # *quarter* wave, so nothing ever wraps and the phase-delta predictor was
        # modelling a discontinuity that does not exist. A scalar target also
        # makes them real-valued, which is what lets them carry a Gaussian
        # uncertainty; the phase predictor cannot, and would need von Mises.
        #
        # The input encoder keeps its quarter-wave, which is a smooth and
        # perfectly good representation to read -- the mis-modelling was only
        # ever on the prediction side.
        #
        # label_scale is 1.0 rather than a fitted constant because these now
        # train under a scale-free likelihood; it survives only to condition the
        # mean, and ``next_state_label_scale/*`` reports what would centre it.
        Feature(
            name="health",
            accessor=Accessor(ObsKey.HEALTH),
            input_encoder=UnitCircle(scales=ship_config.max_health),
            target_encoder=Normalize(scales=ship_config.max_health),
            predictor=AbsolutePredictor(),
            label_scale=1.0,
        ),
        Feature(
            name="power",
            accessor=Accessor(ObsKey.POWER),
            input_encoder=UnitCircle(scales=ship_config.max_power),
            target_encoder=Normalize(scales=ship_config.max_power),
            predictor=AbsolutePredictor(),
            label_scale=1.0,
            scope=FeatureScope.SHIP,
        ),
        Feature(
            name="cooldown",
            accessor=Accessor(ObsKey.COOLDOWN),
            input_encoder=UnitCircle(scales=ship_config.firing_cooldown),
            target_encoder=Normalize(scales=ship_config.firing_cooldown),
            predictor=AbsolutePredictor(),
            label_scale=1.0,
            scope=FeatureScope.SHIP,
        ),
        # Categoricals and static (no predictor)
        Feature("team_id", Accessor(ObsKey.TEAM_ID), OneHot(3), Identity()),
        Feature("alive", Accessor(ObsKey.ALIVE), Identity(), Identity()),
        Feature(
            "visible", Accessor(ObsKey.VISIBLE), Identity(), Identity(), scope=FeatureScope.SHIP
        ),
        Feature(
            "belief_valid",
            Accessor(ObsKey.BELIEF_VALID),
            Identity(),
            Identity(),
            scope=FeatureScope.SHIP,
        ),
        Feature(
            "time_since_observation",
            Accessor(ObsKey.TIME_SINCE_OBSERVATION),
            Symlog(),
            Identity(),
            scope=FeatureScope.SHIP,
        ),
        Feature("object_type", Accessor(ObsKey.OBJECT_TYPE), OneHot(4), Identity()),
        Feature("zone_role", Accessor(ObsKey.ZONE_ROLE), OneHot(6), Identity()),
        Feature(
            "prev_power",
            Accessor(ObsKey.PREVIOUS_ACTION, [0]),
            OneHot(3),
            Identity(),
            scope=FeatureScope.SHIP,
        ),
        Feature(
            "prev_turn",
            Accessor(ObsKey.PREVIOUS_ACTION, [1]),
            OneHot(7),
            Identity(),
            scope=FeatureScope.SHIP,
        ),
        Feature(
            "prev_shoot",
            Accessor(ObsKey.PREVIOUS_ACTION, [2]),
            OneHot(2),
            Identity(),
            scope=FeatureScope.SHIP,
        ),
        Feature(
            "radius",
            Accessor(ObsKey.RADIUS),
            Normalize(0.5 * min(ship_config.world_size)),
            Identity(),
        ),
        # Field material features are numeric physical quantities. Ship slots are
        # zero for field-only channels; field slots are zero for ship-local index.
        Feature(
            "field_transition_width",
            Accessor(ObsKey.FIELD_TRANSITION_WIDTH),
            Normalize(ship_config.field_transition_width_max),
            Identity(),
            scope=FeatureScope.FIELD,
        ),
        Feature(
            "field_target_log_index",
            Accessor(ObsKey.FIELD_TARGET_LOG_INDEX),
            Identity(),
            Identity(),
            scope=FeatureScope.FIELD,
        ),
        Feature(
            "capture_progress",
            Accessor(ObsKey.CAPTURE_PROGRESS),
            Identity(),
            Identity(),
            scope=FeatureScope.ZONE,
        ),
        Feature(
            "capture_direction",
            Accessor(ObsKey.CAPTURE_DIRECTION),
            Identity(),
            Identity(),
            scope=FeatureScope.ZONE,
        ),
        Feature(
            "zone_offensive_distance",
            Accessor(ObsKey.ZONE_OFFENSIVE_DISTANCE),
            Symlog(),
            Identity(),
            scope=FeatureScope.ZONE,
        ),
        Feature(
            "zone_defensive_distance",
            Accessor(ObsKey.ZONE_DEFENSIVE_DISTANCE),
            Symlog(),
            Identity(),
            scope=FeatureScope.ZONE,
        ),
        Feature(
            "front_position",
            Accessor(ObsKey.FRONT_POSITION),
            Symlog(),
            Identity(),
            scope=FeatureScope.BOUNDARY,
        ),
        Feature(
            "front_win_threshold",
            Accessor(ObsKey.FRONT_WIN_THRESHOLD),
            Symlog(),
            Identity(),
            scope=FeatureScope.BOUNDARY,
        ),
        Feature(
            "time_remaining",
            Accessor(ObsKey.TIME_REMAINING),
            Identity(),
            Identity(),
            scope=FeatureScope.BOUNDARY,
        ),
        Feature(
            "game_mode",
            Accessor(ObsKey.GAME_MODE),
            Identity(),
            Identity(),
            scope=FeatureScope.BOUNDARY,
        ),
        Feature(
            name="local_log_index",
            accessor=Accessor(ObsKey.LOCAL_LOG_INDEX),
            input_encoder=Identity(),
            target_encoder=Identity(),
            predictor=AdditivePredictor(),
            label_scale=10.0,
            scope=FeatureScope.SHIP,
        ),
        # grad(n) at the ship, already normalised in observation_from_state. Input
        # only: it is a deterministic function of position given the static map, and
        # setting an aux label_scale for it would need a measurement we do not have.
        Feature(
            name="local_index_gradient",
            accessor=Accessor(ObsKey.LOCAL_INDEX_GRADIENT),
            input_encoder=Identity(),
            target_encoder=Identity(),
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
    encoding ships use for bounded resources: that encoding exists to give smooth
    phase-delta prediction targets, and bullets are never predicted.

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
            target_encoder=Identity(),
        ),
        Feature(
            name="bullet_position_y",
            accessor=BulletAccessor(BulletObsKey.POS, channels=[1]),
            input_encoder=Fourier(n_freqs=position_fourier_frequencies(world_h), periods=world_h),
            target_encoder=Identity(),
        ),
        Feature(
            name="bullet_velocity",
            accessor=BulletAccessor(BulletObsKey.VEL),
            input_encoder=SymlogVelocity(),
            target_encoder=Identity(),
        ),
        Feature(
            name="bullet_lifetime",
            accessor=BulletAccessor(BulletObsKey.LIFETIME),
            input_encoder=Identity(),
            target_encoder=Identity(),
        ),
        Feature(
            name="bullet_local_log_index",
            accessor=BulletAccessor(BulletObsKey.LOCAL_LOG_INDEX),
            input_encoder=Identity(),
            target_encoder=Identity(),
        ),
        Feature(
            name="bullet_local_index_gradient",
            accessor=BulletAccessor(BulletObsKey.LOCAL_INDEX_GRADIENT),
            input_encoder=Identity(),
            target_encoder=Identity(),
        ),
        Feature(
            name="bullet_team_id",
            accessor=BulletAccessor(BulletObsKey.TEAM_ID),
            input_encoder=OneHot(2),
            target_encoder=Identity(),
        ),
        Feature(
            name="bullet_active",
            accessor=BulletAccessor(BulletObsKey.ACTIVE),
            input_encoder=Identity(),
            target_encoder=Identity(),
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
    """Encode heading phase, retaining Cartesian targets for phase prediction."""

    def __init__(self):
        super().__init__(n_freqs=ATTITUDE_FOURIER_FREQUENCIES, periods=2.0 * math.pi)

    def out_dim(self, in_dim):
        return 2 * self.n_freqs

    def __call__(self, x):
        return super().__call__(torch.atan2(x[..., 1:2], x[..., 0:1]))
