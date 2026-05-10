"""Composable feature pipeline for observation encoding and aux prediction.

Each Feature bundles:
  - Accessor:  extracts raw channels from MVPObservation
  - Transform: encodes raw values into network-ready representation (input path)
  - Transform: encodes raw values into target space (aux prediction path)
  - Predictor: defines label computation and how predictions update the target

FeatureCoordinator integrates a list of Features into:
  - get_input_vector(obs)  → flat encoded observation for the encoder MLP
  - get_target_vector(obs) → flat target representation for aux loss
  - compute_labels(curr, next) → ground-truth labels (deltas or absolutes)
  - apply_all_predictions(curr, preds) → apply predicted updates to targets
"""

import math
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

from boost_and_broadside.env.observation import MVPObservation, ObsKey


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
    """Reads specific channels from an MVPObservation tensor."""

    def __init__(self, key: ObsKey, channels: list[int] | None = None):
        self.key = key
        self.channels = channels

    def get(self, obs: MVPObservation) -> torch.Tensor:
        val = obs[self.key]
        if self.channels is not None:
            return val[..., self.channels]
        return val


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
        if x.shape[-1] == 1:
            x = x.squeeze(-1)
        return F.one_hot(x, self.n).float()


class Normalize(Transform):
    """Divide by a scale factor."""

    def __init__(self, scales: float | list[float]):
        self.scales = scales

    def out_dim(self, in_dim: int) -> int:
        return in_dim

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.scales, list):
            s = torch.tensor(self.scales, device=x.device, dtype=x.dtype)
            return x.float() / s
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

    def out_dim(self, in_dim: int) -> int:
        return in_dim * 2 * self.n_freqs

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        ps = [self.periods] * x.shape[-1] if isinstance(self.periods, (float, int)) else self.periods
        results = []
        for i, period in enumerate(ps):
            xi = x[..., i]
            k = torch.arange(self.n_freqs, device=x.device, dtype=x.dtype)
            freqs = (2.0 * math.pi / period) * (2.0 ** k)
            args = xi.unsqueeze(-1) * freqs
            results.append(torch.sin(args))
            results.append(torch.cos(args))
        return torch.cat(results, dim=-1)


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


class Directional(Transform):
    """Map 2D velocity vector to (cos θ, sin θ, symlog_speed)."""

    def out_dim(self, in_dim: int) -> int:
        return 3

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        mag = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-6)
        direction = x / mag
        symlog_speed = symmetric_logarithm(mag)
        return torch.cat([direction, symlog_speed], dim=-1)


# ---------------------------------------------------------------------------
# Predictors (define label computation and prediction application)
# ---------------------------------------------------------------------------


class Predictor(ABC):
    @abstractmethod
    def target_dim(self, in_channels: int) -> int: ...

    @abstractmethod
    def prediction_dim(self, in_channels: int) -> int: ...

    @abstractmethod
    def compute_labels(self, curr: torch.Tensor, next_: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def apply_prediction(self, curr: torch.Tensor, pred: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def decode(self, target: torch.Tensor) -> torch.Tensor:
        """Invert target encoding back toward raw physical space."""
        ...


class AbsolutePredictor(Predictor):
    """Predict next state directly (absolute, no delta)."""

    def target_dim(self, in_channels: int) -> int:
        return in_channels

    def prediction_dim(self, in_channels: int) -> int:
        return in_channels

    def compute_labels(self, curr: torch.Tensor, next_: torch.Tensor) -> torch.Tensor:
        return next_

    def apply_prediction(self, curr: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        return pred

    def decode(self, target: torch.Tensor) -> torch.Tensor:
        return torch.sign(target) * torch.expm1(target.abs())


class AdditivePredictor(Predictor):
    """Predict delta: next − curr in target space."""

    def target_dim(self, in_channels: int) -> int:
        return in_channels

    def prediction_dim(self, in_channels: int) -> int:
        return in_channels

    def compute_labels(self, curr: torch.Tensor, next_: torch.Tensor) -> torch.Tensor:
        return next_ - curr

    def apply_prediction(self, curr: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        return curr + pred

    def decode(self, target: torch.Tensor) -> torch.Tensor:
        return target


class UnitCirclePredictor(Predictor):
    """Predict a phase shift for a 2-channel (sin,cos) or (cos,sin) unit circle.

    Label: scalar phase delta wrapped to [-π, π].
    Application: rotation — preserves unit norm exactly.
    """

    def __init__(self, cosine_first: bool = False):
        self.cosine_first = cosine_first

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

    def decode(self, target: torch.Tensor) -> torch.Tensor:
        if self.cosine_first:
            return torch.atan2(target[..., 1], target[..., 0]) % (2.0 * math.pi)
        return torch.atan2(target[..., 0], target[..., 1]) % (2.0 * math.pi)


class VelocityPredictor(Predictor):
    """Predict (Δphase, Δsymlog_speed) for a (cos θ, sin θ, symlog_speed) triple.

    Directional component uses phase rotation; speed uses additive delta.
    Both preserve the geometry of each sub-representation.
    """

    def target_dim(self, in_channels: int) -> int:
        return 3

    def prediction_dim(self, in_channels: int) -> int:
        return 2  # (delta_phase, delta_symlog_speed)

    def compute_labels(self, curr: torch.Tensor, next_: torch.Tensor) -> torch.Tensor:
        curr_angle = torch.atan2(curr[..., 1], curr[..., 0])
        next_angle = torch.atan2(next_[..., 1], next_[..., 0])
        delta_phase = (next_angle - curr_angle + math.pi) % (2.0 * math.pi) - math.pi
        delta_speed = next_[..., 2] - curr[..., 2]
        return torch.stack([delta_phase, delta_speed], dim=-1)

    def apply_prediction(self, curr: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        new_dir = phase_shift_circle(curr[..., 0:2], pred[..., 0], cosine_first=True)
        new_speed = curr[..., 2:3] + pred[..., 1:2]
        return torch.cat([new_dir, new_speed], dim=-1)

    def decode(self, target: torch.Tensor) -> torch.Tensor:
        direction = target[..., 0:2]
        symlog_speed = target[..., 2:3]
        speed = torch.sign(symlog_speed) * torch.expm1(symlog_speed.abs())
        return direction * speed


# ---------------------------------------------------------------------------
# Feature
# ---------------------------------------------------------------------------


class Feature:
    def __init__(
        self,
        name: str,
        accessor: Accessor,
        input_encoder: Transform,
        target_encoder: Transform,
        predictor: Predictor | None = None,
        weight: float | tuple[float, ...] = 1.0,
    ):
        self.name = name
        self.accessor = accessor
        self.input_encoder = input_encoder
        self.target_encoder = target_encoder
        self.predictor = predictor
        self.weight = weight

    def get_input(self, obs: MVPObservation) -> torch.Tensor:
        return self.input_encoder(self.accessor.get(obs))

    def get_target(self, obs: MVPObservation) -> torch.Tensor:
        return self.target_encoder(self.accessor.get(obs))


# ---------------------------------------------------------------------------
# FeatureCoordinator
# ---------------------------------------------------------------------------


class FeatureCoordinator:
    """Integrates a list of Features into cohesive input/target vectors."""

    def __init__(self, features: list[Feature]):
        self.features = features
        self._init_dims()

    def _init_dims(self) -> None:
        dummy = self._dummy_obs()
        self.total_input_dimension = 0
        self.total_target_dimension = 0
        self.total_prediction_dimension = 0

        for f in self.features:
            raw = f.accessor.get(dummy)
            in_c = raw.shape[-1] if raw.dim() > 2 else 1
            self.total_input_dimension += f.input_encoder.out_dim(in_c)

            if f.predictor:
                target_dummy = f.get_target(dummy)
                t_c = target_dummy.shape[-1]
                self.total_target_dimension += t_c
                self.total_prediction_dimension += f.predictor.prediction_dim(t_c)

    def _dummy_obs(self) -> MVPObservation:
        from boost_and_broadside.env.observation import MVPObservation, ObsKey
        return MVPObservation(data={
            ObsKey.POS:             torch.zeros((1, 1, 2)),
            ObsKey.VEL:             torch.zeros((1, 1, 2)),
            ObsKey.ATT:             torch.zeros((1, 1, 2)),
            ObsKey.ANG_VEL:         torch.zeros((1, 1, 1)),
            ObsKey.HEALTH:          torch.zeros((1, 1, 1)),
            ObsKey.POWER:           torch.zeros((1, 1, 1)),
            ObsKey.COOLDOWN:        torch.zeros((1, 1, 1)),
            ObsKey.TEAM_ID:         torch.zeros((1, 1), dtype=torch.long),
            ObsKey.ALIVE:           torch.zeros((1, 1), dtype=torch.bool),
            ObsKey.RADIUS:          torch.zeros((1, 1, 1)),
            ObsKey.PREVIOUS_ACTION: torch.zeros((1, 1, 3), dtype=torch.long),
        })

    # ------------------------------------------------------------------
    # Forward paths
    # ------------------------------------------------------------------

    def get_input_vector(self, obs: MVPObservation) -> torch.Tensor:
        return torch.cat([f.get_input(obs) for f in self.features], dim=-1)

    def get_target_vector(self, obs: MVPObservation) -> torch.Tensor:
        parts = [f.get_target(obs) for f in self.features if f.predictor]
        if not parts:
            return obs.pos.new_zeros((*obs.pos.shape[:-1], 0))
        return torch.cat(parts, dim=-1)

    # ------------------------------------------------------------------
    # Aux loss label computation
    # ------------------------------------------------------------------

    def compute_labels(
        self, curr_targets: torch.Tensor, next_targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute prediction labels from curr/next target vectors.

        Both curr_targets and next_targets come from get_target_vector() and
        have the same per-feature layout: t_dim channels per feature.
        Bug fix: both offsets advance by t_dim (not p_dim for next).
        """
        results = []
        curr_offset = 0
        next_offset = 0
        dummy = self._dummy_obs()

        for f in self.features:
            if not f.predictor:
                continue
            t_dim = f.get_target(dummy).shape[-1]
            p_dim = f.predictor.prediction_dim(t_dim)

            curr_slice = curr_targets[..., curr_offset: curr_offset + t_dim]
            next_slice = next_targets[..., next_offset: next_offset + t_dim]

            results.append(f.predictor.compute_labels(curr_slice, next_slice))

            curr_offset += t_dim
            next_offset += t_dim  # Fix: both advance by t_dim, not p_dim

        return torch.cat(results, dim=-1)

    def apply_all_predictions(
        self, curr_targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.Tensor:
        results = []
        t_offset = 0
        p_offset = 0
        dummy = self._dummy_obs()

        for f in self.features:
            if not f.predictor:
                continue
            t_dim = f.get_target(dummy).shape[-1]
            p_dim = f.predictor.prediction_dim(t_dim)

            t_slice = curr_targets[..., t_offset: t_offset + t_dim]
            p_slice = predictions[..., p_offset: p_offset + p_dim]

            results.append(f.predictor.apply_prediction(t_slice, p_slice))

            t_offset += t_dim
            p_offset += p_dim

        return torch.cat(results, dim=-1)

    # ------------------------------------------------------------------
    # Loss weights and feature names
    # ------------------------------------------------------------------

    def get_loss_weights(self, device: torch.device) -> torch.Tensor:
        weights = []
        dummy = self._dummy_obs()
        for f in self.features:
            if not f.predictor:
                continue
            t_dim = f.get_target(dummy).shape[-1]
            p_dim = f.predictor.prediction_dim(t_dim)
            if isinstance(f.weight, (list, tuple)):
                if len(f.weight) != p_dim:
                    raise ValueError(
                        f"Feature {f.name}: weight length {len(f.weight)} != prediction_dim {p_dim}"
                    )
                weights.extend(f.weight)
            else:
                weights.extend([f.weight] * p_dim)
        return torch.tensor(weights, device=device, dtype=torch.float32)

    def get_feature_names(self) -> list[str]:
        names = []
        dummy = self._dummy_obs()
        for f in self.features:
            if not f.predictor:
                continue
            t_dim = f.get_target(dummy).shape[-1]
            p_dim = f.predictor.prediction_dim(t_dim)
            names.extend(f"{f.name}_{i}" for i in range(p_dim))
        return names


# ---------------------------------------------------------------------------
# Standard coordinator factory
# ---------------------------------------------------------------------------


def build_standard_coordinator(ship_config) -> FeatureCoordinator:
    """Standard feature pipeline matching the current game's physics."""
    world_w, world_h = ship_config.world_size

    features = [
        # Position: Fourier input (rich freq) + unit-circle target (phase prediction)
        Feature(
            name="position_x",
            accessor=Accessor(ObsKey.POS, channels=[0]),
            input_encoder=Fourier(n_freqs=4, periods=world_w),
            target_encoder=Fourier(n_freqs=1, periods=world_w),
            predictor=UnitCirclePredictor(cosine_first=False),  # Fourier gives (sin, cos)
            weight=31485.6,
        ),
        Feature(
            name="position_y",
            accessor=Accessor(ObsKey.POS, channels=[1]),
            input_encoder=Fourier(n_freqs=4, periods=world_h),
            target_encoder=Fourier(n_freqs=1, periods=world_h),
            predictor=UnitCirclePredictor(cosine_first=False),
            weight=31485.6,
        ),
        # Velocity: (cos θ, sin θ, symlog_speed) — phase + speed delta
        Feature(
            name="velocity",
            accessor=Accessor(ObsKey.VEL),
            input_encoder=Directional(),
            target_encoder=Directional(),
            predictor=VelocityPredictor(),
            weight=(1794.5, 13011.6),
        ),
        # Attitude: Fourier input, raw (cos,sin) target — phase prediction
        # wrapper produces (cos θ, sin θ) — cosine_first=True
        Feature(
            name="attitude",
            accessor=Accessor(ObsKey.ATT),
            input_encoder=Fourier(n_freqs=4, periods=2.0 * math.pi),
            target_encoder=Identity(),
            predictor=UnitCirclePredictor(cosine_first=True),
            weight=219.7,
        ),
        # Angular velocity: symlog scalar, absolute prediction
        Feature(
            name="angular_velocity",
            accessor=Accessor(ObsKey.ANG_VEL),
            input_encoder=Symlog(),
            target_encoder=Symlog(),
            predictor=AbsolutePredictor(),
            weight=0.2,
        ),
        # Resources: quarter-wave (sin,cos), additive delta in that space
        Feature(
            name="health",
            accessor=Accessor(ObsKey.HEALTH),
            input_encoder=UnitCircle(scales=ship_config.max_health),
            target_encoder=UnitCircle(scales=ship_config.max_health),
            predictor=AdditivePredictor(),
            weight=(3162.4, 2150.1),
        ),
        Feature(
            name="power",
            accessor=Accessor(ObsKey.POWER),
            input_encoder=UnitCircle(scales=ship_config.max_power),
            target_encoder=UnitCircle(scales=ship_config.max_power),
            predictor=AdditivePredictor(),
            weight=(21044.4, 12478.6),
        ),
        Feature(
            name="cooldown",
            accessor=Accessor(ObsKey.COOLDOWN),
            input_encoder=UnitCircle(scales=ship_config.firing_cooldown),
            target_encoder=UnitCircle(scales=ship_config.firing_cooldown),
            predictor=AdditivePredictor(),
            weight=(11.3, 11.3),
        ),
        # Categoricals and static (no predictor)
        Feature("team_id",  Accessor(ObsKey.TEAM_ID),  OneHot(3),   Identity()),
        Feature("alive",    Accessor(ObsKey.ALIVE),     Identity(),  Identity()),
        Feature(
            "prev_power",
            Accessor(ObsKey.PREVIOUS_ACTION, [0]),
            OneHot(3),
            Identity(),
        ),
        Feature(
            "prev_turn",
            Accessor(ObsKey.PREVIOUS_ACTION, [1]),
            OneHot(7),
            Identity(),
        ),
        Feature(
            "prev_shoot",
            Accessor(ObsKey.PREVIOUS_ACTION, [2]),
            OneHot(2),
            Identity(),
        ),
        Feature("radius",   Accessor(ObsKey.RADIUS),    Normalize(40.0), Identity()),
    ]

    return FeatureCoordinator(features)
