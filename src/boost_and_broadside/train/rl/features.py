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

import torch
import torch.nn.functional as F

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.env.observation import ObsKey, YemongObservation

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

    def get(self, obs: YemongObservation) -> torch.Tensor:
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
            k = torch.arange(self.n_freqs, device=x.device, dtype=x.dtype)
            freqs = (2.0 * math.pi / period) * (2.0**k)
            args = xi.unsqueeze(-1) * freqs
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


class Predictor(ABC):
    @abstractmethod
    def target_dim(self, in_channels: int) -> int: ...

    @abstractmethod
    def prediction_dim(self, in_channels: int) -> int: ...

    @abstractmethod
    def compute_labels(self, curr: torch.Tensor, next_: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def apply_prediction(self, curr: torch.Tensor, pred: torch.Tensor) -> torch.Tensor: ...


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


# ---------------------------------------------------------------------------
# Windowed cumulative loss
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class WindowedLoss:
    """Triangle convolution cumulative loss spec for a Feature's predictions.

    Convolves per-step prediction errors with a symmetric triangle kernel and
    penalises the squared output. This amplifies systematic bias (which scales
    as window²) relative to random noise (which scales as window), catching
    drift that teacher-forced per-step loss cannot see.

    Attributes:
        window: Length of the triangle kernel (steps).
        weight: Scalar multiplier applied to this feature's windowed loss before
                it is averaged with other windowed features in the coordinator.
    """

    window: int = 32
    weight: float = 1.0


def _triangle_conv_loss(
    errors: torch.Tensor,  # (T, B, N, D)
    ns_mask: torch.Tensor,  # (T, B, N) — True where prediction is valid
    terminated: torch.Tensor,  # (T, B)    — True where episode ended
    window: int,
) -> torch.Tensor:
    """MSE of triangle-filtered prediction errors along the time axis.

    Steps:
      1. Zero errors at invalid positions (ns_mask guards dead ships + terminals).
      2. Apply a causal symmetric-triangle conv1d along T.
      3. Mask out output positions whose window spans an episode boundary.
      4. Return mean squared conv output over valid (T, B, N) positions.
    """
    T, B, N, D = errors.shape
    device = errors.device

    # Zero errors at invalid steps so they don't contaminate the conv sum.
    errors = errors * ns_mask.float().unsqueeze(-1)

    # Symmetric triangle kernel: h[s] = min(s+1, window-s), normalized.
    # Peaks in the centre; provides better spectral side-lobe suppression than
    # a box filter so high-frequency per-step noise leaks less into the loss.
    s = torch.arange(window, device=device, dtype=torch.float32)
    h = torch.minimum(s + 1, torch.full_like(s, window) - s)
    h = h / h.sum()

    # Causal conv1d: output[t] = Σ h[s] * error[t-window+1+s] for s in [0,window).
    # Reshape to (B*N*D, 1, T), pad left by window-1, apply conv.
    x = errors.permute(1, 2, 3, 0).reshape(B * N * D, 1, T)
    x_padded = F.pad(x, (window - 1, 0))
    conv_out = F.conv1d(x_padded, h.view(1, 1, window))  # (B*N*D, 1, T)
    conv_out = conv_out.view(B, N, D, T).permute(3, 0, 1, 2)  # (T, B, N, D)

    # Temporal validity: position t is invalid if any termination falls inside
    # the window [t-window+1, t]. Max-pool with explicit left padding checks this.
    term_f = terminated.T.unsqueeze(1).float()  # (B, 1, T)
    spread = F.max_pool1d(F.pad(term_f, (window - 1, 0)), window, stride=1)
    t_valid = (spread.squeeze(1) == 0).T  # (T, B)

    # Per-ship validity: ship must be alive and non-terminal at window endpoint.
    valid = t_valid.unsqueeze(-1) & ns_mask  # (T, B, N)
    valid_4d = valid.float().unsqueeze(-1)  # (T, B, N, 1)
    n_valid = valid_4d.sum() * D

    return (conv_out.pow(2) * valid_4d).sum() / n_valid.clamp(min=1.0)


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
        label_scale: float | tuple[float, ...] = 1.0,
        windowed_loss: WindowedLoss | None = None,
    ):
        self.name = name
        self.accessor = accessor
        self.input_encoder = input_encoder
        self.target_encoder = target_encoder
        self.predictor = predictor
        self.label_scale = label_scale
        self.windowed_loss = windowed_loss

    def get_input(self, obs: YemongObservation) -> torch.Tensor:
        return self.input_encoder(self.accessor.get(obs))

    def get_target(self, obs: YemongObservation) -> torch.Tensor:
        return self.target_encoder(self.accessor.get(obs))


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
        # One cached spec per predictor feature — the single source of truth for
        # every per-feature offset/dimension lookup below.
        self._predictor_specs: list[_PredictorSpec] = []
        # (pred_offset, pred_dim, WindowedLoss) for features that opt in
        self._windowed_loss_specs: list[tuple[int, int, WindowedLoss]] = []
        # Lazily-built label-scale tensor, cached per device (see label_scale_vector).
        self._label_scale_cache: torch.Tensor | None = None

        t_offset = 0
        p_offset = 0
        for f in self.features:
            raw = f.accessor.get(dummy)
            in_c = raw.shape[-1] if raw.dim() > 2 else 1
            self.total_input_dimension += f.input_encoder.out_dim(in_c)

            if f.predictor:
                t_dim = f.get_target(dummy).shape[-1]
                p_dim = f.predictor.prediction_dim(t_dim)
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
                    )
                )
                self.total_target_dimension += t_dim
                self.total_prediction_dimension += p_dim
                if f.windowed_loss is not None:
                    self._windowed_loss_specs.append((p_offset, p_dim, f.windowed_loss))
                t_offset += t_dim
                p_offset += p_dim

    def _dummy_obs(self) -> YemongObservation:
        from boost_and_broadside.env.observation import ObsKey, YemongObservation

        return YemongObservation(
            data={
                ObsKey.POS: torch.zeros((1, 1, 2)),
                ObsKey.VEL: torch.zeros((1, 1, 2)),
                ObsKey.ATT: torch.zeros((1, 1, 2)),
                ObsKey.ANG_VEL: torch.zeros((1, 1, 1)),
                ObsKey.HEALTH: torch.zeros((1, 1, 1)),
                ObsKey.POWER: torch.zeros((1, 1, 1)),
                ObsKey.COOLDOWN: torch.zeros((1, 1, 1)),
                ObsKey.TEAM_ID: torch.zeros((1, 1), dtype=torch.long),
                ObsKey.ALIVE: torch.zeros((1, 1), dtype=torch.bool),
                ObsKey.RADIUS: torch.zeros((1, 1, 1)),
                ObsKey.PREVIOUS_ACTION: torch.zeros((1, 1, 3), dtype=torch.long),
                ObsKey.LOCAL_LOG_INDEX: torch.zeros((1, 1, 1)),
                ObsKey.FIELD_TRANSITION_WIDTH: torch.zeros((1, 1, 1)),
                ObsKey.FIELD_INSIDE_LOG_INDEX: torch.zeros((1, 1, 1)),
                ObsKey.FIELD_OUTSIDE_LOG_INDEX: torch.zeros((1, 1, 1)),
                ObsKey.FIELD_LOG_INDEX_RATIO: torch.zeros((1, 1, 1)),
                ObsKey.FIELD_DAMAGE: torch.zeros((1, 1, 1)),
            }
        )

    # ------------------------------------------------------------------
    # Forward paths
    # ------------------------------------------------------------------

    def get_input_vector(self, obs: YemongObservation) -> torch.Tensor:
        return torch.cat([f.get_input(obs) for f in self.features], dim=-1)

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
        """
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

    def compute_windowed_loss(
        self,
        pred_rollout: torch.Tensor,  # (T, B, N, D_pred)
        label_rollout: torch.Tensor,  # (T, B, N, D_pred)
        ns_mask: torch.Tensor,  # (T, B, N) — alive & non-terminal
        terminated: torch.Tensor,  # (T, B)
    ) -> torch.Tensor:
        """Triangle convolution cumulative loss for features that opt in via WindowedLoss.

        Errors are extracted per-feature using stored prediction-dimension offsets,
        passed through _triangle_conv_loss, and averaged (with per-feature weights
        absorbed before averaging).

        Returns a scalar; zero if no features have windowed_loss set.
        """
        if not self._windowed_loss_specs:
            return pred_rollout.new_zeros(())

        errors = pred_rollout.float() - label_rollout.float()  # (T, B, N, D_pred)
        total = pred_rollout.new_zeros(())

        for p_offset, p_dim, wl in self._windowed_loss_specs:
            feat_errs = errors[..., p_offset : p_offset + p_dim]
            total = total + wl.weight * _triangle_conv_loss(
                feat_errs, ns_mask, terminated, wl.window
            )

        return total / len(self._windowed_loss_specs)

    def get_feature_names(self) -> list[str]:
        names = []
        for spec in self._predictor_specs:
            names.extend(f"{spec.name}_{i}" for i in range(spec.p_dim))
        return names


# ---------------------------------------------------------------------------
# Standard coordinator factory
# ---------------------------------------------------------------------------


def build_standard_coordinator(ship_config: ShipConfig) -> FeatureCoordinator:
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
            input_encoder=Fourier(n_freqs=4, periods=world_w),
            target_encoder=Fourier(n_freqs=1, periods=world_w),
            predictor=UnitCirclePredictor(cosine_first=False),  # Fourier gives (sin, cos)
            label_scale=177.4,
            windowed_loss=WindowedLoss(window=32),
        ),
        Feature(
            name="position_y",
            accessor=Accessor(ObsKey.POS, channels=[1]),
            input_encoder=Fourier(n_freqs=4, periods=world_h),
            target_encoder=Fourier(n_freqs=1, periods=world_h),
            predictor=UnitCirclePredictor(cosine_first=False),
            label_scale=177.4,
            windowed_loss=WindowedLoss(window=32),
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
            windowed_loss=WindowedLoss(window=32),
        ),
        # Attitude: Fourier input, raw (cos,sin) target — phase prediction
        # wrapper produces (cos θ, sin θ) — cosine_first=True
        Feature(
            name="attitude",
            accessor=Accessor(ObsKey.ATT),
            input_encoder=Fourier(n_freqs=4, periods=2.0 * math.pi),
            target_encoder=Identity(),
            predictor=UnitCirclePredictor(cosine_first=True),
            label_scale=1.5,
        ),
        # Angular velocity: symlog scalar, absolute prediction
        Feature(
            name="angular_velocity",
            accessor=Accessor(ObsKey.ANG_VEL),
            input_encoder=Symlog(),
            target_encoder=Symlog(),
            predictor=AbsolutePredictor(),
            label_scale=0.447,
        ),
        # Resources: quarter-wave (sin,cos) target + phase-delta prediction.
        # UnitCirclePredictor is geometrically correct for circular quantities
        # (1D phase delta vs 2D Cartesian delta for a 1 DoF variable).
        Feature(
            name="health",
            accessor=Accessor(ObsKey.HEALTH),
            input_encoder=UnitCircle(scales=ship_config.max_health),
            target_encoder=UnitCircle(scales=ship_config.max_health),
            predictor=UnitCirclePredictor(cosine_first=False),
            label_scale=36.0,
        ),
        Feature(
            name="power",
            accessor=Accessor(ObsKey.POWER),
            input_encoder=UnitCircle(scales=ship_config.max_power),
            target_encoder=UnitCircle(scales=ship_config.max_power),
            predictor=UnitCirclePredictor(cosine_first=False),
            label_scale=93.0,
        ),
        Feature(
            name="cooldown",
            accessor=Accessor(ObsKey.COOLDOWN),
            input_encoder=UnitCircle(scales=ship_config.firing_cooldown),
            target_encoder=UnitCircle(scales=ship_config.firing_cooldown),
            predictor=UnitCirclePredictor(cosine_first=False),
            label_scale=2.1,
        ),
        # Categoricals and static (no predictor)
        Feature("team_id", Accessor(ObsKey.TEAM_ID), OneHot(3), Identity()),
        Feature("alive", Accessor(ObsKey.ALIVE), Identity(), Identity()),
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
        ),
        Feature(
            "field_inside_log_index",
            Accessor(ObsKey.FIELD_INSIDE_LOG_INDEX),
            Identity(),
            Identity(),
        ),
        Feature(
            "field_outside_log_index",
            Accessor(ObsKey.FIELD_OUTSIDE_LOG_INDEX),
            Identity(),
            Identity(),
        ),
        Feature(
            "field_log_index_ratio",
            Accessor(ObsKey.FIELD_LOG_INDEX_RATIO),
            Identity(),
            Identity(),
        ),
        Feature("field_damage", Accessor(ObsKey.FIELD_DAMAGE), Identity(), Identity()),
        Feature(
            name="local_log_index",
            accessor=Accessor(ObsKey.LOCAL_LOG_INDEX),
            input_encoder=Identity(),
            target_encoder=Identity(),
            predictor=AdditivePredictor(),
            label_scale=10.0,
        ),
    ]

    return FeatureCoordinator(features)
