"""GPU-resident rollout buffer for recurrent PPO with decomposed per-component critic.

All data is pre-allocated on-device. No CPU-GPU transfers during rollout
collection. GAE is computed fully on GPU.

Shape conventions throughout:
    T = num_steps (rollout length)
    B = num_envs
    N = num_ships
    K = num_value_components (per-component critic decomposition)
    D = d_model
    H = packed recurrent state width (CONV_KERNEL * D)
"""

from collections.abc import Generator
from typing import NamedTuple

import torch

from boost_and_broadside.env.observation import BulletObsKey, ObsKey, YemongObservation


class MicroBatch(NamedTuple):
    """One recurrent PPO micro-batch produced by :class:`RolloutBuffer`."""

    obs: YemongObservation
    actions: torch.Tensor
    old_logprobs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    alive: torch.Tensor
    hidden: torch.Tensor
    actor_mask: torch.Tensor
    expert_probs: torch.Tensor
    terminated: torch.Tensor
    adv_agg: torch.Tensor
    ret_agg: torch.Tensor
    ns_labels: torch.Tensor | None

    def pin_memory(self) -> "MicroBatch":
        """Copy one CPU micro-batch into page-locked transfer memory.

        Returns:
            A structurally identical micro-batch backed by pinned CPU tensors.

        Raises:
            ValueError: If the batch is not CPU-resident.
        """
        if self.actions.device.type != "cpu":
            raise ValueError("pin_memory() requires a CPU micro-batch")
        return MicroBatch(
            obs=self.obs.map(lambda t: t.pin_memory()),
            actions=self.actions.pin_memory(),
            old_logprobs=self.old_logprobs.pin_memory(),
            advantages=self.advantages.pin_memory(),
            returns=self.returns.pin_memory(),
            alive=self.alive.pin_memory(),
            hidden=self.hidden.pin_memory(),
            actor_mask=self.actor_mask.pin_memory(),
            expert_probs=self.expert_probs.pin_memory(),
            terminated=self.terminated.pin_memory(),
            adv_agg=self.adv_agg.pin_memory(),
            ret_agg=self.ret_agg.pin_memory(),
            ns_labels=self.ns_labels.pin_memory() if self.ns_labels is not None else None,
        )

    def to(self, device: torch.device, non_blocking: bool = False) -> "MicroBatch":
        """Move one micro-batch to the update device.

        Args:
            device: Destination device.
            non_blocking: Request asynchronous copies when the source supports them.

        Returns:
            A micro-batch whose tensors reside on ``device``.
        """
        if self.actions.device == device:
            return self
        return MicroBatch(
            obs=self.obs.map(lambda t: t.to(device=device, non_blocking=non_blocking)),
            actions=self.actions.to(device=device, non_blocking=non_blocking),
            old_logprobs=self.old_logprobs.to(device=device, non_blocking=non_blocking),
            advantages=self.advantages.to(device=device, non_blocking=non_blocking),
            returns=self.returns.to(device=device, non_blocking=non_blocking),
            alive=self.alive.to(device=device, non_blocking=non_blocking),
            hidden=self.hidden.to(device=device, non_blocking=non_blocking),
            actor_mask=self.actor_mask.to(device=device, non_blocking=non_blocking),
            expert_probs=self.expert_probs.to(device=device, non_blocking=non_blocking),
            terminated=self.terminated.to(device=device, non_blocking=non_blocking),
            adv_agg=self.adv_agg.to(device=device, non_blocking=non_blocking),
            ret_agg=self.ret_agg.to(device=device, non_blocking=non_blocking),
            ns_labels=(
                self.ns_labels.to(device=device, non_blocking=non_blocking)
                if self.ns_labels is not None
                else None
            ),
        )

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        """Record a CUDA consumer stream for every tensor in this batch.

        Args:
            stream: Stream that will consume tensors allocated by a copy stream.
        """
        for _, value in self.obs.items():
            value.record_stream(stream)
        for value in self[1:]:
            if isinstance(value, torch.Tensor):
                value.record_stream(stream)

    def slice_envs(self, start: int, end: int, num_recurrent: int) -> "MicroBatch":
        """Slice a contiguous environment range from a staged shard minibatch.

        Args:
            start: Inclusive environment offset.
            end: Exclusive environment offset.
            num_recurrent: Recurrent tokens per environment (ships), used to reshape
                hidden state. Fields are non-recurrent, so this is not N+M.

        Returns:
            A view-only micro-batch over ``[start:end]``.
        """
        batch_envs = self.actions.shape[1]
        n_layers, _, hidden_width = self.hidden.shape
        hidden = self.hidden.reshape(
            n_layers,
            batch_envs,
            num_recurrent,
            hidden_width,
        )[:, start:end]
        return MicroBatch(
            obs=self.obs.map(lambda t: t[:, start:end]),
            actions=self.actions[:, start:end],
            old_logprobs=self.old_logprobs[:, start:end],
            advantages=self.advantages[:, start:end],
            returns=self.returns[:, start:end],
            alive=self.alive[:, start:end],
            hidden=hidden.reshape(n_layers, (end - start) * num_recurrent, hidden_width),
            actor_mask=self.actor_mask[:, start:end],
            expert_probs=self.expert_probs[:, start:end],
            terminated=self.terminated[:, start:end],
            adv_agg=self.adv_agg[:, start:end],
            ret_agg=self.ret_agg[:, start:end],
            ns_labels=self.ns_labels[:, start:end] if self.ns_labels is not None else None,
        )

    def split_envs(self, num_chunks: int, num_recurrent: int) -> list["MicroBatch"]:
        """Split a staged shard minibatch into near-even contiguous views."""
        batch_envs = self.actions.shape[1]
        base, remainder = divmod(batch_envs, num_chunks)
        chunks = []
        start = 0
        for index in range(num_chunks):
            width = base + (1 if index < remainder else 0)
            chunks.append(self.slice_envs(start, start + width, num_recurrent))
            start += width
        return chunks


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric log transform: sign(x) * log(1 + |x|).

    Compresses large reward magnitudes while preserving sign and the zero point.
    Applied to raw rewards at storage time [symlog #1].
    """
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog: sign(x) * (exp(|x|) - 1)."""
    return torch.sign(x) * torch.expm1(x.abs())


# --------------------------------------------------------------------------
# Reduced-precision storage for the GPU-resident rollout buffer
# --------------------------------------------------------------------------
# The buffer is the batch-axis memory hog: every array scales with num_envs. We
# halve its float channels while holding two hard rules that keep this a pure
# memory optimization rather than a training change:
#
#   1. bf16, never fp16. bf16 keeps fp32's full exponent range, so a reward or
#      value spike cannot silently overflow to inf the way fp16 (max 65504) can.
#      The stored channels are read-once leaf data whose ~0.4% mantissa rounding
#      is negligible; range safety is what matters.
#
#   2. Accumulators stay fp32. Anything that sums or runs an EMA over the stored
#      data upcasts first (see compute_gae, AdvantageScaler/ReturnScaler.update
#      and PPOTrainer._precompute_lambda_aggregates). bf16's ~0.4% resolution
#      would let small increments vanish under a large running value — the classic
#      swamping failure — so no running statistic is ever held in bf16.
#
# Positions are the deliberate exception and stay fp32: the Fourier position
# encoder needs sub-pixel accuracy, and bf16's 8-bit mantissa resolves only ~1
# part in 256 (~4 px on a 1024 map, and linearly worse as maps grow). fp16 would
# reach 2048 px but violates rule 1 and still caps out on large maps.
_STORAGE_FLOAT: torch.dtype = torch.bfloat16

# Per-observation-channel storage dtype overrides. Channels not listed fall back
# by kind (see _obs_storage_dtype): float → bf16, int → uint8, bool → bool.
_OBS_STORAGE_OVERRIDES: dict[ObsKey | BulletObsKey, torch.dtype] = {
    ObsKey.POS: torch.float32,  # keep full precision — needed now and for large maps
    BulletObsKey.POS: torch.float32,  # same Fourier basis as ship position
}


def _obs_storage_dtype(key: ObsKey | BulletObsKey, dt: torch.dtype) -> torch.dtype:
    """Reduced storage dtype for observation channel ``key`` of source dtype ``dt``."""
    override = _OBS_STORAGE_OVERRIDES.get(key)
    if override is not None:
        return override
    if dt.is_floating_point:
        return _STORAGE_FLOAT
    if dt == torch.bool:
        return torch.bool
    # team_id (0-2) and previous_action indices (0-6) are small non-negatives; the
    # feature read path upcasts via .long()/.float() before any arithmetic.
    return torch.uint8


class ReturnScaler:
    """Per-component EMA of 5th/95th percentiles for return normalization.

    Maps symlog-reward space returns to roughly [-1, 1] per component so that
    MSE value loss is comparable across components that have very different
    natural scales (e.g. victory ±80 vs turn_rate ±0.001).

    The scaler is updated once per rollout from the buffer's computed returns.
    Lambdas in the PPO advantage aggregation then act as pure importance weights
    (sign + magnitude) rather than also implicitly controlling scale.

    ``min_span`` is a divide-by-zero guard, not a scale. It must sit far below
    every active component's real span: a floor that binds on a live component
    silently shrinks that component's critic targets, and so its share of the
    value loss, by the ratio between the floor and the truth. Check
    ``floor_bound`` — the trainer logs it per component — before raising it.

    Args:
        num_components: K — number of value components.
        device:         Torch device (must match the returns tensor).
        ema_alpha:      EMA decay rate per rollout update (default 0.005 ≈ 200-update
                        memory). Slower = more stable but slower adaptation.
        min_span:       Degeneracy epsilon on p95−p5 (symlog-space). Guards a
                        component whose returns collapse to a constant — e.g. one
                        whose reward group scale is scheduled to zero.
    """

    def __init__(
        self,
        num_components: int,
        device: torch.device,
        ema_alpha: float = 0.005,
        min_span: float = 1e-3,
    ) -> None:
        self.alpha = ema_alpha
        self.min_span = min_span
        self._initialized = False
        self._p5 = torch.zeros(num_components, device=device)
        self._p95 = torch.zeros(num_components, device=device)

    @torch.no_grad()
    def update(self, returns: torch.Tensor) -> None:
        """Update EMA percentiles from this rollout's returns.

        On the first call the EMA is seeded with the observed percentiles directly
        (no prior) so that normalization is correct from the very first update.

        Args:
            returns: (T, B, N, K) bf16-stored — GAE returns in symlog-reward space
                     (upcast to fp32 internally for the percentile reduction).
        """
        K = returns.shape[-1]
        flat = returns.reshape(-1, K)  # (T*B*N, K)
        p5 = torch.quantile(flat.float(), 0.05, dim=0)  # (K,)
        p95 = torch.quantile(flat.float(), 0.95, dim=0)
        self._update_percentiles(p5, p95)

    @torch.no_grad()
    def update_chunks(
        self,
        returns_chunks: list[torch.Tensor],
        max_samples: int | None = None,
    ) -> None:
        """Update percentiles from one logical rollout split across host shards.

        Args:
            returns_chunks: Shards shaped ``(T, B, N, K)`` on a common device.
            max_samples: Maximum flattened entities retained across all shards.
                None computes exact quantiles.
        """
        if len(returns_chunks) == 0:
            raise ValueError("returns_chunks must contain at least one shard")
        K = returns_chunks[0].shape[-1]
        flat_chunks = [returns.reshape(-1, K) for returns in returns_chunks]
        if max_samples is not None:
            samples_per_chunk = max(1, max_samples // len(flat_chunks))
            flat_chunks = [
                self._evenly_sample_entities(flat, samples_per_chunk) for flat in flat_chunks
            ]
        flat = torch.cat(flat_chunks, dim=0).float()  # (samples, K)
        quantiles = torch.tensor(
            (0.05, 0.95),
            dtype=torch.float32,
            device=returns_chunks[0].device,
        )
        percentile_matrix = torch.quantile(flat, quantiles, dim=0)  # (2, K)
        p5, p95 = percentile_matrix[0], percentile_matrix[1]
        self._update_percentiles(p5, p95)

    @staticmethod
    def _evenly_sample_entities(flat: torch.Tensor, max_samples: int) -> torch.Tensor:
        """Select a deterministic, evenly spaced entity sample."""
        if flat.shape[0] <= max_samples:
            return flat
        indices = torch.linspace(
            0,
            flat.shape[0] - 1,
            steps=max_samples,
            device=flat.device,
        ).long()  # (max_samples,)
        return flat[indices]  # (max_samples, K)

    def _update_percentiles(self, p5: torch.Tensor, p95: torch.Tensor) -> None:
        """Apply one observed percentile pair to the running EMA."""
        p5 = p5.to(self._p5.device)
        p95 = p95.to(self._p95.device)
        if not self._initialized:
            self._p5 = p5
            self._p95 = p95
            self._initialized = True
        else:
            self._p5 = (1.0 - self.alpha) * self._p5 + self.alpha * p5
            self._p95 = (1.0 - self.alpha) * self._p95 + self.alpha * p95

    def _half_span(self) -> torch.Tensor:
        """Half the p95−p5 range, clamped to at least min_span/2."""
        return ((self._p95 - self._p5) * 0.5).clamp(min=self.min_span * 0.5)

    @property
    def floor_bound(self) -> torch.Tensor:
        """(K,) bool — components whose span is being held up by ``min_span``.

        True for an active component means its critic targets are compressed by
        the guard rather than scaled by its own statistics.
        """
        return (self._p95 - self._p5) < self.min_span

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Map from symlog-reward space → normalized space (≈ [-1, 1] per component).

        Args:
            x: (..., K) float — values in symlog-reward space.

        Returns:
            (..., K) float — normalized values.
        """
        center = (self._p95 + self._p5) * 0.5  # (K,)
        return (x - center) / self._half_span()

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Map from normalized space → symlog-reward space.

        Args:
            x: (..., K) float — normalized values.

        Returns:
            (..., K) float — values in symlog-reward space.
        """
        center = (self._p95 + self._p5) * 0.5
        return x * self._half_span() + center

    def state_dict(self) -> dict:
        return {
            "p5": self._p5.cpu(),
            "p95": self._p95.cpu(),
            "initialized": self._initialized,
            "min_span": self.min_span,
        }

    @property
    def percentiles(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Current p5 and p95 vectors."""
        return self._p5, self._p95

    def load_state_dict(self, d: dict) -> None:
        self._p5 = d["p5"].to(self._p5.device)
        self._p95 = d["p95"].to(self._p95.device)
        self._initialized = d.get("initialized", True)  # assume initialized if loading old ckpt
        # A checkpoint written under a different floor carries percentiles shaped
        # by that floor. Re-seeding costs one rollout; EMA-ing a stale floor out
        # takes ~1/alpha updates, during which the component is mis-scaled.
        if d.get("min_span") != self.min_span:
            self._initialized = False


class AdvantageScaler:
    """Per-component EMA of advantage RMS for policy gradient normalization.

    The actor-side counterpart to ReturnScaler. While ReturnScaler normalizes
    value targets so MSE loss is comparable across components, AdvantageScaler
    normalizes advantages before lambda aggregation so each component contributes
    equally to the policy gradient regardless of raw reward magnitude or density.

    Follows Dreamer v3's approach: running scale statistics make the system
    invariant to reward magnitude across all K components. Uses RMS rather than
    p5/p95 because advantages are already approximately zero-mean from GAE's
    baseline subtraction.

    ``min_rms`` is a divide-by-zero guard, not a scale. Sparse components have
    genuinely small advantages — a terminal win reward spread over a whole
    episode by GAE lands two orders of magnitude below a per-step damage signal —
    so a floor set at "small" rather than "epsilon" binds on them forever and
    silently downweights them in the policy gradient by the ratio between the
    floor and the truth. Check ``floor_bound``, which the trainer logs per
    component, before raising it.

    Args:
        num_components: K — number of value components.
        device:         Torch device.
        ema_alpha:      EMA decay per rollout (default 0.005 ≈ 200-rollout memory,
                        matching ReturnScaler).
        min_rms:        Degeneracy epsilon on the advantage RMS. Guards a
                        component whose advantages collapse to zero — e.g. one
                        whose reward group scale is scheduled to zero.
    """

    def __init__(
        self,
        num_components: int,
        device: torch.device,
        ema_alpha: float = 0.005,
        min_rms: float = 1e-4,
    ) -> None:
        self.alpha = ema_alpha
        self.min_rms = min_rms
        self._initialized = False
        self._rms = torch.ones(num_components, device=device)

    @torch.no_grad()
    def update(self, advantages: torch.Tensor, alive_mask: torch.Tensor) -> None:
        """Update EMA RMS from this rollout's per-component advantages.

        Args:
            advantages: (T, B, N, K) bf16-stored — GAE advantages in symlog-reward space
                        (upcast to fp32 internally for the RMS reduction).
            alive_mask: (T, B, N) bool — which ships were alive each step.
        """
        self.update_chunks([advantages], [alive_mask])

    @torch.no_grad()
    def update_chunks(
        self,
        advantage_chunks: list[torch.Tensor],
        alive_chunks: list[torch.Tensor],
    ) -> None:
        """Update RMS from one logical rollout split across host shards.

        Args:
            advantage_chunks: Shards shaped ``(T, B, N, K)``.
            alive_chunks: Matching alive masks shaped ``(T, B, N)``.
        """
        if len(advantage_chunks) == 0 or len(advantage_chunks) != len(alive_chunks):
            raise ValueError("advantage_chunks and alive_chunks must be non-empty and aligned")
        K = advantage_chunks[0].shape[-1]
        square_sum = torch.zeros(K, dtype=torch.float32, device=advantage_chunks[0].device)
        mask_sum = torch.zeros((), dtype=torch.float32, device=advantage_chunks[0].device)
        for advantages, alive_mask in zip(advantage_chunks, alive_chunks):
            alive_k = alive_mask.float().unsqueeze(-1)  # (T, B, N, 1)
            square_sum += (advantages.float().pow(2) * alive_k).sum((0, 1, 2))
            mask_sum += alive_mask.float().sum()
        mask_sum.clamp_(min=1.0)
        # Upcast before squaring/reducing: advantages are bf16-stored, and a bf16
        # sum-of-squares over T*B*N elements would swamp small terms under the total.
        rms_k = (square_sum / mask_sum).sqrt().clamp(min=self.min_rms)  # (K,)
        rms_k = rms_k.to(self._rms.device)
        if not self._initialized:
            self._rms = rms_k
            self._initialized = True
        else:
            self._rms = (1.0 - self.alpha) * self._rms + self.alpha * rms_k

    def normalize(self, advantages: torch.Tensor) -> torch.Tensor:
        """Divide advantages by per-component EMA RMS → approximately unit RMS per component.

        Args:
            advantages: (..., K) float — advantages in symlog-reward space.

        Returns:
            (..., K) float — normalized advantages.
        """
        return advantages / (self._rms.clamp(min=self.min_rms) + 1e-8)

    def state_dict(self) -> dict:
        return {
            "rms": self._rms.cpu(),
            "initialized": self._initialized,
            "min_rms": self.min_rms,
        }

    @property
    def rms(self) -> torch.Tensor:
        """Current per-component advantage RMS vector."""
        return self._rms

    @property
    def floor_bound(self) -> torch.Tensor:
        """(K,) bool — components whose RMS is being held up by ``min_rms``.

        True for an active component means its policy-gradient share is set by
        the guard rather than by its own statistics.
        """
        return self._rms <= self.min_rms

    def load_state_dict(self, d: dict) -> None:
        self._rms = d["rms"].to(self._rms.device)
        self._initialized = d.get("initialized", True)
        # A checkpoint written under a different floor carries an RMS shaped by
        # that floor. Re-seeding costs one rollout; EMA-ing a stale floor out
        # takes ~1/alpha updates, during which the component is mis-scaled.
        if d.get("min_rms") != self.min_rms:
            self._initialized = False


class RolloutBuffer:
    """Pre-allocated GPU rollout buffer for one PPO rollout.

    Supports per-ship per-component rewards and values for the decomposed critic.
    Stores one initial GRU hidden state per rollout for recurrent re-evaluation.
    Stores T+1 observations (one extra for aux loss label computation at update time).

    Args:
        num_steps:       Rollout horizon T.
        num_envs:        Parallel environments B.
        num_ships:       Ships per environment N.
        num_components:  Value components K (len(REWARD_COMPONENT_NAMES)).
        obs_sample:      Sample YemongObservation (B, N+M, ...) — used to infer shapes/dtypes.
        gamma:           Discount factor.
        gae_lambda:      GAE lambda.
        device:          GPU device for all storage.
        num_tokens:      N+M total entity tokens (ships + fields).
    """

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        num_ships: int,
        num_components: int,
        obs_sample: YemongObservation,
        gamma: torch.Tensor,
        gae_lambda: torch.Tensor,
        device: torch.device,
        num_tokens: int | None = None,
    ) -> None:
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.num_ships = num_ships
        self.num_tokens = num_tokens if num_tokens is not None else num_ships
        self.num_components = num_components
        self.gamma = gamma.to(device=device)  # (K,)
        self.gae_lambda = gae_lambda.to(device=device)  # (K,)
        self.device = device

        T, B, N, K = num_steps, num_envs, num_ships, num_components

        # Observations — T+1 slots per key: obs[t] = obs at time t, obs[T] = final obs
        # Stored at reduced precision (bf16 floats / uint8 indices) except positions;
        # the feature transforms upcast every channel to fp32 on read. See
        # _obs_storage_dtype for the per-channel policy.
        self.obs: dict = {
            key: torch.zeros(
                (T + 1, B, *val.shape[1:]), device=device, dtype=_obs_storage_dtype(key, val.dtype)
            )
            for key, val in obs_sample.items()
        }
        # Bullet channels live on their own (B, N*K, ...) axis. Allocated only when
        # the policy reads them — this is the largest single tensor the change adds.
        self.bullet_obs: dict | None = (
            None
            if obs_sample.bullets is None
            else {
                key: torch.zeros(
                    (T + 1, B, *val.shape[1:]),
                    device=device,
                    dtype=_obs_storage_dtype(key, val.dtype),
                )
                for key, val in obs_sample.bullets.items()
            }
        )

        self.actions = torch.zeros((T, B, N, 3), device=device, dtype=torch.int32)
        # logprobs stay fp32: PPO's ratio exp(new - old) is precision-sensitive.
        self.logprobs = torch.zeros((T, B, N), device=device, dtype=torch.float32)
        # Per-component float arrays are read-once leaf data → bf16 (see _STORAGE_FLOAT).
        self.rewards = torch.zeros((T, B, N, K), device=device, dtype=_STORAGE_FLOAT)
        self.values = torch.zeros((T, B, N, K), device=device, dtype=_STORAGE_FLOAT)
        self.alive_mask = torch.zeros((T, B, N), device=device, dtype=torch.bool)

        self.advantages = torch.zeros((T, B, N, K), device=device, dtype=_STORAGE_FLOAT)
        self.returns = torch.zeros((T, B, N, K), device=device, dtype=_STORAGE_FLOAT)

        # Lambda-aggregated advantages/returns — filled once per update by
        # PPOTrainer._precompute_lambda_aggregates before the epoch loop (they
        # depend only on rollout data, not the policy).
        self.adv_agg = torch.zeros((T, B, N), device=device, dtype=torch.float32)
        self.ret_agg = torch.zeros((T, B, N), device=device, dtype=torch.float32)
        # Mean squared aggregated advantage over actor tokens — set once per
        # update by _precompute_lambda_aggregates. Global (whole-buffer) so the
        # advantage normalization is independent of minibatch/micro-batch splits.
        self.adv_rms = torch.ones((), device=device, dtype=torch.float32)
        # Next-state prediction labels (T, B, N, pred_dim) — set once per update
        # by PPOTrainer._precompute_ns_labels; None for aux scales or when the
        # aux losses are disabled.
        self.ns_labels: torch.Tensor | None = None

        self.actor_masks = torch.ones((T, B, N), device=device, dtype=torch.bool)
        self.expert_probs = torch.zeros((T, B, N, 12), device=device, dtype=_STORAGE_FLOAT)

        # Episode termination mask: done | truncated — used to exclude terminal transitions
        # from the aux next-state prediction loss.
        self.terminated = torch.zeros((T, B), device=device, dtype=torch.bool)

        # Initial GRU hidden state at the start of this rollout
        self.initial_hidden: torch.Tensor | None = None

        self.ptr = 0

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear the write pointer (tensors are overwritten, not zeroed)."""
        self.ptr = 0
        self.initial_hidden = None
        self.expert_probs.zero_()  # only filled for scripted-group envs; rest must be zero
        self.terminated.zero_()
        # obs[T] slot is overwritten by store_final_obs() — no need to zero it

    def store_initial_hidden(self, hidden: torch.Tensor) -> None:
        """Store the GRU hidden state at rollout start.

        Args:
            hidden: (n_layers, B*num_tokens, H) float32.
        """
        self.initial_hidden = hidden.clone()

    def add(
        self,
        obs: YemongObservation,
        action: torch.Tensor,
        logprob: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor,
        alive: torch.Tensor,
        actor_mask: torch.Tensor | None = None,
        expert_probs: torch.Tensor | None = None,
        terminated: torch.Tensor | None = None,
    ) -> None:
        """Store one step.

        Args:
            obs:          YemongObservation with (B, N+M, ...) tensors.
            action:       (B, N, 3) int.
            logprob:      (B, N) float.
            reward:       (B, N, K) float — raw per-component per-ship rewards.
            value:        (B, N, K) float — critic expected values (symlog-reward space).
            alive:        (B, N) bool.
            actor_mask:   (B, N) bool — True for ships that should contribute to actor loss.
                          Defaults to all-True (pure self-play).
            expert_probs: (B, N, 12) float — scripted-agent marginal probs for BC loss.
                          Zero for envs without a scripted opponent.
            terminated:   (B,) bool — True when the episode ended (done | truncated).
                          Cuts the GAE trace and masks the aux loss at boundaries.
        """
        if self.ptr >= self.num_steps:
            raise IndexError("Buffer is full — call reset() before reuse.")

        t = self.ptr
        for key, val in obs.items():
            self.obs[key][t].copy_(val)
        if self.bullet_obs is not None and obs.bullets is not None:
            for key, val in obs.bullets.items():
                self.bullet_obs[key][t].copy_(val)

        self.actions[t] = action.int()
        self.logprobs[t] = logprob
        self.rewards[t] = symlog(reward)  # symlog #1: compress raw reward scale
        self.values[t] = value
        self.alive_mask[t] = alive
        self.actor_masks[t] = actor_mask if actor_mask is not None else torch.ones_like(alive)
        if expert_probs is not None:
            self.expert_probs[t] = expert_probs
        if terminated is not None:
            self.terminated[t] = terminated

        self.ptr += 1

    def store_final_obs(self, obs: YemongObservation) -> None:
        """Store the observation at the end of the rollout (the T+1-th obs slot).

        Called once after the rollout loop completes. This final obs enables
        computing aux next-state prediction labels at update time without a
        separate pre-computed target buffer.
        """
        T = self.num_steps
        for key, val in obs.items():
            self.obs[key][T].copy_(val)
        if self.bullet_obs is not None and obs.bullets is not None:
            for key, val in obs.bullets.items():
                self.bullet_obs[key][T].copy_(val)

    # ------------------------------------------------------------------
    # GAE computation
    # ------------------------------------------------------------------

    def compute_gae(self, next_value: torch.Tensor, next_done: torch.Tensor) -> None:
        """Compute GAE advantages and returns in-place over K components.

        All tensor ops broadcast over the K dimension automatically — the
        loop body is identical to the scalar case.

        Episode boundaries use ``terminated`` (done | truncated), not ``dones``
        (physics termination only). The wrapper auto-resets a finished
        environment *before* returning its observation, so ``values[t+1]`` after
        a truncation is the value of a freshly spawned episode. Bootstrapping a
        truncated episode's return off it would carry value across the boundary
        — and with the win component at gamma 0.999 that leak runs the length of
        the trace. Cutting the trace at truncation instead is mildly conservative
        (a time-limited episode is treated as if it genuinely ended) but it never
        mixes two episodes. Recovering the exact bootstrap would mean storing the
        pre-reset final observation, which is not worth it at the ~2% of episodes
        that reach the horizon.

        Args:
            next_value: (B, N, K) float — critic expected values at step T+1,
                        in symlog-reward space (symexp of expected bin).
            next_done:  (B,) float — whether step T+1 ended an episode
                        (done | truncated).
        """
        with torch.no_grad():
            # Accumulate in fp32 even though rewards/values are stored bf16: gamma/lam
            # are fp32 so every product promotes, and lastgaelam is fp32-seeded, so the
            # recursion runs at full precision. Only the stored advantage is downcast.
            lastgaelam = torch.zeros_like(next_value, dtype=torch.float32)  # (B, N, K)
            gamma = self.gamma.view(1, 1, -1)  # (1, 1, K) — broadcasts over (B, N, K)
            lam = self.gae_lambda.view(1, 1, -1)  # (1, 1, K)

            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    non_terminal = 1.0 - next_done.view(-1, 1, 1)  # (B, 1, 1)
                    next_val = next_value  # (B, N, K)
                else:
                    non_terminal = 1.0 - self.terminated[t].float().view(-1, 1, 1)  # (B, 1, 1)
                    next_val = self.values[t + 1]  # (B, N, K)

                delta = self.rewards[t] + gamma * next_val * non_terminal - self.values[t]
                lastgaelam = delta + gamma * lam * non_terminal * lastgaelam
                self.advantages[t] = lastgaelam.to(self.advantages.dtype)  # store bf16

            # returns = advantages + values, summed in fp32 then stored bf16.
            self.returns = (self.advantages.float() + self.values.float()).to(self.returns.dtype)

    # ------------------------------------------------------------------
    # Minibatch iteration for PPO update
    # ------------------------------------------------------------------

    def get_minibatch_iterator(
        self, num_minibatches: int, microbatch_tokens: int | None = None
    ) -> Generator[list[MicroBatch]]:
        """Yield minibatches of environments for PPO update epochs.

        Shuffles environments and slices them into num_minibatches chunks.
        Each chunk yields the full (T+1)-step observation sequence plus T-step
        non-observation data for those environments.

        Each minibatch is yielded as a list of ``MicroBatch`` objects: when
        microbatch_tokens is set and the minibatch exceeds it (tokens =
        envs × T × num_tokens), the minibatch's envs are split near-evenly into
        the fewest micro-batches that each fit the budget. Callers accumulate
        gradients over the list before stepping. With microbatch_tokens=None
        the list always has exactly one entry (the whole minibatch).

        The extra T+1-th obs step enables computing aux next-state prediction
        labels at update time: coordinator.compute_labels(target(obs[t]), target(obs[t+1])).

        Yields:
            List of named micro-batches, each containing:
                mb_obs:          YemongObservation (T+1, B_mb, N+M, ...)
                mb_actions:      (T, B_mb, N, 3) int32
                mb_logprobs:     (T, B_mb, N) float32
                mb_advantages:   (T, B_mb, N, K) float32
                mb_returns:      (T, B_mb, N, K) float32
                mb_alive:        (T, B_mb, N) bool
                mb_hidden:       (n_layers, B_mb*num_tokens, H) float32
                mb_actor_mask:   (T, B_mb, N) bool
                mb_expert_probs: (T, B_mb, N, 12) float32
                mb_terminated:   (T, B_mb) bool
                mb_adv_agg:      (T, B_mb, N) float32 — precomputed lambda-aggregated advantages
                mb_ret_agg:      (T, B_mb, N) float32 — precomputed lambda-aggregated returns
                mb_ns_labels:    (T, B_mb, N, pred_dim) float32 or None — precomputed aux labels
        """
        assert self.initial_hidden is not None, "Call store_initial_hidden() before iterating."

        envs_per_batch = self.num_envs // num_minibatches
        env_order = torch.randperm(self.num_envs)
        D = self.initial_hidden.shape[-1]

        tokens_per_env = self.num_steps * self.num_tokens
        n_micro = 1
        if microbatch_tokens is not None:
            n_micro = -(-envs_per_batch * tokens_per_env // microbatch_tokens)
            n_micro = min(max(n_micro, 1), envs_per_batch)

        n_layers = self.initial_hidden.shape[0]
        hidden_full = self.initial_hidden.reshape(n_layers, self.num_envs, self.num_ships, D)

        for start in range(0, self.num_envs, envs_per_batch):
            end = start + envs_per_batch
            chunks = []
            for idx in torch.tensor_split(env_order[start:end], n_micro):
                # T+1 obs for this micro-batch
                mb_obs = YemongObservation(
                    data={k: v[:, idx] for k, v in self.obs.items()},
                    bullets=(
                        None
                        if self.bullet_obs is None
                        else {k: v[:, idx] for k, v in self.bullet_obs.items()}
                    ),
                )

                # Reconstruct initial hidden: (n_layers, B_mb*N, H) — ships only
                mb_hidden = hidden_full[:, idx, :, :].reshape(
                    n_layers, len(idx) * self.num_ships, D
                )

                chunks.append(
                    MicroBatch(
                        obs=mb_obs,
                        actions=self.actions[:, idx],
                        old_logprobs=self.logprobs[:, idx],
                        advantages=self.advantages[:, idx],
                        returns=self.returns[:, idx],
                        alive=self.alive_mask[:, idx],
                        hidden=mb_hidden.contiguous(),
                        actor_mask=self.actor_masks[:, idx],
                        expert_probs=self.expert_probs[:, idx],
                        terminated=self.terminated[:, idx],
                        adv_agg=self.adv_agg[:, idx],
                        ret_agg=self.ret_agg[:, idx],
                        ns_labels=self.ns_labels[:, idx] if self.ns_labels is not None else None,
                    )
                )
            yield chunks


class StoredRollout:
    """One completed rollout shard stored in pageable CPU memory.

    Raw experience is copied here after GPU GAE, allowing the fixed-width device
    buffer to be reused for another shard collected under the same policy. Derived
    PPO tensors are filled later after logical-batch scaler statistics are known.
    """

    def __init__(self, source: RolloutBuffer) -> None:
        if source.initial_hidden is None:
            raise ValueError("source rollout has no initial hidden state")
        self.num_steps = source.num_steps
        self.num_envs = source.num_envs
        self.num_ships = source.num_ships
        self.num_tokens = source.num_tokens
        self.num_components = source.num_components

        self.obs = {
            key: value.detach().to(device="cpu", copy=True) for key, value in source.obs.items()
        }
        self.bullet_obs = (
            None
            if source.bullet_obs is None
            else {
                key: value.detach().to(device="cpu", copy=True)
                for key, value in source.bullet_obs.items()
            }
        )
        self.actions = source.actions.detach().to(device="cpu", copy=True)
        self.logprobs = source.logprobs.detach().to(device="cpu", copy=True)
        self.advantages = source.advantages.detach().to(device="cpu", copy=True)
        self.returns = source.returns.detach().to(device="cpu", copy=True)
        self.alive_mask = source.alive_mask.detach().to(device="cpu", copy=True)
        self.actor_masks = source.actor_masks.detach().to(device="cpu", copy=True)
        self.expert_probs = source.expert_probs.detach().to(device="cpu", copy=True)
        self.terminated = source.terminated.detach().to(device="cpu", copy=True)
        self.initial_hidden = source.initial_hidden.detach().to(device="cpu", copy=True)

        self.adv_agg: torch.Tensor | None = None
        self.ret_agg: torch.Tensor | None = None
        self.ns_labels = (
            source.ns_labels.detach().to(device="cpu", copy=True)
            if source.ns_labels is not None
            else None
        )

    def restore_aggregate_inputs(self, destination: RolloutBuffer) -> None:
        """Restore only tensors required for lambda aggregation.

        Args:
            destination: Fixed-width rollout buffer with matching dimensions.
        """
        self._validate_destination(destination)
        destination.obs[ObsKey.TEAM_ID].copy_(self.obs[ObsKey.TEAM_ID])
        destination.advantages.copy_(self.advantages)
        destination.returns.copy_(self.returns)
        destination.alive_mask.copy_(self.alive_mask)
        destination.actor_masks.copy_(self.actor_masks)

    def capture_aggregates(self, source: RolloutBuffer) -> None:
        """Copy device-computed lambda aggregates into host storage.

        Args:
            source: Fixed-width rollout buffer containing derived tensors.
        """
        self._validate_destination(source)
        self.adv_agg = source.adv_agg.detach().to(device="cpu", copy=True)
        self.ret_agg = source.ret_agg.detach().to(device="cpu", copy=True)

    def get_minibatch_iterator(
        self,
        num_minibatches: int,
        microbatch_tokens: int | None = None,
    ) -> Generator[list[MicroBatch]]:
        """Yield CPU micro-batches for one PPO epoch.

        Args:
            num_minibatches: Number of optimizer minibatches per epoch.
            microbatch_tokens: Maximum entity tokens per backward pass.

        Yields:
            Lists of CPU micro-batches whose gradients form one optimizer step.
        """
        if self.adv_agg is None or self.ret_agg is None:
            raise RuntimeError("capture_aggregates() must run before minibatch iteration")

        del microbatch_tokens  # device staging performs the split after one host gather
        envs_per_batch = self.num_envs // num_minibatches
        env_order = torch.randperm(self.num_envs)

        n_layers = self.initial_hidden.shape[0]
        hidden_width = self.initial_hidden.shape[-1]
        hidden_full = self.initial_hidden.reshape(
            n_layers,
            self.num_envs,
            self.num_ships,
            hidden_width,
        )

        for start in range(0, self.num_envs, envs_per_batch):
            indices = env_order[start : start + envs_per_batch]
            obs = YemongObservation(
                data={key: value[:, indices] for key, value in self.obs.items()},
                bullets=(
                    None
                    if self.bullet_obs is None
                    else {k: v[:, indices] for k, v in self.bullet_obs.items()}
                ),
            )
            hidden = hidden_full[:, indices].reshape(
                n_layers,
                len(indices) * self.num_ships,
                hidden_width,
            )
            yield [
                MicroBatch(
                    obs=obs,
                    actions=self.actions[:, indices],
                    old_logprobs=self.logprobs[:, indices],
                    advantages=self.advantages[:, indices],
                    returns=self.returns[:, indices],
                    alive=self.alive_mask[:, indices],
                    hidden=hidden.contiguous(),
                    actor_mask=self.actor_masks[:, indices],
                    expert_probs=self.expert_probs[:, indices],
                    terminated=self.terminated[:, indices],
                    adv_agg=self.adv_agg[:, indices],
                    ret_agg=self.ret_agg[:, indices],
                    ns_labels=(self.ns_labels[:, indices] if self.ns_labels is not None else None),
                )
            ]

    def _validate_destination(self, destination: RolloutBuffer) -> None:
        expected = (
            self.num_steps,
            self.num_envs,
            self.num_ships,
            self.num_tokens,
            self.num_components,
        )
        actual = (
            destination.num_steps,
            destination.num_envs,
            destination.num_ships,
            destination.num_tokens,
            destination.num_components,
        )
        if actual != expected:
            raise ValueError(f"rollout buffer shape mismatch: expected {expected}, got {actual}")


class LogicalRolloutBuffer:
    """Host-backed logical PPO batch composed of fixed-width rollout shards."""

    def __init__(self, shards: list[StoredRollout], adv_rms: torch.Tensor) -> None:
        if len(shards) == 0:
            raise ValueError("shards must contain at least one stored rollout")
        first = shards[0]
        if any(shard.num_envs != first.num_envs for shard in shards):
            raise ValueError("all logical rollout shards must have the same width")
        self.shards = shards
        self.num_steps = first.num_steps
        self.num_envs = sum(shard.num_envs for shard in shards)
        self.num_ships = first.num_ships
        self.num_tokens = first.num_tokens
        self.num_components = first.num_components
        self.adv_rms = adv_rms

    def get_minibatch_iterator(
        self,
        num_minibatches: int,
        microbatch_tokens: int | None = None,
    ) -> Generator[list[MicroBatch]]:
        """Yield aligned minibatches from every host shard.

        Each shard is shuffled independently. Corresponding shard minibatches are
        combined into one optimizer step, producing the same loss normalization as
        a single larger environment batch.
        """
        iterators = [
            shard.get_minibatch_iterator(num_minibatches, microbatch_tokens)
            for shard in self.shards
        ]
        for shard_batches in zip(*iterators, strict=True):
            yield [chunk for shard_batch in shard_batches for chunk in shard_batch]
