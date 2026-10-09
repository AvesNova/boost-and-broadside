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

from boost_and_broadside.constants import NUM_EXPERT_MARGINALS, NUM_OUTCOME_CLASSES
from boost_and_broadside.env.observation import (
    BulletObsKey,
    ObsKey,
    YemongObservation,
)
from boost_and_broadside.train.rl.critic import outcome_targets


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
    decision_committed: torch.Tensor
    expert_probs: torch.Tensor
    terminated: torch.Tensor
    transition_contiguous: torch.Tensor
    adv_agg: torch.Tensor
    ret_agg: torch.Tensor
    ns_labels: torch.Tensor | None
    # (T, B, 4) categorical lambda-return of the match outcome per environment,
    # in the observer's frame: win, tie, loss, unresolved (see train/rl/critic.py).
    outcome_targets: torch.Tensor
    # (T, B, N) bool, the ship is on the observer's team.
    observer_side: torch.Tensor
    # (T, B, 2C) privileged ally/enemy hex density per step, or None for a scale
    # that never trains the global density head. Deliberately outside ``obs``,
    # like ``privileged_means``: no policy input path can reach it by key.
    density_targets: torch.Tensor | None = None
    # (T, B, N, SHOT_LABEL_DIM) completed shot examples filed under their launch
    # decision, validity first (train/rl/shot_codes.py), or None without shot
    # heads. Outside ``obs`` for ``privileged_means``' reason: it is the future.
    shot_labels: torch.Tensor | None = None

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
            decision_committed=self.decision_committed.pin_memory(),
            expert_probs=self.expert_probs.pin_memory(),
            terminated=self.terminated.pin_memory(),
            outcome_targets=self.outcome_targets.pin_memory(),
            observer_side=self.observer_side.pin_memory(),
            transition_contiguous=self.transition_contiguous.pin_memory(),
            adv_agg=self.adv_agg.pin_memory(),
            ret_agg=self.ret_agg.pin_memory(),
            ns_labels=self.ns_labels.pin_memory() if self.ns_labels is not None else None,
            density_targets=(
                self.density_targets.pin_memory() if self.density_targets is not None else None
            ),
            shot_labels=self.shot_labels.pin_memory() if self.shot_labels is not None else None,
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
            decision_committed=self.decision_committed.to(device=device, non_blocking=non_blocking),
            expert_probs=self.expert_probs.to(device=device, non_blocking=non_blocking),
            terminated=self.terminated.to(device=device, non_blocking=non_blocking),
            outcome_targets=self.outcome_targets.to(device=device, non_blocking=non_blocking),
            observer_side=self.observer_side.to(device=device, non_blocking=non_blocking),
            transition_contiguous=self.transition_contiguous.to(
                device=device, non_blocking=non_blocking
            ),
            adv_agg=self.adv_agg.to(device=device, non_blocking=non_blocking),
            ret_agg=self.ret_agg.to(device=device, non_blocking=non_blocking),
            ns_labels=(
                self.ns_labels.to(device=device, non_blocking=non_blocking)
                if self.ns_labels is not None
                else None
            ),
            density_targets=(
                self.density_targets.to(device=device, non_blocking=non_blocking)
                if self.density_targets is not None
                else None
            ),
            shot_labels=(
                self.shot_labels.to(device=device, non_blocking=non_blocking)
                if self.shot_labels is not None
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

    def slice_envs(self, start: int, end: int) -> "MicroBatch":
        """Slice a contiguous environment range from a staged shard minibatch.

        Recurrent tokens per environment are read off the hidden tensor rather
        than passed in, so the split follows however the policy sized its state
        (ships plus the global token) and cannot assume it is the ship count.

        Args:
            start: Inclusive environment offset.
            end: Exclusive environment offset.

        Returns:
            A view-only micro-batch over ``[start:end]``.
        """
        batch_envs = self.actions.shape[1]
        n_layers, batch_tokens, hidden_width = self.hidden.shape
        num_recurrent = batch_tokens // batch_envs  # N+G
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
            decision_committed=self.decision_committed[:, start:end],
            expert_probs=self.expert_probs[:, start:end],
            terminated=self.terminated[:, start:end],
            outcome_targets=self.outcome_targets[:, start:end],
            observer_side=self.observer_side[:, start:end],
            transition_contiguous=self.transition_contiguous[:, start:end],
            adv_agg=self.adv_agg[:, start:end],
            ret_agg=self.ret_agg[:, start:end],
            ns_labels=self.ns_labels[:, start:end] if self.ns_labels is not None else None,
            density_targets=(
                self.density_targets[:, start:end] if self.density_targets is not None else None
            ),
            shot_labels=self.shot_labels[:, start:end] if self.shot_labels is not None else None,
        )

    def split_envs(self, num_chunks: int) -> list["MicroBatch"]:
        """Split a staged shard minibatch into near-even contiguous views."""
        batch_envs = self.actions.shape[1]
        base, remainder = divmod(batch_envs, num_chunks)
        chunks = []
        start = 0
        for index in range(num_chunks):
            width = base + (1 if index < remainder else 0)
            chunks.append(self.slice_envs(start, start + width))
            start += width
        return chunks


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
#      data upcasts first (see compute_gae and
#      PPOTrainer._precompute_lambda_aggregates). bf16's ~0.4% resolution
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
    # The other ten physical ship channels, for a second reason: with a physical
    # belief plane these *are* the believed state the next-state label steps
    # from, so their storage precision lands directly in the label rather than
    # only in the encoder's input. bf16's ~0.4% resolution puts about 1.2 px/s of
    # noise on a 300 px/s velocity, against a calibrated velocity delta scale of
    # 4.0 px/s -- a third of the label's own spread, which is not a rounding.
    ObsKey.VEL: torch.float32,
    ObsKey.ATT: torch.float32,
    ObsKey.ANG_VEL: torch.float32,
    ObsKey.SHIELD_DELAY: torch.float32,
    ObsKey.HEALTH: torch.float32,
    ObsKey.POWER: torch.float32,
    ObsKey.COOLDOWN: torch.float32,
    ObsKey.LOCAL_LOG_INDEX: torch.float32,
    # The belief's spreads, from which the update rebuilds the same code the
    # rollout read, as the next-state head's baseline and the encoder's input.
    ObsKey.BELIEF_UNCERTAINTY: torch.float32,
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
        prediction_target_dim: int = 0,
        prediction_dim: int = 0,
        uncertainty_dim: int = 0,
        density_dim: int = 0,
        store_expert_probs: bool = True,
        shot_label_dim: int = 0,
        shot_label_delay: int = 0,
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
        # The belief tracker adds BELIEF_UNCERTAINTY to every composed
        # observation, while the raw environment view this sample comes from has
        # none. The update replays stored observations through the same encoder
        # that read them during the rollout, so storage has to cover the channel
        # either way or the two would see different inputs.
        #
        # A caller that will never compose a belief into an observation passes
        # zero and gets no storage.
        sampled_obs = dict(obs_sample.items())
        if ObsKey.BELIEF_UNCERTAINTY not in sampled_obs and uncertainty_dim:
            # Shaped from ``pos`` rather than from ``team_id`` -- a channel the
            # compact test fixtures omit.
            tokens = obs_sample.pos
            sampled_obs[ObsKey.BELIEF_UNCERTAINTY] = torch.zeros(
                (*tokens.shape[:2], uncertainty_dim),
                device=tokens.device,
                dtype=torch.float32,
            )
        self.obs: dict = {
            key: torch.zeros(
                (T + 1, B, *val.shape[1:]), device=device, dtype=_obs_storage_dtype(key, val.dtype)
            )
            for key, val in sampled_obs.items()
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
        # Raw per-level rewards, values, advantages and returns, in win units.
        # fp32 rather than bf16: they are no longer compressed, and at a value of
        # 50 bf16's resolution is 0.25 -- larger than a typical per-step reward.
        self.rewards = torch.zeros((T, B, N, K), device=device, dtype=torch.float32)
        self.values = torch.zeros((T, B, N, K), device=device, dtype=torch.float32)
        self.alive_mask = torch.zeros((T, B, N), device=device, dtype=torch.bool)

        self.advantages = torch.zeros((T, B, N, K), device=device, dtype=torch.float32)
        self.returns = torch.zeros((T, B, N, K), device=device, dtype=torch.float32)
        # The outcome critic's rollout-time class probabilities and the signed
        # result per transition, both per environment in the observer's frame,
        # and the categorical targets built from them after the rollout.
        self.outcome_probs = torch.zeros(
            (T, B, NUM_OUTCOME_CLASSES), device=device, dtype=torch.float32
        )
        self.outcome_result = torch.zeros((T, B), device=device, dtype=torch.float32)
        self.outcome_targets = torch.zeros(
            (T, B, NUM_OUTCOME_CLASSES), device=device, dtype=torch.float32
        )
        # Which ships are on the observer's (Team 0's) side, from authoritative
        # team ids: the observation's team channel zeroes hidden enemies. The
        # outcome critic's estimate is the observer's, so this signs it per ship.
        self.observer_side = torch.ones((T, B, N), device=device, dtype=torch.bool)

        # Lambda-aggregated advantages/returns — filled once per update by
        # PPOTrainer._precompute_lambda_aggregates before the epoch loop (they
        # depend only on rollout data, not the policy).
        self.adv_agg = torch.zeros((T, B, N), device=device, dtype=torch.float32)
        self.ret_agg = torch.zeros((T, B, N), device=device, dtype=torch.float32)
        # The return normaliser's scale the summed advantage is divided by --
        # set once per update, so the normalisation is independent of
        # minibatch/micro-batch splits.
        self.return_scale = torch.ones((), device=device, dtype=torch.float32)
        # Next-state targets (T, B, N, 11): the true physical state one decision
        # later, whose exact code the head is trained toward. Set once per update
        # by PPOTrainer._precompute_ns_labels; None for aux scales or when the
        # next-state loss is disabled.
        self.ns_labels: torch.Tensor | None = None
        # Authoritative physical ship state, ``(T+1, B, N, 11)``. Auxiliary
        # supervision only, and deliberately stored outside ``obs`` so no
        # actor/critic path can consume hidden enemy truth by key lookup.
        #
        # fp32 rather than the buffer's usual bf16: these are the far end of
        # every next-state target, and a coordinate spends its bits on magnitude,
        # so bf16 would quantise a 65536 px world into 128 px steps, against a
        # finest position cell of 3.3 px. The whole tensor
        # is eleven channels wide, which is cheap enough that there is nothing to
        # trade off.
        self.privileged_means: torch.Tensor | None = (
            torch.zeros((T + 1, B, N, prediction_target_dim), device=device, dtype=torch.float32)
            if prediction_target_dim > 0
            else None
        )
        # Privileged ally/enemy density over the hex grid, ``(T, B, 2C)``, for
        # the global density head. Only the step's own field is a target -- there
        # is nothing to predict one step ahead -- so this needs no T+1 slot.
        #
        # ``_STORAGE_FLOAT`` unlike ``privileged_means``: a cell holds a
        # ``log1p``-compressed count divided by the presence scale, so it is an
        # O(1) number whose bf16 rounding is far below the head's own error. The
        # tensor is the widest thing per step the buffer stores after the
        # observation, and fp32 would double that for no accuracy that matters.
        self.density_targets: torch.Tensor | None = (
            torch.zeros((T, B, density_dim), device=device, dtype=_STORAGE_FLOAT)
            if density_dim > 0
            else None
        )
        # Completed shot examples, ``(T, B, N, shot_label_dim)``, validity first.
        # The environment emits an example exactly ``shot_label_delay`` decisions
        # after its launch, so the row it belongs to is a fixed offset back and
        # filing it is a slice, never a scatter. The last ``shot_label_delay``
        # rows of every rollout complete only in the next one and stay invalid.
        self.shot_label_delay = shot_label_delay
        self.shot_labels: torch.Tensor | None = (
            torch.zeros((T, B, N, shot_label_dim), device=device, dtype=torch.float32)
            if shot_label_dim > 0
            else None
        )
        # The behaviour policy's next-state forecasts, decoded to moments
        # (T, B, N, 25), for the belief diagnostics.
        self.rollout_predictions: torch.Tensor | None = (
            torch.zeros((T, B, N, prediction_dim), device=device) if prediction_dim > 0 else None
        )
        self.belief_diagnostics: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        self.actor_masks = torch.ones((T, B, N), device=device, dtype=torch.bool)
        self.decision_committed = torch.ones((T, B, N), device=device, dtype=torch.bool)
        # A zero-width tensor keeps MicroBatch structurally uniform without
        # reserving rollout memory in schedules that never enable BC.
        expert_width = NUM_EXPERT_MARGINALS if store_expert_probs else 0
        self.expert_probs = torch.zeros(
            (T, B, N, expert_width), device=device, dtype=_STORAGE_FLOAT
        )

        # Episode termination mask: done | truncated — used to exclude terminal transitions
        # from the aux next-state prediction loss.
        self.terminated = torch.zeros((T, B), device=device, dtype=torch.bool)
        # Per-ship physical continuity. False excludes a death->respawn teleport
        # from auxiliary dynamics targets without ending the strategic episode.
        self.transition_contiguous = torch.ones((T, B, N), device=device, dtype=torch.bool)

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
        self.transition_contiguous.fill_(True)
        if self.shot_labels is not None:
            self.shot_labels[..., 0].zero_()
        self.belief_diagnostics = {}
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
        decision_committed: torch.Tensor | None = None,
        expert_probs: torch.Tensor | None = None,
        terminated: torch.Tensor | None = None,
        transition_contiguous: torch.Tensor | None = None,
        privileged_means: torch.Tensor | None = None,
        scaled_predictions: torch.Tensor | None = None,
        density_target: torch.Tensor | None = None,
        outcome_probs: torch.Tensor | None = None,
        outcome_result: torch.Tensor | None = None,
        observer_side: torch.Tensor | None = None,
    ) -> None:
        """Store one step.

        Args:
            obs:          YemongObservation with (B, N+M, ...) tensors.
            action:       (B, N, 3) int.
            logprob:      (B, N) float.
            reward:       (B, N, K) float — raw per-level per-ship rewards, win units.
            value:        (B, N, K) float — critic expected returns, same units.
            alive:        (B, N) bool.
            actor_mask:   (B, N) bool — True for ships that should contribute to actor loss.
                          Defaults to all-True (pure self-play).
            decision_committed: (B, N) bool — True where the selected action entered
                          the continuing actuator queue. Used for causal PPO credit,
                          not BC supervision or entropy regularization.
            expert_probs: (B, N, 10) float — scripted-agent marginal probs for BC loss.
                          Zero for envs without a scripted opponent.
            terminated:   (B,) bool — True when the episode ended (done | truncated).
                          Cuts the GAE trace and masks the aux loss at boundaries.
            transition_contiguous: (B, N) bool — False where the physical next
                          state is a respawn teleport. Does not cut GAE/recurrent state.
            outcome_probs: (B, 4) outcome-class probabilities, observer's frame.
            outcome_result: (B,) signed match result on this transition,
                          observer's frame; zero until the terminal one.
            observer_side: (B, N) bool — the ship is on the observer's team,
                          from authoritative team ids.
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
        self.rewards[t] = reward
        self.values[t] = value
        self.alive_mask[t] = alive
        self.actor_masks[t] = actor_mask if actor_mask is not None else torch.ones_like(alive)
        self.decision_committed[t] = (
            decision_committed if decision_committed is not None else torch.ones_like(alive)
        )
        if expert_probs is not None:
            if self.expert_probs.shape[-1] == 0:
                raise ValueError("expert probabilities supplied to a buffer with BC storage off")
            self.expert_probs[t] = expert_probs
        if terminated is not None:
            self.terminated[t] = terminated
        if transition_contiguous is not None:
            self.transition_contiguous[t] = transition_contiguous
        if outcome_probs is not None:
            self.outcome_probs[t] = outcome_probs
        if outcome_result is not None:
            self.outcome_result[t] = outcome_result
        if observer_side is not None:
            self.observer_side[t] = observer_side
        if self.privileged_means is not None:
            if privileged_means is None:
                raise ValueError("primary rollout requires privileged next-state targets")
            self.privileged_means[t].copy_(privileged_means)
        if self.rollout_predictions is not None:
            if scaled_predictions is None:
                raise ValueError("primary rollout requires rollout-time predictions")
            self.rollout_predictions[t].copy_(scaled_predictions)
        if self.density_targets is not None:
            if density_target is None:
                raise ValueError("the global density head requires a density target")
            self.density_targets[t].copy_(density_target)

        self.ptr += 1

    def add_shot_examples(self, example: torch.Tensor, valid: torch.Tensor) -> None:
        """File the shot examples the step just added completed.

        Call right after :meth:`add`. An example completing on this step belongs
        to the decision ``shot_label_delay`` steps back; one whose launch fell in
        the previous rollout has no row here and is dropped.

        Args:
            example: ``(B, N, EXAMPLE_DIM)`` the environment's per-ship example.
            valid:   ``(B, N)`` bool, whether the ship completed one this step.
        """
        if self.shot_labels is None:
            return
        row = self.ptr - 1 - self.shot_label_delay
        if row < 0:
            return
        self.shot_labels[row, ..., 0] = valid.float()
        self.shot_labels[row, ..., 1:] = example

    def store_final_obs(
        self,
        obs: YemongObservation,
        privileged_means: torch.Tensor | None = None,
    ) -> None:
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
        if self.privileged_means is not None:
            if privileged_means is None:
                raise ValueError("primary rollout requires final privileged targets")
            self.privileged_means[T].copy_(privileged_means)

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
            next_value: (B, N, K) float — critic expected returns at step T+1.
            next_done:  (B,) float — whether step T+1 ended an episode
                        (done | truncated).
        """
        with torch.no_grad():
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
                self.advantages[t] = lastgaelam

            self.returns = self.advantages + self.values

    def compute_outcome_targets(
        self, next_probs: torch.Tensor, gamma: float, gae_lambda: float
    ) -> None:
        """Build the outcome critic's categorical lambda-return targets.

        Args:
            next_probs: (B, 4) outcome-class probabilities at step T+1.
            gamma, gae_lambda: The outcome level's discount and GAE lambda.
        """
        self.outcome_targets = outcome_targets(
            self.outcome_probs,
            self.outcome_result,
            self.terminated,
            next_probs,
            self.terminated[-1],
            gamma,
            gae_lambda,
        )

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
        labels at update time: see ``PPOTrainer._precompute_ns_labels``.

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
                mb_expert_probs: (T, B_mb, N, 10) float32
                mb_terminated:   (T, B_mb) bool
                mb_adv_agg:      (T, B_mb, N) float32 — precomputed lambda-aggregated advantages
                mb_ret_agg:      (T, B_mb, N) float32 — precomputed lambda-aggregated returns
                mb_ns_labels:    (T, B_mb, N, 11) float32 or None — next-state truth
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
        # Read off the stored state, not assumed to be the ship count: the
        # policy's recurrent set is ships plus the global token.
        num_recurrent = self.initial_hidden.shape[1] // self.num_envs  # N+G
        hidden_full = self.initial_hidden.reshape(n_layers, self.num_envs, num_recurrent, D)

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

                # Reconstruct initial hidden: (n_layers, B_mb*(N+G), H)
                mb_hidden = hidden_full[:, idx, :, :].reshape(n_layers, len(idx) * num_recurrent, D)

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
                        decision_committed=self.decision_committed[:, idx],
                        expert_probs=self.expert_probs[:, idx],
                        terminated=self.terminated[:, idx],
                        outcome_targets=self.outcome_targets[:, idx],
                        observer_side=self.observer_side[:, idx],
                        transition_contiguous=self.transition_contiguous[:, idx],
                        adv_agg=self.adv_agg[:, idx],
                        ret_agg=self.ret_agg[:, idx],
                        ns_labels=self.ns_labels[:, idx] if self.ns_labels is not None else None,
                        density_targets=(
                            self.density_targets[:, idx]
                            if self.density_targets is not None
                            else None
                        ),
                        shot_labels=(
                            self.shot_labels[:, idx] if self.shot_labels is not None else None
                        ),
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
        self.decision_committed = source.decision_committed.detach().to(device="cpu", copy=True)
        self.expert_probs = source.expert_probs.detach().to(device="cpu", copy=True)
        self.terminated = source.terminated.detach().to(device="cpu", copy=True)
        self.outcome_targets = source.outcome_targets.detach().to(device="cpu", copy=True)
        self.observer_side = source.observer_side.detach().to(device="cpu", copy=True)
        self.transition_contiguous = source.transition_contiguous.detach().to(
            device="cpu", copy=True
        )
        self.initial_hidden = source.initial_hidden.detach().to(device="cpu", copy=True)

        self.adv_agg: torch.Tensor | None = None
        self.ret_agg: torch.Tensor | None = None
        self.ns_labels = (
            source.ns_labels.detach().to(device="cpu", copy=True)
            if source.ns_labels is not None
            else None
        )
        self.density_targets = (
            source.density_targets.detach().to(device="cpu", copy=True)
            if source.density_targets is not None
            else None
        )
        self.shot_labels = (
            source.shot_labels.detach().to(device="cpu", copy=True)
            if source.shot_labels is not None
            else None
        )
        self.belief_diagnostics = {
            key: (total.detach().cpu(), count.detach().cpu())
            for key, (total, count) in source.belief_diagnostics.items()
        }

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
        destination.decision_committed.copy_(self.decision_committed)

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
        num_recurrent = self.initial_hidden.shape[1] // self.num_envs  # N+G
        hidden_full = self.initial_hidden.reshape(
            n_layers,
            self.num_envs,
            num_recurrent,
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
                len(indices) * num_recurrent,
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
                    decision_committed=self.decision_committed[:, indices],
                    expert_probs=self.expert_probs[:, indices],
                    terminated=self.terminated[:, indices],
                    outcome_targets=self.outcome_targets[:, indices],
                    observer_side=self.observer_side[:, indices],
                    transition_contiguous=self.transition_contiguous[:, indices],
                    adv_agg=self.adv_agg[:, indices],
                    ret_agg=self.ret_agg[:, indices],
                    ns_labels=(self.ns_labels[:, indices] if self.ns_labels is not None else None),
                    density_targets=(
                        self.density_targets[:, indices]
                        if self.density_targets is not None
                        else None
                    ),
                    shot_labels=(
                        self.shot_labels[:, indices] if self.shot_labels is not None else None
                    ),
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

    def __init__(self, shards: list[StoredRollout], return_scale: torch.Tensor) -> None:
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
        self.return_scale = return_scale
        self.belief_diagnostics: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for shard in shards:
            for key, (total, count) in shard.belief_diagnostics.items():
                if key not in self.belief_diagnostics:
                    self.belief_diagnostics[key] = (total.clone(), count.clone())
                else:
                    old_total, old_count = self.belief_diagnostics[key]
                    self.belief_diagnostics[key] = (old_total + total, old_count + count)

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
