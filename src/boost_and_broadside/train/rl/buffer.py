"""GPU-resident rollout buffer for recurrent PPO with decomposed per-component critic.

All data is pre-allocated on-device. No CPU-GPU transfers during rollout
collection. GAE is computed fully on GPU.

Shape conventions throughout:
    T = num_steps (rollout length)
    B = num_envs
    N = num_ships
    K = num_value_components (per-component critic decomposition)
    D = d_model
"""

import numpy as np
import torch
from typing import Generator


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric log transform: sign(x) * log(1 + |x|).

    Compresses large reward magnitudes while preserving sign and the zero point.
    Applied to raw rewards at storage time [symlog #1].
    """
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog: sign(x) * (exp(|x|) - 1)."""
    return torch.sign(x) * torch.expm1(x.abs())


class ReturnScaler:
    """Per-component EMA of 5th/95th percentiles for return normalization.

    Maps symlog-reward space returns to roughly [-1, 1] per component so that
    MSE value loss is comparable across components that have very different
    natural scales (e.g. victory ±80 vs turn_rate ±0.001).

    The scaler is updated once per rollout from the buffer's computed returns.
    Lambdas in the PPO advantage aggregation then act as pure importance weights
    (sign + magnitude) rather than also implicitly controlling scale.

    Args:
        num_components: K — number of value components.
        device:         Torch device (must match the returns tensor).
        ema_alpha:      EMA decay rate per rollout update (default 0.005 ≈ 200-update
                        memory). Slower = more stable but slower adaptation.
        min_span:       Minimum allowed p95−p5 value (symlog-space). Guards
                        disabled components (weight=0) whose returns are all zero.
    """

    def __init__(
        self,
        num_components: int,
        device: torch.device,
        ema_alpha: float = 0.005,
        min_span: float  = 1.0,
    ) -> None:
        self.alpha    = ema_alpha
        self.min_span = min_span
        self._initialized = False
        self._p5  = torch.zeros(num_components, device=device)
        self._p95 = torch.zeros(num_components, device=device)

    @torch.no_grad()
    def update(self, returns: torch.Tensor) -> None:
        """Update EMA percentiles from this rollout's returns.

        On the first call the EMA is seeded with the observed percentiles directly
        (no prior) so that normalization is correct from the very first update.

        Args:
            returns: (T, B, N, K) float32 — GAE returns in symlog-reward space.
        """
        T, B, N, K = returns.shape
        flat = returns.reshape(-1, K)                                    # (T*B*N, K)
        p5  = torch.quantile(flat.float(), 0.05, dim=0)                 # (K,)
        p95 = torch.quantile(flat.float(), 0.95, dim=0)
        if not self._initialized:
            self._p5  = p5
            self._p95 = p95
            self._initialized = True
        else:
            self._p5  = (1.0 - self.alpha) * self._p5  + self.alpha * p5
            self._p95 = (1.0 - self.alpha) * self._p95 + self.alpha * p95

    def _half_span(self) -> torch.Tensor:
        """Half the p95−p5 range, clamped to at least min_span/2."""
        return ((self._p95 - self._p5) * 0.5).clamp(min=self.min_span * 0.5)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Map from symlog-reward space → normalized space (≈ [-1, 1] per component).

        Args:
            x: (..., K) float — values in symlog-reward space.

        Returns:
            (..., K) float — normalized values.
        """
        center = (self._p95 + self._p5) * 0.5   # (K,)
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
        return {"p5": self._p5.cpu(), "p95": self._p95.cpu(),
                "initialized": self._initialized}

    def load_state_dict(self, d: dict) -> None:
        self._p5          = d["p5"].to(self._p5.device)
        self._p95         = d["p95"].to(self._p95.device)
        self._initialized = d.get("initialized", True)  # assume initialized if loading old ckpt


class RolloutBuffer:
    """Pre-allocated GPU rollout buffer for one PPO rollout.

    Supports per-ship per-component rewards and values for the decomposed critic.
    Stores one initial GRU hidden state per rollout for recurrent re-evaluation.

    Args:
        num_steps:       Rollout horizon T.
        num_envs:        Parallel environments B.
        num_ships:       Ships per environment N.
        num_components:  Value components K (len(REWARD_COMPONENT_NAMES)).
        obs_shapes:      Dict mapping obs key → trailing shape, e.g. {"pos": (N, 2)}.
        gamma:           Discount factor.
        gae_lambda:      GAE lambda.
        device:          GPU device for all storage.
    """

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        num_ships: int,
        num_components: int,
        obs_shapes: dict[str, tuple],
        gamma: float,
        gae_lambda: float,
        device: torch.device,
    ) -> None:
        self.num_steps      = num_steps
        self.num_envs       = num_envs
        self.num_ships      = num_ships
        self.num_components = num_components
        self.gamma          = gamma
        self.gae_lambda     = gae_lambda
        self.device         = device

        T, B, N, K = num_steps, num_envs, num_ships, num_components

        # Observations — one entry per obs key
        self.obs: dict[str, torch.Tensor] = {
            key: torch.zeros((T, B, *shape), device=device, dtype=torch.float32)
            for key, shape in obs_shapes.items()
        }

        self.actions    = torch.zeros((T, B, N, 3), device=device, dtype=torch.int32)
        self.logprobs   = torch.zeros((T, B, N),    device=device, dtype=torch.float32)
        self.rewards    = torch.zeros((T, B, N, K), device=device, dtype=torch.float32)
        self.values     = torch.zeros((T, B, N, K), device=device, dtype=torch.float32)
        self.dones      = torch.zeros((T, B),       device=device, dtype=torch.float32)
        self.alive_mask = torch.zeros((T, B, N),    device=device, dtype=torch.bool)

        self.advantages   = torch.zeros((T, B, N, K),  device=device, dtype=torch.float32)
        self.returns      = torch.zeros((T, B, N, K),  device=device, dtype=torch.float32)
        self.actor_masks  = torch.ones( (T, B, N),     device=device, dtype=torch.bool)
        self.expert_probs = torch.zeros((T, B, N, 12), device=device, dtype=torch.float32)

        # Initial GRU hidden state at the start of this rollout
        self.initial_hidden: torch.Tensor | None = None

        self.ptr = 0

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear the write pointer (tensors are overwritten, not zeroed)."""
        self.ptr            = 0
        self.initial_hidden = None
        self.expert_probs.zero_()  # only filled for scripted-group envs; rest must be zero

    def store_initial_hidden(self, hidden: torch.Tensor) -> None:
        """Store the GRU hidden state at rollout start.

        Args:
            hidden: (1, B*N, D) float32.
        """
        self.initial_hidden = hidden.clone()

    def add(
        self,
        obs: dict[str, torch.Tensor],
        action: torch.Tensor,
        logprob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
        alive: torch.Tensor,
        actor_mask: torch.Tensor | None = None,
        expert_probs: torch.Tensor | None = None,
    ) -> None:
        """Store one step.

        Args:
            obs:          Dict with (B, N, ...) tensors — float32.
            action:       (B, N, 3) int.
            logprob:      (B, N) float.
            reward:       (B, N, K) float — raw per-component per-ship rewards.
            done:         (B,) float (0.0 or 1.0).
            value:        (B, N, K) float — critic expected values (symlog-reward space).
            alive:        (B, N) bool.
            actor_mask:   (B, N) bool — True for ships whose actions came from the training
                          policy and should contribute to the actor/entropy loss. Defaults to
                          all-True (pure self-play: all ships are training ships).
            expert_probs: (B, N, 12) float — scripted-agent marginal probs [power|turn|shoot]
                          for BC loss. Zero for envs without a scripted opponent.
        """
        if self.ptr >= self.num_steps:
            raise IndexError("Buffer is full — call reset() before reuse.")

        t = self.ptr
        for key, val in obs.items():
            if key in self.obs:
                self.obs[key][t] = val.float()

        self.actions    [t] = action.int()
        self.logprobs   [t] = logprob
        self.rewards    [t] = symlog(reward)   # symlog #1: compress raw reward scale
        self.dones      [t] = done.float()
        self.values     [t] = value
        self.alive_mask [t] = alive
        self.actor_masks[t] = actor_mask if actor_mask is not None else torch.ones_like(alive)
        if expert_probs is not None:
            self.expert_probs[t] = expert_probs

        self.ptr += 1

    # ------------------------------------------------------------------
    # GAE computation
    # ------------------------------------------------------------------

    def compute_gae(
        self, next_value: torch.Tensor, next_done: torch.Tensor
    ) -> None:
        """Compute GAE advantages and returns in-place over K components.

        All tensor ops broadcast over the K dimension automatically — the
        loop body is identical to the scalar case.

        Args:
            next_value: (B, N, K) float — critic expected values at step T+1,
                        in symlog-reward space (symexp of expected bin).
            next_done:  (B,) float — whether step T+1 is terminal.
        """
        with torch.no_grad():
            lastgaelam = torch.zeros_like(next_value)    # (B, N, K)

            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    non_terminal = 1.0 - next_done.view(-1, 1, 1)   # (B, 1, 1)
                    next_val     = next_value                         # (B, N, K)
                else:
                    non_terminal = 1.0 - self.dones[t].view(-1, 1, 1)  # (B, 1, 1)
                    next_val     = self.values[t + 1]                    # (B, N, K)

                delta              = self.rewards[t] + self.gamma * next_val * non_terminal - self.values[t]
                lastgaelam         = delta + self.gamma * self.gae_lambda * non_terminal * lastgaelam
                self.advantages[t] = lastgaelam

            self.returns = self.advantages + self.values

    # ------------------------------------------------------------------
    # Minibatch iteration for PPO update
    # ------------------------------------------------------------------

    def get_minibatch_iterator(
        self, num_minibatches: int
    ) -> Generator[tuple, None, None]:
        """Yield minibatches of environments for PPO update epochs.

        Shuffles environments and slices them into num_minibatches chunks.
        Each chunk yields the full T-length sequence for those environments,
        enabling recurrent re-evaluation from the stored initial hidden state.

        Yields:
            Tuple of:
                mb_obs:           dict[str, (T, B_mb, N, ...)] float32
                mb_actions:       (T, B_mb, N, 3) int32
                mb_logprobs:      (T, B_mb, N) float32
                mb_advantages:    (T, B_mb, N, K) float32
                mb_returns:       (T, B_mb, N, K) float32
                mb_values:        (T, B_mb, N, K) float32
                mb_alive:         (T, B_mb, N) bool
                mb_hidden:        (1, B_mb*N, D) float32
                mb_actor_mask:    (T, B_mb, N) bool
                mb_expert_probs:  (T, B_mb, N, 12) float32
        """
        assert self.initial_hidden is not None, "Call store_initial_hidden() before iterating."

        envs_per_batch = self.num_envs // num_minibatches
        env_order      = np.random.permutation(self.num_envs)
        D              = self.initial_hidden.shape[-1]

        for start in range(0, self.num_envs, envs_per_batch):
            end    = start + envs_per_batch
            idx    = env_order[start:end]                          # (B_mb,)

            mb_obs = {key: val[:, idx] for key, val in self.obs.items()}

            # Reconstruct initial hidden for this minibatch: (1, B_mb*N, D)
            hidden_full = self.initial_hidden.reshape(1, self.num_envs, self.num_ships, D)
            mb_hidden   = hidden_full[:, idx, :, :].reshape(1, len(idx) * self.num_ships, D)

            yield (
                mb_obs,
                self.actions      [:, idx],
                self.logprobs     [:, idx],
                self.advantages   [:, idx],
                self.returns      [:, idx],
                self.values       [:, idx],
                self.alive_mask   [:, idx],
                mb_hidden.contiguous(),
                self.actor_masks  [:, idx],
                self.expert_probs [:, idx],
            )
