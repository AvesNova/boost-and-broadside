"""GPU-resident rollout buffer for recurrent PPO.

All data is pre-allocated on-device. No CPU-GPU transfers during rollout
collection. GAE is computed fully on GPU.

Shape conventions throughout:
    T = num_steps (rollout length)
    B = num_envs
    N = num_ships
    D = d_model
"""

import numpy as np
import torch
from typing import Generator


class RolloutBuffer:
    """Pre-allocated GPU rollout buffer for one PPO rollout.

    Supports per-ship rewards and values (zero-sum multi-agent PPO).
    Stores one initial GRU hidden state per rollout for recurrent re-evaluation.

    Args:
        num_steps:    Rollout horizon T.
        num_envs:     Parallel environments B.
        num_ships:    Ships per environment N.
        obs_shapes:   Dict mapping obs key → trailing shape, e.g. {"pos": (N, 2)}.
        gamma:        Discount factor.
        gae_lambda:   GAE lambda.
        device:       GPU device for all storage.
    """

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        num_ships: int,
        obs_shapes: dict[str, tuple],
        gamma: float,
        gae_lambda: float,
        device: torch.device,
    ) -> None:
        self.num_steps   = num_steps
        self.num_envs    = num_envs
        self.num_ships   = num_ships
        self.gamma       = gamma
        self.gae_lambda  = gae_lambda
        self.device      = device

        T, B, N = num_steps, num_envs, num_ships

        # Observations — one entry per obs key
        self.obs: dict[str, torch.Tensor] = {
            key: torch.zeros((T, B, *shape), device=device, dtype=torch.float32)
            for key, shape in obs_shapes.items()
        }
        # Store integer obs separately (team_id, alive, prev_action)
        # We cast them to float for the obs dict above; keep originals as float too.

        self.actions    = torch.zeros((T, B, N, 3), device=device, dtype=torch.int32)
        self.logprobs   = torch.zeros((T, B, N),    device=device, dtype=torch.float32)
        self.rewards    = torch.zeros((T, B, N),    device=device, dtype=torch.float32)
        self.values     = torch.zeros((T, B, N),    device=device, dtype=torch.float32)
        self.dones      = torch.zeros((T, B),       device=device, dtype=torch.float32)
        self.alive_mask = torch.zeros((T, B, N),    device=device, dtype=torch.bool)

        self.advantages = torch.zeros_like(self.values)
        self.returns    = torch.zeros_like(self.values)

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
    ) -> None:
        """Store one step.

        Args:
            obs:     Dict with (B, N, ...) tensors — float32.
            action:  (B, N, 3) int.
            logprob: (B, N) float.
            reward:  (B, N) float.
            done:    (B,) float (0.0 or 1.0).
            value:   (B, N) float.
            alive:   (B, N) bool.
        """
        if self.ptr >= self.num_steps:
            raise IndexError("Buffer is full — call reset() before reuse.")

        t = self.ptr
        for key, val in obs.items():
            if key in self.obs:
                self.obs[key][t] = val.float()

        self.actions   [t] = action.int()
        self.logprobs  [t] = logprob
        self.rewards   [t] = reward
        self.dones     [t] = done.float()
        self.values    [t] = value
        self.alive_mask[t] = alive

        self.ptr += 1

    # ------------------------------------------------------------------
    # GAE computation
    # ------------------------------------------------------------------

    def compute_gae(
        self, next_value: torch.Tensor, next_done: torch.Tensor
    ) -> None:
        """Compute GAE advantages and returns in-place.

        Args:
            next_value: (B, N) float — critic value at step T+1.
            next_done:  (B,) float — whether step T+1 is terminal.
        """
        with torch.no_grad():
            lastgaelam = torch.zeros_like(next_value)    # (B, N)

            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    # After the last step: bootstrap from next_value unless next ep is done
                    non_terminal = 1.0 - next_done.unsqueeze(-1)   # (B, 1)
                    next_val     = next_value                        # (B, N)
                else:
                    # Step t is terminal → the observation at t+1 is a new episode
                    # nextnonterminal blocks bootstrap from t+1 if THIS step was terminal
                    non_terminal = 1.0 - self.dones[t].unsqueeze(-1)  # (B, 1)
                    next_val     = self.values[t + 1]                  # (B, N)

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
                mb_obs:         dict[str, (T, B_mb, N, ...)] float32
                mb_actions:     (T, B_mb, N, 3) int32
                mb_logprobs:    (T, B_mb, N) float32
                mb_advantages:  (T, B_mb, N) float32
                mb_returns:     (T, B_mb, N) float32
                mb_values:      (T, B_mb, N) float32
                mb_alive:       (T, B_mb, N) bool
                mb_hidden:      (1, B_mb*N, D) float32
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
            # initial_hidden is (1, B*N, D); select envs by reshaping
            hidden_full = self.initial_hidden.reshape(1, self.num_envs, self.num_ships, D)
            mb_hidden   = hidden_full[:, idx, :, :].reshape(1, len(idx) * self.num_ships, D)

            yield (
                mb_obs,
                self.actions   [:, idx],
                self.logprobs  [:, idx],
                self.advantages[:, idx],
                self.returns   [:, idx],
                self.values    [:, idx],
                self.alive_mask[:, idx],
                mb_hidden.contiguous(),
            )
