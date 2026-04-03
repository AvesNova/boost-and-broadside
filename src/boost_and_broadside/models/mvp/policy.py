"""MVPPolicy: the full per-ship actor-critic policy.

Architecture (per timestep):
    obs → ShipEncoder → (B, N, D)
         → N × TransformerBlock    → (B, N, D)      [ships attend to each other]
         → per-ship GRU             → (B, N, D)      [temporal memory per ship]
         → ActionHead               → (B, N, 12)     [logits: power|turn|shoot]
         → ValueHead                → (B, N, K)      [MSE critic: K components in normalized space]

K = num_value_components (one head per reward component).
Value head outputs in normalized space. The ReturnScaler in PPOTrainer maps
between symlog-reward space (GAE) and normalized space (value head I/O).

GRU hidden state shape: (1, B*N, D) — ships are treated as independent sequences.
The wrapper zeros hidden states for ships in reset environments.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical

from boost_and_broadside.config import ModelConfig, ShipConfig
from boost_and_broadside.constants import (
    TOTAL_ACTION_LOGITS,
    NUM_POWER_ACTIONS, NUM_TURN_ACTIONS, NUM_SHOOT_ACTIONS,
    POWER_SLICE, TURN_SLICE, SHOOT_SLICE,
)
from boost_and_broadside.models.mvp.encoder import ShipEncoder
from boost_and_broadside.models.mvp.attention import TransformerBlock


class MVPPolicy(nn.Module):
    """Actor-critic policy with shared trunk: Encoder → Attention → GRU.

    Args:
        model_config: Architecture hyperparameters.
        ship_config: Physics constants (used by encoder for world_size).
    """

    def __init__(
        self,
        model_config: ModelConfig,
        ship_config: ShipConfig,
        num_value_components: int,
    ) -> None:
        super().__init__()
        D        = model_config.d_model
        self._K  = num_value_components

        self.encoder            = ShipEncoder(model_config, ship_config)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(model_config)
            for _ in range(model_config.n_transformer_blocks)
        ])
        self.gru = nn.GRU(D, D, num_layers=1, batch_first=False)

        self.action_head = nn.Linear(D, TOTAL_ACTION_LOGITS)
        # K independent scalar heads — one per reward component, in normalized space
        self.value_head  = nn.Linear(D, self._K)

        # Orthogonal init — standard PPO practice
        for layer in [self.action_head, self.value_head]:
            nn.init.orthogonal_(layer.weight, gain=0.01)
            nn.init.zeros_(layer.bias)

    # ------------------------------------------------------------------
    # Hidden state management
    # ------------------------------------------------------------------

    def initial_hidden(
        self, num_envs: int, num_ships: int, device: torch.device
    ) -> torch.Tensor:
        """Return a zeroed GRU hidden state.

        Returns:
            (1, B*N, D) float32 — ready to pass to forward().
        """
        D = self.gru.hidden_size
        return torch.zeros(1, num_envs * num_ships, D, device=device)

    def reset_hidden_for_envs(
        self,
        hidden: torch.Tensor,
        done_mask: torch.Tensor,
        num_ships: int,
    ) -> torch.Tensor:
        """Zero hidden states for all ships in done environments.

        Args:
            hidden:    (1, B*N, D) current hidden state.
            done_mask: (B,) bool — True for envs that finished.
            num_ships: N.

        Returns:
            Updated hidden state with done envs zeroed.
        """
        if not done_mask.any():
            return hidden
        # done_mask (B,) → expand to (B*N,)
        ship_done = done_mask.repeat_interleave(num_ships)  # (B*N,)
        hidden    = hidden.clone()
        hidden[0, ship_done, :] = 0.0
        return hidden

    # ------------------------------------------------------------------
    # Rollout-time forward (single step)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_action_and_value(
        self,
        obs: dict[str, torch.Tensor],
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action and estimate value for one environment step.

        Args:
            obs:    Dict with (B, N, ...) tensors.
            hidden: (1, B*N, D) GRU hidden state.

        Returns:
            action:     (B, N, 3) int — sampled [power, turn, shoot].
            logprob:    (B, N) float — sum of log probs for each sub-action.
            value:      (B, N, K) float — per-component value in normalized space.
                        Caller must denormalize via ReturnScaler before using for GAE.
            new_hidden: (1, B*N, D) updated GRU state.
        """
        alive = obs["alive"]              # (B, N) bool
        x     = self.encoder(obs)         # (B, N, D)
        for block in self.transformer_blocks:
            x = block(x, alive)           # (B, N, D)

        B, N, D = x.shape
        x_step  = x.reshape(B * N, D).unsqueeze(0)      # (1, B*N, D)
        gru_out, new_hidden = self.gru(x_step, hidden)  # (1, B*N, D)
        gru_out = gru_out.squeeze(0).reshape(B, N, D)   # (B, N, D)

        logits = self.action_head(gru_out)               # (B, N, 12)
        value  = self.value_head(gru_out)                # (B, N, K) — normalized space

        action, logprob = _sample_action(logits)

        return action, logprob, value, new_hidden

    # ------------------------------------------------------------------
    # Update-time forward (full rollout re-evaluation)
    # ------------------------------------------------------------------

    def evaluate_actions(
        self,
        obs: dict[str, torch.Tensor],
        actions: torch.Tensor,
        initial_hidden: torch.Tensor,
        alive_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-evaluate actions over a full rollout for PPO update.

        The encoder and attention run over all (T*B) tokens in parallel
        (no temporal dependency there). The GRU runs sequentially over T
        from initial_hidden, preserving recurrent correctness.

        Args:
            obs:            Dict with (T, B, N, ...) tensors.
            actions:        (T, B, N, 3) int actions taken during rollout.
            initial_hidden: (1, B*N, D) hidden state at rollout start.
            alive_mask:     (T, B, N) bool — alive ships per timestep.

        Returns:
            logprob:   (T, B, N) float.
            entropy:   (T, B, N) float.
            new_value: (T, B, N, K) float — per-component value in normalized space.
        """
        T, B, N = actions.shape[:3]
        D       = self.gru.hidden_size

        # Flatten T into B for encoder and attention
        flat_obs   = {k: v.reshape(T * B, *v.shape[2:]) for k, v in obs.items()}
        flat_alive = alive_mask.reshape(T * B, N)

        x = self.encoder(flat_obs)                      # (T*B, N, D)
        for block in self.transformer_blocks:
            x = block(x, flat_alive)                    # (T*B, N, D)

        # GRU over time — reshape to (T, B*N, D) for sequential processing
        x_seq      = x.reshape(T, B, N, D).reshape(T, B * N, D)  # (T, B*N, D)
        gru_out, _ = self.gru(x_seq, initial_hidden)              # (T, B*N, D)
        gru_out    = gru_out.reshape(T, B, N, D)                  # (T, B, N, D)

        logits    = self.action_head(gru_out)   # (T, B, N, 12)
        new_value = self.value_head(gru_out)    # (T, B, N, K) — normalized space

        logprob, entropy = _evaluate_action(logits, actions)

        return logprob, entropy, new_value, logits


# ---------------------------------------------------------------------------
# Action sampling helpers (pure functions, no state)
# ---------------------------------------------------------------------------

def _sample_action(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one action per ship from factored categorical distributions.

    Args:
        logits: (..., TOTAL_ACTION_LOGITS) — [power | turn | shoot] logit slices.

    Returns:
        action:  (..., 3) int — sampled indices.
        logprob: (...) float — sum of log-probs across sub-actions.
    """
    power_dist = Categorical(logits=logits[..., POWER_SLICE])
    turn_dist  = Categorical(logits=logits[..., TURN_SLICE])
    shoot_dist = Categorical(logits=logits[..., SHOOT_SLICE])

    power_a = power_dist.sample()   # (...,)
    turn_a  = turn_dist.sample()
    shoot_a = shoot_dist.sample()

    action  = torch.stack([power_a, turn_a, shoot_a], dim=-1)   # (..., 3)
    logprob = (
        power_dist.log_prob(power_a)
        + turn_dist.log_prob(turn_a)
        + shoot_dist.log_prob(shoot_a)
    )                                                             # (...)
    return action, logprob


def _evaluate_action(
    logits: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute log-probs and entropy for given actions under the policy.

    Args:
        logits:  (..., TOTAL_ACTION_LOGITS).
        actions: (..., 3) int.

    Returns:
        logprob: (...) float.
        entropy: (...) float.
    """
    power_dist = Categorical(logits=logits[..., POWER_SLICE])
    turn_dist  = Categorical(logits=logits[..., TURN_SLICE])
    shoot_dist = Categorical(logits=logits[..., SHOOT_SLICE])

    logprob = (
        power_dist.log_prob(actions[..., 0])
        + turn_dist.log_prob(actions[..., 1])
        + shoot_dist.log_prob(actions[..., 2])
    )
    entropy = (
        power_dist.entropy()
        + turn_dist.entropy()
        + shoot_dist.entropy()
    )
    return logprob, entropy
