"""MVPPolicy: the full per-ship actor-critic policy.

Architecture (per timestep):
    obs → EntityEncoder → (B, N+M, D)     [ships N then obstacles M]
         → N+M x YemongBlock → (B, N+M, D)   [spatial + temporal over all tokens]
         → slice [:N]                    → (B, N, D)    [ship tokens only]
         → ActionHead                   → (B, N, 12)   [logits: power|turn|shoot]
         → TeamPMA                      → (B, N, D)    [pool per team, broadcast back]
         → ValueHead                    → (B, N, K)    [MSE critic: K components]

Obstacle tokens (team_id=2) participate in attention and carry temporal hidden
state, but receive no action or value heads.

K = num_value_components (one head per reward component).
Value head outputs in normalized space. The ReturnScaler in PPOTrainer maps
between symlog-reward space (GAE) and normalized space (value head I/O).

Hidden state shape: (n_layers, B*(N+M), CONV_KERNEL * D), packed as:
  hidden[:, :, :D]   -- RG-LRU recurrent state
  hidden[:, :, D:]   -- causal conv buffer (CONV_KERNEL-1 past linear1 outputs, flattened)
"""

import math
import torch
import torch.nn as nn
from torch.distributions import Categorical

from boost_and_broadside.config import ModelConfig, ShipConfig
from boost_and_broadside.constants import (
    TOTAL_ACTION_LOGITS,
    NUM_POWER_ACTIONS,
    NUM_TURN_ACTIONS,
    NUM_SHOOT_ACTIONS,
    POWER_SLICE,
    TURN_SLICE,
    SHOOT_SLICE,
)
from boost_and_broadside.models.mvp.encoder import ShipEncoder
from boost_and_broadside.models.mvp.griffin import CONV_KERNEL, YemongBlock


class TeamPMA(nn.Module):
    """Pooling by Multi-head Attention over per-team ship embeddings.

    For each team t ∈ {0, 1}, a learned seed attends over the GRU outputs of
    alive ships on that team. Dead ships and ships from the opposite team are
    masked out as keys. The two team embeddings are broadcast back so every
    ship holds its team's pooled embedding — preserving the (B, N, D) shape
    expected by the value head.

    Args:
        d_model: Token embedding dimension D.
        n_heads:  Attention heads (must divide d_model evenly).
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.seeds = nn.Parameter(torch.zeros(2, d_model))
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, bias=False
        )
        self.norm = nn.RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,  # (B, N, D)
        team_id: torch.Tensor,  # (B, N) int
        alive: torch.Tensor,  # (B, N) bool
    ) -> torch.Tensor:  # (B, N, D)
        B, N, D = x.shape
        team_pool = x.new_zeros(B, 2, D)

        for t in range(2):
            mask = (team_id == t) & alive  # (B, N) — alive ships on team t
            has_ship = mask.any(dim=1)  # (B,) bool

            seed = self.seeds[t].view(1, 1, D).expand(B, 1, D)  # (B, 1, D)
            # key_padding_mask: True = ignore that key position
            out, _ = self.attn(seed, x, x, key_padding_mask=~mask, need_weights=False)
            out = out.squeeze(1).nan_to_num(0.0)  # (B, D) — guard: all-dead → NaN → 0
            out = out * has_ship.unsqueeze(1).float()  # zero dead-team envs before norm
            team_pool[:, t] = self.norm(out)  # (B, D)

        # Each ship gets its team's pooled embedding
        idx = team_id.clamp(0, 1).long().unsqueeze(-1).expand(B, N, D)
        return team_pool.gather(1, idx)  # (B, N, D)


class MVPPolicy(nn.Module):
    """Actor-critic policy with shared trunk: Encoder → N × YemongBlock.

    Args:
        model_config: Architecture hyperparameters.
        ship_config: Physics constants (used by encoder for world_size).
    """

    def __init__(
        self,
        model_config: ModelConfig,
        ship_config: ShipConfig,
        num_value_components: int,
        num_ships: int,
    ) -> None:
        super().__init__()
        D = model_config.d_model
        self._d_model = D
        self._K = num_value_components
        self._num_ships = num_ships  # N — first N tokens are ships; rest are obstacles

        self.encoder = ShipEncoder(model_config, ship_config)
        self.yemong_layers = nn.ModuleList(
            [YemongBlock(model_config) for _ in range(model_config.n_transformer_blocks)]
        )
        self.team_pma = TeamPMA(d_model=D, n_heads=model_config.n_heads)

        hidden_dim = D * 2

        self.action_head = nn.Sequential(
            nn.Linear(D, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, TOTAL_ACTION_LOGITS),
        )
        # K independent scalar heads — one per reward component, in normalized space
        self.value_head = nn.Sequential(
            nn.Linear(D, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self._K),
        )

        # Orthogonal init — standard PPO practice
        for head in [self.action_head, self.value_head]:
            # Hidden layer (ReLU/GELU activation) gets sqrt(2) gain
            nn.init.orthogonal_(head[0].weight, gain=math.sqrt(2))
            nn.init.zeros_(head[0].bias)
            # Output layers get 0.01 gain for stable early predictions
            nn.init.orthogonal_(head[3].weight, gain=0.01)
            nn.init.zeros_(head[3].bias)

        # TeamPMA init — seeds are small and asymmetric; projections use orthogonal
        nn.init.normal_(self.team_pma.seeds, mean=0.0, std=0.02)
        nn.init.orthogonal_(self.team_pma.attn.in_proj_weight, gain=math.sqrt(2))
        nn.init.orthogonal_(self.team_pma.attn.out_proj.weight, gain=1.0)

    # ------------------------------------------------------------------
    # Hidden state management
    # ------------------------------------------------------------------

    def initial_hidden(
        self, num_envs: int, num_tokens: int, device: torch.device
    ) -> torch.Tensor:
        """Return zeroed hidden states for all Yemong layers.

        Args:
            num_tokens: N+M (ships + obstacles) — all entity tokens carry hidden state.

        Returns:
            (n_layers, B*(N+M), CONV_KERNEL*D) float32 — packed RG-LRU state + conv buffer.
        """
        n_layers = len(self.yemong_layers)
        return torch.zeros(
            n_layers, num_envs * num_tokens, CONV_KERNEL * self._d_model, device=device
        )

    def reset_hidden_for_envs(
        self,
        hidden: torch.Tensor,
        done_mask: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        """Zero hidden states for all tokens in done environments.

        Args:
            hidden:     (n_layers, B*(N+M), D) current hidden state.
            done_mask:  (B,) bool — True for envs that finished.
            num_tokens: N+M.

        Returns:
            Updated hidden state with done envs zeroed.
        """
        if not done_mask.any():
            return hidden
        token_done = done_mask.repeat_interleave(num_tokens)  # (B*(N+M),)
        hidden = hidden.clone()
        hidden[:, token_done, :] = 0.0
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
            hidden: (n_layers, B*N, D) RG-LRU hidden state.

        Returns:
            action:     (B, N, 3) int — sampled [power, turn, shoot].
            logprob:    (B, N) float — sum of log probs for each sub-action.
            value:      (B, N, K) float — per-component value in normalized space.
                        Caller must denormalize via ReturnScaler before using for GAE.
            new_hidden: (n_layers, B*N, D) updated hidden state.
        """
        alive = obs["alive"]  # (B, N+M) bool — ships then obstacles
        x = self.encoder(obs)  # (B, N+M, D)

        B, NM, D = x.shape
        BNM = B * NM
        n_layers = len(self.yemong_layers)
        rglru_states = hidden[:, :, :D]                                      # (n_layers, B*(N+M), D)
        conv_bufs = hidden[:, :, D:].reshape(n_layers, BNM, CONV_KERNEL - 1, D)

        new_rglru, new_cbs = [], []
        for i, layer in enumerate(self.yemong_layers):
            x, new_h, new_cb = layer.step(x, alive, rglru_states[i], conv_bufs[i])
            new_rglru.append(new_h)
            new_cbs.append(new_cb)

        new_rglru_t = torch.stack(new_rglru, dim=0)                          # (n_layers, B*(N+M), D)
        new_cbs_t = torch.stack(new_cbs, dim=0).reshape(n_layers, BNM, -1)  # (n_layers, B*(N+M), (K-1)*D)
        new_hidden = torch.cat([new_rglru_t, new_cbs_t], dim=-1)             # (n_layers, B*(N+M), K*D)

        # Slice ship tokens only for action and value heads
        N = self._num_ships
        x_ships = x[:, :N, :]                    # (B, N, D)
        alive_ships = alive[:, :N]               # (B, N)
        team_id_ships = obs["team_id"][:, :N]    # (B, N) — obstacles (team_id=2) excluded by TeamPMA

        logits = self.action_head(x_ships)                                   # (B, N, 12)
        x_value = self.team_pma(x_ships, team_id_ships, alive_ships)        # (B, N, D)
        value = self.value_head(x_value)                                     # (B, N, K)

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
        return_encoder_output: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Re-evaluate actions over a full rollout for PPO update.

        The encoder runs over all T*B*(N+M) tokens in parallel. Each YemongBlock
        runs its spatial attention over (T*B, N+M, D) in parallel, then its
        temporal RG-LRU sequentially over T from the layer's initial hidden.
        Action and value heads are applied only to the first N ship tokens.

        Args:
            obs:                  Dict with (T, B, N+M, ...) tensors.
            actions:              (T, B, N, 3) int actions taken during rollout.
            initial_hidden:       (n_layers, B*(N+M), D) hidden state at rollout start.
            alive_mask:           (T, B, N+M) bool — alive entities per timestep.
            return_encoder_output: If True, return raw encoder embeddings as 5th value.
                                   Pass False (default) when sigreg_coef=0 to avoid
                                   keeping the encoder output tensor alive in RAM.

        Returns:
            logprob:   (T, B, N) float.
            entropy:   (T, B, N) float.
            new_value: (T, B, N, K) float — per-component value in normalized space.
            logits:    (T, B, N, TOTAL_ACTION_LOGITS) float — raw action logits.
            z:         (T, B, N+M, D) float — raw encoder embeddings before Yemong layers,
                       or None if return_encoder_output=False.
        """
        T, B, N = actions.shape[:3]  # N = num_ships (actions only for ships)
        D = self._d_model
        n_layers = len(self.yemong_layers)
        BNM = initial_hidden.shape[1]  # B*(N+M)

        rglru_states = initial_hidden[:, :, :D]                                        # (n_layers, B*(N+M), D)
        conv_bufs = initial_hidden[:, :, D:].reshape(n_layers, BNM, CONV_KERNEL - 1, D)

        # obs has (T, B, N+M, ...) — flatten T into B for encoder
        NM = obs["pos"].shape[2]  # N+M total tokens
        flat_obs = {k: v.reshape(T * B, *v.shape[2:]) for k, v in obs.items()}

        x = self.encoder(flat_obs)              # (T*B, N+M, D)
        x = x.reshape(T, B, NM, D)             # (T, B, N+M, D)
        z = x if return_encoder_output else None

        for i, layer in enumerate(self.yemong_layers):
            x, _, _ = layer.sequence(x, alive_mask, rglru_states[i], conv_bufs[i])

        # Slice ship tokens for heads
        x_ships = x[:, :, :N, :]                      # (T, B, N, D)
        alive_ships = alive_mask[:, :, :N]             # (T, B, N)
        team_id_ships = obs["team_id"][:, :, :N]       # (T, B, N)

        logits = self.action_head(x_ships)             # (T, B, N, 12)

        # TeamPMA over ship tokens: fold T into B, run PMA, unfold
        x_s_flat = x_ships.reshape(T * B, N, D)
        alive_s_flat = alive_ships.reshape(T * B, N)
        tid_s_flat = team_id_ships.reshape(T * B, N)
        xv_flat = self.team_pma(x_s_flat, tid_s_flat, alive_s_flat)  # (T*B, N, D)
        xv = xv_flat.reshape(T, B, N, D)

        new_value = self.value_head(xv)                # (T, B, N, K)

        logprob, entropy = _evaluate_action(logits, actions)

        return logprob, entropy, new_value, logits, z


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
    turn_dist = Categorical(logits=logits[..., TURN_SLICE])
    shoot_dist = Categorical(logits=logits[..., SHOOT_SLICE])

    power_a = power_dist.sample()  # (...,)
    turn_a = turn_dist.sample()
    shoot_a = shoot_dist.sample()

    action = torch.stack([power_a, turn_a, shoot_a], dim=-1)  # (..., 3)
    logprob = (
        power_dist.log_prob(power_a)
        + turn_dist.log_prob(turn_a)
        + shoot_dist.log_prob(shoot_a)
    )  # (...)
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
    turn_dist = Categorical(logits=logits[..., TURN_SLICE])
    shoot_dist = Categorical(logits=logits[..., SHOOT_SLICE])

    logprob = (
        power_dist.log_prob(actions[..., 0])
        + turn_dist.log_prob(actions[..., 1])
        + shoot_dist.log_prob(actions[..., 2])
    )
    entropy = power_dist.entropy() + turn_dist.entropy() + shoot_dist.entropy()
    return logprob, entropy
