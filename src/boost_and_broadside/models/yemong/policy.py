"""YemongPolicy: the full per-ship actor-critic policy.

Architecture (per timestep):
    obs → EntityEncoder → (B, N+M, D)     [ships N then fields M]
    bullets → BulletEncoder → (B, N*K, D) [key/value only; optional]
         → n_yemong_blocks x YemongBlock → (B, N+M, D)
              [n_spatial_per_block spatial sublayers, the first
               n_bullet_cross_per_block of which cross-attend to bullets,
               then n_temporal_per_block temporal sublayers]
         → slice [:N]                    → (B, N, D)    [ship tokens only]
         → ActionHead                   → (B, N, 12)   [logits: power|turn|shoot]
         → NextStateHead                → (B, N, P)    [aux: pred next state deltas; P from coord.]
         → TeamPMA                      → (B, N, D)    [pool per team, broadcast back]
         → ValueHead                    → (B, N, K)    [MSE critic: K components]

Three entity kinds, three levels of participation:
  ships  (team_id 0/1) — attention, recurrence, and all three heads.
  fields (team_id 2)   — attention only. Static within an episode, so they take
                         the non-recurrent Griffin path and receive no heads.
  bullets              — key/value only. Never queried, never recurrent, never
                         updated; they exist solely as things ships can look at.

K = num_value_components (one head per reward component).
Value head outputs in normalized space. The ReturnScaler in PPOTrainer maps
between symlog-reward space (GAE) and normalized space (value head I/O).

Hidden state shape: (n_layers, B*N, CONV_KERNEL * D) — ships only — packed as:
  hidden[:, :, :D]   -- RG-LRU recurrent state
  hidden[:, :, D:]   -- causal conv buffer (CONV_KERNEL-1 past linear1 outputs, flattened)

n_layers is n_yemong_blocks * n_temporal_per_block: every temporal sublayer owns
one slot, and block i's slots are the contiguous run [i*n_temporal, (i+1)*n_temporal).
The trunk reads its ship/field split off this tensor's width rather than tracking
it separately, so sizing and splitting cannot disagree.
"""

import math

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.utils.checkpoint import checkpoint

from boost_and_broadside.config import ModelConfig
from boost_and_broadside.constants import (
    POWER_SLICE,
    SHOOT_SLICE,
    TOTAL_ACTION_LOGITS,
    TURN_SLICE,
)
from boost_and_broadside.env.observation import BulletObsKey, YemongObservation
from boost_and_broadside.models.yemong.encoder import BulletEncoder, ShipEncoder
from boost_and_broadside.models.yemong.griffin import CONV_KERNEL, YemongBlock
from boost_and_broadside.train.rl.features import FeatureCoordinator


class NextStateHead(nn.Module):
    """Predicts next-state deltas and absolutes for each ship.

    Output dimension is determined by the FeatureCoordinator's total_prediction_dimension.
    """

    def __init__(self, d_model: int, pred_dim: int) -> None:
        super().__init__()
        self.pred_dim = pred_dim
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.RMSNorm(d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, pred_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (..., D). Returns: (..., pred_dim)."""
        return self.net(x)


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
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, bias=False)
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
            out = out * has_ship.unsqueeze(1).to(out.dtype)  # zero dead-team envs before norm
            team_pool[:, t] = self.norm(out)  # (B, D)

        # Each ship gets its team's pooled embedding
        idx = team_id.clamp(0, 1).long().unsqueeze(-1).expand(B, N, D)
        return team_pool.gather(1, idx)  # (B, N, D)


def _init_head_orthogonal(head: nn.Sequential) -> None:
    """Orthogonal-init a Linear+Norm+Act+Linear head's first and last Linear layers.

    Locates layers by type instead of a fixed index, so the head's Sequential can
    grow or reorder non-Linear layers without corrupting or missing this init.
    """
    linears = [m for m in head if isinstance(m, nn.Linear)]
    nn.init.orthogonal_(linears[0].weight, gain=math.sqrt(2))
    nn.init.zeros_(linears[0].bias)
    nn.init.orthogonal_(linears[-1].weight, gain=0.01)
    nn.init.zeros_(linears[-1].bias)


class YemongPolicy(nn.Module):
    """Actor-critic policy with shared trunk: Encoder → N × YemongBlock.

    Args:
        model_config:  Architecture hyperparameters.
        coordinator:   Feature pipeline; drives encoder input dim and aux pred dim.
        num_value_components: K — one value head output per reward component.
        num_ships:     N — first N tokens are ships; rest are fields.
        bullet_coordinator: Bullet feature pipeline; required when the model
            config enables bullet cross-attention.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        coordinator: FeatureCoordinator,
        num_value_components: int,
        num_ships: int,
        team_pma_k: tuple[int, ...] = (),
        bullet_coordinator: FeatureCoordinator | None = None,
    ) -> None:
        super().__init__()
        D = model_config.d_model
        self._d_model = D
        self._K = num_value_components
        self._num_ships = num_ships  # N — first N tokens are ships; rest are fields
        self._team_pma_k = team_pma_k  # K indices that use TeamPMA path for value
        self._team_pma_k_set = set(team_pma_k)
        self.coordinator = coordinator

        self.encoder = ShipEncoder(model_config, coordinator, num_ships=num_ships)
        # Bullets are encoded once per timestep and reused by every spatial
        # sublayer that reads them — re-encoding per layer would multiply the
        # dominant encoder cost for identical data.
        self.bullet_encoder = (
            BulletEncoder(model_config, bullet_coordinator)
            if model_config.reads_bullets and bullet_coordinator is not None
            else None
        )
        self.yemong_layers = nn.ModuleList(
            [YemongBlock(model_config) for _ in range(model_config.n_yemong_blocks)]
        )
        # Hidden-state slots per block — the trunk's recurrent state is indexed
        # [block * n_temporal + sublayer], so this stride is load-bearing.
        self._n_temporal = model_config.n_temporal_per_block
        self._grad_checkpoint = model_config.grad_checkpoint

        hidden_dim = D * 2

        self.action_head = nn.Sequential(
            nn.Linear(D, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, TOTAL_ACTION_LOGITS),
        )
        # Local value head: per-ship embedding → all K components.
        # For indices in team_pma_k, outputs are overridden by value_head_win.
        self.value_head_local = nn.Sequential(
            nn.Linear(D, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self._K),
        )
        # TeamPMA + win/loss head: explicit team-pool → win/loss components only.
        # Only instantiated when team_pma_k is non-empty.
        if team_pma_k:
            self.team_pma = TeamPMA(d_model=D, n_heads=model_config.n_heads)
            self.value_head_win = nn.Sequential(
                nn.Linear(D, hidden_dim),
                nn.RMSNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, len(team_pma_k)),
            )
        self.next_state_head = NextStateHead(D, pred_dim=coordinator.total_prediction_dimension)

        # Orthogonal init — standard PPO practice. Located by type (first/last Linear)
        # rather than fixed Sequential index, so inserting a non-Linear layer (e.g.
        # Dropout) into a head can't silently init the wrong module.
        for head in [self.action_head, self.value_head_local, self.next_state_head.net]:
            _init_head_orthogonal(head)
        if team_pma_k:
            _init_head_orthogonal(self.value_head_win)
            nn.init.normal_(self.team_pma.seeds, mean=0.0, std=0.02)
            nn.init.orthogonal_(self.team_pma.attn.in_proj_weight, gain=math.sqrt(2))
            nn.init.orthogonal_(self.team_pma.attn.out_proj.weight, gain=1.0)

    def _encode_bullets(
        self, obs: YemongObservation
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Encode the bullet axis once, returning tokens and their active mask."""
        if self.bullet_encoder is None or obs.bullets is None:
            return None, None
        return self.bullet_encoder(obs), obs.bullets[BulletObsKey.ACTIVE]

    # ------------------------------------------------------------------
    # Hidden state management
    # ------------------------------------------------------------------

    def initial_hidden(
        self, num_envs: int, num_recurrent_tokens: int, device: torch.device
    ) -> torch.Tensor:
        """Return zeroed hidden states for all temporal sublayers.

        Args:
            num_recurrent_tokens: N — only ship tokens carry recurrent state. Field
                tokens are static within an episode and take the non-recurrent path
                (see ``GriffinTemporalBlock.forward_nonrecurrent``), so passing N+M
                here would allocate a third more state than the trunk consumes.

        Returns:
            (n_layers, B*N, CONV_KERNEL*D) float32 — packed RG-LRU state + conv
            buffer, where n_layers is n_yemong_blocks * n_temporal_per_block.
        """
        return torch.zeros(
            self.n_hidden_layers,
            num_envs * num_recurrent_tokens,
            CONV_KERNEL * self._d_model,
            device=device,
        )

    @property
    def n_hidden_layers(self) -> int:
        """Recurrent state slots — one per temporal sublayer across the trunk."""

        return len(self.yemong_layers) * self._n_temporal

    @property
    def num_recurrent_tokens(self) -> int:
        """Tokens carrying recurrent state — ships only; fields are static."""

        return self._num_ships

    def reset_hidden_for_envs(
        self,
        hidden: torch.Tensor,
        done_mask: torch.Tensor,
        num_recurrent_tokens: int,
    ) -> torch.Tensor:
        """Zero hidden states for all recurrent tokens in done environments.

        Args:
            hidden:     (n_layers, B*N, CONV_KERNEL*D) current hidden state.
            done_mask:  (B,) bool — True for envs that finished.
            num_recurrent_tokens: N — must match what ``initial_hidden`` was given.

        Returns:
            Updated hidden state with done envs zeroed.
        """
        token_keep = (~done_mask.repeat_interleave(num_recurrent_tokens)).to(hidden.dtype)
        return hidden * token_keep[None, :, None]

    # ------------------------------------------------------------------
    # Rollout-time forward (single step)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_action_and_value(
        self,
        obs: YemongObservation,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action and estimate value for one environment step.

        Args:
            obs:    YemongObservation with (B, N+M, ...) tensors.
            hidden: (n_layers, B*(N+M), CONV_KERNEL*D) packed recurrent state.

        Returns:
            action:     (B, N, 3) int — sampled [power, turn, shoot].
            logprob:    (B, N) float — sum of log probs for each sub-action.
            value:      (B, N, K) float — per-component value in normalized space.
                        Caller must denormalize via ReturnScaler before using for GAE.
            pred_next:  (B, N, pred_dim) float — predicted next-state deltas/phase shifts.
            new_hidden: (n_layers, B*(N+M), CONV_KERNEL*D) updated packed state.
        """
        alive = obs["alive"]  # (B, N+M) bool — ships then fields
        x = self.encoder(obs)  # (B, N+M, D)
        bullets, bullet_mask = self._encode_bullets(obs)  # (B, N*K, D), (B, N*K)

        B, NM, D = x.shape
        n_layers = hidden.shape[0]
        n_temporal = self._n_temporal
        # Recurrent token count is read off the hidden tensor rather than tracked
        # separately, so the split can never disagree with how the caller sized it.
        # A caller that still allocates B*(N+M) therefore keeps every token recurrent.
        B_rec = hidden.shape[1]  # B*N — recurrent (ship) tokens only
        n_rec = B_rec // B
        rglru_states = hidden[:, :, :D]  # (n_layers, B*N, D)
        conv_bufs = hidden[:, :, D:].reshape(n_layers, B_rec, CONV_KERNEL - 1, D)

        new_rglru, new_cbs = [], []
        for i, layer in enumerate(self.yemong_layers):
            # Each block owns a contiguous run of n_temporal hidden slots.
            block_slice = slice(i * n_temporal, (i + 1) * n_temporal)
            x, new_h, new_cb = layer.step(
                x,
                alive,
                rglru_states[block_slice],
                conv_bufs[block_slice],
                n_rec,
                bullets,
                bullet_mask,
            )
            new_rglru.append(new_h)
            new_cbs.append(new_cb)

        # cat, not stack: each entry already carries this block's n_temporal slots.
        # The conv-buffer width is spelled out rather than inferred with -1, which is
        # ambiguous when n_layers is 0 (a purely spatial trunk).
        conv_width = (CONV_KERNEL - 1) * D
        new_rglru_t = torch.cat(new_rglru, dim=0) if new_rglru else rglru_states
        new_cbs_t = (
            torch.cat(new_cbs, dim=0).reshape(n_layers, B_rec, conv_width)
            if new_cbs
            else conv_bufs.reshape(n_layers, B_rec, conv_width)
        )
        new_hidden = torch.cat(
            [new_rglru_t, new_cbs_t], dim=-1
        )  # (n_layers, B*(N+M), CONV_KERNEL*D)

        # Slice ship tokens only for action and value heads
        N = self._num_ships
        x_ships = x[:, :N, :]  # (B, N, D)
        alive_ships = alive[:, :N]  # (B, N)
        team_id_ships = obs["team_id"][:, :N]  # (B, N) — fields excluded by TeamPMA

        logits = self.action_head(x_ships)  # (B, N, 12)
        pred_next = self.next_state_head(x_ships)  # (B, N, AUX_PRED_DIM)
        value = self.value_head_local(x_ships)  # (B, N, K)
        if self._team_pma_k:
            x_team = self.team_pma(x_ships, team_id_ships, alive_ships)  # (B, N, D)
            win_val = self.value_head_win(x_team)  # (B, N, K_win)
            for i, k in enumerate(self._team_pma_k):
                value[:, :, k] = win_val[:, :, i]

        action, logprob = _sample_action(logits)

        return action, logprob, value, pred_next, new_hidden

    # ------------------------------------------------------------------
    # Update-time forward (full rollout re-evaluation)
    # ------------------------------------------------------------------

    def evaluate_actions(
        self,
        obs: YemongObservation,
        actions: torch.Tensor,
        initial_hidden: torch.Tensor,
        alive_mask: torch.Tensor,
        done_mask: torch.Tensor | None = None,
        return_encoder_output: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
    ]:
        """Re-evaluate actions over a full rollout for PPO update.

        The encoder runs over all T*B*(N+M) tokens in parallel. Each YemongBlock
        runs its spatial attention over (T*B, N+M, D) in parallel, then its
        temporal RG-LRU via parallel scan over T from the layer's initial hidden.
        Action and value heads are applied only to the first N ship tokens.

        Args:
            obs:                  YemongObservation with (T, B, N+M, ...) tensors.
            actions:              (T, B, N, 3) int actions taken during rollout.
            initial_hidden:       (n_layers, B*(N+M), CONV_KERNEL*D) rollout-start state.
            alive_mask:           (T, B, N+M) bool — alive entities per timestep.
            done_mask:            (T, B) bool — True at step t means the episode ended
                                  at t; the RG-LRU resets hidden state for step t+1.
            return_encoder_output: If True, return raw encoder embeddings as 5th value.
                                   Pass False (default) when sigreg_coef=0 to avoid
                                   keeping the encoder output tensor alive in RAM.

        Returns:
            logprob:    (T, B, N) float.
            entropy:    (T, B, N) float.
            new_value:  (T, B, N, K) float — per-component value in normalized space.
            logits:     (T, B, N, TOTAL_ACTION_LOGITS) float — raw action logits.
            z:          (T, B, N+M, D) float — raw encoder embeddings before Yemong layers,
                        or None if return_encoder_output=False.
            pred_next:  (T, B, N, pred_dim) float — predicted next-state predictions (with grad).
        """
        T, B, N = actions.shape[:3]  # N = num_ships (actions only for ships)
        D = self._d_model
        n_layers = initial_hidden.shape[0]
        n_temporal = self._n_temporal
        B_rec = initial_hidden.shape[1]  # B*N — recurrent (ship) tokens only
        n_rec = B_rec // B  # see get_action_and_value: split follows hidden sizing

        rglru_states = initial_hidden[:, :, :D]  # (n_layers, B*(N+M), D)
        conv_bufs = initial_hidden[:, :, D:].reshape(n_layers, B_rec, CONV_KERNEL - 1, D)

        # obs has (T, B, N+M, ...) — flatten T into B for encoder
        NM = obs["pos"].shape[2]  # N+M total tokens
        flat_obs = YemongObservation(
            data={k: v.reshape(T * B, *v.shape[2:]) for k, v in obs.items()},
            bullets=(
                None
                if obs.bullets is None
                else {k: v.reshape(T * B, *v.shape[2:]) for k, v in obs.bullets.items()}
            ),
        )

        x = self.encoder(flat_obs)  # (T*B, N+M, D)
        x = x.reshape(T, B, NM, D)  # (T, B, N+M, D)
        bullets, bullet_mask = self._encode_bullets(flat_obs)  # (T*B, N*K, D)
        z = x if return_encoder_output else None

        for i, layer in enumerate(self.yemong_layers):
            # Each block owns a contiguous run of n_temporal hidden slots.
            block_slice = slice(i * n_temporal, (i + 1) * n_temporal)
            if self._grad_checkpoint and torch.is_grad_enabled():
                # Recompute this block's activations in backward instead of storing
                # them: activation memory stops scaling with depth. use_reentrant=False
                # is the modern checkpoint API and composes with torch.compile.
                x = checkpoint(
                    _yemong_forward,
                    layer,
                    x,
                    alive_mask,
                    rglru_states[block_slice],
                    conv_bufs[block_slice],
                    done_mask,
                    n_rec,
                    bullets,
                    bullet_mask,
                    use_reentrant=False,
                )
            else:
                x, _, _ = layer.sequence(
                    x,
                    alive_mask,
                    rglru_states[block_slice],
                    conv_bufs[block_slice],
                    done_mask,
                    n_rec,
                    bullets,
                    bullet_mask,
                )

        # Slice ship tokens for heads
        x_ships = x[:, :, :N, :]  # (T, B, N, D)
        alive_ships = alive_mask[:, :, :N]  # (T, B, N)
        team_id_ships = obs["team_id"][:, :, :N]  # (T, B, N)

        logits = self.action_head(x_ships)  # (T, B, N, 12)
        pred_next = self.next_state_head(x_ships)  # (T, B, N, AUX_PRED_DIM)

        # Local value path: per-ship embedding, no team pooling.
        local_value = self.value_head_local(x_ships)  # (T, B, N, K)

        if self._team_pma_k:
            # Win/loss path: TeamPMA over ship tokens, then fold T into B.
            x_s_flat = x_ships.reshape(T * B, N, D)
            alive_s_flat = alive_ships.reshape(T * B, N)
            tid_s_flat = team_id_ships.reshape(T * B, N)
            xv_flat = self.team_pma(x_s_flat, tid_s_flat, alive_s_flat)  # (T*B, N, D)
            xv = xv_flat.reshape(T, B, N, D)
            win_val = self.value_head_win(xv)  # (T, B, N, K_win)

            # Merge: cat approach preserves gradients through both paths.
            K = local_value.shape[-1]
            pieces = []
            win_i = 0
            for k in range(K):
                if k in self._team_pma_k_set:
                    pieces.append(win_val[..., win_i : win_i + 1])
                    win_i += 1
                else:
                    pieces.append(local_value[..., k : k + 1])
            new_value = torch.cat(pieces, dim=-1)  # (T, B, N, K)
        else:
            new_value = local_value

        logprob, entropy = _evaluate_action(logits, actions)

        return logprob, entropy, new_value, logits, z, pred_next


# ---------------------------------------------------------------------------
# Checkpoint helper
# ---------------------------------------------------------------------------


def _yemong_forward(
    layer: YemongBlock,
    x: torch.Tensor,
    alive_mask: torch.Tensor,
    h0: torch.Tensor,
    conv_buf0: torch.Tensor,
    done_mask: torch.Tensor | None,
    num_recurrent: int,
    bullets: torch.Tensor | None,
    bullet_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Run one Yemong block's full-sequence forward, returning only the output.

    Module-level (no ``self`` capture) so ``torch.utils.checkpoint`` can rematerialize
    it cleanly. The final hidden/conv states are unused by the update-time re-evaluation.
    """
    out, _, _ = layer.sequence(
        x, alive_mask, h0, conv_buf0, done_mask, num_recurrent, bullets, bullet_mask
    )
    return out


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
        power_dist.log_prob(power_a) + turn_dist.log_prob(turn_a) + shoot_dist.log_prob(shoot_a)
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
