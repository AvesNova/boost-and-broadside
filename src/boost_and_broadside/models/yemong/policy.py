"""YemongPolicy: the full per-ship actor-critic policy.

Architecture (per timestep):
    obs → EntityEncoder → (B, N+G+M, D)   [ships N, global token G, map objects M]
    bullets → BulletEncoder → (B, N*K, D) [key/value only; optional]
    full_attention:
         → n_yemong_blocks x YemongBlock → (B, N+G+M, D)
    kv_memory:
         → ships + global query projected (B, M, D_map) map K/V in every spatial layer
         → n_yemong_blocks x YemongBlock → (B, N+G, D)
              [n_spatial_per_block spatial sublayers, the first
               n_bullet_cross_per_block of which cross-attend to bullets,
               then n_temporal_per_block temporal sublayers]
         → slice [:N]                    → (B, N, D)    [ship tokens only]
         → ActionHead                   → (B, N, 30)   [joint command logits]
         → EnemyActionHead              → (B, N, 30)   [next enemy-command logits]
         → NextStateHead                → (B, N, P)    [aux: pred next state deltas; P from coord.]
         → ValueHead                    → (B, N, K_local, bins) [categorical critic]
         → slice [N] (global token)      → (B, D)
         → GlobalValueHead              → (B, 4)       [outcome: win/tie/loss/unresolved]
         → slice [N] (global token)      → (B, D)
         → GlobalDensityHead            → (B, 2C)      [aux: hex ally/enemy density]

Four object kinds, four levels of participation:
  ships  (team_id 0/1) — attention, recurrence, and every head.
  global token (G = 1 when ``ModelConfig.global_token``, else 0) — a query with
                         recurrent state, updated by every sublayer exactly as a
                         ship is. Read by no ship head, and by GlobalDensityHead
                         alone. Off, it is a map object.
  map objects (team_id 2) — either full attention plus a non-recurrent temporal
                            adapter, or K/V-only reads with no trunk updates.
  bullets              — key/value only. Never queried, never recurrent, never
                         updated; they exist solely as things ships can look at.

K = num_value_components (one critic head per reward level). Per-ship levels are
categorical over fixed symlog-spaced bins and valued by their expectation in raw
reward units; the outcome is four classes off the global token (see
``train/rl/critic.py``).

Hidden state shape: (n_layers, B*(N+G), CONV_KERNEL * D) — ships and the global
token, which lead the token axis — packed as:
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
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.checkpoint import checkpoint

from boost_and_broadside.config import ModelConfig, ShipConfig
from boost_and_broadside.config.core import NUM_GLOBAL_TOKENS
from boost_and_broadside.constants import (
    NUM_OUTCOME_CLASSES,
    TOTAL_ACTION_LOGITS,
)
from boost_and_broadside.env.observation import BulletObsKey, ObsKey, YemongObservation
from boost_and_broadside.models.yemong.attention import SpatialGeometry
from boost_and_broadside.models.yemong.encoder import BulletEncoder, ShipEncoder
from boost_and_broadside.models.yemong.griffin import CONV_KERNEL, YemongBlock
from boost_and_broadside.models.yemong.relation import relation_inputs_from_observation
from boost_and_broadside.models.yemong.rope import SpatialRotary, check_rotary_budget
from boost_and_broadside.runtime.actions import (
    decode_joint_action_unchecked,
    encode_joint_action_unchecked,
)
from boost_and_broadside.train.rl.critic import (
    CriticOutput,
    expectation,
    outcome_values,
    value_bins,
)
from boost_and_broadside.train.rl.features import FeatureCoordinator
from boost_and_broadside.train.rl.hex_density import HEX_DENSITY_DIM
from boost_and_broadside.train.rl.ship_codes import SHIP_CODE_DIM, ShipStateCodec
from boost_and_broadside.train.rl.shot_codes import (
    OUTCOME_CLASSES,
    TRAJECTORY_DIM,
    ShotCodec,
    shot_time_features,
)


class NextStateHead(nn.Module):
    """Predicts the next decision's ship-state code as a residual on the current one.

    ``logits = log(code + eps) + f(h)`` per softmax group of the code
    (``train/rl/ship_codes.py``), with ``f``'s last layer zero-initialised, so
    the head starts as "nothing changes" and learning the dynamics is all it
    does (``frontline-redesign-plan.md`` §8.2). For a hidden ship the code is
    the belief's, already smoothed by its spreads, so an unresolvable level
    costs nothing to leave alone.
    """

    def __init__(self, d_model: int, code_dim: int = SHIP_CODE_DIM) -> None:
        super().__init__()
        self.code_dim = code_dim
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.RMSNorm(d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, code_dim),
        )

    def forward(self, x: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        """Args: x (..., D), code (..., code_dim). Returns float32 logits (..., code_dim)."""
        return ShipStateCodec.baseline(code.float()) + self.net(x).float()


class ShotHeads(nn.Module):
    """Counterfactual shot prediction from a ship's launch-decision latent.

    Trained only on shots, real or ghost, launched on the decision whose
    observation carried the applied action (``env/shot_labels.py``); never read
    by the policy.

    The trajectory head is queried at bullet ages: ``[h; phi(age)] -> Linear ->
    RMSNorm -> GELU -> logits``, with ``phi`` the normalized age and a small
    Fourier basis. The first Linear is split into its latent and age parts, so
    the latent's share is computed once per ship-step and broadcast over the
    queries -- the same function as concatenating, for less work. The outcome
    head reads the latent alone.

    Args:
        d_model: Token embedding dimension D.
        max_age: The oldest queried age, which normalizes ``phi``.
    """

    HARMONICS = 4

    def __init__(self, d_model: int, max_age: int) -> None:
        super().__init__()
        self.max_age = max_age
        self.latent = nn.Linear(d_model, d_model)
        self.age = nn.Linear(1 + 2 * self.HARMONICS, d_model, bias=False)
        self.trajectory_out = nn.Sequential(
            nn.RMSNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, TRAJECTORY_DIM),
        )
        self.outcome_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.RMSNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, OUTCOME_CLASSES),
        )

    def trajectory(self, x: torch.Tensor, ages: torch.Tensor) -> torch.Tensor:
        """Args: x (..., D), ages (..., Q). Returns float32 logits (..., Q, TRAJECTORY_DIM)."""
        phi = shot_time_features(ages, self.max_age, self.HARMONICS).to(x.dtype)
        hidden = self.latent(x).unsqueeze(-2) + self.age(phi)  # (..., Q, D)
        return self.trajectory_out(hidden).float()

    def outcome(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (..., D). Returns float32 logits (..., OUTCOME_CLASSES)."""
        return self.outcome_net(x).float()


class GlobalDensityHead(nn.Module):
    """Predicts where both fleets are, as a Poisson intensity over hex cells.

    The one head that reads the global token rather than a ship token: it asks
    that token to carry where both fleets are, which is a property of the game
    rather than of any ship. Output is ``HEX_DENSITY_DIM`` wide -- every hex
    cell's ally log-rate, then every cell's enemy log-rate, in the cell order
    ``train/rl/hex_density.py`` fixes. Nothing it predicts re-enters the
    policy's input.

    The output is a **log-rate, not a probability**: the target is a soft ship
    count per cell, summing to the living count per side rather than to one, so
    a softmax here would pin the total mass and discard the count. The loss is
    the Poisson negative log likelihood, whose gradient in this output is the
    bounded ``exp(logit) - count``.

    Args:
        d_model:       Token embedding dimension D.
        out_dim:       Target width; the grid decides it, not a free choice.
        init_log_rate: Bias the output layer starts at. Orthogonal init leaves
            the weights near zero, so this alone sets the head's opening
            prediction, and a rate of one ship per cell -- what a zero bias
            means -- is two orders of magnitude too crowded. Starting at the
            true mean rate makes the first updates about *where* the ships are
            rather than about how many there are in total.
    """

    def __init__(
        self,
        d_model: int,
        out_dim: int = HEX_DENSITY_DIM,
        init_log_rate: float = -4.0,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.init_log_rate = init_log_rate
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.RMSNorm(d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (..., D) global-token embedding. Returns: (..., out_dim) log-rates."""

        return self.net(x)


class GlobalValueHead(nn.Module):
    """The match outcome as four classes, read once per environment off the global token.

    The outcome level pays every ship on a side the same number -- a function of
    team and result alone, paid to the living and the dead alike -- so its
    return is the same for every teammate and one estimate per environment is
    enough. The global token is a recurrent, attended game-level summary, so it
    is the token to read it from.

    The classes are win, tie, loss and unresolved (the discount's leak; see
    ``train/rl/critic.py``). The observer is always Team 0 -- every view is
    canonicalized before the policy sees it -- so "team 0 wins" is "I win"; the
    trainer signs the estimate per ship by authoritative team.

    Args:
        d_model:    Token embedding dimension D.
        hidden_dim: Width of the hidden layer.
    """

    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, NUM_OUTCOME_CLASSES),
        )

    def forward(self, x: torch.Tensor, num_ships: int) -> torch.Tensor:
        """Args: x (..., tokens, D), num_ships N. Returns: (..., 4) class logits."""

        return self.net(x[..., num_ships, :])


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
        num_ships:     N — first N tokens are ships; the global token follows.
        bullet_coordinator: Bullet feature pipeline; required when the model
            config enables bullet cross-attention.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        coordinator: FeatureCoordinator,
        num_value_components: int,
        num_ships: int,
        global_value_k: tuple[int, ...],
        bullet_coordinator: FeatureCoordinator | None = None,
        predict_density: bool = False,
        ship_config: ShipConfig | None = None,
        predict_shots: bool = False,
    ) -> None:
        super().__init__()
        D = model_config.d_model
        self._d_model = D
        self._K = num_value_components
        self._num_ships = num_ships  # N — first N tokens are ships
        # G — the global token that follows the ships, when promoted. Ships plus
        # it are the query/recurrent set; everything after them is map memory.
        self._num_global = NUM_GLOBAL_TOKENS if model_config.global_token else 0
        self._map_is_memory = model_config.map_read_mode == "kv_memory"
        self.coordinator = coordinator

        # Rotary spatial attention needs the world's physical periods, which is
        # the one thing the feature coordinator holds implicitly and the policy
        # does not. ``build_policy`` always supplies it; the argument stays
        # optional so an un-rotated policy can still be built from a bare config.
        if model_config.spatial_rope:
            if ship_config is None:
                raise ValueError("spatial_rope requires ship_config to derive its frequencies")
            check_rotary_budget(model_config, ship_config)
            self.rotary = SpatialRotary(ship_config, model_config.spatial_head_dim)
        else:
            self.rotary = None
        # The relation function wraps on the same toroid the rotation does.
        if model_config.relational_bias and ship_config is None:
            raise ValueError("relational_bias requires ship_config to derive its toroid")
        self._relation_world = (
            (float(ship_config.world_size[0]), float(ship_config.world_size[1]))
            if model_config.relational_bias and ship_config is not None
            else None
        )

        self.encoder = ShipEncoder(model_config, coordinator, num_ships=num_ships)
        self.map_memory_proj = (
            nn.Sequential(
                nn.Linear(D, model_config.map_memory_dim, bias=False),
                nn.RMSNorm(model_config.map_memory_dim),
            )
            if self._map_is_memory
            else None
        )
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
        self.enemy_action_head = nn.Sequential(
            nn.Linear(D, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, TOTAL_ACTION_LOGITS),
        )
        # The outcome level is read off the global token as four classes. With
        # the promotion off there is no global embedding to read, and it falls
        # back to an ordinary per-ship categorical value like every other level.
        self._global_value_k = global_value_k if self._num_global else ()
        if len(self._global_value_k) > 1:
            raise ValueError("only the outcome level is valued off the global token")
        self._local_value_k = tuple(k for k in range(self._K) if k not in self._global_value_k)
        # Categorical critic for every per-ship level: logits over fixed bins at
        # symexp(linspace(-L, L, n)), valued by their expectation.
        self.register_buffer(
            "value_support",
            value_bins(model_config.value_bins, model_config.value_symlog_limit),
            persistent=False,
        )
        self._value_bins = model_config.value_bins
        self.value_head_local = nn.Sequential(
            nn.Linear(D, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(self._local_value_k) * model_config.value_bins),
        )
        self.value_head_global = GlobalValueHead(D, hidden_dim) if self._global_value_k else None
        # Predicts the ship-state code the encoder reads; the coordinator owns it.
        if coordinator.ship_codec is None:
            raise ValueError("the policy's feature pipeline must carry the ship-state code")
        self.next_state_head = NextStateHead(D)
        # Reads the global token, so it cannot exist without one in the query set:
        # with the promotion off that token is K/V-only map memory and no final
        # embedding for it ever leaves the trunk. Built only when something trains
        # it: an untrained head is dead weight in every checkpoint and trips the
        # optimizer-moment integrity check.
        if predict_density and not self._num_global:
            raise ValueError(
                "the global density head reads the global token; it needs "
                "ModelConfig.global_token on"
            )
        # The opening rate is the true mean: one side's ships spread over the
        # grid's cells. Read off the fleet and the grid rather than guessed.
        self.density_head = (
            GlobalDensityHead(
                D,
                init_log_rate=math.log(max(num_ships, 2) / 2.0 / (HEX_DENSITY_DIM / 2)),
            )
            if predict_density
            else None
        )

        # Orthogonal init — standard PPO practice. Located by type (first/last Linear)
        # rather than fixed Sequential index, so inserting a non-Linear layer (e.g.
        # Dropout) into a head can't silently init the wrong module.
        for head in [
            self.action_head,
            self.enemy_action_head,
            self.value_head_local,
            self.next_state_head.net,
        ]:
            _init_head_orthogonal(head)
        if self.density_head is not None:
            _init_head_orthogonal(self.density_head.net)
            # After the orthogonal pass, which zeroes it: the opening log-rate.
            final = [m for m in self.density_head.net if isinstance(m, nn.Linear)][-1]
            nn.init.constant_(final.bias, self.density_head.init_log_rate)
        if self.value_head_global is not None:
            _init_head_orthogonal(self.value_head_global.net)
        # Counterfactual shot heads, built only when something trains them, for
        # the density head's reason.
        if predict_shots and ship_config is None:
            raise ValueError("the shot heads size their codes from the ship config")
        self.shot_codec = ShotCodec.from_ship_config(ship_config) if predict_shots else None
        self.shot_heads = ShotHeads(D, self.shot_codec.max_age) if predict_shots else None
        if self.shot_heads is not None:
            nn.init.orthogonal_(self.shot_heads.latent.weight, gain=math.sqrt(2))
            nn.init.zeros_(self.shot_heads.latent.bias)
            nn.init.orthogonal_(self.shot_heads.trajectory_out[-1].weight, gain=0.01)
            nn.init.zeros_(self.shot_heads.trajectory_out[-1].bias)
            _init_head_orthogonal(self.shot_heads.outcome_net)
        # The residual starts at exactly zero: "nothing changes" (§8.2).
        final = [m for m in self.next_state_head.net if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(final.weight)

    def trunk_modules(self) -> tuple[nn.Module, ...]:
        """The submodules every head reads from.

        Stated as module references rather than name patterns so a renamed or
        newly added head cannot silently redefine what "shared" means. Anything
        not listed here is head-specific: the action head, the two value heads
        and their pooling, and the next-state head.
        """
        modules: list[nn.Module] = [self.encoder, self.yemong_layers]
        if self.map_memory_proj is not None:
            modules.append(self.map_memory_proj)
        if self.bullet_encoder is not None:
            modules.append(self.bullet_encoder)
        return tuple(modules)

    def trunk_parameter_ids(self) -> frozenset[int]:
        """Identities of the shared-trunk parameters, for scoping diagnostics."""

        return frozenset(
            id(parameter) for module in self.trunk_modules() for parameter in module.parameters()
        )

    def _spatial_geometry(
        self,
        obs: YemongObservation,
        num_entity_tokens: int,
        bullets_present: bool,
        map_is_memory: bool,
    ) -> SpatialGeometry | None:
        """Rotary tables for one forward pass, shared by every spatial sublayer.

        Built from the flattened ``(B, tokens, ...)`` observation the spatial
        layers actually see, so the rollout and the full-sequence path use one
        code path with ``B`` standing for ``B`` or ``T*B`` respectively.

        Args:
            obs: Observation whose leading dims match the spatial layers' batch.
            num_entity_tokens: N+G — where the query set ends and map objects
                begin, needed only in K/V-memory mode, where the two are rotated
                separately because they are passed to attention as separate
                tensors.
            bullets_present: Whether bullet K/V tokens are attached this call.
            map_is_memory: Whether map objects are K/V-only rather than queries.
        """

        if self.rotary is None and self._relation_world is None:
            return None
        relation = (
            None
            if self._relation_world is None
            else relation_inputs_from_observation(obs, self._relation_world)
        )
        if self.rotary is None:
            return SpatialGeometry(relation=relation)
        position = obs[ObsKey.POS]
        attitude = obs[ObsKey.ATT]
        cos, sin = self.rotary.tables(position, attitude)
        map_tables = None
        if map_is_memory:
            map_tables = (cos[:, num_entity_tokens:], sin[:, num_entity_tokens:])
            cos, sin = cos[:, :num_entity_tokens], sin[:, :num_entity_tokens]
        bullet_tables = None
        if bullets_present and obs.bullets is not None:
            # A bullet has a world position and no heading, so it is rotated on
            # the same x/y basis and left unrotated on the attitude axis.
            bullet_tables = self.rotary.tables(obs.bullets[BulletObsKey.POS], None)
        return SpatialGeometry(
            entity=(cos, sin),
            bullet=bullet_tables,
            map_memory=map_tables,
            relation=relation,
        )

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
            num_recurrent_tokens: N+G — ``num_recurrent_tokens``. Map tokens are
                static within an episode and take the non-recurrent path (see
                ``GriffinTemporalBlock.forward_nonrecurrent``), so passing N+G+M
                here would allocate more state than the trunk consumes.

        Returns:
            (n_layers, B*(N+G), CONV_KERNEL*D) float32 — packed RG-LRU state +
            conv buffer, where n_layers is n_yemong_blocks * n_temporal_per_block.
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
    def num_ships(self) -> int:
        """Ship tokens: the leading N of the token axis, and all any head reads."""

        return self._num_ships

    @property
    def num_recurrent_tokens(self) -> int:
        """Tokens carrying recurrent state: ships plus the promoted global token."""

        return self._num_ships + self._num_global

    def _recurrent_count(self, hidden: torch.Tensor, batch: int) -> int:
        """Recurrent tokens per environment, read off the hidden tensor's width.

        Read rather than assumed, so the split can never disagree with how the
        caller sized the state. It may not be smaller than the query set -- the
        global token would silently lose its recurrence -- and in K/V-memory mode
        it must equal it, because nothing after the query set enters the trunk.
        """
        n_rec = hidden.shape[1] // batch
        if n_rec < self.num_recurrent_tokens or (
            self._map_is_memory and n_rec != self.num_recurrent_tokens
        ):
            raise ValueError(
                f"hidden state carries {n_rec} recurrent tokens per environment; this "
                f"policy's query set is {self.num_recurrent_tokens} (size it with "
                "num_recurrent_tokens)"
            )
        return n_rec

    def reset_hidden_for_envs(
        self,
        hidden: torch.Tensor,
        done_mask: torch.Tensor,
        num_recurrent_tokens: int,
    ) -> torch.Tensor:
        """Zero hidden states for all recurrent tokens in done environments.

        Args:
            hidden:     (n_layers, B*(N+G), CONV_KERNEL*D) current hidden state.
            done_mask:  (B,) bool — True for envs that finished.
            num_recurrent_tokens: N+G — must match what ``initial_hidden`` was given.

        Returns:
            Updated hidden state with done envs zeroed.
        """
        token_keep = (~done_mask.repeat_interleave(num_recurrent_tokens)).to(hidden.dtype)
        return hidden * token_keep[None, :, None]

    def _critic(self, x: torch.Tensor, x_ships: torch.Tensor, return_logits: bool) -> CriticOutput:
        """Every reward level's value, and the logits that produced it.

        Args:
            x: (..., N+G, D) final query-set embeddings.
            x_ships: (..., N, D) the ship slice of ``x``.
            return_logits: Keep the per-ship bin logits for the update's loss.

        Returns:
            ``CriticOutput`` with ``value`` (..., N, K): per-ship levels as the
            expectation over the value bins, the outcome level as
            ``P(win) - P(loss)`` in the observer's frame.
        """
        N = x_ships.shape[-2]
        local_logits = self.value_head_local(x_ships).unflatten(
            -1, (len(self._local_value_k), self._value_bins)
        )  # (..., N, K_local, bins)
        local_value = expectation(local_logits, self.value_support)  # (..., N, K_local)
        outcome_logits = None
        columns: list[torch.Tensor | None] = [None] * self._K
        for position, k in enumerate(self._local_value_k):
            columns[k] = local_value[..., position]
        if self.value_head_global is not None:
            outcome_logits = self.value_head_global(x, N)  # (..., 4)
            outcome = outcome_values(F.softmax(outcome_logits.float(), dim=-1))  # (...,)
            columns[self._global_value_k[0]] = outcome.unsqueeze(-1).expand(local_value.shape[:-1])
        value = torch.stack(columns, dim=-1)  # (..., N, K)
        return CriticOutput(value, local_logits if return_logits else None, outcome_logits)

    # ------------------------------------------------------------------
    # Rollout-time forward (single step)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_action_and_value(
        self,
        obs: YemongObservation,
        hidden: torch.Tensor,
        return_enemy_action: bool = False,
    ) -> tuple:
        """Sample an action and estimate value for one environment step.

        Args:
            obs:    YemongObservation with (B, N+G+M, ...) tensors.
            hidden: (n_layers, B*(N+G), CONV_KERNEL*D) packed recurrent state.

        Returns:
            action:     (B, N, 3) int — sampled [power, turn, shoot].
            logprob:    (B, N) float — log probability of the joint command.
            critic:     ``CriticOutput``; ``value`` (B, N, K) per-level expected
                        return, the outcome column in the observer's frame.
            pred_next:  (B, N, 25) float — next-state prediction decoded to moments.
            enemy_action_logits: optional (B, N, 30) next-command prediction.
            new_hidden: (n_layers, B*(N+G), CONV_KERNEL*D) updated packed state.
        """
        # Hidden-but-remembered enemies remain attention/recurrent tokens. Their
        # predicted ALIVE value is an input feature, never the existence mask --
        # and there is no existence mask any more: every ship is revealed on the
        # decision it spawns and validity is sticky, so attention carries no key
        # padding at all and SDPA can reach the flash kernel.
        encoded = self.encoder(obs)  # (B, N+G+M, D)
        N = self._num_ships
        Q = self.num_recurrent_tokens  # N+G
        if self.map_memory_proj is not None:
            map_memory = self.map_memory_proj(encoded[:, Q:, :])  # (B, M, D_map)
            x = encoded[:, :Q, :]  # (B, N+G, D)
        else:
            map_memory = None
            x = encoded
        bullets, bullet_mask = self._encode_bullets(obs)  # (B, N*K, D), (B, N*K)
        geometry = self._spatial_geometry(
            obs, Q, bullets is not None, self.map_memory_proj is not None
        )

        B, NM, D = x.shape
        n_layers = hidden.shape[0]
        n_temporal = self._n_temporal
        n_rec = self._recurrent_count(hidden, B)  # N+G
        B_rec = hidden.shape[1]  # B*(N+G)
        rglru_states = hidden[:, :, :D]  # (n_layers, B*(N+G), D)
        conv_bufs = hidden[:, :, D:].reshape(n_layers, B_rec, CONV_KERNEL - 1, D)

        new_rglru, new_cbs = [], []
        for i, layer in enumerate(self.yemong_layers):
            # Each block owns a contiguous run of n_temporal hidden slots.
            block_slice = slice(i * n_temporal, (i + 1) * n_temporal)
            x, new_h, new_cb = layer.step(
                x,
                rglru_states[block_slice],
                conv_bufs[block_slice],
                n_rec,
                bullets,
                bullet_mask,
                map_memory,
                geometry,
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
        new_hidden = torch.cat([new_rglru_t, new_cbs_t], dim=-1)  # (n_layers, B*(N+G), CK*D)

        # Per-ship heads read ships only; the global token shapes them through
        # the trunk and is decoded only by the two heads whose subject is the
        # game -- the team-level value head and the density head.
        x_ships = x[:, :N, :]  # (B, N, D)

        logits = self.action_head(x_ships)  # (B, N, 30)
        enemy_action_logits = (
            self.enemy_action_head(x_ships) if return_enemy_action else None
        )  # (B, N, 30) when requested
        # Decoded here, once: everything downstream of a rollout step -- the
        # belief, the buffer, the diagnostics -- works in moments.
        codec = self.coordinator.ship_codec
        pred_next = codec.decode_logits(
            self.next_state_head(x_ships, self.coordinator.ship_codes(obs, N))
        )  # (B, N, 25)
        critic = self._critic(x, x_ships, return_logits=False)

        action, logprob = _sample_action(logits)

        base = (action, logprob, critic, pred_next)
        if return_enemy_action:
            return (*base, enemy_action_logits, new_hidden)
        return (*base, new_hidden)

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
        return_enemy_action: bool = False,
        return_density: bool = False,
        shot_ages: torch.Tensor | None = None,
    ) -> tuple:
        """Re-evaluate actions over a full rollout for PPO update.

        The encoder runs over all T*B*(N+G+M) inputs in parallel. Full-attention
        mode sends every encoded token through spatial layers; K/V-memory mode
        sends only ships and the global token through the trunk and exposes map
        objects as a smaller read-only memory. Temporal scans run over ships and
        the global token; heads read ships only.

        Args:
            obs:                  YemongObservation with (T, B, N+G+M, ...) tensors.
            actions:              (T, B, N, 3) int actions taken during rollout.
            initial_hidden:       (n_layers, B*(N+G), CONV_KERNEL*D) rollout-start state.
            alive_mask:           (T, B, N+M) bool — used only for team pooling in
                                  the value head; attention carries no key padding.
            done_mask:            (T, B) bool — True at step t means the episode ended
                                  at t; the RG-LRU resets hidden state for step t+1.
            return_encoder_output: If True, return raw encoder embeddings as 5th value.
                                   Pass False (default) when sigreg_coef=0 to avoid
                                   keeping the encoder output tensor alive in RAM.

        Returns:
            logprob:    (T, B, N) float.
            entropy:    (T, B, N) float.
            critic:     ``CriticOutput`` with the per-ship bin logits and the
                        outcome class logits for the critic loss.
            logits:     (T, B, N, TOTAL_ACTION_LOGITS) float — raw action logits.
            z:          (T, B, N+G+M, D) float — raw encoder embeddings before Yemong layers,
                        or None if return_encoder_output=False.
            enemy_action_logits: optional (T, B, N, 30) next-command prediction.
            pred_next:  (T, B, N, 469) float — next-state code logits (with grad).
            density:    optional (T, B, 2C) float — global ally/enemy density
                        prediction, or None when this policy has no density head.
            shot_trajectory, shot_outcome: with ``shot_ages`` (T, B, N, Q), the
                        trajectory logits (T, B, N, Q, TRAJECTORY_DIM) at those
                        bullet ages and the outcome logits (T, B, N, 19).
        """
        T, B, N = actions.shape[:3]  # N = num_ships (actions only for ships)
        Q = self.num_recurrent_tokens  # N+G
        D = self._d_model
        n_layers = initial_hidden.shape[0]
        n_temporal = self._n_temporal
        n_rec = self._recurrent_count(initial_hidden, B)  # N+G
        B_rec = initial_hidden.shape[1]  # B*(N+G)

        rglru_states = initial_hidden[:, :, :D]  # (n_layers, B*(N+G), D)
        conv_bufs = initial_hidden[:, :, D:].reshape(n_layers, B_rec, CONV_KERNEL - 1, D)

        # obs has (T, B, N+G+M, ...) — flatten T into B for encoder
        NM = obs["pos"].shape[2]  # N+G+M total tokens
        flat_obs = YemongObservation(
            data={k: v.reshape(T * B, *v.shape[2:]) for k, v in obs.items()},
            bullets=(
                None
                if obs.bullets is None
                else {k: v.reshape(T * B, *v.shape[2:]) for k, v in obs.bullets.items()}
            ),
        )

        encoded = self.encoder(flat_obs)  # (T*B, N+G+M, D)
        encoded_sequence = encoded.reshape(T, B, NM, D)
        if self.map_memory_proj is not None:
            map_memory = self.map_memory_proj(encoded[:, Q:, :])  # (T*B, M, D_map)
            x = encoded_sequence[:, :, :Q, :]  # (T, B, N+G, D)
        else:
            map_memory = None
            x = encoded_sequence
        bullets, bullet_mask = self._encode_bullets(flat_obs)  # (T*B, N*K, D)
        geometry = self._spatial_geometry(
            flat_obs, Q, bullets is not None, self.map_memory_proj is not None
        )
        z = encoded_sequence if return_encoder_output else None

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
                    rglru_states[block_slice],
                    conv_bufs[block_slice],
                    done_mask,
                    n_rec,
                    bullets,
                    bullet_mask,
                    map_memory,
                    geometry,
                    use_reentrant=False,
                )
            else:
                x, _, _ = layer.sequence(
                    x,
                    rglru_states[block_slice],
                    conv_bufs[block_slice],
                    done_mask,
                    n_rec,
                    bullets,
                    bullet_mask,
                    map_memory,
                    geometry,
                )

        # Ship tokens for the per-ship heads; the global token is sliced by the
        # two heads whose subject is the game, at [N].
        x_ships = x[:, :, :N, :]  # (T, B, N, D)

        logits = self.action_head(x_ships)  # (T, B, N, 30)
        codes = self.coordinator.ship_codes(flat_obs, N).reshape(T, B, N, -1)
        pred_next = self.next_state_head(x_ships, codes)  # (T, B, N, 469) logits

        critic = self._critic(x, x_ships, return_logits=True)
        enemy_action_logits = (
            self.enemy_action_head(x_ships) if return_enemy_action else None
        )  # (T, B, N, 30) when requested

        logprob, entropy = _evaluate_action(logits, actions)

        base = (logprob, entropy, critic, logits, z, pred_next)
        if return_enemy_action:
            base = (*base, enemy_action_logits)
        if return_density:
            # The global token's own final embedding, the one thing no ship head
            # sees. G is one, so the token axis is squeezed out rather than kept.
            density = (
                None
                if self.density_head is None
                else self.density_head(x[:, :, N, :])  # (T, B, 2C)
            )
            base = (*base, density)
        if shot_ages is not None:
            if self.shot_heads is None:
                raise ValueError("shot_ages given but this policy has no shot heads")
            base = (
                *base,
                self.shot_heads.trajectory(x_ships, shot_ages),
                self.shot_heads.outcome(x_ships),
            )
        return base


# ---------------------------------------------------------------------------
# Checkpoint helper
# ---------------------------------------------------------------------------


def _yemong_forward(
    layer: YemongBlock,
    x: torch.Tensor,
    h0: torch.Tensor,
    conv_buf0: torch.Tensor,
    done_mask: torch.Tensor | None,
    num_recurrent: int,
    bullets: torch.Tensor | None,
    bullet_mask: torch.Tensor | None,
    map_memory: torch.Tensor | None,
    geometry: SpatialGeometry | None,
) -> torch.Tensor:
    """Run one Yemong block's full-sequence forward, returning only the output.

    Module-level (no ``self`` capture) so ``torch.utils.checkpoint`` can rematerialize
    it cleanly. The final hidden/conv states are unused by the update-time re-evaluation.
    """
    out, _, _ = layer.sequence(
        x,
        h0,
        conv_buf0,
        done_mask,
        num_recurrent,
        bullets,
        bullet_mask,
        map_memory,
        geometry,
    )
    return out


# ---------------------------------------------------------------------------
# Action sampling helpers (pure functions, no state)
# ---------------------------------------------------------------------------


def _sample_action(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one physical command per ship from the joint categorical.

    Args:
        logits: (..., 30) joint physical-command logits.

    Returns:
        action:  (..., 3) int — [power, turn, shoot].
        logprob: (...) float — joint-command log probability.
    """
    distribution = Categorical(logits=logits)
    action_id = distribution.sample()
    return decode_joint_action_unchecked(action_id), distribution.log_prob(action_id)


def _evaluate_action(
    logits: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute log-probs and entropy for given actions under the policy.

    Args:
        logits:  (..., 30) joint physical-command logits.
        actions: (..., 3) int.

    Returns:
        logprob: (...) float.
        entropy: (...) float.
    """
    distribution = Categorical(logits=logits)
    action_id = encode_joint_action_unchecked(actions)
    return distribution.log_prob(action_id), distribution.entropy()
