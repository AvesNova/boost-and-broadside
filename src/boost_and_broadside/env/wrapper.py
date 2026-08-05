"""YemongEnvWrapper: observation builder and episode manager around TensorEnv.

Responsibilities:
  - Convert TensorState into the raw obs dict consumed by YemongPolicy.
  - Concatenate ship and refractive-field tokens into one (B, N+M, ...) obs dict.
  - Optionally attach the bullet cross-attention axis, (B, N*K, ...), when the
    policy reads it (include_bullets).
  - Compute per-ship per-component rewards via the reward components
    (zero-sum accounting happens later, in PPO's lambda aggregation).
  - Reset done / truncated environments and zero GRU hidden states.
  - Track per-ship episode statistics for logging.
"""

import dataclasses
from typing import Any

import torch

from boost_and_broadside.config import EnvConfig, RewardConfig, ShipConfig
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.field_cache import FieldMapCache
from boost_and_broadside.env.observation import (
    ObservationBuffers,
    YemongObservation,
    observation_from_state,
)
from boost_and_broadside.env.rewards import (
    REWARD_COMPONENT_NAMES,
    RewardComponent,
    build_reward_components,
)
from boost_and_broadside.env.state import TensorState


class YemongEnvWrapper:
    """Wraps TensorEnv to produce policy-ready observations and zero-sum rewards.

    YemongObservation keys and shapes (B = num_envs, N = num_ships, M = num_fields).
    All values are RAW — no normalization applied. All encoding decisions
    (Fourier expand, symlog, normalize, one-hot) live in FeatureCoordinator feature chains.

        "pos"             (B, N+M, 2)  — [x, y] raw pixels
        "vel"             (B, N+M, 2)  — [vx, vy] raw px/s
        "att"             (B, N+M, 2)  — [cos θ, sin θ]; zero for fields
        "ang_vel"         (B, N+M, 1)  — rad/s; zero for fields
        "health"          (B, N+M, 1)  — raw [0, max_health]; fields = max_health
        "power"           (B, N+M, 1)  — raw [0, max_power]; fields = 0
        "cooldown"        (B, N+M, 1)  — raw seconds; fields = 0
        "team_id"         (B, N+M)     — int32; 0/1 for ships, 2 for fields
        "alive"           (B, N+M)     — bool; fields are always True
        "previous_action" (B, N+M, 3)  — int actions; zero for fields
        "radius"          (B, N+M, 1)  — raw px; ship collision or nominal field radius
        "local_index_gradient" (B, N+M, 2) — normalized grad(n); zero for fields
        field material     (B, N+M, 1)  — numeric width/index-ratio/damage channels

    All reward computations remain (B, N) — field tokens are never reward recipients.
    """

    def __init__(
        self,
        num_envs: int,
        ship_config: ShipConfig,
        env_config: EnvConfig,
        rewards: RewardConfig,
        device: str | torch.device,
        field_map: FieldMapCache | None = None,
        collision_compile_mode: str | None = None,
        include_bullets: bool = False,
    ) -> None:
        self.env = TensorEnv(
            num_envs,
            ship_config,
            env_config,
            device,
            field_map,
            collision_compile_mode,
        )
        self.ship_config = ship_config
        self.env_config = env_config
        self.device = torch.device(device)
        # Attach the bullet cross-attention axis only when the policy reads it —
        # otherwise the profile pays the reduction and the rollout storage for
        # channels nothing consumes.
        self.include_bullets = include_bullets

        # All components (group-scale multipliers update individual weights each training step).
        self._all_components: list[RewardComponent] = build_reward_components(rewards, ship_config)

        # Active components: weight != 0 and registered in REWARD_COMPONENT_NAMES,
        # in canonical REWARD_COMPONENT_NAMES order.
        _comp_by_name = {c.name: c for c in self._all_components}
        self._active_names: list[str] = [
            name
            for name in REWARD_COMPONENT_NAMES
            if name in _comp_by_name and _comp_by_name[name].weight != 0
        ]
        self._active_components: list[RewardComponent] = [
            _comp_by_name[name] for name in self._active_names
        ]

        self._obs_buffers = ObservationBuffers.allocate(
            num_envs,
            env_config.num_ships,
            env_config.num_fields,
            ship_config,
            self.device,
        )

        # Per-episode trackers — active components only. Components are stored as
        # one (B, N, K) tensor (not a per-name dict) so per-step accumulation is
        # a single kernel.
        B, N = num_envs, env_config.num_ships
        K_active = len(self._active_names)
        self._ep_reward = torch.zeros((B, N), device=self.device)
        self._ep_length = torch.zeros((B,), device=self.device, dtype=torch.int32)
        self._ep_comp = torch.zeros((B, N, K_active), device=self.device)
        # Scaled rewards: raw compute output × (individual_weight × group_scale).
        # comp.weight is mutated each update step by ppo.py; the trainer must call
        # refresh_component_weights() afterwards to re-sync the cached tensor.
        self._ep_comp_scaled = torch.zeros((B, N, K_active), device=self.device)
        # Win flag: +1 for ships on the surviving team, 0 otherwise (draws = 0).
        self._ep_wins = torch.zeros((B, N), device=self.device)
        # Steps each ship has been alive this episode (stops at death, resets on episode end).
        self._ship_age = torch.zeros((B, N), device=self.device, dtype=torch.int32)

        self.refresh_component_weights()
        self._zero_stat_accumulators()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        options: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> YemongObservation:
        """Reset all environments and return initial observations."""
        self.env.reset(options=options, seed=seed)
        self._refresh_field_obs_all()
        self._ep_reward.zero_()
        self._ep_length.zero_()
        self._ep_comp.zero_()
        self._ep_comp_scaled.zero_()
        self._ep_wins.zero_()
        self._ship_age.zero_()
        self._zero_stat_accumulators()
        return self._get_obs()

    # ------------------------------------------------------------------
    # Episode statistics (GPU-accumulated, flushed once per update)
    # ------------------------------------------------------------------

    def refresh_component_weights(self) -> None:
        """Re-sync the cached (K,) weight tensor from the active components.

        Must be called after mutating component weights (ppo.py does this once
        per update when applying schedule group scales).
        """
        self._weight_t = torch.tensor(
            [c.weight for c in self._active_components],
            device=self.device,
            dtype=torch.float32,
        )

    def _zero_stat_accumulators(self) -> None:
        d = self.device
        K = len(self._active_names)
        self._acc_episodes = torch.zeros((), device=d)
        self._acc_reward_sum = torch.zeros((), device=d)
        self._acc_reward_min = torch.full((), float("inf"), device=d)
        self._acc_reward_max = torch.full((), float("-inf"), device=d)
        self._acc_length_sum = torch.zeros((), device=d)
        self._acc_comp_sum = torch.zeros((K,), device=d)
        self._acc_comp_scaled_sum = torch.zeros((K,), device=d)
        self._acc_wins_sum = torch.zeros((), device=d)
        self._acc_lifespan_sum = torch.zeros((), device=d)
        # field damage, combat damage, field deaths, combat deaths,
        # field-damage steps, non-ambient live steps, total live steps.
        self._acc_source_stats = torch.zeros((7,), device=d)

    def pop_episode_stats(self) -> dict[str, torch.Tensor]:
        """Return finished-episode stats accumulated since the last call, and reset.

        All values are device tensors — the caller decides when to synchronize
        (ppo.py does so once per update). Keys:
            episodes:         () — number of finished env-episodes.
            reward_sum:       () — total reward over finished ship-episodes.
            reward_min/max:   () — extremes over finished ship-episodes
                              (±inf when episodes == 0).
            length_sum:       () — total episode length (per env-episode).
            comp_sum:         (K,) — per-component reward sums (ship-episodes).
            comp_scaled_sum:  (K,) — same, scaled by component weights.
            wins_sum:         () — total win flags over finished ship-episodes.
            lifespan_sum:     () — total ship lifespans (steps alive).
        """
        stats = {
            "episodes": self._acc_episodes,
            "reward_sum": self._acc_reward_sum,
            "reward_min": self._acc_reward_min,
            "reward_max": self._acc_reward_max,
            "length_sum": self._acc_length_sum,
            "comp_sum": self._acc_comp_sum,
            "comp_scaled_sum": self._acc_comp_scaled_sum,
            "wins_sum": self._acc_wins_sum,
            "lifespan_sum": self._acc_lifespan_sum,
            "source_stats": self._acc_source_stats,
        }
        self._zero_stat_accumulators()
        return stats

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        actions: torch.Tensor,
        *,
        unlimited_resources: bool = False,
    ) -> tuple[YemongObservation, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Advance all environments and return (obs, rewards, dones, truncated, info).

        The wrapper snapshots health/alive before physics, computes rewards from
        the post-physics state, then resets done environments.

        Fully branchless on the GPU: episode stats for finished envs fold into
        on-device accumulators (see pop_episode_stats) instead of being copied
        to the CPU here, so a step never forces a host-device sync.

        Args:
            actions: (B, N, 3) int tensor — [power, turn, shoot].
            unlimited_resources: Protect and refill alive ships for interactive play.

        One call is one *decision*: the action is held for ``action_repeat``
        physics ticks. Physics, collisions and projectile integration always run
        at ``ShipConfig.dt``, so the simulation is unchanged — only the rate at
        which the policy may change its mind moves.

        Rewards are summed across the held ticks, which is scale-preserving:
        over a fixed span of game time both the dense per-tick terms and the
        one-off event terms total exactly what they would at repeat 1, so the
        component ratios that RewardConfig sets are untouched.

        An environment that finishes partway through the hold stops contributing
        — its rewards, its win flag and its episode statistics are all frozen at
        that tick. It keeps being simulated for the remainder (masking the
        physics would cost more than the wasted ticks) but nothing it produces
        afterwards is read, and it is reset once at the end.

        Args:
            actions: (B, N, 3) int tensor — [power, turn, shoot].
            unlimited_resources: Protect and refill alive ships for interactive play.

        Returns:
            obs:          dict of (B, N, ...) tensors.
            comp_rewards: (B, N, K) float32 — per-component per-ship rewards
                          summed over the held ticks (no zero-sum).
            dones:        (B,) bool — game-over (physics termination).
            truncated:    (B,) bool — episode length limit reached.
            info:         empty dict (episode stats moved to pop_episode_stats).
        """
        B, N = self.env.state.ship_health.shape
        K = len(self._active_names)
        comp_rewards = torch.zeros(B, N, K, device=self.device, dtype=torch.float32)
        dones = torch.zeros(B, dtype=torch.bool, device=self.device)
        truncated = torch.zeros(B, dtype=torch.bool, device=self.device)

        for _ in range(self.env_config.action_repeat):
            # Envs that already finished earlier in this hold contribute nothing.
            running = ~(dones | truncated)
            tick_dones, tick_truncated = self._physics_tick(
                actions, comp_rewards, running, unlimited_resources
            )
            dones = dones | (tick_dones & running)
            truncated = truncated | (tick_truncated & running)

        done_mask = dones | truncated
        done_n = done_mask.unsqueeze(1)

        # Reset done environments (state mutated in-place) and their trackers
        self.env.reset_envs(done_mask)
        self._refresh_field_obs(done_mask)
        self._ep_reward.masked_fill_(done_n, 0.0)
        self._ep_length.masked_fill_(done_mask, 0)
        self._ep_comp.masked_fill_(done_mask.view(B, 1, 1), 0.0)
        self._ep_comp_scaled.masked_fill_(done_mask.view(B, 1, 1), 0.0)
        self._ep_wins.masked_fill_(done_n, 0.0)
        self._ship_age.masked_fill_(done_n, 0)

        return self._get_obs(), comp_rewards, dones, truncated, {}

    def _physics_tick(
        self,
        actions: torch.Tensor,
        comp_rewards: torch.Tensor,
        running: torch.Tensor,
        unlimited_resources: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance one physics tick, accumulating rewards and episode statistics.

        Fully branchless on the GPU: episode stats for finished envs fold into
        on-device accumulators (see pop_episode_stats) instead of being copied
        to the CPU here, so a tick never forces a host-device sync.

        Args:
            actions: (B, N, 3) — the held action.
            comp_rewards: (B, N, K) accumulator, added to in place.
            running: (B,) bool — envs still live in this decision's hold.
            unlimited_resources: Interactive-play protection toggle.

        Returns:
            (dones, truncated) for this tick, before the running mask.
        """
        # Snapshot pre-physics state fields needed for reward delta
        prev_health = self.env.state.ship_health.clone()  # (B, N)
        prev_alive = self.env.state.ship_alive.clone()  # (B, N)
        prev_state = _make_prev_state_proxy(self.env.state, prev_health, prev_alive)

        # Physics step (no auto-reset)
        dones, truncated = self.env.step(
            actions,
            unlimited_resources=unlimited_resources,
        )

        running_n = running.unsqueeze(1)  # (B, 1)

        source_state = self.env.state
        live = prev_alive & running_n
        self._acc_source_stats += torch.stack(
            [
                (source_state.ship_field_damage * running_n).sum(),
                (source_state.ship_combat_damage * running_n).sum(),
                (source_state.ship_field_death & running_n).sum(),
                (source_state.ship_combat_death & running_n).sum(),
                ((source_state.ship_field_damage > 0.0) & running_n).sum(),
                ((source_state.ship_local_index - 1.0).abs() > 1e-6).logical_and(live).sum(),
                live.sum(),
            ]
        )

        # Compute rewards for active components only — (B, N, K_active)
        tick_rewards = torch.zeros_like(comp_rewards)
        for k, comp in enumerate(self._active_components):
            tick_rewards[:, :, k] = comp.compute(prev_state, actions, self.env.state, dones)

        # Normalize all rewards by total ship count so reward scale is invariant
        # to game size across 1v1, 2v2, 4v4, etc. Win rewards are included: in 2v2
        # both allies each contribute +1, so without normalization the win signal
        # would be 2× stronger than in 1v1 after lambda aggregation.
        tick_rewards /= self.env_config.num_ships
        tick_rewards *= running_n.unsqueeze(-1)  # finished envs stop earning
        comp_rewards += tick_rewards

        # Accumulate per-episode trackers (active components only). Lengths and
        # ages are in physics ticks, so they stay comparable across action_repeat.
        self._ep_reward += tick_rewards.sum(dim=-1)
        self._ep_length += running.int()
        self._ship_age += (prev_alive & running_n).int()  # freeze at death step
        self._ep_comp += tick_rewards
        self._ep_comp_scaled += tick_rewards * self._weight_t

        # Only envs finishing on *this* tick fold into the per-update stats; an
        # env that ended earlier in the hold was already counted.
        done_mask = (dones | truncated) & running

        # Win tracking — +1 for ships on the surviving team, 0 otherwise. Read at
        # the tick the env finished, so extra held ticks cannot rewrite the result.
        s = self.env.state
        team0 = s.ship_team_id == 0  # (B, N)
        team1 = s.ship_team_id == 1  # (B, N)
        t0_alive = (team0 & s.ship_alive).sum(dim=1)  # (B,)
        t1_alive = (team1 & s.ship_alive).sum(dim=1)  # (B,)
        t0_wins = ((t0_alive > 0) & (t1_alive == 0) & done_mask).unsqueeze(1)
        t1_wins = ((t1_alive > 0) & (t0_alive == 0) & done_mask).unsqueeze(1)
        self._ep_wins += ((team0 & t0_wins) | (team1 & t1_wins)).float()

        # Fold finished episodes into the per-update accumulators.
        done_f = done_mask.float()  # (B,)
        done_n = done_mask.unsqueeze(1)  # (B, 1)
        done_nf = done_n.float()
        self._acc_episodes += done_f.sum()
        self._acc_reward_sum += (self._ep_reward * done_nf).sum()
        self._acc_reward_min = torch.minimum(
            self._acc_reward_min,
            torch.where(done_n, self._ep_reward, float("inf")).min(),
        )
        self._acc_reward_max = torch.maximum(
            self._acc_reward_max,
            torch.where(done_n, self._ep_reward, float("-inf")).max(),
        )
        self._acc_length_sum += (self._ep_length.float() * done_f).sum()
        self._acc_comp_sum += (self._ep_comp * done_nf.unsqueeze(-1)).sum(dim=(0, 1))
        self._acc_comp_scaled_sum += (self._ep_comp_scaled * done_nf.unsqueeze(-1)).sum(dim=(0, 1))
        self._acc_wins_sum += (self._ep_wins * done_nf).sum()
        self._acc_lifespan_sum += (self._ship_age.float() * done_nf).sum()

        return dones, truncated

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------

    def _get_obs(self) -> YemongObservation:
        """Build the combined (ship + field) raw observation as YemongObservation.

        All values are in native units — no normalization. Feature chains in
        FeatureCoordinator handle all encoding (Fourier, symlog, one-hot, etc.).
        """
        return observation_from_state(
            self.env.state,
            self.ship_config,
            self._obs_buffers,
            include_bullets=self.include_bullets,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _refresh_field_obs_all(self) -> None:
        self._obs_buffers.refresh_field_state_all(self.env.state)

    def _refresh_field_obs(self, mask: torch.Tensor) -> None:
        self._obs_buffers.refresh_field_state(self.env.state, mask)

    @property
    def state(self) -> TensorState:
        """Direct access to the underlying physics state."""
        return self.env.state

    @property
    def active_names(self) -> list[str]:
        """Reward component names that are active (weight != 0), in canonical order."""
        return self._active_names

    @property
    def reward_components(self) -> tuple[RewardComponent, ...]:
        """All configured reward components in canonical order."""
        return tuple(self._all_components)

    @property
    def active_components(self) -> tuple[RewardComponent, ...]:
        """Active reward components in canonical order."""
        return tuple(self._active_components)

    @property
    def component_weights(self) -> torch.Tensor:
        """Cached weights for the active reward components."""
        return self._weight_t

    @property
    def num_active_components(self) -> int:
        """Number of active reward components (= K, value head width)."""
        return len(self._active_names)

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    @property
    def num_ships(self) -> int:
        return self.env_config.num_ships


def _make_prev_state_proxy(
    state: TensorState,
    prev_health: torch.Tensor,
    prev_alive: torch.Tensor,
) -> TensorState:
    """Lightweight snapshot: shares all tensors but swaps in pre-damage health/alive.

    This avoids a full state clone while giving reward components the correct
    delta (health before → health after damage).

    Invariant (see TensorState): every non-swapped field aliases the live,
    already-advanced state, so this snapshot is correct only because physics
    advances fields by reassignment (never in-place). Reward components must read
    *only* `ship_health`/`ship_alive` from the returned proxy; reading any other
    field here would see post-step values with no error. A new component needing a
    genuine pre-step value for another field must take a full `.clone()` instead.
    """
    return dataclasses.replace(
        state,
        ship_health=prev_health,
        ship_alive=prev_alive,
    )
