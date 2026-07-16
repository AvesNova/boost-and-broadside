"""MVPEnvWrapper: observation builder and episode manager around TensorEnv.

Responsibilities:
  - Convert TensorState into the raw obs dict consumed by MVPPolicy.
  - Concatenate ship and obstacle tokens into a single (B, N+M, ...) obs dict.
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
from boost_and_broadside.env.observation import (
    MVPObservation,
    ObservationBuffers,
    observation_from_state,
)
from boost_and_broadside.env.obstacle_cache import ObstacleCache
from boost_and_broadside.env.rewards import (
    REWARD_COMPONENT_NAMES,
    RewardComponent,
    build_reward_components,
)
from boost_and_broadside.env.state import TensorState


class MVPEnvWrapper:
    """Wraps TensorEnv to produce policy-ready observations and zero-sum rewards.

    MVPObservation keys and shapes (B = num_envs, N = num_ships, M = num_obstacles).
    All values are RAW — no normalization applied. All encoding decisions
    (Fourier expand, symlog, normalize, one-hot) live in FeatureCoordinator feature chains.

        "pos"             (B, N+M, 2)  — [x, y] raw pixels
        "vel"             (B, N+M, 2)  — [vx, vy] raw px/s
        "att"             (B, N+M, 2)  — [cos θ, sin θ] unit heading; vel_dir for obstacles
        "ang_vel"         (B, N+M, 1)  — rad/s; zeroed for obstacles
        "health"          (B, N+M, 1)  — raw [0, max_health]; obstacles = max_health
        "power"           (B, N+M, 1)  — raw [0, max_power]; obstacles = 0
        "cooldown"        (B, N+M, 1)  — raw [0, firing_cooldown]; obstacles = 0
        "team_id"         (B, N+M)     — int32; 0/1 for ships, 2 for obstacles
        "alive"           (B, N+M)     — bool; obstacles are always True
        "previous_action" (B, N+M, 3)  — int [power, turn, shoot]; zeroed for obstacles
        "radius"          (B, N+M, 1)  — raw px; collision_radius for ships, actual for obstacles

    All reward computations remain (B, N) — obstacle tokens are never in the reward signal.
    """

    def __init__(
        self,
        num_envs: int,
        ship_config: ShipConfig,
        env_config: EnvConfig,
        rewards: RewardConfig,
        device: str | torch.device,
        obstacle_cache: ObstacleCache | None = None,
    ) -> None:
        self.env = TensorEnv(num_envs, ship_config, env_config, device, obstacle_cache)
        self.ship_config = ship_config
        self.env_config = env_config
        self.device = torch.device(device)

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
            env_config.num_obstacles,
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
    ) -> MVPObservation:
        """Reset all environments and return initial observations."""
        self.env.reset(options=options, seed=seed)
        self._refresh_obs_radius_all()
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
        }
        self._zero_stat_accumulators()
        return stats

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        actions: torch.Tensor,
    ) -> tuple[MVPObservation, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Advance all environments and return (obs, rewards, dones, truncated, info).

        The wrapper snapshots health/alive before physics, computes rewards from
        the post-physics state, then resets done environments.

        Fully branchless on the GPU: episode stats for finished envs fold into
        on-device accumulators (see pop_episode_stats) instead of being copied
        to the CPU here, so a step never forces a host-device sync.

        Args:
            actions: (B, N, 3) int tensor — [power, turn, shoot].

        Returns:
            obs:          dict of (B, N, ...) tensors.
            comp_rewards: (B, N, K) float32 — per-component per-ship rewards (no zero-sum).
            dones:        (B,) bool — game-over (physics termination).
            truncated:    (B,) bool — episode length limit reached.
            info:         GPU terminal-team flags used by league outcome recording.
        """
        # Snapshot pre-physics state fields needed for reward delta
        prev_health = self.env.state.ship_health.clone()  # (B, N)
        prev_alive = self.env.state.ship_alive.clone()  # (B, N)
        prev_state = _make_prev_state_proxy(self.env.state, prev_health, prev_alive)

        # Physics step (no auto-reset)
        dones, truncated = self.env.step(actions)

        # Compute rewards for active components only — (B, N, K_active)
        B, N = self.env.state.ship_health.shape
        K = len(self._active_names)
        comp_rewards = torch.zeros(B, N, K, device=self.device, dtype=torch.float32)
        for k, comp in enumerate(self._active_components):
            comp_rewards[:, :, k] = comp.compute(prev_state, actions, self.env.state, dones)

        # Normalize all rewards by total ship count so reward scale is invariant
        # to game size across 1v1, 2v2, 4v4, etc. Win rewards are included: in 2v2
        # both allies each contribute +1, so without normalization the win signal
        # would be 2× stronger than in 1v1 after lambda aggregation.
        _n_ships = self.env_config.num_ships
        comp_rewards /= _n_ships

        # Accumulate per-episode trackers (active components only)
        self._ep_reward += comp_rewards.sum(dim=-1)
        self._ep_length += 1
        self._ship_age += prev_alive.int()  # freeze at death step
        self._ep_comp += comp_rewards
        self._ep_comp_scaled += comp_rewards * self._weight_t

        done_mask = dones | truncated

        # Win tracking — +1 for ships on the surviving team, 0 otherwise.
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

        # Reset done environments (state mutated in-place) and their trackers
        self.env.reset_envs(done_mask)
        self._refresh_obs_radius(done_mask)
        self._ep_reward.masked_fill_(done_n, 0.0)
        self._ep_length.masked_fill_(done_mask, 0)
        self._ep_comp.masked_fill_(done_mask.view(B, 1, 1), 0.0)
        self._ep_comp_scaled.masked_fill_(done_mask.view(B, 1, 1), 0.0)
        self._ep_wins.masked_fill_(done_n, 0.0)
        self._ship_age.masked_fill_(done_n, 0)

        info = {
            "team0_won": t0_wins.squeeze(1),
            "team1_won": t1_wins.squeeze(1),
        }
        return self._get_obs(), comp_rewards, dones, truncated, info

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------

    def _get_obs(self) -> MVPObservation:
        """Build the combined (ship + obstacle) raw observation as MVPObservation.

        All values are in native units — no normalization. Feature chains in
        FeatureCoordinator handle all encoding (Fourier, symlog, one-hot, etc.).
        """
        return observation_from_state(self.env.state, self.ship_config, self._obs_buffers)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _refresh_obs_radius_all(self) -> None:
        self._obs_buffers.refresh_obstacle_radius_all(self.env.state)

    def _refresh_obs_radius(self, mask: torch.Tensor) -> None:
        self._obs_buffers.refresh_obstacle_radius(self.env.state, mask)

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
    """
    return dataclasses.replace(
        state,
        ship_health=prev_health,
        ship_alive=prev_alive,
    )
