"""MVPEnvWrapper: observation builder and episode manager around TensorEnv.

Responsibilities:
  - Convert TensorState into the raw obs dict consumed by MVPPolicy.
  - Concatenate ship and obstacle tokens into a single (B, N+M, ...) obs dict.
  - Compute zero-sum rewards via the reward components.
  - Reset done / truncated environments and zero GRU hidden states.
  - Track per-ship episode statistics for logging.
"""

import torch
from typing import Any

_EPS = 1e-6  # division safety guard for direction normalization

from boost_and_broadside.config import ShipConfig, EnvConfig, RewardConfig
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.observation import MVPObservation, ObsKey
from boost_and_broadside.env.obstacle_cache import ObstacleCache
from boost_and_broadside.env.rewards import (
    RewardComponent,
    REWARD_COMPONENT_NAMES,
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
        self._all_components: list[RewardComponent] = build_reward_components(
            rewards, ship_config
        )

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

        # Pre-allocate static obstacle padding tensors (reused every _get_obs call)
        M = env_config.num_obstacles
        N = env_config.num_ships
        if M > 0:
            self._obs_ang_vel      = torch.zeros(num_envs, M, 1, device=self.device)
            self._obs_health       = torch.full((num_envs, M, 1), ship_config.max_health, device=self.device)
            self._obs_power        = torch.zeros(num_envs, M, 1, device=self.device)
            self._obs_cooldown     = torch.zeros(num_envs, M, 1, device=self.device)
            self._obs_team_id      = torch.full((num_envs, M), 2, device=self.device, dtype=torch.int32)
            self._obs_alive        = torch.ones(num_envs, M, device=self.device, dtype=torch.bool)
            self._obs_prev_action  = torch.zeros(num_envs, M, 3, device=self.device, dtype=torch.long)
            self._obs_radius       = torch.zeros(num_envs, M, 1, device=self.device)
        self._ship_radius = torch.full(
            (num_envs, N, 1),
            ship_config.collision_radius,
            device=self.device,
            dtype=torch.float32,
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
            info:         empty dict (episode stats moved to pop_episode_stats).
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
            comp_rewards[:, :, k] = comp.compute(
                prev_state, actions, self.env.state, dones
            )

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
        done_f = done_mask.float()        # (B,)
        done_n = done_mask.unsqueeze(1)   # (B, 1)
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
        self._acc_comp_scaled_sum += (
            self._ep_comp_scaled * done_nf.unsqueeze(-1)
        ).sum(dim=(0, 1))
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

        return self._get_obs(), comp_rewards, dones, truncated, {}

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------

    def _get_obs(self) -> MVPObservation:
        """Build the combined (ship + obstacle) raw observation as MVPObservation.

        All values are in native units — no normalization. Feature chains in
        FeatureCoordinator handle all encoding (Fourier, symlog, one-hot, etc.).
        """
        s = self.env.state
        M = s.num_obstacles

        # --- Ship features (raw) ---
        ship_pos     = torch.stack([s.ship_pos.real, s.ship_pos.imag], dim=-1)           # (B, N, 2)
        ship_vel     = torch.stack([s.ship_vel.real, s.ship_vel.imag], dim=-1)           # (B, N, 2)
        ship_att     = torch.stack([s.ship_attitude.real, s.ship_attitude.imag], dim=-1) # (B, N, 2)
        ship_ang     = s.ship_ang_vel.unsqueeze(-1)                                       # (B, N, 1)
        ship_health  = s.ship_health.unsqueeze(-1)                                        # (B, N, 1)
        ship_power   = s.ship_power.unsqueeze(-1)                                         # (B, N, 1)
        ship_cooldown = s.ship_cooldown.unsqueeze(-1)                                     # (B, N, 1)
        ship_prev_action = s.prev_action.long()                                           # (B, N, 3)

        if M > 0:
            obs_pos = torch.stack([s.obstacle_pos.real, s.obstacle_pos.imag], dim=-1)  # (B, M, 2)
            obs_vel = torch.stack([s.obstacle_vel.real, s.obstacle_vel.imag], dim=-1)  # (B, M, 2)
            obs_speed = torch.norm(obs_vel, dim=-1, keepdim=True).clamp(min=_EPS)
            obs_att = obs_vel / obs_speed  # (B, M, 2) — unit heading = velocity direction
            return MVPObservation(data={
                ObsKey.POS:             torch.cat([ship_pos,          obs_pos],              dim=1),
                ObsKey.VEL:             torch.cat([ship_vel,          obs_vel],              dim=1),
                ObsKey.ATT:             torch.cat([ship_att,          obs_att],              dim=1),
                ObsKey.ANG_VEL:         torch.cat([ship_ang,          self._obs_ang_vel],    dim=1),
                ObsKey.HEALTH:          torch.cat([ship_health,       self._obs_health],     dim=1),
                ObsKey.POWER:           torch.cat([ship_power,        self._obs_power],      dim=1),
                ObsKey.COOLDOWN:        torch.cat([ship_cooldown,     self._obs_cooldown],   dim=1),
                ObsKey.TEAM_ID:         torch.cat([s.ship_team_id,    self._obs_team_id],    dim=1),
                ObsKey.ALIVE:           torch.cat([s.ship_alive,      self._obs_alive],      dim=1),
                ObsKey.PREVIOUS_ACTION: torch.cat([ship_prev_action,  self._obs_prev_action], dim=1),
                ObsKey.RADIUS:          torch.cat([self._ship_radius, self._obs_radius],     dim=1),
            })

        return MVPObservation(data={
            ObsKey.POS:             ship_pos,
            ObsKey.VEL:             ship_vel,
            ObsKey.ATT:             ship_att,
            ObsKey.ANG_VEL:         ship_ang,
            ObsKey.HEALTH:          ship_health,
            ObsKey.POWER:           ship_power,
            ObsKey.COOLDOWN:        ship_cooldown,
            ObsKey.TEAM_ID:         s.ship_team_id,
            ObsKey.ALIVE:           s.ship_alive,
            ObsKey.PREVIOUS_ACTION: ship_prev_action,
            ObsKey.RADIUS:          self._ship_radius,
        })

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _refresh_obs_radius_all(self) -> None:
        if self.env_config.num_obstacles > 0:
            self._obs_radius.copy_(self.env.state.obstacle_radius.unsqueeze(-1))

    def _refresh_obs_radius(self, mask: torch.Tensor) -> None:
        if self.env_config.num_obstacles > 0:
            self._obs_radius.copy_(
                torch.where(
                    mask.view(-1, 1, 1),
                    self.env.state.obstacle_radius.unsqueeze(-1),
                    self._obs_radius,
                )
            )

    @property
    def state(self) -> TensorState:
        """Direct access to the underlying physics state."""
        return self.env.state

    @property
    def active_names(self) -> list[str]:
        """Reward component names that are active (weight != 0), in canonical order."""
        return self._active_names

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
    return TensorState(
        step_count=state.step_count,
        ship_pos=state.ship_pos,
        ship_vel=state.ship_vel,
        ship_attitude=state.ship_attitude,
        ship_ang_vel=state.ship_ang_vel,
        ship_health=prev_health,
        ship_power=state.ship_power,
        ship_cooldown=state.ship_cooldown,
        ship_team_id=state.ship_team_id,
        ship_alive=prev_alive,
        ship_is_shooting=state.ship_is_shooting,
        prev_action=state.prev_action,
        bullet_pos=state.bullet_pos,
        bullet_vel=state.bullet_vel,
        bullet_time=state.bullet_time,
        bullet_active=state.bullet_active,
        bullet_cursor=state.bullet_cursor,
        damage_matrix=state.damage_matrix,
        cumulative_damage_matrix=state.cumulative_damage_matrix,
        obstacle_pos=state.obstacle_pos,
        obstacle_vel=state.obstacle_vel,
        obstacle_radius=state.obstacle_radius,
        obstacle_gcenter=state.obstacle_gcenter,
        ship_hit_obstacle=state.ship_hit_obstacle,
    )
