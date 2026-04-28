"""MVPEnvWrapper: observation builder and episode manager around TensorEnv.

Responsibilities:
  - Convert TensorState into the float obs dict consumed by MVPPolicy.
  - Concatenate ship and obstacle tokens into a single (B, N+M, ...) obs dict.
  - Compute zero-sum rewards via the reward components.
  - Reset done / truncated environments and zero GRU hidden states.
  - Track per-ship episode statistics for logging.
"""

import torch
from typing import Any

from boost_and_broadside.config import ShipConfig, EnvConfig, RewardConfig
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.obstacle_cache import ObstacleCache
from boost_and_broadside.env.rewards import (
    RewardComponent,
    REWARD_COMPONENT_NAMES,
    build_reward_components,
)
from boost_and_broadside.env.state import TensorState


class MVPEnvWrapper:
    """Wraps TensorEnv to produce policy-ready observations and zero-sum rewards.

    Obs dict keys and shapes (B = num_envs, N = num_ships, M = num_obstacles):
        "pos"         (B, N+M, 2)  — x, y in [0, 1] (normalized by world_size)
        "vel"         (B, N+M, 2)  — vx, vy (raw; zeroed for obstacles)
        "att"         (B, N+M, 2)  — cos/sin of heading (zeroed for obstacles)
        "ang_vel"     (B, N+M, 1)  — angular velocity (zeroed for obstacles)
        "scalars"     (B, N+M, 3)  — [health, power, cooldown] normed (zeroed for obstacles)
        "team_id"     (B, N+M)     — int32; 0/1 for ships, 2 for obstacles
        "alive"       (B, N+M)     — bool; obstacles are always True
        "prev_action" (B, N+M, 3)  — int [power, turn, shoot] (zeroed for obstacles)
        "radius"      (B, N+M, 1)  — normalized radius (collision_radius for ships, actual for obstacles)

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
            self._obs_zeros_2 = torch.zeros(num_envs, M, 2, device=self.device)
            self._obs_zeros_1 = torch.zeros(num_envs, M, 1, device=self.device)
            self._obs_zeros_3 = torch.zeros(num_envs, M, 3, device=self.device)
            self._obs_team_id = torch.full((num_envs, M), 2, device=self.device, dtype=torch.int32)
            self._obs_alive   = torch.ones(num_envs, M, device=self.device, dtype=torch.bool)
            self._obs_action  = torch.zeros(num_envs, M, 3, device=self.device, dtype=torch.long)
            self._obs_radius  = torch.zeros(num_envs, M, 1, device=self.device)
        self._ship_radius = torch.full(
            (num_envs, N, 1),
            ship_config.collision_radius / ship_config.obstacle_radius_max,
            device=self.device,
            dtype=torch.float32,
        )

        # Episode stat accumulators — active components only
        B, N = num_envs, env_config.num_ships
        self._ep_reward = torch.zeros((B, N), device=self.device)
        self._ep_length = torch.zeros((B,), device=self.device, dtype=torch.int32)
        self._ep_reward_components: dict[str, torch.Tensor] = {
            name: torch.zeros((B, N), device=self.device) for name in self._active_names
        }
        # Scaled rewards: raw compute output × (individual_weight × group_scale).
        # comp.weight is mutated each update step by ppo.py to include the group scale.
        self._ep_scaled_reward_components: dict[str, torch.Tensor] = {
            name: torch.zeros((B, N), device=self.device) for name in self._active_names
        }
        # Win flag: +1 for alive ships on the surviving team, 0 otherwise (draws = 0).
        self._ep_wins = torch.zeros((B, N), device=self.device)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        options: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Reset all environments and return initial observations."""
        self.env.reset(options=options, seed=seed)
        self._refresh_obs_radius_all()
        self._ep_reward.zero_()
        self._ep_length.zero_()
        for t in self._ep_reward_components.values():
            t.zero_()
        for t in self._ep_scaled_reward_components.values():
            t.zero_()
        self._ep_wins.zero_()
        return self._get_obs()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        actions: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Advance all environments and return (obs, rewards, dones, truncated, info).

        The wrapper snapshots health/alive before physics, computes rewards from
        the post-physics state, then resets done environments.

        Args:
            actions: (B, N, 3) int tensor — [power, turn, shoot].

        Returns:
            obs:          dict of (B, N, ...) tensors.
            comp_rewards: (B, N, K) float32 — per-component per-ship rewards (no zero-sum).
            dones:        (B,) bool — game-over (physics termination).
            truncated:    (B,) bool — episode length limit reached.
            info:         dict with optional "ep_reward" and "ep_length" for done envs.
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

        # Accumulate episode stats (active components only)
        self._ep_reward += comp_rewards.sum(dim=-1)
        self._ep_length += 1
        for k, name in enumerate(self._active_names):
            self._ep_reward_components[name] += comp_rewards[:, :, k]
            self._ep_scaled_reward_components[name] += (
                comp_rewards[:, :, k] * self._active_components[k].weight
            )

        # Collect episode info for done envs before resetting
        done_mask = dones | truncated
        info: dict = {}
        if done_mask.any():
            # Win tracking — +1 for alive ships on the surviving team, 0 otherwise.
            s = self.env.state
            team0 = s.ship_team_id == 0  # (B, N)
            team1 = s.ship_team_id == 1  # (B, N)
            t0_alive = (team0 & s.ship_alive).sum(dim=1)  # (B,)
            t1_alive = (team1 & s.ship_alive).sum(dim=1)  # (B,)
            t0_wins = ((t0_alive > 0) & (t1_alive == 0) & done_mask).unsqueeze(1)
            t1_wins = ((t1_alive > 0) & (t0_alive == 0) & done_mask).unsqueeze(1)
            self._ep_wins[team0 & t0_wins.expand_as(team0)] += 1.0
            self._ep_wins[team1 & t1_wins.expand_as(team1)] += 1.0

            info["ep_reward"] = self._ep_reward[done_mask].detach().cpu()
            info["ep_length"] = self._ep_length[done_mask].detach().cpu()
            info["ep_reward_components"] = {
                name: t[done_mask].detach().cpu()
                for name, t in self._ep_reward_components.items()
            }
            info["ep_scaled_reward_components"] = {
                name: t[done_mask].detach().cpu()
                for name, t in self._ep_scaled_reward_components.items()
            }
            info["ep_wins"] = self._ep_wins[done_mask].detach().cpu()  # (done_envs, N)

        # Reset done environments (state mutated in-place)
        if done_mask.any():
            self.env.reset_envs(done_mask)
            self._refresh_obs_radius(done_mask)
            self._ep_reward[done_mask] = 0.0
            self._ep_length[done_mask] = 0
            for t in self._ep_reward_components.values():
                t[done_mask] = 0.0
            for t in self._ep_scaled_reward_components.values():
                t[done_mask] = 0.0
            self._ep_wins[done_mask] = 0.0

        return self._get_obs(), comp_rewards, dones, truncated, info

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------

    def _get_obs(self) -> dict[str, torch.Tensor]:
        """Build the combined (ship + obstacle) float observation dict."""
        s = self.env.state
        world_w, world_h = self.ship_config.world_size
        M = s.num_obstacles

        # --- Ship features ---
        ship_pos = torch.stack(
            [s.ship_pos.real / world_w, s.ship_pos.imag / world_h], dim=-1
        )  # (B, N, 2)
        ship_vel = torch.stack([s.ship_vel.real, s.ship_vel.imag], dim=-1)  # (B, N, 2)
        ship_att = torch.stack([s.ship_attitude.real, s.ship_attitude.imag], dim=-1)  # (B, N, 2)
        ship_ang = s.ship_ang_vel.unsqueeze(-1)  # (B, N, 1)
        ship_scalars = torch.stack(
            [
                s.ship_health / self.ship_config.max_health,
                s.ship_power / self.ship_config.max_power,
                (s.ship_cooldown / self.ship_config.firing_cooldown).clamp(0.0, 1.0),
            ],
            dim=-1,
        )  # (B, N, 3)

        # --- Obstacle features (ship-specific fields use pre-allocated zero/const tensors) ---
        if M > 0:
            obs_pos = torch.stack(
                [s.obstacle_pos.real / world_w, s.obstacle_pos.imag / world_h], dim=-1
            )  # (B, M, 2)
            return {
                "pos":         torch.cat([ship_pos, obs_pos], dim=1),                    # (B, N+M, 2)
                "vel":         torch.cat([ship_vel, self._obs_zeros_2], dim=1),
                "att":         torch.cat([ship_att, self._obs_zeros_2], dim=1),
                "ang_vel":     torch.cat([ship_ang, self._obs_zeros_1], dim=1),
                "scalars":     torch.cat([ship_scalars, self._obs_zeros_3], dim=1),
                "team_id":     torch.cat([s.ship_team_id, self._obs_team_id], dim=1),
                "alive":       torch.cat([s.ship_alive, self._obs_alive], dim=1),
                "prev_action": torch.cat([s.prev_action.long(), self._obs_action], dim=1),
                "radius":      torch.cat([self._ship_radius, self._obs_radius], dim=1),  # (B, N+M, 1)
            }

        return {
            "pos":         ship_pos,
            "vel":         ship_vel,
            "att":         ship_att,
            "ang_vel":     ship_ang,
            "scalars":     ship_scalars,
            "team_id":     s.ship_team_id,
            "alive":       s.ship_alive,
            "prev_action": s.prev_action.long(),
            "radius":      self._ship_radius,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _refresh_obs_radius_all(self) -> None:
        if self.env_config.num_obstacles > 0:
            self._obs_radius.copy_(
                (self.env.state.obstacle_radius / self.ship_config.obstacle_radius_max).unsqueeze(-1)
            )

    def _refresh_obs_radius(self, mask: torch.Tensor) -> None:
        if self.env_config.num_obstacles > 0:
            self._obs_radius[mask] = (
                self.env.state.obstacle_radius[mask] / self.ship_config.obstacle_radius_max
            ).unsqueeze(-1)

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
