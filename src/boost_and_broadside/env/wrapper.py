"""MVPEnvWrapper: observation builder and episode manager around TensorEnv.

Responsibilities:
  - Convert TensorState into the float obs dict consumed by MVPPolicy.
  - Compute zero-sum rewards via the reward components.
  - Reset done / truncated environments and zero GRU hidden states.
  - Track per-ship episode statistics for logging.
"""

import torch
from typing import Any

from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config import ShipConfig, EnvConfig, RewardConfig
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.rewards import (
    RewardComponent,
    REWARD_COMPONENT_NAMES,
    build_reward_components,
)
from boost_and_broadside.env.state import TensorState


class MVPEnvWrapper:
    """Wraps TensorEnv to produce policy-ready observations and zero-sum rewards.

    Obs dict keys and shapes (B = num_envs, N = num_ships):
        "pos"         (B, N, 2)  — x, y in [0, 1] (normalized by world_size)
        "vel"         (B, N, 2)  — vx, vy (raw)
        "att"         (B, N, 2)  — cos/sin of heading (unit vector components)
        "ang_vel"     (B, N, 1)  — angular velocity
        "scalars"     (B, N, 3)  — [health_norm, power_norm, cooldown_norm]
        "team_id"     (B, N)     — int32 team identifier
        "alive"       (B, N)     — bool
        "prev_action" (B, N, 3)  — int [power, turn, shoot] from last step
    """

    def __init__(
        self,
        num_envs: int,
        ship_config: ShipConfig,
        env_config: EnvConfig,
        reward_config: RewardConfig,
        device: str | torch.device,
        scripted_agent: StochasticScriptedAgent | None = None,
    ) -> None:
        self.env = TensorEnv(num_envs, ship_config, env_config, device)
        self.ship_config = ship_config
        self.env_config = env_config
        self.device = torch.device(device)

        # All components (decay scheduler and other callers may need access to inactive ones)
        self._all_components: list[RewardComponent] = build_reward_components(
            reward_config, ship_config, scripted_agent
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

        # Episode stat accumulators — active components only
        B, N = num_envs, env_config.num_ships
        self._ep_reward = torch.zeros((B, N), device=self.device)
        self._ep_length = torch.zeros((B,), device=self.device, dtype=torch.int32)
        self._ep_reward_components: dict[str, torch.Tensor] = {
            name: torch.zeros((B, N), device=self.device) for name in self._active_names
        }

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
        self._ep_reward.zero_()
        self._ep_length.zero_()
        for t in self._ep_reward_components.values():
            t.zero_()
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

        # Accumulate episode stats (active components only)
        self._ep_reward += comp_rewards.sum(dim=-1)
        self._ep_length += 1
        for k, name in enumerate(self._active_names):
            self._ep_reward_components[name] += comp_rewards[:, :, k]

        # Collect episode info for done envs before resetting
        done_mask = dones | truncated
        info: dict = {}
        if done_mask.any():
            info["ep_reward"] = self._ep_reward[done_mask].detach().cpu()
            info["ep_length"] = self._ep_length[done_mask].detach().cpu()
            info["ep_reward_components"] = {
                name: t[done_mask].detach().cpu()
                for name, t in self._ep_reward_components.items()
            }

        # Reset done environments (state mutated in-place)
        if done_mask.any():
            self.env.reset_envs(done_mask)
            self._ep_reward[done_mask] = 0.0
            self._ep_length[done_mask] = 0
            for t in self._ep_reward_components.values():
                t[done_mask] = 0.0

        return self._get_obs(), comp_rewards, dones, truncated, info

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------

    def _get_obs(self) -> dict[str, torch.Tensor]:
        """Build the float observation dict from the current TensorState."""
        s = self.env.state
        world_w, world_h = self.ship_config.world_size

        pos_norm = torch.stack(
            [
                s.ship_pos.real / world_w,
                s.ship_pos.imag / world_h,
            ],
            dim=-1,
        )  # (B, N, 2)

        vel = torch.stack([s.ship_vel.real, s.ship_vel.imag], dim=-1)  # (B, N, 2)

        att = torch.stack(
            [s.ship_attitude.real, s.ship_attitude.imag], dim=-1
        )  # (B, N, 2)

        ang_vel = s.ship_ang_vel.unsqueeze(-1)  # (B, N, 1)

        scalars = torch.stack(
            [
                s.ship_health / self.ship_config.max_health,
                s.ship_power / self.ship_config.max_power,
                (s.ship_cooldown / self.ship_config.firing_cooldown).clamp(0.0, 1.0),
            ],
            dim=-1,
        )  # (B, N, 3)

        return {
            "pos": pos_norm,
            "vel": vel,
            "att": att,
            "ang_vel": ang_vel,
            "scalars": scalars,
            "team_id": s.ship_team_id,
            "alive": s.ship_alive,
            "prev_action": s.prev_action.long(),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
    )
