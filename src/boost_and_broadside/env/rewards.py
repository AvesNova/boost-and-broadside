"""Modular reward components for the MVP RL pipeline.

All computations are GPU-vectorized. No Python loops over ships or envs.
Rewards are computed from the perspective of each ship's team, then the
zero-sum wrapper negates team-1 rewards so the shared policy trains correctly.

Adding a new reward
-------------------
1. Create a subclass of RewardComponent with a unique `name` class attribute.
2. Optionally add a weight field to RewardConfig in config.py and set it in main.py.
3. Add an instance to the list in build_reward_components().

No other files need to change — tracking and logging pick up the new component automatically.
"""

import torch
from abc import ABC, abstractmethod

from boost_and_broadside.env.state import TensorState
from boost_and_broadside.config import RewardConfig, ShipConfig


class RewardComponent(ABC):
    """Base class for a single reward signal.

    Subclasses must define:
        name: str — unique key used as W&B metric label (e.g. "damage", "victory").
    """

    name: str

    @abstractmethod
    def compute(
        self,
        prev_state: TensorState,
        actions: torch.Tensor,
        next_state: TensorState,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-ship rewards from the ship's own-team perspective.

        Args:
            prev_state: State snapshot immediately before physics/damage.
            actions: (B, N, 3) actions that produced next_state.
            next_state: State after physics + damage (before env reset).
            dones: (B,) bool — game-over flags.

        Returns:
            (B, N) float32 reward tensor.
        """


class DamageReward(RewardComponent):
    """Reward each ship for dealing damage and penalize for receiving damage."""

    name = "damage"

    def __init__(self, damage_weight: float) -> None:
        self.damage_weight = damage_weight

    def compute(
        self,
        prev_state: TensorState,
        actions: torch.Tensor,
        next_state: TensorState,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        # Positive delta = damage taken (health decreased)
        delta = (prev_state.ship_health - next_state.ship_health).clamp(min=0.0)  # (B, N)

        team0 = next_state.ship_team_id == 0   # (B, N)
        team1 = next_state.ship_team_id == 1   # (B, N)

        # From team-0's perspective
        reward = torch.zeros_like(delta)
        reward[team0]  = -delta[team0]  # ally hurt = bad
        reward[team1]  =  delta[team1]  # enemy hurt = good

        # From team-1's perspective (symmetric)
        reward_t1 = torch.zeros_like(delta)
        reward_t1[team1] = -delta[team1]
        reward_t1[team0] =  delta[team0]

        return torch.where(team0, reward, reward_t1) * self.damage_weight


class DeathReward(RewardComponent):
    """Penalty for dying; bonus for killing."""

    name = "death"

    def __init__(self, kill_weight: float, death_weight: float) -> None:
        self.kill_weight  = kill_weight
        self.death_weight = death_weight

    def compute(
        self,
        prev_state: TensorState,
        actions: torch.Tensor,
        next_state: TensorState,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        just_died = prev_state.ship_alive & ~next_state.ship_alive  # (B, N)

        team0 = next_state.ship_team_id == 0
        team1 = next_state.ship_team_id == 1

        reward = torch.zeros_like(next_state.ship_health)
        reward[just_died & team0] -= self.death_weight
        reward[just_died & team1] += self.kill_weight

        reward_t1 = torch.zeros_like(reward)
        reward_t1[just_died & team1] -= self.death_weight
        reward_t1[just_died & team0] += self.kill_weight

        return torch.where(team0, reward, reward_t1)


class VictoryReward(RewardComponent):
    """Terminal reward: win, lose, or draw."""

    name = "victory"

    def __init__(self, victory_weight: float) -> None:
        self.victory_weight = victory_weight

    def compute(
        self,
        prev_state: TensorState,
        actions: torch.Tensor,
        next_state: TensorState,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        reward = torch.zeros_like(next_state.ship_health)

        if not dones.any():
            return reward

        t0_alive = ((next_state.ship_team_id == 0) & next_state.ship_alive).sum(dim=1)
        t1_alive = ((next_state.ship_team_id == 1) & next_state.ship_alive).sum(dim=1)

        team0 = next_state.ship_team_id == 0   # (B, N)

        t0_wins  = ((t0_alive > 0) & (t1_alive == 0) & dones).unsqueeze(1)  # (B, 1)
        t0_loses = ((t0_alive == 0) & (t1_alive > 0) & dones).unsqueeze(1)

        reward[team0 & t0_wins.expand_as(team0)]  =  self.victory_weight
        reward[team0 & t0_loses.expand_as(team0)] = -self.victory_weight

        reward[~team0 & t0_wins.expand_as(team0)]  = -self.victory_weight
        reward[~team0 & t0_loses.expand_as(team0)] =  self.victory_weight

        return reward


class PositioningReward(RewardComponent):
    """Offensive / defensive positioning reward.

    Formula (per ship s):
        r_s = [ w(r_target) * alpha_max  -  sum_i( w(r_i) * beta_i ) ] / (1 + N_enemies)

    Where:
        w(r)      = max(0, (1 - r/R)^2)    proximity weight
        r_target  = distance to nearest enemy
        alpha_max = max over enemies of dot(my_attitude, dir_to_enemy)  — offensive alignment
        r_i       = distance from enemy i to me
        beta_i    = dot(enemy_i_attitude, dir_from_enemy_to_me)        — defensive exposure
        N_enemies = number of enemies within radius R
        R         = config.positioning_radius
    """

    name = "positioning"

    def __init__(self, positioning_weight: float, positioning_radius: float,
                 world_size: tuple[float, float]) -> None:
        self.positioning_weight  = positioning_weight
        self.positioning_radius  = positioning_radius
        self.world_size          = world_size

    def compute(
        self,
        prev_state: TensorState,
        actions: torch.Tensor,
        next_state: TensorState,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        # GPU kernel: kept together for performance
        pos   = next_state.ship_pos        # (B, N) complex64
        att   = next_state.ship_attitude   # (B, N) complex64
        alive = next_state.ship_alive      # (B, N) bool
        teams = next_state.ship_team_id    # (B, N) int32

        B, N  = pos.shape
        R     = self.positioning_radius
        W, H  = self.world_size

        d       = pos.unsqueeze(2) - pos.unsqueeze(1)   # (B, N_i, N_j)
        d.real  = (d.real + W / 2) % W - W / 2
        d.imag  = (d.imag + H / 2) % H - H / 2
        dist    = d.abs()

        w          = (1.0 - dist / R).clamp(min=0.0) ** 2
        safe_dist  = dist.clamp(min=1e-6)
        dir_j_to_i = d / safe_dist

        att_i  = att.unsqueeze(2)
        alpha  = (att_i * torch.conj(-dir_j_to_i)).real

        att_j  = att.unsqueeze(1)
        beta   = (att_j * torch.conj(dir_j_to_i)).real

        is_enemy  = teams.unsqueeze(2) != teams.unsqueeze(1)
        alive_j   = alive.unsqueeze(1).expand(B, N, N)
        alive_i   = alive.unsqueeze(2).expand(B, N, N)
        valid     = is_enemy & alive_j & alive_i & ~torch.eye(N, device=pos.device, dtype=torch.bool).unsqueeze(0)

        alpha_masked = torch.where(valid, alpha, torch.full_like(alpha, -1.0))
        alpha_max, _ = (w * alpha_masked * valid.float()).max(dim=2)

        beta_sum  = (w * beta * valid.float()).sum(dim=2)
        n_enemies = (valid & (dist < R)).sum(dim=2).float()

        reward = (alpha_max - beta_sum) / (1.0 + n_enemies) * alive.float()
        return reward * self.positioning_weight


def build_reward_components(
    reward_config: RewardConfig, ship_config: ShipConfig
) -> list[RewardComponent]:
    """Construct the list of active reward components from config.

    To add a new reward: create a RewardComponent subclass above, then append
    an instance here. Tracking and logging require no further changes.

    Args:
        reward_config: Reward weights and radii.
        ship_config:   Physics config (used for world_size).

    Returns:
        List of RewardComponent instances to apply each step.
    """
    return [
        DamageReward(damage_weight=reward_config.damage_weight),
        DeathReward(kill_weight=reward_config.kill_weight, death_weight=reward_config.death_weight),
        VictoryReward(victory_weight=reward_config.victory_weight),
        PositioningReward(
            positioning_weight=reward_config.positioning_weight,
            positioning_radius=reward_config.positioning_radius,
            world_size=ship_config.world_size,
        ),
    ]


def compute_rewards(
    components: list[RewardComponent],
    prev_state: TensorState,
    actions: torch.Tensor,
    next_state: TensorState,
    dones: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute total rewards and a per-component breakdown.

    Each component computes rewards from each ship's own-team perspective.
    The zero-sum transformation (negating team-1 rewards) is applied to every
    tensor before returning, so callers always receive zero-sum values.

    Args:
        components: Built by build_reward_components().
        prev_state: State before this step's physics.
        actions:    Actions taken.
        next_state: State after physics + damage (before auto-reset).
        dones:      (B,) game-over flags.

    Returns:
        total:     (B, N) float32 — summed zero-sum rewards.
        breakdown: dict mapping component.name → (B, N) zero-sum reward tensor.
    """
    team1_mask = next_state.ship_team_id == 1  # (B, N)

    breakdown: dict[str, torch.Tensor] = {}
    total = torch.zeros(next_state.ship_health.shape, dtype=torch.float32, device=next_state.device)

    for comp in components:
        r = comp.compute(prev_state, actions, next_state, dones)
        r[team1_mask] = -r[team1_mask]   # zero-sum transform
        breakdown[comp.name] = r
        total = total + r

    return total, breakdown
