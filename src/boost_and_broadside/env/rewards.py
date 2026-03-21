"""Modular reward components for the MVP RL pipeline.

All computations are GPU-vectorized. No Python loops over ships or envs.
Rewards are computed from the perspective of each ship's team, then the
zero-sum wrapper negates team-1 rewards so the shared policy trains correctly.
"""

import torch
from abc import ABC, abstractmethod

from boost_and_broadside.env.state import TensorState
from boost_and_broadside.config import RewardConfig, ShipConfig


class RewardComponent(ABC):
    """Base class for a single reward signal."""

    @abstractmethod
    def compute(
        self,
        prev_state: TensorState,
        actions: torch.Tensor,
        next_state: TensorState,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-ship rewards.

        Args:
            prev_state: State snapshot immediately before physics/damage.
            actions: (B, N, 3) actions that produced next_state.
            next_state: State after physics + damage (before env reset).
            dones: (B,) bool — game-over flags.

        Returns:
            (B, N) float32 reward tensor.
        """


class DamageReward(RewardComponent):
    """Reward each ship for dealing damage and penalize for receiving damage.

    From each ship's perspective: dealing damage to enemies = positive,
    receiving damage = negative. The sign is applied per-team below.
    """

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

        # From a team-0 ship's perspective: enemy damage = good, ally damage = bad
        team0 = next_state.ship_team_id == 0   # (B, N)
        team1 = next_state.ship_team_id == 1   # (B, N)

        reward = torch.zeros_like(delta)
        reward[team0]  = -delta[team0]  # own team hurt = bad for team-0 ships
        reward[team1]  =  delta[team1]  # enemy hurt = good for team-0 ships

        # Negate for team-1 ships: their enemies are team-0
        # (zero-sum wrapper negates team-1 rewards later, so we keep sign consistent here)
        # Actually: compute from each ship's own-team perspective so the
        # zero-sum negation in the wrapper is clean.
        #
        # For team-1 ships: damage TO team-0 ships is good, damage TO team-1 ships is bad.
        # We recompute their reward symmetrically.
        reward_t1     = torch.zeros_like(delta)
        reward_t1[team1] = -delta[team1]   # own team hurt = bad for team-1 ships
        reward_t1[team0] =  delta[team0]   # enemy (team-0) hurt = good for team-1 ships

        # Each ship uses its own-team reward
        final = torch.where(team0, reward, reward_t1)

        return final * self.damage_weight


class DeathReward(RewardComponent):
    """Penalty for dying; bonus for killing."""

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

        # Team-0 perspective
        t0_penalty = just_died & team0    # ally died
        t0_bonus   = just_died & team1    # enemy died
        reward[t0_penalty] -= self.death_weight
        reward[t0_bonus]   += self.kill_weight

        # Team-1 perspective (symmetric, stored in reward negated)
        # For team-1 ships their "ally died" is team1, "enemy died" is team0
        t1_penalty = just_died & team1
        t1_bonus   = just_died & team0
        reward_t1  = torch.zeros_like(reward)
        reward_t1[t1_penalty] -= self.death_weight
        reward_t1[t1_bonus]   += self.kill_weight

        return torch.where(team0, reward, reward_t1)


class VictoryReward(RewardComponent):
    """Terminal reward: win, lose, or draw."""

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

        # (B,) counts of alive ships per team
        t0_alive = ((next_state.ship_team_id == 0) & next_state.ship_alive).sum(dim=1)
        t1_alive = ((next_state.ship_team_id == 1) & next_state.ship_alive).sum(dim=1)

        team0 = next_state.ship_team_id == 0   # (B, N)

        # From team-0's perspective (broadcast to ship dim)
        t0_wins  = ((t0_alive > 0) & (t1_alive == 0) & dones).unsqueeze(1)  # (B, 1)
        t0_loses = ((t0_alive == 0) & (t1_alive > 0) & dones).unsqueeze(1)

        # Team-0 ships get the team-0 outcome
        reward[team0 & t0_wins.expand_as(team0)]  =  self.victory_weight
        reward[team0 & t0_loses.expand_as(team0)] = -self.victory_weight

        # Team-1 ships get the mirror
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

    def __init__(self, positioning_weight: float, positioning_radius: float, world_size: tuple[float, float]) -> None:
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

        # Pairwise toroidal displacement: from ship j to ship i
        d       = pos.unsqueeze(2) - pos.unsqueeze(1)   # (B, N_i, N_j) — i receives from j
        d.real  = (d.real + W / 2) % W - W / 2
        d.imag  = (d.imag + H / 2) % H - H / 2
        dist    = d.abs()                               # (B, N_i, N_j)

        # Proximity weight w(r) = (1 - r/R)^2, zero beyond R
        w = (1.0 - dist / R).clamp(min=0.0) ** 2       # (B, N_i, N_j)

        # Unit vector from j to i
        safe_dist = dist.clamp(min=1e-6)
        dir_j_to_i = d / safe_dist                     # (B, N_i, N_j) complex — j→i direction

        # Offensive alignment: how well ship i points at enemy j
        # dot(att_i, dir_i_to_j) = Re(att_i * conj(-dir_j_to_i))
        att_i   = att.unsqueeze(2)                      # (B, N_i, 1)
        alpha   = (att_i * torch.conj(-dir_j_to_i)).real  # (B, N_i, N_j)

        # Defensive exposure: how well enemy j points at ship i
        att_j   = att.unsqueeze(1)                      # (B, 1, N_j)
        beta    = (att_j * torch.conj(dir_j_to_i)).real   # (B, N_i, N_j)

        # Enemy mask: j is an enemy of i if team_id differs
        is_enemy = (teams.unsqueeze(2) != teams.unsqueeze(1))  # (B, N_i, N_j)
        alive_j  = alive.unsqueeze(1).expand(B, N, N)          # (B, N_i, N_j)
        alive_i  = alive.unsqueeze(2).expand(B, N, N)          # (B, N_i, N_j)
        valid    = is_enemy & alive_j & alive_i & ~torch.eye(N, device=pos.device, dtype=torch.bool).unsqueeze(0)

        # Offensive: best alignment to any enemy within R
        alpha_masked  = torch.where(valid, alpha, torch.full_like(alpha, -1.0))
        alpha_max, _  = (w * alpha_masked * valid.float()).max(dim=2)  # (B, N)

        # Defensive: weighted sum of enemy exposures
        beta_sum      = (w * beta * valid.float()).sum(dim=2)          # (B, N)

        # Number of enemies in range
        n_enemies     = (valid & (dist < R)).sum(dim=2).float()        # (B, N)

        reward = (alpha_max - beta_sum) / (1.0 + n_enemies)            # (B, N)

        # Dead ships get no reward
        reward = reward * alive.float()

        return reward * self.positioning_weight


def build_reward_components(
    reward_config: RewardConfig, ship_config: ShipConfig
) -> list[RewardComponent]:
    """Construct the list of active reward components from config.

    Args:
        reward_config: Reward weights and radii.
        ship_config: Physics config (used for world_size).

    Returns:
        List of RewardComponent instances to apply each step.
    """
    return [
        DamageReward(damage_weight=reward_config.damage_weight),
        DeathReward(
            kill_weight=reward_config.kill_weight,
            death_weight=reward_config.death_weight,
        ),
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
) -> torch.Tensor:
    """Sum all reward components into a single (B, N) tensor.

    Each component computes rewards from the ship's own-team perspective.
    The zero-sum transformation (negating team-1 rewards) is applied here
    so the PPO buffer always receives zero-sum rewards.

    Args:
        components: Built by build_reward_components().
        prev_state: State before this step's physics.
        actions: Actions taken.
        next_state: State after physics + damage (before auto-reset).
        dones: (B,) game-over flags.

    Returns:
        (B, N) float32 rewards — zero-sum across teams.
    """
    total = torch.zeros(
        next_state.ship_health.shape,
        dtype=torch.float32,
        device=next_state.device,
    )
    for comp in components:
        total = total + comp.compute(prev_state, actions, next_state, dones)

    # Zero-sum: negate team-1 rewards so the shared policy optimises symmetrically
    team1_mask = next_state.ship_team_id == 1   # (B, N)
    total[team1_mask] = -total[team1_mask]

    return total
