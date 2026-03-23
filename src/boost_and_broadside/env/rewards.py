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
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent


class RewardComponent(ABC):
    """Base class for a single reward signal.

    Subclasses must define:
        name: str — unique key used as W&B metric label (e.g. "damage", "victory").

    Optionally override log_keys / log_breakdown to split a component into
    sub-metrics for logging (e.g. "damage_given" + "damage_taken") without
    changing the training signal — the sub-tensors must sum to the total reward.
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

    @property
    def log_keys(self) -> list[str]:
        """W&B keys this component contributes to the breakdown dict."""
        return [self.name]

    def log_breakdown(self, r: torch.Tensor) -> dict[str, torch.Tensor]:
        """Split the (already zero-sum-transformed) reward tensor into sub-metrics.

        The values must sum to r for every ship. Default: single key = self.name.

        Args:
            r: (B, N) zero-sum-transformed reward for this component.

        Returns:
            Dict mapping log key → (B, N) tensor.
        """
        return {self.name: r}


class DamageReward(RewardComponent):
    """Reward each ship for dealing damage and penalize for receiving damage."""

    name = "damage"

    @property
    def log_keys(self) -> list[str]:
        return ["damage_given", "damage_taken"]

    def log_breakdown(self, r: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"damage_given": r.clamp(min=0.0), "damage_taken": r.clamp(max=0.0)}

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

        alive = next_state.ship_alive
        team0 = next_state.ship_team_id == 0   # (B, N)
        team1 = next_state.ship_team_id == 1   # (B, N)

        # Per-ship penalty for own damage taken.
        reward = -delta

        # Cross-team bonus: each ship gets a share of the mean damage dealt to enemies.
        # Distributing enemy damage across friendly ships keeps the signal per-ship
        # without inflating total reward magnitude.
        n_team0 = (team0 & alive).float().sum(dim=1, keepdim=True).clamp(min=1.0)
        n_team1 = (team1 & alive).float().sum(dim=1, keepdim=True).clamp(min=1.0)
        team0_dmg = (delta * team0.float()).sum(dim=1, keepdim=True)   # (B, 1) total dmg to team0
        team1_dmg = (delta * team1.float()).sum(dim=1, keepdim=True)   # (B, 1) total dmg to team1

        reward += torch.where(team0, team1_dmg / n_team0, team0_dmg / n_team1)
        reward *= alive.float()

        # Pre-invert team-1 so the zero-sum negation in compute_rewards produces
        # the correct sign: damage taken → penalty, damage dealt → bonus.
        reward[team1] = -reward[team1]
        return reward * self.damage_weight


class DeathReward(RewardComponent):
    """Penalty for dying; bonus for killing."""

    name = "death"

    @property
    def log_keys(self) -> list[str]:
        return ["kill", "death"]

    def log_breakdown(self, r: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"kill": r.clamp(min=0.0), "death": r.clamp(max=0.0)}

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

        # Count deaths per team per batch for distributing kill bonuses.
        team0_deaths = (just_died & team0).float().sum(dim=1, keepdim=True)  # (B, 1)
        team1_deaths = (just_died & team1).float().sum(dim=1, keepdim=True)  # (B, 1)

        n_team0 = team0.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        n_team1 = team1.float().sum(dim=1, keepdim=True).clamp(min=1.0)

        reward = torch.zeros_like(next_state.ship_health)

        # Own-ship death penalty.
        reward[just_died & team0] = -self.death_weight
        # Pre-inverted: +death_weight raw → -death_weight after zero-sum (penalty ✓).
        reward[just_died & team1] = +self.death_weight

        # Kill bonus distributed to every surviving friendly ship.
        # team0 ships get +kill_weight per team1 death (not pre-inverted: team0 slots unchanged).
        reward[team0] += self.kill_weight * (team1_deaths / n_team0).expand_as(reward)[team0]
        # Pre-inverted for team1: −bonus raw → +bonus after zero-sum (reward for killing ✓).
        reward[team1] += -self.kill_weight * (team0_deaths / n_team1).expand_as(reward)[team1]

        return reward


class VictoryReward(RewardComponent):
    """Terminal reward: win, lose, or draw."""

    name = "victory"

    @property
    def log_keys(self) -> list[str]:
        return ["win", "loss"]

    def log_breakdown(self, r: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"win": r.clamp(min=0.0), "loss": r.clamp(max=0.0)}

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

        # Pre-inverted for team-1: +raw → -victory after zero-sum (penalty for losing ✓)
        #                          -raw → +victory after zero-sum (reward for winning ✓)
        reward[~team0 & t0_wins.expand_as(team0)]  = +self.victory_weight
        reward[~team0 & t0_loses.expand_as(team0)] = -self.victory_weight

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


class FacingReward(RewardComponent):
    """Reward for pointing nose toward a nearby enemy.

    Proximity-weighted: facing a close enemy scores higher than facing a distant one.
    Takes the max over enemies — reward for your best target, not the sum.

    Score = max over enemies of w(dist) * dot(my_attitude, dir_to_enemy).clamp(0)
    where w(dist) = (1 - dist/R).clamp(0)  — linear falloff to zero at radius R.

    Team-1 rewards are pre-inverted (same trick as ProximityReward) so that after
    the zero-sum negation both teams receive a positive signal for facing their
    enemies — not a perverse incentive to turn away.
    """

    name = "facing"

    def __init__(self, facing_weight: float, radius: float,
                 world_size: tuple[float, float]) -> None:
        self.facing_weight = facing_weight
        self.radius        = radius
        self.world_size    = world_size

    def compute(
        self,
        prev_state: TensorState,
        actions: torch.Tensor,
        next_state: TensorState,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        pos   = next_state.ship_pos       # (B, N) complex64
        att   = next_state.ship_attitude  # (B, N) complex64
        alive = next_state.ship_alive     # (B, N) bool
        teams = next_state.ship_team_id   # (B, N) int32

        B, N = pos.shape
        W, H = self.world_size
        R    = self.radius

        d      = pos.unsqueeze(2) - pos.unsqueeze(1)   # pos_i - pos_j  (B, N, N)
        d.real = (d.real + W / 2) % W - W / 2
        d.imag = (d.imag + H / 2) % H - H / 2
        dist   = d.abs()

        dir_j_to_i = d / dist.clamp(min=1e-6)

        att_i     = att.unsqueeze(2)                               # (B, N, 1)
        alignment = (att_i * torch.conj(-dir_j_to_i)).real        # (B, N, N)

        prox  = (1.0 - dist / R).clamp(min=0.0)                   # (B, N, N)
        score = prox * alignment.clamp(min=0.0)                    # (B, N, N)

        is_enemy = teams.unsqueeze(2) != teams.unsqueeze(1)
        alive_j  = alive.unsqueeze(1).expand(B, N, N)
        alive_i  = alive.unsqueeze(2).expand(B, N, N)
        valid    = is_enemy & alive_j & alive_i

        score_masked = score.masked_fill(~valid, 0.0)
        best_score   = score_masked.max(dim=2).values              # (B, N)
        best_score   = best_score * valid.any(dim=2).float()

        reward = self.facing_weight * best_score * alive.float()
        team1 = teams == 1
        reward[team1] = -reward[team1]
        return reward


class ExposureReward(RewardComponent):
    """Penalty for being in nearby enemies' crosshairs.

    Proximity-weighted: a close enemy aiming at you is more dangerous than a distant one.
    Takes the sum over all threatening enemies — multiple ships can target you at once.

    Score = sum over enemies of w(dist) * dot(enemy_attitude, dir_from_enemy_to_me).clamp(0)
    where w(dist) = (1 - dist/R).clamp(0)  — applied as a negative reward.

    Team-1 rewards are pre-inverted so that after zero-sum negation both teams
    receive a penalty for being in crosshairs — not a reward for presenting as targets.
    """

    name = "exposure"

    def __init__(self, exposure_weight: float, radius: float,
                 world_size: tuple[float, float]) -> None:
        self.exposure_weight = exposure_weight
        self.radius          = radius
        self.world_size      = world_size

    def compute(
        self,
        prev_state: TensorState,
        actions: torch.Tensor,
        next_state: TensorState,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        pos   = next_state.ship_pos       # (B, N) complex64
        att   = next_state.ship_attitude  # (B, N) complex64
        alive = next_state.ship_alive     # (B, N) bool
        teams = next_state.ship_team_id   # (B, N) int32

        B, N = pos.shape
        W, H = self.world_size
        R    = self.radius

        d      = pos.unsqueeze(2) - pos.unsqueeze(1)   # pos_i - pos_j  (B, N, N)
        d.real = (d.real + W / 2) % W - W / 2
        d.imag = (d.imag + H / 2) % H - H / 2
        dist   = d.abs()

        dir_j_to_i = d / dist.clamp(min=1e-6)  # unit vector from j to i

        # beta[b, i, j] = how much enemy j is aimed at ship i
        att_j = att.unsqueeze(1)                               # (B, 1, N)
        beta  = (att_j * torch.conj(dir_j_to_i)).real         # (B, N, N)

        prox  = (1.0 - dist / R).clamp(min=0.0)               # (B, N, N)
        score = prox * beta.clamp(min=0.0)                     # (B, N, N)

        is_enemy = teams.unsqueeze(2) != teams.unsqueeze(1)
        alive_j  = alive.unsqueeze(1).expand(B, N, N)
        alive_i  = alive.unsqueeze(2).expand(B, N, N)
        valid    = is_enemy & alive_j & alive_i

        score_masked   = score.masked_fill(~valid, 0.0)
        total_exposure = score_masked.sum(dim=2)               # (B, N) — sum over threatening enemies

        reward = -self.exposure_weight * total_exposure * alive.float()
        team1 = teams == 1
        reward[team1] = -reward[team1]
        return reward


class ProximityReward(RewardComponent):
    """Reward for being close to any alive enemy.

    Score = max over enemies of (1 - dist/R).clamp(0) — linear falloff
    from 1.0 at distance 0 to 0.0 at distance R.

    Distance is symmetric, so without correction both ships would receive the
    same raw score and the zero-sum transform would cancel them to zero.  To
    fix this the component pre-inverts team-1's reward so that after the
    zero-sum negation in compute_rewards both teams end up with a positive
    proximity signal — i.e. proximity becomes a cooperative shaping reward
    that incentivises both sides to close the gap.
    """

    name = "proximity"

    def __init__(self, proximity_weight: float, proximity_radius: float,
                 world_size: tuple[float, float]) -> None:
        self.proximity_weight = proximity_weight
        self.proximity_radius = proximity_radius
        self.world_size       = world_size

    def compute(
        self,
        prev_state: TensorState,
        actions: torch.Tensor,
        next_state: TensorState,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        pos   = next_state.ship_pos      # (B, N) complex64
        alive = next_state.ship_alive    # (B, N) bool
        teams = next_state.ship_team_id  # (B, N) int32

        B, N = pos.shape
        W, H = self.world_size
        R    = self.proximity_radius

        d      = pos.unsqueeze(2) - pos.unsqueeze(1)
        d.real = (d.real + W / 2) % W - W / 2
        d.imag = (d.imag + H / 2) % H - H / 2
        dist   = d.abs()  # (B, N, N)

        is_enemy = teams.unsqueeze(2) != teams.unsqueeze(1)
        alive_j  = alive.unsqueeze(1).expand(B, N, N)
        alive_i  = alive.unsqueeze(2).expand(B, N, N)
        valid    = is_enemy & alive_j & alive_i

        prox        = (1.0 - dist / R).clamp(min=0.0)       # (B, N, N)
        prox_masked = prox.masked_fill(~valid, 0.0)
        best_prox   = prox_masked.max(dim=2).values          # (B, N)

        reward = self.proximity_weight * best_prox * alive.float()

        # Pre-invert team-1 so the zero-sum negation in compute_rewards
        # results in both teams receiving +proximity (not a cancelling pair).
        team1 = teams == 1
        reward[team1] = -reward[team1]
        return reward


class ClosingSpeedReward(RewardComponent):
    """Reward for velocity aligned toward the nearest alive enemy.

    Score = dot(my_velocity, dir_from_me_to_nearest_enemy), clamped to [0, inf).
    Encourages actively closing the gap, not just being close.
    """

    name = "closing_speed"

    def __init__(self, closing_speed_weight: float, world_size: tuple[float, float]) -> None:
        self.closing_speed_weight = closing_speed_weight
        self.world_size           = world_size

    def compute(
        self,
        prev_state: TensorState,
        actions: torch.Tensor,
        next_state: TensorState,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        pos   = next_state.ship_pos       # (B, N) complex64
        vel   = next_state.ship_vel       # (B, N) complex64
        alive = next_state.ship_alive     # (B, N) bool
        teams = next_state.ship_team_id   # (B, N) int32

        B, N = pos.shape
        W, H = self.world_size

        d      = pos.unsqueeze(2) - pos.unsqueeze(1)   # pos_i - pos_j  (B, N, N)
        d.real = (d.real + W / 2) % W - W / 2
        d.imag = (d.imag + H / 2) % H - H / 2
        dist   = d.abs()

        dir_j_to_i = d / dist.clamp(min=1e-6)

        is_enemy = teams.unsqueeze(2) != teams.unsqueeze(1)
        alive_j  = alive.unsqueeze(1).expand(B, N, N)
        alive_i  = alive.unsqueeze(2).expand(B, N, N)
        valid    = is_enemy & alive_j & alive_i

        # Approach score toward each enemy j: dot(vel_i, dir_i_to_j)
        vel_i    = vel.unsqueeze(2)                                  # (B, N, 1)
        approach = (vel_i * torch.conj(-dir_j_to_i)).real            # (B, N, N)

        # Score for the nearest enemy
        dist_masked  = dist.masked_fill(~valid, float('inf'))
        nearest_idx  = dist_masked.argmin(dim=2, keepdim=True)       # (B, N, 1)
        best_approach = approach.gather(2, nearest_idx).squeeze(2)   # (B, N)
        best_approach = best_approach.clamp(min=0.0)
        best_approach = best_approach * valid.any(dim=2).float()

        return self.closing_speed_weight * best_approach * alive.float()


class TurnRateReward(RewardComponent):
    """Reward for angular velocity directed toward the nearest alive enemy.

    Score = ang_vel_i * sin(angle from heading to nearest enemy direction).
    Positive when rotating toward the nearest enemy, negative when rotating away.
    This gives a direct rate-of-change signal rather than rewarding the absolute
    facing angle.
    """

    name = "turn_rate"

    def __init__(self, turn_rate_weight: float, world_size: tuple[float, float]) -> None:
        self.turn_rate_weight = turn_rate_weight
        self.world_size       = world_size

    def compute(
        self,
        prev_state: TensorState,
        actions: torch.Tensor,
        next_state: TensorState,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        pos     = next_state.ship_pos       # (B, N) complex64
        att     = next_state.ship_attitude  # (B, N) complex64, unit vector
        ang_vel = next_state.ship_ang_vel   # (B, N) float32, + = counterclockwise
        alive   = next_state.ship_alive     # (B, N) bool
        teams   = next_state.ship_team_id   # (B, N) int32

        B, N = pos.shape
        W, H = self.world_size

        # Wrap-around displacement from j to i
        d      = pos.unsqueeze(2) - pos.unsqueeze(1)   # (B, N, N) complex64
        d.real = (d.real + W / 2) % W - W / 2
        d.imag = (d.imag + H / 2) % H - H / 2
        dist   = d.abs()

        # Unit vector from i toward j (negate d which is i-j)
        dir_i_to_j = (-d) / dist.clamp(min=1e-6)      # (B, N, N) complex64

        is_enemy = teams.unsqueeze(2) != teams.unsqueeze(1)
        alive_j  = alive.unsqueeze(1).expand(B, N, N)
        alive_i  = alive.unsqueeze(2).expand(B, N, N)
        valid    = is_enemy & alive_j & alive_i

        # Find nearest enemy for each ship
        dist_masked = dist.masked_fill(~valid, float('inf'))
        nearest_idx = dist_masked.argmin(dim=2, keepdim=True)  # (B, N, 1)

        # sin(angle from heading to nearest-enemy direction)
        # Im(conj(att_i) * dir_i_to_j) > 0 means enemy is counterclockwise from heading
        att_i   = att.unsqueeze(2)                                       # (B, N, 1)
        cross   = (torch.conj(att_i) * dir_i_to_j).imag                 # (B, N, N)
        nearest_cross = cross.gather(2, nearest_idx).squeeze(2)          # (B, N)

        # Reward = ang_vel * cross: positive when turning toward enemy
        has_enemy = valid.any(dim=2).float()
        score     = ang_vel * nearest_cross * has_enemy                  # (B, N)

        return self.turn_rate_weight * score * alive.float()


class PowerRangeReward(RewardComponent):
    """Reward for keeping power in the target fraction range [lo, hi] of max_power.

    Uses a smooth trapezoidal function: 1.0 inside the range, linearly
    decaying to 0.0 outside.

    Team-1 rewards are pre-inverted so both teams are rewarded (not penalized)
    for staying in the target power range after zero-sum negation.
    """

    name = "power_range"

    def __init__(self, power_range_weight: float, power_range_lo: float,
                 power_range_hi: float, max_power: float) -> None:
        self.power_range_weight = power_range_weight
        self.power_range_lo     = power_range_lo
        self.power_range_hi     = power_range_hi
        self.max_power          = max_power

    def compute(
        self,
        prev_state: TensorState,
        actions: torch.Tensor,
        next_state: TensorState,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        power_norm = next_state.ship_power / self.max_power  # (B, N) in [0, 1]
        alive      = next_state.ship_alive

        below  = (self.power_range_lo - power_norm).clamp(min=0.0)
        above  = (power_norm - self.power_range_hi).clamp(min=0.0)
        reward = 1.0 - (below + above).clamp(max=1.0)

        reward = self.power_range_weight * reward * alive.float()
        team1 = next_state.ship_team_id == 1
        reward[team1] = -reward[team1]
        return reward


class SpeedRangeReward(RewardComponent):
    """Reward for keeping speed in the target range [lo, hi] (world units/s).

    Uses a smooth trapezoidal function: 1.0 inside the range, linearly
    decaying to 0.0 outside. Speed is normalised by hi for the outer clamp.

    Team-1 rewards are pre-inverted so both teams are rewarded (not penalized)
    for staying in the target speed range after zero-sum negation.
    """

    name = "speed_range"

    def __init__(self, speed_range_weight: float, speed_range_lo: float,
                 speed_range_hi: float) -> None:
        self.speed_range_weight = speed_range_weight
        self.speed_range_lo     = speed_range_lo
        self.speed_range_hi     = speed_range_hi

    def compute(
        self,
        prev_state: TensorState,
        actions: torch.Tensor,
        next_state: TensorState,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        speed = next_state.ship_vel.abs()  # (B, N) float32
        alive = next_state.ship_alive

        below  = (self.speed_range_lo - speed).clamp(min=0.0)
        above  = (speed - self.speed_range_hi).clamp(min=0.0)
        # Normalise deviation by the range width so reward decays over a
        # sensible distance rather than requiring exact pixel values.
        span   = max(self.speed_range_hi - self.speed_range_lo, 1.0)
        reward = 1.0 - ((below + above) / span).clamp(max=1.0)

        reward = self.speed_range_weight * reward * alive.float()
        team1 = next_state.ship_team_id == 1
        reward[team1] = -reward[team1]
        return reward


class ShootQualityReward(RewardComponent):
    """Penalise shooting when far away or poorly aimed; reward shooting when close and aimed.

    Shot quality per enemy j:
        facing = dot(my_attitude, dir_to_j)          in [-1, 1]
        prox   = (1 - dist_ij / R).clamp(0)          in [0, 1]
        quality = 2 * facing.clamp(0) * prox - 1     in [-1, 1]

    The ×2 − 1 shift ensures:
      - Any shot outside the kill zone (not aimed OR too far) yields quality < 0 (penalty).
      - Only shots that are both aimed AND close yield quality > 0 (reward).

    The best quality over all valid enemies is used so the ship is judged
    against its best available target.  Not shooting always gives 0.
    """

    name = "shoot_quality"

    def __init__(self, shoot_quality_weight: float, shoot_quality_radius: float,
                 world_size: tuple[float, float]) -> None:
        self.shoot_quality_weight = shoot_quality_weight
        self.shoot_quality_radius = shoot_quality_radius
        self.world_size           = world_size

    def compute(
        self,
        prev_state: TensorState,
        actions: torch.Tensor,
        next_state: TensorState,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        pos      = next_state.ship_pos          # (B, N) complex64
        att      = next_state.ship_attitude     # (B, N) complex64
        alive    = next_state.ship_alive        # (B, N) bool
        teams    = next_state.ship_team_id      # (B, N) int32
        shooting = next_state.ship_is_shooting.float()  # (B, N)

        B, N = pos.shape
        W, H = self.world_size
        R    = self.shoot_quality_radius

        d      = pos.unsqueeze(2) - pos.unsqueeze(1)   # pos_i - pos_j  (B, N, N)
        d.real = (d.real + W / 2) % W - W / 2
        d.imag = (d.imag + H / 2) % H - H / 2
        dist   = d.abs()

        dir_j_to_i = d / dist.clamp(min=1e-6)

        # Facing: dot(att_i, dir_i_to_j)  where dir_i_to_j = -dir_j_to_i
        att_i  = att.unsqueeze(2)                              # (B, N, 1)
        facing = (att_i * torch.conj(-dir_j_to_i)).real        # (B, N, N)

        # Proximity
        prox = (1.0 - dist / R).clamp(min=0.0)                # (B, N, N)

        # Shot quality: only positive when aimed AND close
        quality = 2.0 * facing.clamp(min=0.0) * prox - 1.0   # (B, N, N) in [-1, 1]

        is_enemy = teams.unsqueeze(2) != teams.unsqueeze(1)
        alive_j  = alive.unsqueeze(1).expand(B, N, N)
        alive_i  = alive.unsqueeze(2).expand(B, N, N)
        valid    = is_enemy & alive_j & alive_i

        # Best quality over all valid enemies (judge against most favourable target)
        quality_masked = quality.masked_fill(~valid, -1.0)
        best_quality   = quality_masked.max(dim=2).values      # (B, N)
        best_quality   = best_quality * valid.any(dim=2).float()  # 0 when no enemies

        return self.shoot_quality_weight * shooting * best_quality * alive.float()


class ScriptedAgentReward(RewardComponent):
    """Reward the RL agent for taking actions the scripted expert recommends.

    Computes log P_scripted(rl_action) = sum of log-probs of each sub-action
    (power, turn, shoot) under the scripted agent's marginal distributions.
    A 1% uniform mixture is applied before taking logs to prevent divergence
    from zero-probability actions (e.g. air brakes, which the scripted agent
    never uses).

    The scripted distributions are evaluated on prev_state — the same state
    the RL policy observed when selecting its action.

    Both teams are rewarded for following their own scripted expert, so
    team-1 rewards are pre-inverted (like ProximityReward) so the zero-sum
    transform leaves both teams with a positive signal for expert-matching.
    """

    name = "scripted_agent"

    def __init__(self, weight: float, agent: StochasticScriptedAgent) -> None:
        self.weight = weight
        self.agent  = agent

    def compute(
        self,
        prev_state: TensorState,
        actions: torch.Tensor,
        next_state: TensorState,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        closest_dist, target_idx, has_target = self.agent._select_targets(prev_state)
        dir_pred    = self.agent._predict_interception(prev_state, target_idx, closest_dist)
        active_mask = prev_state.ship_alive & has_target

        p_power, p_turn, p_shoot = self.agent._compute_action_probs(
            prev_state, closest_dist, dir_pred, active_mask
        )

        a_pow = actions[..., 0].long()   # (B, N)
        a_trn = actions[..., 1].long()
        a_sht = actions[..., 2].long()

        # Mix 1% uniform into each head to prevent log(0) divergence when the
        # scripted agent assigns zero probability to an action (e.g. air brakes).
        p_power = 0.99 * p_power + 0.01 / 3
        p_turn  = 0.99 * p_turn  + 0.01 / 7
        p_shoot = 0.99 * p_shoot + 0.01 / 2

        log_p = (
            p_power.gather(-1, a_pow.unsqueeze(-1)).squeeze(-1).log()
          + p_turn.gather(-1,  a_trn.unsqueeze(-1)).squeeze(-1).log()
          + p_shoot.gather(-1, a_sht.unsqueeze(-1)).squeeze(-1).log()
        )

        reward = self.weight * log_p * prev_state.ship_alive.float()

        # Pre-invert team-1 so the zero-sum transform rewards both teams
        # positively for matching their own expert (same trick as ProximityReward).
        team1 = prev_state.ship_team_id == 1
        reward[team1] = -reward[team1]
        return reward


def build_reward_components(
    reward_config: RewardConfig,
    ship_config: ShipConfig,
    scripted_agent: StochasticScriptedAgent | None = None,
) -> list[RewardComponent]:
    """Construct the list of active reward components from config.

    To add a new reward: create a RewardComponent subclass above, then append
    an instance here. Tracking and logging require no further changes.
    To disable a reward: set its weight to 0.0 in RewardConfig, or add its name
    to reward_config.disabled_rewards.

    Args:
        reward_config:   Reward weights and radii.
        ship_config:     Physics config (used for world_size).
        scripted_agent:  Optional stochastic scripted agent. When provided and
                         reward_config.scripted_agent_weight > 0, a
                         ScriptedAgentReward component is appended.

    Returns:
        List of RewardComponent instances to apply each step.
    """
    rc = reward_config
    all_components: list[RewardComponent] = []

    if rc.damage_weight:
        all_components.append(DamageReward(damage_weight=rc.damage_weight))
    if rc.kill_weight or rc.death_weight:
        all_components.append(DeathReward(kill_weight=rc.kill_weight, death_weight=rc.death_weight))
    if rc.victory_weight:
        all_components.append(VictoryReward(victory_weight=rc.victory_weight))
    if rc.positioning_weight:
        all_components.append(PositioningReward(
            positioning_weight=rc.positioning_weight,
            positioning_radius=rc.positioning_radius,
            world_size=ship_config.world_size,
        ))
    if rc.facing_weight:
        all_components.append(FacingReward(
            facing_weight=rc.facing_weight,
            radius=rc.proximity_radius,
            world_size=ship_config.world_size,
        ))
    if rc.exposure_weight:
        all_components.append(ExposureReward(
            exposure_weight=rc.exposure_weight,
            radius=rc.proximity_radius,
            world_size=ship_config.world_size,
        ))
    if rc.proximity_weight:
        all_components.append(ProximityReward(
            proximity_weight=rc.proximity_weight,
            proximity_radius=rc.proximity_radius,
            world_size=ship_config.world_size,
        ))
    if rc.closing_speed_weight:
        all_components.append(ClosingSpeedReward(
            closing_speed_weight=rc.closing_speed_weight,
            world_size=ship_config.world_size,
        ))
    if rc.turn_rate_weight:
        all_components.append(TurnRateReward(
            turn_rate_weight=rc.turn_rate_weight,
            world_size=ship_config.world_size,
        ))
    if rc.power_range_weight:
        all_components.append(PowerRangeReward(
            power_range_weight=rc.power_range_weight,
            power_range_lo=rc.power_range_lo,
            power_range_hi=rc.power_range_hi,
            max_power=ship_config.max_power,
        ))
    if rc.speed_range_weight:
        all_components.append(SpeedRangeReward(
            speed_range_weight=rc.speed_range_weight,
            speed_range_lo=rc.speed_range_lo,
            speed_range_hi=rc.speed_range_hi,
        ))
    if rc.shoot_quality_weight:
        all_components.append(ShootQualityReward(
            shoot_quality_weight=rc.shoot_quality_weight,
            shoot_quality_radius=rc.shoot_quality_radius,
            world_size=ship_config.world_size,
        ))
    if scripted_agent is not None and rc.scripted_agent_weight:
        all_components.append(ScriptedAgentReward(rc.scripted_agent_weight, scripted_agent))

    # Filter out any components listed in disabled_rewards
    components = [c for c in all_components if c.name not in rc.disabled_rewards]
    return components


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
        breakdown.update(comp.log_breakdown(r))
        total = total + r

    return total, breakdown
