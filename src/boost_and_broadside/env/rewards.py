"""Modular reward components for the decomposed per-ship per-component critic.

All computations are GPU-vectorized. No Python loops over ships or envs.
Each component returns the reward from the perspective of that specific ship —
no zero-sum pre-inversion. Zero-sum accounting is handled at PPO update time
via the lambda aggregation matrix (allied ships: +1, enemy ships: component-dependent).

Adding a new reward
-------------------
1. Create a subclass of RewardComponent with a unique `name` class attribute.
2. Add its name to REWARD_COMPONENT_NAMES (fixes K and value head ordering).
3. Optionally add a weight field to RewardConfig in config.py and set it in main.py.
4. Add an instance to the list in build_reward_components().
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
    """Penalty for damage taken by this ship.

    Each ship predicts its own future damage taken. No kill/damage-dealt bonus —
    that would require bullet tracking and risks double-counting with DeathReward.
    Zero-sum accounting is deferred to the PPO lambda aggregation (enemy ships: lambda=-1).
    """

    name = "damage"

    def __init__(self, damage_weight: float) -> None:
        self.damage_weight = damage_weight

    @property
    def weight(self) -> float:
        return self.damage_weight

    def compute(
        self,
        prev_state: TensorState,
        actions: torch.Tensor,
        next_state: TensorState,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        # delta > 0 when health decreased (damage taken)
        delta = (prev_state.ship_health - next_state.ship_health).clamp(min=0.0)  # (B, N)
        return -delta * next_state.ship_alive.float()


class DeathReward(RewardComponent):
    """Penalty for the ship that dies.

    Each ship predicts its own probability of dying this step. No kill bonus —
    the kill incentive is captured by the enemy ships' DeathReward with lambda=-1.
    """

    name = "death"

    def __init__(self, death_weight: float) -> None:
        self.death_weight = death_weight

    @property
    def weight(self) -> float:
        return self.death_weight

    def compute(
        self,
        prev_state: TensorState,
        actions: torch.Tensor,
        next_state: TensorState,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        just_died = prev_state.ship_alive & ~next_state.ship_alive  # (B, N)
        reward = torch.zeros_like(next_state.ship_health)
        reward[just_died] = -1.0
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

    @property
    def weight(self) -> float:
        return self.victory_weight

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

        team0 = next_state.ship_team_id == 0   # (B, N)
        team1 = next_state.ship_team_id == 1   # (B, N)

        t0_alive = (team0 & next_state.ship_alive).sum(dim=1)
        t1_alive = (team1 & next_state.ship_alive).sum(dim=1)

        t0_wins  = ((t0_alive > 0) & (t1_alive == 0) & dones).unsqueeze(1)  # (B, 1)
        t0_loses = ((t0_alive == 0) & (t1_alive > 0) & dones).unsqueeze(1)

        # Each ship gets ±1 from its OWN team's perspective — no pre-inversion.
        # Lambda aggregation applies victory_weight and handles zero-sum (enemy lambda=-1).
        reward[team0 & t0_wins.expand_as(team0)]  = +1.0
        reward[team0 & t0_loses.expand_as(team0)] = -1.0
        reward[team1 & t0_loses.expand_as(team1)] = +1.0  # team1 wins
        reward[team1 & t0_wins.expand_as(team1)]  = -1.0  # team1 loses

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

    @property
    def weight(self) -> float:
        return self.positioning_weight

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
        return reward


class FacingReward(RewardComponent):
    """Reward for pointing nose toward a nearby enemy.

    Proximity-weighted: facing a close enemy scores higher than facing a distant one.
    Takes the max over enemies — reward for your best target, not the sum.

    Score = max over enemies of w(dist) * dot(my_attitude, dir_to_enemy).clamp(0)
    where w(dist) = (1 - dist/R).clamp(0)  — linear falloff to zero at radius R.

    Both teams receive a positive signal for facing their enemies directly from
    compute(). Zero-sum accounting is handled by lambda aggregation in the PPO
    trainer (lambda=0 for enemy ships on this shaping component).
    """

    name = "facing"

    def __init__(self, facing_weight: float, radius: float,
                 world_size: tuple[float, float]) -> None:
        self.facing_weight = facing_weight
        self.radius        = radius
        self.world_size    = world_size

    @property
    def weight(self) -> float:
        return self.facing_weight

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

        return best_score * alive.float()


class ExposureReward(RewardComponent):
    """Penalty for being in nearby enemies' crosshairs.

    Proximity-weighted: a close enemy aiming at you is more dangerous than a distant one.
    Takes the sum over all threatening enemies — multiple ships can target you at once.

    Score = sum over enemies of w(dist) * dot(enemy_attitude, dir_from_enemy_to_me).clamp(0)
    where w(dist) = (1 - dist/R).clamp(0)  — applied as a negative reward.

    Both teams receive a negative signal for being in crosshairs directly from
    compute(). Zero-sum accounting is handled by lambda aggregation in the PPO
    trainer (lambda=-1 for enemy ships on this outcome component).
    """

    name = "exposure"

    def __init__(self, exposure_weight: float, radius: float,
                 world_size: tuple[float, float]) -> None:
        self.exposure_weight = exposure_weight
        self.radius          = radius
        self.world_size      = world_size

    @property
    def weight(self) -> float:
        return self.exposure_weight

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

        return -total_exposure * alive.float()


class ProximityReward(RewardComponent):
    """Reward for being close to any alive enemy.

    Score = max over enemies of (1 - dist/R).clamp(0) — linear falloff
    from 1.0 at distance 0 to 0.0 at distance R.

    Both teams receive a positive proximity signal directly from compute().
    Zero-sum accounting is handled by lambda aggregation in the PPO trainer
    (lambda=0 for enemy ships on this shaping component — proximity is a
    cooperative incentive that encourages both sides to close the gap).
    """

    name = "proximity"

    def __init__(self, proximity_weight: float, proximity_radius: float,
                 world_size: tuple[float, float]) -> None:
        self.proximity_weight = proximity_weight
        self.proximity_radius = proximity_radius
        self.world_size       = world_size

    @property
    def weight(self) -> float:
        return self.proximity_weight

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

        return best_prox * alive.float()


class ClosingSpeedReward(RewardComponent):
    """Reward for velocity aligned toward the nearest alive enemy.

    Score = dot(my_velocity, dir_from_me_to_nearest_enemy), clamped to [0, inf).
    Encourages actively closing the gap, not just being close.
    """

    name = "closing_speed"

    def __init__(self, closing_speed_weight: float, world_size: tuple[float, float]) -> None:
        self.closing_speed_weight = closing_speed_weight
        self.world_size           = world_size

    @property
    def weight(self) -> float:
        return self.closing_speed_weight

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

        return best_approach * alive.float()


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

    @property
    def weight(self) -> float:
        return self.turn_rate_weight

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

        return score * alive.float()


class PowerRangeReward(RewardComponent):
    """Reward for keeping power in the target fraction range [lo, hi] of max_power.

    Uses a smooth trapezoidal function: 1.0 inside the range, linearly
    decaying to 0.0 outside. Both teams receive a positive signal directly
    from compute(). Lambda=0 for enemy ships in the PPO aggregation.
    """

    name = "power_range"

    def __init__(self, power_range_weight: float, power_range_lo: float,
                 power_range_hi: float, max_power: float) -> None:
        self.power_range_weight = power_range_weight
        self.power_range_lo     = power_range_lo
        self.power_range_hi     = power_range_hi
        self.max_power          = max_power

    @property
    def weight(self) -> float:
        return self.power_range_weight

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

        return reward * alive.float()


class SpeedRangeReward(RewardComponent):
    """Reward for keeping speed in the target range [lo, hi] (world units/s).

    Uses a smooth trapezoidal function: 1.0 inside the range, linearly
    decaying to 0.0 outside. Speed is normalised by hi for the outer clamp.
    Both teams receive a positive signal directly from compute(). Lambda=0
    for enemy ships in the PPO aggregation.
    """

    name = "speed_range"

    def __init__(self, speed_range_weight: float, speed_range_lo: float,
                 speed_range_hi: float) -> None:
        self.speed_range_weight = speed_range_weight
        self.speed_range_lo     = speed_range_lo
        self.speed_range_hi     = speed_range_hi

    @property
    def weight(self) -> float:
        return self.speed_range_weight

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

        return reward * alive.float()


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

    @property
    def weight(self) -> float:
        return self.shoot_quality_weight

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

        return shooting * best_quality * alive.float()


class ScriptedAgentReward(RewardComponent):
    """Reward the RL agent for taking actions the scripted expert recommends.

    Computes log P_scripted(rl_action) = sum of log-probs of each sub-action
    (power, turn, shoot) under the scripted agent's marginal distributions.
    A 1% uniform mixture is applied before taking logs to prevent divergence
    from zero-probability actions (e.g. air brakes, which the scripted agent
    never uses).

    The scripted distributions are evaluated on prev_state — the same state
    the RL policy observed when selecting its action.

    Both teams are rewarded for following their own scripted expert. Lambda=0
    for enemy ships in the PPO aggregation (scripted_agent is not in
    REWARD_COMPONENT_NAMES and is excluded from the decomposed critic).
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

        return self.weight * log_p * prev_state.ship_alive.float()


# Fixed ordered tuple of K=12 reward component names.
# The index of each name determines the corresponding slice in (B, N, K) tensors.
# ScriptedAgentReward is intentionally excluded — it's an optional auxiliary signal
# not suitable for per-ship per-component critic decomposition.
REWARD_COMPONENT_NAMES: tuple[str, ...] = (
    "damage",        # 0 — damage taken (negative)
    "death",         # 1 — death penalty (negative)
    "victory",       # 2 — team win/loss (positive/negative)
    "exposure",      # 3 — being in crosshairs (negative)
    "facing",        # 4 — facing enemies (positive)
    "turn_rate",     # 5 — turning toward enemies (positive)
    "proximity",     # 6 — being close to enemies (positive)
    "closing_speed", # 7 — approaching enemies (positive)
    "positioning",   # 8 — positioning formula (positive)
    "power_range",   # 9 — power in target range (positive)
    "speed_range",   # 10 — speed in target range (positive)
    "shoot_quality", # 11 — shot quality when firing (positive/negative)
)

_NAME_TO_K: dict[str, int] = {name: k for k, name in enumerate(REWARD_COMPONENT_NAMES)}


def build_reward_components(
    reward_config: RewardConfig,
    ship_config: ShipConfig,
    scripted_agent: StochasticScriptedAgent | None = None,
) -> list[RewardComponent]:
    """Construct the list of active reward components from config.

    To add a new reward: create a RewardComponent subclass, add its name to
    REWARD_COMPONENT_NAMES, then append an instance here.
    To disable at runtime: add its name to reward_config.disabled_rewards.

    Args:
        reward_config:   Reward weights and radii.
        ship_config:     Physics config (used for world_size).
        scripted_agent:  Optional stochastic scripted agent for behavioral cloning.

    Returns:
        List of RewardComponent instances to apply each step.
    """
    all_components: list[RewardComponent] = [
        DamageReward(damage_weight=reward_config.damage_weight),
        DeathReward(death_weight=reward_config.death_weight),
        VictoryReward(victory_weight=reward_config.victory_weight),
        PositioningReward(
            positioning_weight=reward_config.positioning_weight,
            positioning_radius=reward_config.positioning_radius,
            world_size=ship_config.world_size,
        ),
        FacingReward(
            facing_weight=reward_config.facing_weight,
            radius=reward_config.proximity_radius,
            world_size=ship_config.world_size,
        ),
        ExposureReward(
            exposure_weight=reward_config.exposure_weight,
            radius=reward_config.proximity_radius,
            world_size=ship_config.world_size,
        ),
        ProximityReward(
            proximity_weight=reward_config.proximity_weight,
            proximity_radius=reward_config.proximity_radius,
            world_size=ship_config.world_size,
        ),
        ClosingSpeedReward(
            closing_speed_weight=reward_config.closing_speed_weight,
            world_size=ship_config.world_size,
        ),
        TurnRateReward(
            turn_rate_weight=reward_config.turn_rate_weight,
            world_size=ship_config.world_size,
        ),
        PowerRangeReward(
            power_range_weight=reward_config.power_range_weight,
            power_range_lo=reward_config.power_range_lo,
            power_range_hi=reward_config.power_range_hi,
            max_power=ship_config.max_power,
        ),
        SpeedRangeReward(
            speed_range_weight=reward_config.speed_range_weight,
            speed_range_lo=reward_config.speed_range_lo,
            speed_range_hi=reward_config.speed_range_hi,
        ),
        ShootQualityReward(
            shoot_quality_weight=reward_config.shoot_quality_weight,
            shoot_quality_radius=reward_config.shoot_quality_radius,
            world_size=ship_config.world_size,
        ),
    ]
    if scripted_agent is not None and reward_config.scripted_agent_weight > 0.0:
        all_components.append(ScriptedAgentReward(reward_config.scripted_agent_weight, scripted_agent))

    # Filter out any components listed in disabled_rewards
    return [c for c in all_components if c.name not in reward_config.disabled_rewards]


def compute_per_component_rewards(
    components: list[RewardComponent],
    prev_state: TensorState,
    actions: torch.Tensor,
    next_state: TensorState,
    dones: torch.Tensor,
) -> torch.Tensor:
    """Compute per-ship per-component rewards without zero-sum transform.

    Each component predicts events that happen directly to that ship.
    Components not in REWARD_COMPONENT_NAMES (e.g. scripted_agent) are ignored.
    Zero-sum accounting is deferred to the PPO lambda aggregation step.

    Args:
        components: Built by build_reward_components().
        prev_state: State before this step's physics.
        actions:    Actions taken.
        next_state: State after physics + damage (before auto-reset).
        dones:      (B,) game-over flags.

    Returns:
        (B, N, K) float32 — per-component per-ship rewards.
        K = len(REWARD_COMPONENT_NAMES), components in REWARD_COMPONENT_NAMES order.
    """
    B, N = next_state.ship_health.shape
    K = len(REWARD_COMPONENT_NAMES)
    result = torch.zeros(B, N, K, device=next_state.device, dtype=torch.float32)
    for comp in components:
        k = _NAME_TO_K.get(comp.name)
        if k is not None:
            result[:, :, k] = comp.compute(prev_state, actions, next_state, dones)
    return result
