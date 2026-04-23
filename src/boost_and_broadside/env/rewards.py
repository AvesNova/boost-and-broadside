"""Modular reward components for the 9-component decomposed critic.

All computations are GPU-vectorized. No Python loops over ships or envs.
Each component returns the reward from the perspective of that specific ship —
no zero-sum pre-inversion. Zero-sum accounting is handled at PPO update time
via the lambda aggregation matrix.

The 9 components split outcomes into ally/enemy pairs so the critic can
distinguish symmetric from asymmetric situations (e.g. mutual damage vs no
damage, standoff vs close fight). Shaping rewards provide dense feedback to
prevent passive collapse during early RL.

Adding a new reward
-------------------
1. Create a subclass of RewardComponent with a unique `name` class attribute.
2. Add its name to REWARD_COMPONENT_NAMES (fixes K and value head ordering).
3. Add a weight field to RewardConfig in config/core.py and set it in runs/shared.py.
4. Add an instance to the list in build_reward_components().
"""

import torch
from abc import ABC, abstractmethod

from boost_and_broadside.env.state import TensorState
from boost_and_broadside.config import RewardConfig, ShipConfig


class RewardComponent(ABC):
    """Base class for a single reward signal.

    Subclasses must define:
        name: str — unique key used as W&B metric label.

    Optionally override log_keys / log_breakdown to split a component into
    sub-metrics for logging without changing the training signal.
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
        """Split the reward tensor into sub-metrics for logging.

        Args:
            r: (B, N) reward for this component.

        Returns:
            Dict mapping log key → (B, N) tensor.
        """
        return {self.name: r}


# ---------------------------------------------------------------------------
# Outcome reward components — split into ally/enemy pairs
# ---------------------------------------------------------------------------


class AllyDamageReward(RewardComponent):
    """Damage taken by this ship. Lambda=0 for enemies so only ally damage
    aggregates into the advantage. Critic learns expected ally damage taken."""

    name = "ally_damage"

    def __init__(self, weight: float) -> None:
        self._weight = weight

    @property
    def weight(self) -> float:
        return self._weight

    def compute(
        self,
        prev_state: TensorState,
        actions: torch.Tensor,
        next_state: TensorState,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        delta = (prev_state.ship_health - next_state.ship_health).clamp(
            min=0.0
        )  # (B, N)
        return -delta * next_state.ship_alive.float()


class EnemyDamageReward(RewardComponent):
    """Same compute as AllyDamageReward. Lambda=-1 for enemies inverts sign so allies
    benefit when enemies take damage. Critic learns expected enemy damage taken."""

    name = "enemy_damage"

    def __init__(self, weight: float) -> None:
        self._weight = weight

    @property
    def weight(self) -> float:
        return self._weight

    def compute(
        self,
        prev_state: TensorState,
        actions: torch.Tensor,
        next_state: TensorState,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        delta = (prev_state.ship_health - next_state.ship_health).clamp(
            min=0.0
        )  # (B, N)
        return -delta * next_state.ship_alive.float()


class AllyDeathReward(RewardComponent):
    """Death penalty (-1) for this ship. Lambda=0 for enemies so only ally deaths
    aggregate. Critic learns expected ally death count (negative)."""

    name = "ally_death"

    def __init__(self, weight: float) -> None:
        self._weight = weight

    @property
    def weight(self) -> float:
        return self._weight

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


class EnemyDeathReward(RewardComponent):
    """Same compute as AllyDeathReward. Lambda=-1 for enemies so allies benefit when
    enemies die. Critic learns expected enemy death count (positive for allies)."""

    name = "enemy_death"

    def __init__(self, weight: float) -> None:
        self._weight = weight

    @property
    def weight(self) -> float:
        return self._weight

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


class KillShotReward(RewardComponent):
    """Proportional kill credit/penalty based on step-level damage attribution.

    Each ship earns a proportional share of +1.0 per dying enemy, weighted by
    its step-level damage to that ship. Ships that dealt damage to a dying
    friendly take a proportional share of -1.0 (friendly-fire penalty).
    Uses state.damage_matrix (step-level). Lambda=0 for all other ships
    (self-only, diagonal lambda).
    """

    name = "kill_shot"

    def __init__(self, weight: float) -> None:
        self._weight = weight

    @property
    def weight(self) -> float:
        return self._weight

    def compute(
        self,
        prev_state: TensorState,
        actions: torch.Tensor,
        next_state: TensorState,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        just_died = prev_state.ship_alive & ~next_state.ship_alive  # (B, N)
        reward = torch.zeros_like(next_state.ship_health)
        if not just_died.any():
            return reward

        B, N = next_state.ship_health.shape
        dm = next_state.damage_matrix  # (B, N_shooter, N_target)
        is_enemy_target = next_state.ship_team_id.unsqueeze(
            2
        ) != next_state.ship_team_id.unsqueeze(1)  # (B, N_shooter, N_target)
        self_mask = torch.eye(N, dtype=torch.bool, device=dm.device).unsqueeze(0)
        is_friendly_target = ~is_enemy_target & ~self_mask  # same team, not self

        dying = just_died.unsqueeze(1).float()  # (B, 1, N_target)

        # --- Enemy kill credit (proportional share of +1.0 per kill) ---
        dm_enemy = dm * is_enemy_target.float() * dying
        total_enemy = dm_enemy.sum(dim=1, keepdim=True).clamp(min=1e-8)
        reward = (dm_enemy / total_enemy).sum(dim=2)

        # --- Friendly kill penalty (proportional share of -1.0 per friendly kill) ---
        dm_friendly = dm * is_friendly_target.float() * dying
        total_friendly = dm_friendly.sum(dim=1, keepdim=True).clamp(min=1e-8)
        reward -= (dm_friendly / total_friendly).sum(dim=2)

        return reward


class KillAssistReward(RewardComponent):
    """Proportional kill credit/penalty based on cumulative episode damage.

    Each ship earns a proportional share of 1.0 credit per dying enemy,
    weighted by its cumulative damage to that ship. Ships that dealt cumulative
    damage to a dying friendly take a proportional share of -1.0 (friendly-fire
    penalty). Uses state.cumulative_damage_matrix (episode-level).
    Lambda=0 for all other ships (self-only, diagonal lambda).
    """

    name = "kill_assist"

    def __init__(self, weight: float) -> None:
        self._weight = weight

    @property
    def weight(self) -> float:
        return self._weight

    def compute(
        self,
        prev_state: TensorState,
        actions: torch.Tensor,
        next_state: TensorState,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        just_died = prev_state.ship_alive & ~next_state.ship_alive  # (B, N)
        reward = torch.zeros_like(next_state.ship_health)
        if not just_died.any():
            return reward

        B, N = next_state.ship_health.shape
        cdm = next_state.cumulative_damage_matrix  # (B, N_shooter, N_target)
        is_enemy_target = next_state.ship_team_id.unsqueeze(
            2
        ) != next_state.ship_team_id.unsqueeze(1)  # (B, N_shooter, N_target)
        self_mask = torch.eye(N, dtype=torch.bool, device=cdm.device).unsqueeze(0)
        is_friendly_target = ~is_enemy_target & ~self_mask

        dying = just_died.unsqueeze(1).float()  # (B, 1, N_target)

        # --- Enemy kill credit (proportional share of +1.0 per kill) ---
        cdm_enemy = cdm * is_enemy_target.float() * dying
        total_enemy = cdm_enemy.sum(dim=1, keepdim=True).clamp(min=1e-8)
        reward = (cdm_enemy / total_enemy).sum(dim=2)

        # --- Friendly kill penalty (proportional share of -1.0 per friendly kill) ---
        cdm_friendly = cdm * is_friendly_target.float() * dying
        total_friendly = cdm_friendly.sum(dim=1, keepdim=True).clamp(min=1e-8)
        reward -= (cdm_friendly / total_friendly).sum(dim=2)

        return reward


class AllyWinReward(RewardComponent):
    """+1 to ships whose team wins at game end; 0 to losers and draws. Lambda=0 for
    enemies. Critic learns P(ally wins), which distinguishes standoff (≈0) from
    close fight (≈0.5) — unlike a single win/loss component where both look like 0."""

    name = "ally_win"

    def __init__(self, weight: float) -> None:
        self._weight = weight

    @property
    def weight(self) -> float:
        return self._weight

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
        team0 = next_state.ship_team_id == 0  # (B, N)
        team1 = next_state.ship_team_id == 1  # (B, N)
        t0_alive = (team0 & next_state.ship_alive).sum(dim=1)  # (B,)
        t1_alive = (team1 & next_state.ship_alive).sum(dim=1)  # (B,)
        t0_wins = ((t0_alive > 0) & (t1_alive == 0) & dones).unsqueeze(1)  # (B, 1)
        t1_wins = ((t1_alive > 0) & (t0_alive == 0) & dones).unsqueeze(1)  # (B, 1)
        reward[team0 & t0_wins.expand_as(team0)] = +1.0
        reward[team1 & t1_wins.expand_as(team1)] = +1.0
        return reward


class EnemyWinReward(RewardComponent):
    """Same compute as AllyWinReward (+1 to winning-team ships). Lambda=-1 for
    enemies: ally advantage = -1 when enemies win, 0 otherwise. Critic learns
    -P(enemy wins), completing the ally_win/enemy_win pair."""

    name = "enemy_win"

    def __init__(self, weight: float) -> None:
        self._weight = weight

    @property
    def weight(self) -> float:
        return self._weight

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
        team0 = next_state.ship_team_id == 0  # (B, N)
        team1 = next_state.ship_team_id == 1  # (B, N)
        t0_alive = (team0 & next_state.ship_alive).sum(dim=1)  # (B,)
        t1_alive = (team1 & next_state.ship_alive).sum(dim=1)  # (B,)
        t0_wins = ((t0_alive > 0) & (t1_alive == 0) & dones).unsqueeze(1)  # (B, 1)
        t1_wins = ((t1_alive > 0) & (t0_alive == 0) & dones).unsqueeze(1)  # (B, 1)
        reward[team0 & t0_wins.expand_as(team0)] = +1.0
        reward[team1 & t1_wins.expand_as(team1)] = +1.0
        return reward


# ---------------------------------------------------------------------------
# Local per-ship combat rewards — self-only, lambda=0 for all other ships
# ---------------------------------------------------------------------------


class LocalDeathReward(RewardComponent):
    """Penalty of -1 on the step this ship dies.

    Uses just_died = prev_state.ship_alive & ~next_state.ship_alive so the
    reward fires on the exact step of death, before the ship is masked out.
    Never multiplied by next_state.ship_alive — the ship is dead there by
    definition and would always produce zero.

    Unlike ally_death (which propagates to teammates via lambda=1), this is
    self-only: lambda=0 for all other ships.
    """

    name = "death"

    def __init__(self, weight: float) -> None:
        self._weight = weight

    @property
    def weight(self) -> float:
        return self._weight

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


class LocalBulletDeathReward(RewardComponent):
    """Penalty of -1 on the step this ship is killed by a bullet.

    Fires when a ship dies AND took no obstacle damage that step, so it
    distinguishes bullet kills from obstacle kills for separate reward weighting.
    Self-only: lambda=0 for all other ships.
    """

    name = "bullet_death"

    def __init__(self, weight: float) -> None:
        self._weight = weight

    @property
    def weight(self) -> float:
        return self._weight

    def compute(
        self,
        prev_state: TensorState,
        actions: torch.Tensor,
        next_state: TensorState,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        just_died = prev_state.ship_alive & ~next_state.ship_alive  # (B, N)
        died_by_obstacle = just_died & (next_state.ship_obstacle_damage > 0)
        died_by_bullet = just_died & ~died_by_obstacle
        reward = torch.zeros_like(next_state.ship_health)
        reward[died_by_bullet] = -1.0
        return reward


class LocalObstacleDeathReward(RewardComponent):
    """Penalty of -1 on the step this ship is killed by an obstacle.

    Fires when a ship dies AND took obstacle damage that step.
    Self-only: lambda=0 for all other ships.
    """

    name = "obstacle_death"

    def __init__(self, weight: float) -> None:
        self._weight = weight

    @property
    def weight(self) -> float:
        return self._weight

    def compute(
        self,
        prev_state: TensorState,
        actions: torch.Tensor,
        next_state: TensorState,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        just_died = prev_state.ship_alive & ~next_state.ship_alive  # (B, N)
        died_by_obstacle = just_died & (next_state.ship_obstacle_damage > 0)
        reward = torch.zeros_like(next_state.ship_health)
        reward[died_by_obstacle] = -1.0
        return reward


class ObstacleProximityReward(RewardComponent):
    """Negative reward proportional to proximity to the nearest obstacle edge.

    Reward = -1 / (dist_to_nearest_edge + 1), applied every step to alive ships.
    Approaches -1 as a ship touches the obstacle edge, approaches 0 far away.
    Self-only: lambda=0 for all other ships.
    """

    name = "obstacle_proximity"

    def __init__(self, weight: float, world_size: tuple[float, float]) -> None:
        self._weight = weight
        self._world_w, self._world_h = world_size

    @property
    def weight(self) -> float:
        return self._weight

    def compute(
        self,
        prev_state: TensorState,
        actions: torch.Tensor,
        next_state: TensorState,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        M = next_state.obstacle_pos.shape[1]
        if M == 0:
            return torch.zeros_like(next_state.ship_health)

        world_w, world_h = self._world_w, self._world_h

        # Toroidal vector from each ship to each obstacle center
        dx = next_state.obstacle_pos.real.unsqueeze(1) - next_state.ship_pos.real.unsqueeze(2)
        dy = next_state.obstacle_pos.imag.unsqueeze(1) - next_state.ship_pos.imag.unsqueeze(2)
        dx = (dx + world_w / 2) % world_w - world_w / 2
        dy = (dy + world_h / 2) % world_h - world_h / 2

        dist_to_center = torch.sqrt(dx**2 + dy**2)  # (B, N, M)
        dist_to_edge, _ = (dist_to_center - next_state.obstacle_radius.unsqueeze(1)).min(dim=2)
        dist_to_edge = dist_to_edge.clamp(min=0.0)  # (B, N)

        reward = -1.0 / (dist_to_edge + 1.0)  # (B, N), range (-1, 0]
        return reward * next_state.ship_alive.float()


class LocalDamageTakenReward(RewardComponent):
    """Damage received by this ship this step.

    Negative reward proportional to health lost. Unlike ally_damage, this is
    self-only (lambda=0 for all other ships) so the signal is never shared with
    or aggregated across teammates. Gives the critic a separate head to estimate
    individual survivability independent of team-wide damage accounting.
    """

    name = "damage_taken"

    def __init__(self, weight: float) -> None:
        self._weight = weight

    @property
    def weight(self) -> float:
        return self._weight

    def compute(
        self,
        prev_state: TensorState,
        actions: torch.Tensor,
        next_state: TensorState,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        delta = (prev_state.ship_health - next_state.ship_health).clamp(min=0.0)
        return -delta * next_state.ship_alive.float()


class LocalDamageDealtEnemyReward(RewardComponent):
    """Damage dealt by this ship to enemies this step.

    Positive reward proportional to enemy health removed. Reads from
    state.damage_matrix (shooter × target). Self-only: diagonal lambda.
    """

    name = "damage_dealt_enemy"

    def __init__(self, weight: float) -> None:
        self._weight = weight

    @property
    def weight(self) -> float:
        return self._weight

    def compute(
        self,
        prev_state: TensorState,
        actions: torch.Tensor,
        next_state: TensorState,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        dm = next_state.damage_matrix  # (B, N_shooter, N_target)
        B, N = next_state.ship_team_id.shape
        is_enemy = next_state.ship_team_id.unsqueeze(
            2
        ) != next_state.ship_team_id.unsqueeze(1)  # (B, N_shooter, N_target)
        enemy_damage = (dm * is_enemy.float()).sum(dim=2)  # (B, N_shooter)
        return enemy_damage * next_state.ship_alive.float()


class LocalDamageDealtAllyReward(RewardComponent):
    """Friendly-fire penalty: damage dealt by this ship to teammates this step.

    Negative reward proportional to ally health removed. Reads from
    state.damage_matrix (shooter × target). Self-only: diagonal lambda.
    """

    name = "damage_dealt_ally"

    def __init__(self, weight: float) -> None:
        self._weight = weight

    @property
    def weight(self) -> float:
        return self._weight

    def compute(
        self,
        prev_state: TensorState,
        actions: torch.Tensor,
        next_state: TensorState,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        dm = next_state.damage_matrix  # (B, N_shooter, N_target)
        B, N = next_state.ship_team_id.shape
        is_enemy = next_state.ship_team_id.unsqueeze(
            2
        ) != next_state.ship_team_id.unsqueeze(1)  # (B, N_shooter, N_target)
        self_mask = torch.eye(N, dtype=torch.bool, device=dm.device).unsqueeze(0)
        is_friendly = ~is_enemy & ~self_mask  # same team, not self
        friendly_damage = (dm * is_friendly.float()).sum(dim=2)  # (B, N_shooter)
        return -friendly_damage * next_state.ship_alive.float()


# ---------------------------------------------------------------------------
# Shaping reward components — dense signals to prevent passive collapse
# ---------------------------------------------------------------------------


class FacingReward(RewardComponent):
    """Reward for pointing nose toward a nearby enemy.

    Proximity-weighted: facing a close enemy scores higher than facing a distant one.
    Takes the max over enemies — reward for your best target, not the sum.

    Score = max over enemies of w(dist) * dot(my_attitude, dir_to_enemy).clamp(0)
    where w(dist) = (1 - dist/R).clamp(0)  — linear falloff to zero at radius R.

    Both teams receive a positive signal for facing their enemies directly from
    compute(). Lambda=0 for enemy ships in the PPO aggregation (self-shaping only).
    """

    name = "facing"

    def __init__(
        self, facing_weight: float, radius: float, world_size: tuple[float, float]
    ) -> None:
        self.facing_weight = facing_weight
        self.radius = radius
        self.world_size = world_size

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
        pos = next_state.ship_pos  # (B, N) complex64
        att = next_state.ship_attitude  # (B, N) complex64
        alive = next_state.ship_alive  # (B, N) bool
        teams = next_state.ship_team_id  # (B, N) int32

        B, N = pos.shape
        W, H = self.world_size
        R = self.radius

        d = pos.unsqueeze(2) - pos.unsqueeze(1)  # pos_i - pos_j  (B, N, N)
        d.real = (d.real + W / 2) % W - W / 2
        d.imag = (d.imag + H / 2) % H - H / 2
        dist = d.abs()

        dir_j_to_i = d / dist.clamp(min=1e-6)

        att_i = att.unsqueeze(2)  # (B, N, 1)
        alignment = (att_i * torch.conj(-dir_j_to_i)).real  # (B, N, N)

        prox = (1.0 - dist / R).clamp(min=0.0)  # (B, N, N)
        score = prox * alignment.clamp(min=0.0)  # (B, N, N)

        is_enemy = teams.unsqueeze(2) != teams.unsqueeze(1)
        alive_j = alive.unsqueeze(1).expand(B, N, N)
        alive_i = alive.unsqueeze(2).expand(B, N, N)
        valid = is_enemy & alive_j & alive_i

        score_masked = score.masked_fill(~valid, 0.0)
        best_score = score_masked.max(dim=2).values  # (B, N)
        best_score = best_score * valid.any(dim=2).float()

        return best_score * alive.float()


class ClosingSpeedReward(RewardComponent):
    """Reward for velocity aligned toward the nearest alive enemy.

    Score = dot(my_velocity, dir_from_me_to_nearest_enemy) / max_speed,
    clamped to [0, 1]. Dividing by max_speed puts the output in the same
    [0, 1] range as FacingReward and ShootQualityReward so that comp_weights
    reflect true relative importance rather than physics unit differences.
    """

    name = "closing_speed"

    def __init__(
        self,
        closing_speed_weight: float,
        world_size: tuple[float, float],
        max_speed: float,
    ) -> None:
        self.closing_speed_weight = closing_speed_weight
        self.world_size = world_size
        self.max_speed = max_speed

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
        pos = next_state.ship_pos  # (B, N) complex64
        vel = next_state.ship_vel  # (B, N) complex64
        alive = next_state.ship_alive  # (B, N) bool
        teams = next_state.ship_team_id  # (B, N) int32

        B, N = pos.shape
        W, H = self.world_size

        d = pos.unsqueeze(2) - pos.unsqueeze(1)  # pos_i - pos_j  (B, N, N)
        d.real = (d.real + W / 2) % W - W / 2
        d.imag = (d.imag + H / 2) % H - H / 2
        dist = d.abs()

        dir_j_to_i = d / dist.clamp(min=1e-6)

        is_enemy = teams.unsqueeze(2) != teams.unsqueeze(1)
        alive_j = alive.unsqueeze(1).expand(B, N, N)
        alive_i = alive.unsqueeze(2).expand(B, N, N)
        valid = is_enemy & alive_j & alive_i

        # Approach score toward each enemy j: dot(vel_i, dir_i_to_j)
        vel_i = vel.unsqueeze(2)  # (B, N, 1)
        approach = (vel_i * torch.conj(-dir_j_to_i)).real  # (B, N, N)

        # Score for the nearest enemy
        dist_masked = dist.masked_fill(~valid, float("inf"))
        nearest_idx = dist_masked.argmin(dim=2, keepdim=True)  # (B, N, 1)
        best_approach = approach.gather(2, nearest_idx).squeeze(2)  # (B, N)
        best_approach = best_approach.clamp(min=0.0)
        best_approach = best_approach * valid.any(dim=2).float()

        return (best_approach / self.max_speed) * alive.float()


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
    against its best available target. Not shooting always gives 0.
    """

    name = "shoot_quality"

    def __init__(
        self,
        shoot_quality_weight: float,
        shoot_quality_radius: float,
        world_size: tuple[float, float],
    ) -> None:
        self.shoot_quality_weight = shoot_quality_weight
        self.shoot_quality_radius = shoot_quality_radius
        self.world_size = world_size

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
        pos = next_state.ship_pos  # (B, N) complex64
        att = next_state.ship_attitude  # (B, N) complex64
        alive = next_state.ship_alive  # (B, N) bool
        teams = next_state.ship_team_id  # (B, N) int32
        shooting = next_state.ship_is_shooting.float()  # (B, N)

        B, N = pos.shape
        W, H = self.world_size
        R = self.shoot_quality_radius

        d = pos.unsqueeze(2) - pos.unsqueeze(1)  # pos_i - pos_j  (B, N, N)
        d.real = (d.real + W / 2) % W - W / 2
        d.imag = (d.imag + H / 2) % H - H / 2
        dist = d.abs()

        dir_j_to_i = d / dist.clamp(min=1e-6)

        # Facing: dot(att_i, dir_i_to_j)  where dir_i_to_j = -dir_j_to_i
        att_i = att.unsqueeze(2)  # (B, N, 1)
        facing = (att_i * torch.conj(-dir_j_to_i)).real  # (B, N, N)

        # Proximity
        prox = (1.0 - dist / R).clamp(min=0.0)  # (B, N, N)

        # Shot quality: only positive when aimed AND close
        quality = 2.0 * facing.clamp(min=0.0) * prox - 1.0  # (B, N, N) in [-1, 1]

        is_enemy = teams.unsqueeze(2) != teams.unsqueeze(1)
        alive_j = alive.unsqueeze(1).expand(B, N, N)
        alive_i = alive.unsqueeze(2).expand(B, N, N)
        valid = is_enemy & alive_j & alive_i

        # Best quality over all valid enemies (judge against most favourable target)
        quality_masked = quality.masked_fill(~valid, -1.0)
        best_quality = quality_masked.max(dim=2).values  # (B, N)
        best_quality = best_quality * valid.any(dim=2).float()  # 0 when no enemies

        return shooting * best_quality * alive.float()


# ---------------------------------------------------------------------------
# Component registry
# ---------------------------------------------------------------------------

REWARD_COMPONENT_NAMES: tuple[str, ...] = (
    "ally_damage",  #  0 — damage taken by allies (negative)
    "enemy_damage",  #  1 — damage taken by enemies (positive for allies via lambda)
    "ally_death",  #  2 — ally ship deaths (negative)
    "enemy_death",  #  3 — enemy ship deaths (positive for allies via lambda)
    "ally_win",  #  4 — ally team wins (positive)
    "enemy_win",  #  5 — enemy team wins (negative for allies via lambda)
    "facing",  #  6 — pointing at nearest enemy (shaping, self only)
    "closing_speed",  #  7 — velocity toward nearest enemy (shaping, self only)
    "shoot_quality",  #  8 — shot quality when firing (shaping, self only)
    "kill_shot",  #  9 — proportional kill credit from step-level damage (self only)
    "kill_assist",  # 10 — proportional kill credit from episode-level damage (self only)
    "damage_taken",  # 11 — damage received by this ship this step (self only)
    "damage_dealt_enemy",  # 12 — damage dealt to enemies this step (self only)
    "damage_dealt_ally",  # 13 — damage dealt to allies this step — friendly-fire penalty (self only)
    "death",  # 14 — -1 on the step this ship dies (self only)
    "bullet_death",        # 15 — -1 on the step this ship is killed by a bullet (self only)
    "obstacle_death",      # 16 — -1 on the step this ship is killed by an obstacle (self only)
    "obstacle_proximity",  # 17 — -1/(dist_to_edge+1) each step, alive ships only (self only)
)

_NAME_TO_K: dict[str, int] = {name: k for k, name in enumerate(REWARD_COMPONENT_NAMES)}


def build_reward_components(
    rewards: RewardConfig,
    ship_config: ShipConfig,
) -> list[RewardComponent]:
    """Construct the 9 reward components from config.

    Called once at PPOTrainer init. Individual component weights are updated
    live each update by the group-scale multipliers in the training schedule.

    Args:
        rewards:     Reward weights and geometry params.
        ship_config: Physics config (provides world_size).

    Returns:
        List of all 11 RewardComponent instances.
    """
    return [
        AllyDamageReward(weight=rewards.ally_damage_weight),
        EnemyDamageReward(weight=rewards.enemy_damage_weight),
        AllyDeathReward(weight=rewards.ally_death_weight),
        EnemyDeathReward(weight=rewards.enemy_death_weight),
        AllyWinReward(weight=rewards.ally_win_weight),
        EnemyWinReward(weight=rewards.enemy_win_weight),
        FacingReward(
            facing_weight=rewards.facing_weight,
            radius=rewards.proximity_radius,
            world_size=ship_config.world_size,
        ),
        ClosingSpeedReward(
            closing_speed_weight=rewards.closing_speed_weight,
            world_size=ship_config.world_size,
            max_speed=ship_config.max_speed,
        ),
        ShootQualityReward(
            shoot_quality_weight=rewards.shoot_quality_weight,
            shoot_quality_radius=rewards.shoot_quality_radius,
            world_size=ship_config.world_size,
        ),
        KillShotReward(weight=rewards.kill_shot_weight),
        KillAssistReward(weight=rewards.kill_assist_weight),
        LocalDamageTakenReward(weight=rewards.damage_taken_weight),
        LocalDamageDealtEnemyReward(weight=rewards.damage_dealt_enemy_weight),
        LocalDamageDealtAllyReward(weight=rewards.damage_dealt_ally_weight),
        LocalDeathReward(weight=rewards.death_weight),
        LocalBulletDeathReward(weight=rewards.bullet_death_weight),
        LocalObstacleDeathReward(weight=rewards.obstacle_death_weight),
        ObstacleProximityReward(
            weight=rewards.obstacle_proximity_weight,
            world_size=ship_config.world_size,
        ),
    ]


def compute_per_component_rewards(
    components: list[RewardComponent],
    prev_state: TensorState,
    actions: torch.Tensor,
    next_state: TensorState,
    dones: torch.Tensor,
) -> torch.Tensor:
    """Compute per-ship per-component rewards without zero-sum transform.

    Each component predicts events that happen directly to that ship.
    Zero-sum accounting is deferred to the PPO lambda aggregation step.

    Args:
        components: Built by build_reward_components().
        prev_state: State before this step's physics.
        actions:    Actions taken.
        next_state: State after physics + damage (before auto-reset).
        dones:      (B,) game-over flags.

    Returns:
        (B, N, K) float32 — per-component per-ship rewards in REWARD_COMPONENT_NAMES order.
    """
    B, N = next_state.ship_health.shape
    K = len(REWARD_COMPONENT_NAMES)
    result = torch.zeros(B, N, K, device=next_state.device, dtype=torch.float32)
    for comp in components:
        k = _NAME_TO_K.get(comp.name)
        if k is not None:
            result[:, :, k] = comp.compute(prev_state, actions, next_state, dones)
    return result
