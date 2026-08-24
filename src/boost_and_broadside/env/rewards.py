"""Modular reward components for the decomposed critic.

All computations are GPU-vectorized. No Python loops over ships or envs.
Each component returns the reward from the perspective of that specific ship —
no zero-sum pre-inversion. Zero-sum accounting is handled at PPO update time
via the lambda aggregation matrix.

REWARD_COMPONENT_NAMES is the full registry; outcome components are split into
ally/enemy pairs so the critic can distinguish symmetric from asymmetric
situations (e.g. mutual damage vs no damage, standoff vs close fight). Shaping
rewards provide dense feedback to prevent passive collapse during early RL.

Adding a new reward
-------------------
1. Create a subclass of RewardComponent with a unique `name` class attribute.
2. Add its name to REWARD_COMPONENT_NAMES (fixes K and value head ordering).
3. Add a weight field to RewardConfig in config/core.py and set it in config/defaults.py.
4. Add an instance to the list in build_reward_components().
5. Classify it in `_TIER` in train/rl/ppo.py (outcome / kill_death / damage /
   shaping scale), and add it to `_LOCAL_COMPONENTS` there if its signal is
   self-only. The two are independent: the tier decides which schedule scales it,
   locality decides whether the lambda matrix propagates it to teammates.
"""

from abc import ABC, abstractmethod

import torch

from boost_and_broadside.config import RewardConfig, ShipConfig
from boost_and_broadside.constants import EPS
from boost_and_broadside.env.state import TensorState


class RewardComponent(ABC):
    """Base class for a single reward signal.

    Subclasses must define:
        name: str — unique key used as W&B metric label.

    ``weight`` is a plain mutable attribute: ppo.py overwrites it once per
    update with individual_weight × schedule group scale, and the env wrapper
    re-reads it via refresh_component_weights().

    Optionally override log_keys / log_breakdown to split a component into
    sub-metrics for logging without changing the training signal.
    """

    name: str

    def __init__(self, weight: float) -> None:
        self.weight = weight

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
# Shared geometry helpers
# ---------------------------------------------------------------------------


def _toroidal_wrap(d: torch.Tensor, world_size: tuple[float, float]) -> torch.Tensor:
    """Wrap complex displacements to the nearest toroidal image.

    Args:
        d: Complex tensor of raw displacements (any shape).
        world_size: (W, H) world extent.

    Returns:
        Complex tensor with real in [-W/2, W/2) and imag in [-H/2, H/2).
    """
    W, H = world_size
    return torch.complex((d.real + W / 2) % W - W / 2, (d.imag + H / 2) % H - H / 2)


def _valid_enemy_pairs(teams: torch.Tensor, alive: torch.Tensor) -> torch.Tensor:
    """(B, N, N) mask of pairs (i, j) where both ships are alive and on opposing teams."""
    is_enemy = teams.unsqueeze(2) != teams.unsqueeze(1)
    return is_enemy & alive.unsqueeze(2) & alive.unsqueeze(1)


# ---------------------------------------------------------------------------
# Outcome reward components — split into ally/enemy pairs
# ---------------------------------------------------------------------------


class _DamageTakenReward(RewardComponent):
    """Negative applied health loss from one exact physics source."""

    source_attr: str

    def compute(
        self,
        prev_state: TensorState,
        actions: torch.Tensor,
        next_state: TensorState,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        damage = getattr(next_state, self.source_attr)  # (B, N)
        return -damage * prev_state.ship_alive.float()


class AllyCombatDamageReward(_DamageTakenReward):
    name = "ally_combat_damage"
    source_attr = "ship_combat_damage"


class EnemyCombatDamageReward(AllyCombatDamageReward):
    name = "enemy_combat_damage"


class AllyFieldDamageReward(_DamageTakenReward):
    name = "ally_field_damage"
    source_attr = "ship_field_damage"


class EnemyFieldDamageReward(AllyFieldDamageReward):
    name = "enemy_field_damage"


class _DeathReward(RewardComponent):
    """Negative one on a death attributed to one exact physics source."""

    source_attr: str

    def compute(
        self,
        prev_state: TensorState,
        actions: torch.Tensor,
        next_state: TensorState,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        return -getattr(next_state, self.source_attr).float()


class AllyCombatDeathReward(_DeathReward):
    name = "ally_combat_death"
    source_attr = "ship_combat_death"


class EnemyCombatDeathReward(AllyCombatDeathReward):
    name = "enemy_combat_death"


class AllyFieldDeathReward(_DeathReward):
    name = "ally_field_death"
    source_attr = "ship_field_death"


class EnemyFieldDeathReward(AllyFieldDeathReward):
    name = "enemy_field_death"


class KillShotReward(RewardComponent):
    """Proportional kill credit based on step-level damage attribution.

    Each ship earns a proportional share of +1.0 per dying enemy, weighted by
    its step-level damage to that ship. Uses state.damage_matrix (step-level).
    Lambda=0 for all other ships (self-only, diagonal lambda).

    Friendly kills are ``kill_ally``, not a negative term folded in here. One
    head cannot usefully predict the sum of a positive enemy-kill signal and a
    negative friendly-kill one, and folding them together made the friendly
    signal unweightable and invisible to every per-component diagnostic.
    """

    name = "kill_shot"

    def compute(
        self,
        prev_state: TensorState,
        actions: torch.Tensor,
        next_state: TensorState,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        just_died = prev_state.ship_alive & ~next_state.ship_alive  # (B, N)

        B, N = next_state.ship_health.shape
        dm = next_state.damage_matrix  # (B, N_shooter, N_target)
        is_enemy_target = next_state.ship_team_id.unsqueeze(2) != next_state.ship_team_id.unsqueeze(
            1
        )  # (B, N_shooter, N_target)
        dying = just_died.unsqueeze(1).float()  # (B, 1, N_target)

        # --- Enemy kill credit (proportional share of +1.0 per kill) ---
        dm_enemy = dm * is_enemy_target.float() * dying
        total_enemy = dm_enemy.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return (dm_enemy / total_enemy).sum(dim=2)


class KillAssistReward(RewardComponent):
    """Proportional kill credit based on cumulative episode damage.

    Each ship earns a proportional share of 1.0 credit per dying enemy,
    weighted by its cumulative damage to that ship. Uses
    state.cumulative_damage_matrix (episode-level). Lambda=0 for all other ships
    (self-only, diagonal lambda).

    Friendly kills are ``kill_ally``; see KillShotReward.
    """

    name = "kill_assist"

    def compute(
        self,
        prev_state: TensorState,
        actions: torch.Tensor,
        next_state: TensorState,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        just_died = prev_state.ship_alive & ~next_state.ship_alive  # (B, N)

        B, N = next_state.ship_health.shape
        cdm = next_state.cumulative_damage_matrix  # (B, N_shooter, N_target)
        is_enemy_target = next_state.ship_team_id.unsqueeze(2) != next_state.ship_team_id.unsqueeze(
            1
        )  # (B, N_shooter, N_target)
        dying = just_died.unsqueeze(1).float()  # (B, 1, N_target)

        # --- Enemy kill credit (proportional share of +1.0 per kill) ---
        cdm_enemy = cdm * is_enemy_target.float() * dying
        total_enemy = cdm_enemy.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return (cdm_enemy / total_enemy).sum(dim=2)


class KillAllyReward(RewardComponent):
    """Proportional blame for a teammate's death, by cumulative episode damage.

    Each ship takes a proportional share of -1.0 per dying teammate, weighted by
    its cumulative damage to that ship. Uses state.cumulative_damage_matrix, so
    a ship that chipped an ally early and left another to finish them keeps its
    share: friendly-fire accountability is the whole contribution to the death,
    not who happened to land last. The ship that dealt the fatal blow is covered
    as the limiting case, since fatal damage is also cumulative damage.

    Split out of KillShotReward and KillAssistReward, which each carried this as
    a negative term inside their own positive signal. Self-only (diagonal lambda).
    """

    name = "kill_ally"

    def compute(
        self,
        prev_state: TensorState,
        actions: torch.Tensor,
        next_state: TensorState,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        just_died = prev_state.ship_alive & ~next_state.ship_alive  # (B, N)

        B, N = next_state.ship_health.shape
        cdm = next_state.cumulative_damage_matrix  # (B, N_shooter, N_target)
        is_enemy_target = next_state.ship_team_id.unsqueeze(2) != next_state.ship_team_id.unsqueeze(
            1
        )  # (B, N_shooter, N_target)
        self_mask = torch.eye(N, dtype=torch.bool, device=cdm.device).unsqueeze(0)
        is_friendly_target = ~is_enemy_target & ~self_mask  # same team, not self

        dying = just_died.unsqueeze(1).float()  # (B, 1, N_target)

        cdm_friendly = cdm * is_friendly_target.float() * dying
        total_friendly = cdm_friendly.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return -(cdm_friendly / total_friendly).sum(dim=2)


class AllyWinReward(RewardComponent):
    """+1 to ships whose team wins at game end; 0 to losers and draws. Lambda=0 for
    enemies. Critic learns P(ally wins), which distinguishes standoff (≈0) from
    close fight (≈0.5) — unlike a single win/loss component where both look like 0."""

    name = "ally_win"

    def compute(
        self,
        prev_state: TensorState,
        actions: torch.Tensor,
        next_state: TensorState,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        reward = torch.zeros_like(next_state.ship_health)
        team0 = next_state.ship_team_id == 0  # (B, N)
        team1 = next_state.ship_team_id == 1  # (B, N)
        t0_alive = (team0 & next_state.ship_alive).sum(dim=1)  # (B,)
        t1_alive = (team1 & next_state.ship_alive).sum(dim=1)  # (B,)
        t0_wins = ((t0_alive > 0) & (t1_alive == 0) & dones).unsqueeze(1)  # (B, 1)
        t1_wins = ((t1_alive > 0) & (t0_alive == 0) & dones).unsqueeze(1)  # (B, 1)
        reward[team0 & t0_wins.expand_as(team0)] = +1.0
        reward[team1 & t1_wins.expand_as(team1)] = +1.0
        return reward


class EnemyWinReward(AllyWinReward):
    """+1 to each ship on the WINNING team at game end; 0 to losers and draws.

    Mirrors AllyWinReward but is consumed from the enemy perspective: PPO applies
    lambda=-1 so allies see -1 when enemies win and 0 when enemies lose, letting
    the critic distinguish three outcomes:
      win  → ally_win=+1, enemy_win= 0
      draw → ally_win= 0, enemy_win= 0
      loss → ally_win= 0, enemy_win=+1 (enemy sees +1; ally sees lambda*+1=-1)"""

    name = "enemy_win"


# ---------------------------------------------------------------------------
# Local per-ship combat rewards — self-only, lambda=0 for all other ships
# ---------------------------------------------------------------------------


class LocalCombatDeathReward(AllyCombatDeathReward):
    """Self-only projectile death penalty."""

    name = "combat_death"


class LocalFieldDeathReward(AllyFieldDeathReward):
    """Self-only field-boundary death penalty."""

    name = "field_death"


class LocalCombatDamageTakenReward(AllyCombatDamageReward):
    """Self-only applied projectile health-loss penalty."""

    name = "combat_damage_taken"


class LocalFieldDamageTakenReward(AllyFieldDamageReward):
    """Self-only applied field-boundary health-loss penalty."""

    name = "field_damage_taken"


class LocalDamageDealtEnemyReward(RewardComponent):
    """Damage dealt by this ship to enemies this step.

    Positive reward proportional to enemy health removed. Reads from
    state.damage_matrix (shooter × target). Self-only: diagonal lambda.
    """

    name = "damage_dealt_enemy"

    def compute(
        self,
        prev_state: TensorState,
        actions: torch.Tensor,
        next_state: TensorState,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        dm = next_state.damage_matrix  # (B, N_shooter, N_target)
        B, N = next_state.ship_team_id.shape
        is_enemy = next_state.ship_team_id.unsqueeze(2) != next_state.ship_team_id.unsqueeze(
            1
        )  # (B, N_shooter, N_target)
        enemy_damage = (dm * is_enemy.float()).sum(dim=2)  # (B, N_shooter)
        return enemy_damage * next_state.ship_alive.float()


class LocalDamageDealtAllyReward(RewardComponent):
    """Friendly-fire penalty: damage dealt by this ship to teammates this step.

    Negative reward proportional to ally health removed. Reads from
    state.damage_matrix (shooter × target). Self-only: diagonal lambda.
    """

    name = "damage_dealt_ally"

    def compute(
        self,
        prev_state: TensorState,
        actions: torch.Tensor,
        next_state: TensorState,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        dm = next_state.damage_matrix  # (B, N_shooter, N_target)
        B, N = next_state.ship_team_id.shape
        is_enemy = next_state.ship_team_id.unsqueeze(2) != next_state.ship_team_id.unsqueeze(
            1
        )  # (B, N_shooter, N_target)
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

    def __init__(self, weight: float, radius: float, world_size: tuple[float, float]) -> None:
        super().__init__(weight)
        self.radius = radius
        self.world_size = world_size

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

        R = self.radius

        d = _toroidal_wrap(pos.unsqueeze(2) - pos.unsqueeze(1), self.world_size)  # pos_i - pos_j
        dist = d.abs()  # (B, N, N)

        dir_j_to_i = d / dist.clamp(min=EPS)

        att_i = att.unsqueeze(2)  # (B, N, 1)
        alignment = (att_i * torch.conj(-dir_j_to_i)).real  # (B, N, N)

        prox = (1.0 - dist / R).clamp(min=0.0)  # (B, N, N)
        score = prox * alignment.clamp(min=0.0)  # (B, N, N)

        valid = _valid_enemy_pairs(teams, alive)

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
        weight: float,
        world_size: tuple[float, float],
        max_speed: float,
    ) -> None:
        super().__init__(weight)
        self.world_size = world_size
        self.max_speed = max_speed

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

        d = _toroidal_wrap(pos.unsqueeze(2) - pos.unsqueeze(1), self.world_size)  # pos_i - pos_j
        dist = d.abs()  # (B, N, N)

        dir_j_to_i = d / dist.clamp(min=EPS)

        valid = _valid_enemy_pairs(teams, alive)

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
        weight: float,
        radius: float,
        world_size: tuple[float, float],
    ) -> None:
        super().__init__(weight)
        self.radius = radius
        self.world_size = world_size

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

        R = self.radius

        d = _toroidal_wrap(pos.unsqueeze(2) - pos.unsqueeze(1), self.world_size)  # pos_i - pos_j
        dist = d.abs()  # (B, N, N)

        dir_j_to_i = d / dist.clamp(min=EPS)

        # Facing: dot(att_i, dir_i_to_j)  where dir_i_to_j = -dir_j_to_i
        att_i = att.unsqueeze(2)  # (B, N, 1)
        facing = (att_i * torch.conj(-dir_j_to_i)).real  # (B, N, N)

        # Proximity
        prox = (1.0 - dist / R).clamp(min=0.0)  # (B, N, N)

        # Shot quality: only positive when aimed AND close
        quality = 2.0 * facing.clamp(min=0.0) * prox - 1.0  # (B, N, N) in [-1, 1]

        valid = _valid_enemy_pairs(teams, alive)

        # Best quality over all valid enemies (judge against most favourable target)
        quality_masked = quality.masked_fill(~valid, -1.0)
        best_quality = quality_masked.max(dim=2).values  # (B, N)
        best_quality = best_quality * valid.any(dim=2).float()  # 0 when no enemies

        return shooting * best_quality * alive.float()


class ShootingPenaltyReward(RewardComponent):
    """Negative reward on every step this ship fires."""

    name = "shooting_penalty"

    def compute(
        self,
        prev_state: TensorState,
        actions: torch.Tensor,
        next_state: TensorState,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        return -next_state.ship_is_shooting.float() * next_state.ship_alive.float()


class SpeedReward(RewardComponent):
    """Penalty proportional to how far below min_speed the ship is traveling.

    Returns 0 at speed >= min_speed, -1 at speed = 0, linear between.
    """

    name = "speed"

    def __init__(self, weight: float, min_speed: float) -> None:
        super().__init__(weight)
        self.min_speed = min_speed

    def compute(
        self,
        prev_state: TensorState,
        actions: torch.Tensor,
        next_state: TensorState,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        speed = next_state.ship_local_index * next_state.ship_vel.abs()  # proper speed
        penalty = ((speed - self.min_speed) / self.min_speed).clamp(min=-1.0, max=0.0)
        return penalty * next_state.ship_alive.float()


# ---------------------------------------------------------------------------
# Component registry
# ---------------------------------------------------------------------------

REWARD_COMPONENT_NAMES: tuple[str, ...] = (
    "ally_combat_damage",  #  0 — applied projectile damage to allies
    "enemy_combat_damage",  #  1 — applied projectile damage to enemies
    "ally_field_damage",  #  2 — applied boundary damage to allies
    "enemy_field_damage",  #  3 — applied boundary damage to enemies
    "ally_combat_death",  #  4 — ally projectile deaths
    "enemy_combat_death",  #  5 — enemy projectile deaths
    "ally_field_death",  #  6 — ally boundary deaths
    "enemy_field_death",  #  7 — enemy boundary deaths
    "ally_win",  #  8 — ally team wins (positive)
    "enemy_win",  #  9 — enemy team wins (negative for allies via lambda)
    "facing",  # 10 — pointing at nearest enemy (shaping, self only)
    "closing_speed",  # 11 — velocity toward nearest enemy (shaping, self only)
    "shoot_quality",  # 12 — shot quality when firing (shaping, self only)
    "kill_shot",  # 13 — proportional kill credit from step-level damage (self only)
    "kill_assist",  # 14 — cumulative combat credit, including field-finished kills
    "kill_ally",  # 15 — proportional blame for a teammate's death (self only)
    "combat_damage_taken",  # 16 — applied projectile damage to this ship
    "field_damage_taken",  # 17 — applied boundary damage to this ship
    "damage_dealt_enemy",  # 18 — damage dealt to enemies this step (self only)
    "damage_dealt_ally",  # 19 — damage dealt to allies — friendly-fire penalty
    "combat_death",  # 20 — projectile death of this ship (self only)
    "field_death",  # 21 — boundary death of this ship (self only)
    "shooting_penalty",  # 22 — negative reward on every shot (self only)
    "speed",  # 23 — penalty when proper speed < min_speed (self only)
)

_NAME_TO_K: dict[str, int] = {name: k for k, name in enumerate(REWARD_COMPONENT_NAMES)}


def build_reward_components(
    rewards: RewardConfig,
    ship_config: ShipConfig,
) -> list[RewardComponent]:
    """Construct one instance of every registered reward component from config.

    Called once at PPOTrainer init. Individual component weights are updated
    live each update by the group-scale multipliers in the training schedule.

    Args:
        rewards:     Reward weights and geometry params.
        ship_config: Physics config (provides world_size).

    Returns:
        One RewardComponent per entry in REWARD_COMPONENT_NAMES, in order.
    """
    return [
        AllyCombatDamageReward(weight=rewards.ally_combat_damage_weight),
        EnemyCombatDamageReward(weight=rewards.enemy_combat_damage_weight),
        AllyFieldDamageReward(weight=rewards.ally_field_damage_weight),
        EnemyFieldDamageReward(weight=rewards.enemy_field_damage_weight),
        AllyCombatDeathReward(weight=rewards.ally_combat_death_weight),
        EnemyCombatDeathReward(weight=rewards.enemy_combat_death_weight),
        AllyFieldDeathReward(weight=rewards.ally_field_death_weight),
        EnemyFieldDeathReward(weight=rewards.enemy_field_death_weight),
        AllyWinReward(weight=rewards.ally_win_weight),
        EnemyWinReward(weight=rewards.enemy_win_weight),
        FacingReward(
            weight=rewards.facing_weight,
            radius=rewards.proximity_radius,
            world_size=ship_config.world_size,
        ),
        ClosingSpeedReward(
            weight=rewards.closing_speed_weight,
            world_size=ship_config.world_size,
            max_speed=ship_config.max_speed,
        ),
        ShootQualityReward(
            weight=rewards.shoot_quality_weight,
            radius=rewards.shoot_quality_radius,
            world_size=ship_config.world_size,
        ),
        KillShotReward(weight=rewards.kill_shot_weight),
        KillAssistReward(weight=rewards.kill_assist_weight),
        KillAllyReward(weight=rewards.kill_ally_weight),
        LocalCombatDamageTakenReward(weight=rewards.combat_damage_taken_weight),
        LocalFieldDamageTakenReward(weight=rewards.field_damage_taken_weight),
        LocalDamageDealtEnemyReward(weight=rewards.damage_dealt_enemy_weight),
        LocalDamageDealtAllyReward(weight=rewards.damage_dealt_ally_weight),
        LocalCombatDeathReward(weight=rewards.combat_death_weight),
        LocalFieldDeathReward(weight=rewards.field_death_weight),
        ShootingPenaltyReward(weight=rewards.shooting_penalty_weight),
        SpeedReward(weight=rewards.speed_weight, min_speed=rewards.speed_penalty_min),
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
