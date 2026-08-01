"""Shared config constants used across multiple run profiles.

Import these into individual profiles rather than duplicating values.
Override in a profile only when the run genuinely needs a different value.
"""

from boost_and_broadside.config import (
    EloCalibrateConfig,
    EloEvalConfig,
    ModelConfig,
    RewardConfig,
    ShipConfig,
)

SHIP_CONFIG = ShipConfig(bullet_energy_cost=2, bullet_min_damage_frac=1.0)

MODEL_CONFIG = ModelConfig(
    d_model=128,
    n_heads=4,
    n_transformer_blocks=2,
    grad_checkpoint=False,
)

ELO_EVAL = EloEvalConfig(
    # 5 slots × 512 envs, stepped every rollout step: 64 rated games per slot per
    # update, and an episode span of 8 updates. Rollout collection has headroom
    # the eval battery can absorb, and wider slices also push the per-slot policy
    # forward passes further from the launch-bound floor.
    envs_per_matchup=512,
    step_interval=1,
    k_factor=4.0,
    scripted_elo_init=1000.0,
    window_size=100,
    # A floating ladder checkpoint must settle over this many rated games
    # before it can be frozen as an anchor — milestones are deferred until then.
    min_games_to_freeze=1000,
)

# Post-training calibration (`--mode elo_calibrate`). Runs once after a run
# finishes, so this budget costs nothing during training.
#
# Measured on vague-lion-678 (7-player field): 4096 envs took ~2 min per batch
# and reached +/-9.7 Elo after 8 batches. At 16384 a batch carries four times the
# games, so the target should fall inside the first two or three; max_batches is
# left generous as a cap rather than an expectation. Precision goes as
# 1/sqrt(games), so halving target_stderr costs roughly four times the games.
ELO_CALIBRATE = EloCalibrateConfig(
    num_envs=16384,
    target_stderr=10.0,
    max_batches=12,
    prior_games=1.0,
    # Matches the refined semi-random ladder (0 and 1 are the random and
    # scripted players themselves).
    reference_probabilities=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
)

# Reward weights shared by all training profiles.
# true_reward_scale/global_scale/local_scale live in each profile's TrainingSchedule.
REWARDS = RewardConfig(
    # Outcome rewards — ally/enemy split so the critic distinguishes symmetric
    # from asymmetric outcomes (e.g. mutual damage vs no damage, standoff vs
    # close fight).
    ally_damage_weight=0.0,
    enemy_damage_weight=0.0,
    ally_death_weight=0.0,
    enemy_death_weight=0.0,
    ally_win_weight=4.0,
    enemy_win_weight=4.0,
    # Dense shaping rewards — prevent passive collapse during early RL.
    facing_weight=0.1,
    closing_speed_weight=0.1,
    shoot_quality_weight=0.1,
    # Per-ship kill credit — self-only (lambda=0 for all other ships).
    # kill_shot: fatal-step credit split in proportion to each ship's damage that step.
    # kill_assist: proportional share based on cumulative episode damage dealt.
    kill_shot_weight=1.0,
    kill_assist_weight=1.0,
    death_weight=1.0,
    # Per-ship local combat credit — self-only (lambda=0 for all other ships).
    # damage_taken: negative reward proportional to health lost this step.
    # damage_dealt: positive reward proportional to enemy health removed this step.
    damage_taken_weight=0.5,
    damage_dealt_enemy_weight=0.5,
    damage_dealt_ally_weight=0.5,
    # Geometry params
    proximity_radius=400.0,
    shoot_quality_radius=200.0,
    # Lambda configuration:
    #   enemy_neg_lambda_components → enemy ships get lambda=-1
    #   ally_zero_components        → ally ships get lambda=0 (enemy-perspective only)
    # enemy_win is zero-sum: allies see -1 when the enemy team wins, letting the
    # critic distinguish win / draw / loss when paired with ally_win.
    enemy_neg_lambda_components=frozenset({"enemy_damage", "enemy_death", "enemy_win"}),
    ally_zero_components=frozenset({"enemy_damage", "enemy_death", "enemy_win"}),
    # Behaviour shaping — off by default in combat mode
    shooting_penalty_weight=0.0,
    speed_weight=0.0,
    speed_penalty_min=10.0,
)

# Per-component GAE discount factors.
# Bucket rationale (250–500 step episodes, dt=1/60 s):
#   Terminal  γ=0.999 → γ^375≈0.69 — win credit survives full episode; λ high for eligibility trace
#   Kill/death γ=0.995 → γ^200≈0.37 — connects decisions to kills within a fight
#   Damage    γ=0.991 → γ^110≈0.36 — local to the combat exchange
#   Shaping   γ=0.975 → γ^40≈0.36  — immediate behaviour; λ low since dense rewards → TD is accurate
COMPONENT_GAMMAS: dict[str, float] = {
    # Terminal — ally_win (+1 win) and enemy_win (-1 loss) both need full-episode horizon
    "ally_win": 0.999,
    "enemy_win": 0.999,
    # Kill/death
    "ally_death": 0.995,
    "enemy_death": 0.995,
    "death": 0.995,
    "kill_shot": 0.995,
    "kill_assist": 0.995,
    # Damage
    "ally_damage": 0.991,
    "enemy_damage": 0.991,
    "damage_taken": 0.991,
    "damage_dealt_enemy": 0.991,
    "damage_dealt_ally": 0.991,
    # Shaping
    "facing": 0.975,
    "closing_speed": 0.975,
    "shoot_quality": 0.975,
    "speed": 0.975,
    "shooting_penalty": 0.975,
}

COMPONENT_LAMBDAS: dict[str, float] = {
    # Terminal — high λ: sparse signal must be traced back through the full episode
    "ally_win": 0.97,
    "enemy_win": 0.97,
    # Kill/death
    "ally_death": 0.95,
    "enemy_death": 0.95,
    "death": 0.95,
    # kill_shot: sparse fatal-step credit is noisy; shorter trace reduces variance
    # kill_assist: episode-level cumulative credit needs a longer trace
    "kill_shot": 0.87,
    "kill_assist": 0.97,
    # Damage — slightly lower; semi-dense rewards make TD errors more informative
    "ally_damage": 0.90,
    "enemy_damage": 0.90,
    "damage_taken": 0.90,
    "damage_dealt_enemy": 0.90,
    "damage_dealt_ally": 0.90,
    # Shaping — low λ: dense rewards → low-variance TD; prevents "style over substance" compounding
    "facing": 0.80,
    "closing_speed": 0.80,
    "shoot_quality": 0.80,
    "speed": 0.80,
    "shooting_penalty": 0.80,
}
