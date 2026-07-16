"""Shared config constants used across multiple run profiles.

Import these into individual profiles rather than duplicating values.
Override in a profile only when the run genuinely needs a different value.
"""

from boost_and_broadside.config import LeagueEvalConfig, ModelConfig, RewardConfig, ShipConfig

SHIP_CONFIG = ShipConfig(bullet_energy_cost=2, bullet_min_damage_frac=1.0)

MODEL_CONFIG = ModelConfig(
    d_model=128,
    n_heads=4,
    n_transformer_blocks=2,
)

LEAGUE_EVAL = LeagueEvalConfig(
    eval_num_envs=1536,
    step_interval=4,
    eval_pairs=12,
    eval_slots=6,
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
    # kill_shot: winner-take-all to the top-damage dealer in the fatal step.
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
    # Obstacle rewards
    obstacle_death_weight=0.0,
    obstacle_proximity_weight=0.0,
    obstacle_closing_speed_weight=0.0,
    obstacle_tti_weight=0.0,
    obstacle_proximity_radius=80.0,
    obstacle_tti_max=3.0,
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
    "obstacle_death": 0.995,
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
    "obstacle_proximity": 0.975,
    "obstacle_closing_speed": 0.975,
    "obstacle_tti": 0.975,
}

COMPONENT_LAMBDAS: dict[str, float] = {
    # Terminal — high λ: sparse signal must be traced back through the full episode
    "ally_win": 0.97,
    "enemy_win": 0.97,
    # Kill/death
    "ally_death": 0.95,
    "enemy_death": 0.95,
    "death": 0.95,
    # kill_shot: winner-take-all is noisy; shorter trace reduces variance
    # kill_assist: episode-level cumulative credit needs a longer trace
    "kill_shot": 0.87,
    "kill_assist": 0.97,
    "obstacle_death": 0.95,
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
    "obstacle_proximity": 0.80,
    "obstacle_closing_speed": 0.80,
    "obstacle_tti": 0.80,
}
