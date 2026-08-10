"""Project-level constants shared by independent training profiles."""

from __future__ import annotations

from dataclasses import replace

from boost_and_broadside.config.core import ModelConfig, RewardConfig, ShipConfig
from boost_and_broadside.config.schedule_spec import (
    TrainingScheduleSpec,
    constant_spec,
    exponential_spec,
    join_spec,
    linear_spec,
    stepped_spec,
)
from boost_and_broadside.config.training import EloCalibrateConfig, EloEvalConfig

SHIP_CONFIG = ShipConfig(bullet_energy_cost=2, bullet_min_damage_frac=1.0)

MODEL_CONFIG = ModelConfig(
    d_model=128,
    n_heads=4,
    n_yemong_blocks=2,
    # Two spatial sublayers per temporal sublayer buy relational depth cheaply.
    n_spatial_per_block=2,
    n_temporal_per_block=1,
    # Read bullets early enough that a later spatial layer can share the signal.
    n_bullet_cross_per_block=1,
    grad_checkpoint=False,
)

ELO_EVAL = EloEvalConfig(
    # Five 512-env slices advance every rollout step; a floating ladder policy
    # must settle for 1000 games before promotion.
    envs_per_matchup=512,
    step_interval=1,
    k_factor=4.0,
    scripted_elo_init=1000.0,
    window_size=100,
    min_games_to_freeze=1000,
)

ELO_CALIBRATE = EloCalibrateConfig(
    # Post-training only: prefer wide batches, with max_batches as a safety cap.
    num_envs=16384,
    target_stderr=10.0,
    max_batches=12,
    prior_games=1.0,
    reference_probabilities=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
)

# Live reference gauges, fitted by the legacy semi-random tournament on 4v4 at
# action_repeat=2.  They are properties of an environment and its stationary
# controllers, not of a training objective, so every profile that trains in that
# environment rates on the same gauge.  S12 replaces the fitted numbers with the
# accepted live-Elo contract; keeping them here means it edits one place.
ZERO_FIELD_REFERENCE_LADDER: tuple[tuple[float, float], ...] = (
    (0.2, -236.0),
    (0.3, -96.2),
    (0.4, 114.9),
    (0.5, 270.7),
    (0.6, 461.1),
    (0.7, 589.2),
    (0.8, 733.0),
    (0.9, 861.3),
    (0.95, 942.5),
)
ZERO_FIELD_RANDOM_ELO = -363.9

# Fields compress the fitted skill scale, so a field environment cannot borrow
# the zero-field gauge without corrupting Elo-proximity opponent sampling.
FIELD_REFERENCE_LADDER: tuple[tuple[float, float], ...] = (
    (0.2, 238.8),
    (0.3, 322.6),
    (0.4, 435.0),
    (0.5, 550.4),
    (0.6, 656.9),
    (0.7, 753.6),
    (0.8, 824.3),
    (0.9, 939.9),
    (0.95, 988.9),
)
FIELD_RANDOM_ELO = 132.3

REWARDS = RewardConfig(
    # Source-split outcome heads are retained even when their weights are zero.
    ally_combat_damage_weight=0.0,
    enemy_combat_damage_weight=0.0,
    ally_field_damage_weight=0.0,
    enemy_field_damage_weight=0.0,
    ally_combat_death_weight=0.0,
    enemy_combat_death_weight=0.0,
    ally_field_death_weight=0.0,
    enemy_field_death_weight=0.0,
    # The win pair receives 32% of the normalized reward-component mix: strong
    # enough to express the objective without drowning dense engagement shaping.
    ally_win_weight=1.5,
    enemy_win_weight=1.5,
    facing_weight=0.1,
    closing_speed_weight=0.1,
    shoot_quality_weight=0.1,
    # Per-ship credit stays local under the lambda aggregation matrix.
    kill_shot_weight=1.0,
    kill_assist_weight=1.0,
    combat_death_weight=1.0,
    field_death_weight=0.0,
    combat_damage_taken_weight=0.5,
    field_damage_taken_weight=0.0,
    damage_dealt_enemy_weight=0.5,
    damage_dealt_ally_weight=0.5,
    proximity_radius=400.0,
    shoot_quality_radius=200.0,
    enemy_neg_lambda_components=frozenset(
        {
            "enemy_combat_damage",
            "enemy_field_damage",
            "enemy_combat_death",
            "enemy_field_death",
            "enemy_win",
        }
    ),
    ally_zero_components=frozenset(
        {
            "enemy_combat_damage",
            "enemy_field_damage",
            "enemy_combat_death",
            "enemy_field_death",
            "enemy_win",
        }
    ),
    shooting_penalty_weight=0.0,
    speed_weight=0.0,
    speed_penalty_min=10.0,
)

FIELD_REWARDS = replace(
    REWARDS,
    field_damage_taken_weight=0.5,
    field_death_weight=1.0,
)

# Values are expressed per 60 Hz physics tick.  The resolver raises them to
# action_repeat so decision-step horizons remain normalized to game time.
# Gamma buckets are win=.999, kill/death=.995, damage=.991, shaping=.975;
# their approximate horizons are full episode, engagement, exchange, and
# immediate geometry respectively.
COMPONENT_GAMMAS_PER_TICK: dict[str, float] = {
    "ally_win": 0.999,
    "enemy_win": 0.999,
    "ally_combat_death": 0.995,
    "enemy_combat_death": 0.995,
    "ally_field_death": 0.995,
    "enemy_field_death": 0.995,
    "combat_death": 0.995,
    "field_death": 0.995,
    "kill_shot": 0.995,
    "kill_assist": 0.995,
    "ally_combat_damage": 0.991,
    "enemy_combat_damage": 0.991,
    "ally_field_damage": 0.991,
    "enemy_field_damage": 0.991,
    "combat_damage_taken": 0.991,
    "field_damage_taken": 0.991,
    "damage_dealt_enemy": 0.991,
    "damage_dealt_ally": 0.991,
    "facing": 0.975,
    "closing_speed": 0.975,
    "shoot_quality": 0.975,
    "speed": 0.975,
    "shooting_penalty": 0.975,
}

COMPONENT_LAMBDAS_PER_TICK: dict[str, float] = {
    "ally_win": 0.97,
    "enemy_win": 0.97,
    "ally_combat_death": 0.95,
    "enemy_combat_death": 0.95,
    "ally_field_death": 0.95,
    "enemy_field_death": 0.95,
    "combat_death": 0.95,
    "field_death": 0.95,
    "kill_shot": 0.87,
    "kill_assist": 0.97,
    "ally_combat_damage": 0.90,
    "enemy_combat_damage": 0.90,
    "ally_field_damage": 0.90,
    "enemy_field_damage": 0.90,
    "combat_damage_taken": 0.90,
    "field_damage_taken": 0.90,
    "damage_dealt_enemy": 0.90,
    "damage_dealt_ally": 0.90,
    "facing": 0.80,
    "closing_speed": 0.80,
    "shoot_quality": 0.80,
    "speed": 0.80,
    "shooting_penalty": 0.80,
}


def make_rl_schedule_spec() -> TrainingScheduleSpec:
    """Return independent declarative intent for the current RL schedule."""

    return TrainingScheduleSpec(
        learning_rate=join_spec(
            (0, linear_spec((0, 1e-7), (5_000_000, 3e-4))),
            (5_000_000, constant_spec(3e-4)),
            (100_000_000, exponential_spec((100_000_000, 3e-4), (500_000_000, 1e-4))),
        ),
        policy_gradient_coef=constant_spec(1.0),
        entropy_coef=constant_spec(0.005),
        behavior_cloning_coef=constant_spec(2.0),
        value_function_coef=constant_spec(1.0),
        sigreg_coef=constant_spec(0.00),
        true_reward_scale=constant_spec(1.0),
        global_scale=constant_spec(1.0),
        local_scale=constant_spec(1.0),
        league_fraction=constant_spec(0.5),
        checkpoint_interval=constant_spec(50),
        num_epochs=stepped_spec((0, 4)),
        target_kl=stepped_spec((0, 0.1)),
        high_winrate_threshold=constant_spec(0.8),
        high_winrate_target_kl=constant_spec(0.02),
    )


def make_bc_schedule_spec() -> TrainingScheduleSpec:
    """Return independent declarative intent for the supervised BC schedule.

    Every value the behavior-cloning objective does not require is the current
    project value.  The five that differ from :func:`make_rl_schedule_spec` are
    named and tested by the BC-versus-RL allowed-difference invariant.
    """

    return TrainingScheduleSpec(
        # Warm up to the project learning rate, then hold.  RL's decay tail is
        # keyed to keypoints at 100M and 500M steps -- the end of *its* budget --
        # and means nothing on BC's own, much longer budget.
        learning_rate=linear_spec((0, 1e-7), (6_000_000, 3e-4)),
        # No policy gradient: the scripted controller supplies supervised action
        # targets and never takes a side in the rollout.
        policy_gradient_coef=constant_spec(0.0),
        entropy_coef=constant_spec(0.005),
        # In BC this is the policy head's only learning signal, deliberately
        # balanced one-to-one against the next-state auxiliary BC also weights
        # at 1.0.  RL's 2.0 is the strength of an *auxiliary* imitation term
        # carried alongside a live policy gradient.
        behavior_cloning_coef=constant_spec(1.0),
        value_function_coef=constant_spec(1.0),
        sigreg_coef=constant_spec(0.0),
        # All component groups stay active so the critic learns the full reward
        # signal before RL begins.
        true_reward_scale=constant_spec(1.0),
        global_scale=constant_spec(1.0),
        local_scale=constant_spec(1.0),
        # League opposition disabled: no roster opponent plays a BC rollout.
        league_fraction=constant_spec(0.0),
        checkpoint_interval=constant_spec(50),
        num_epochs=stepped_spec((0, 4)),
        # A KL trust region early-stops epochs when the policy moves away from
        # the one that produced the rollout.  Under supervision that movement is
        # the objective, so the PPO stopping criterion does not apply.
        target_kl=constant_spec(None),
        high_winrate_threshold=constant_spec(0.8),
        high_winrate_target_kl=constant_spec(0.02),
    )
