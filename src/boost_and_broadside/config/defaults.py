"""Project-level constants shared by independent training profiles."""

from __future__ import annotations

from boost_and_broadside.config.core import ModelConfig, RewardConfig, ShipConfig
from boost_and_broadside.config.live_elo import LIVE_SCRIPTED_ELO
from boost_and_broadside.config.schedule_spec import TrainingScheduleSpec, hold
from boost_and_broadside.config.training import EloCalibrateConfig, EloEvalConfig

# Frontal armour, re-enabled at 0.3: a head-on hit lands 30% of its damage, and
# the mitigation falls off with the hit angle so a broadside still lands in full.
# 1.0 disables the term entirely, which is what every run from 719 to 730 used.
#
# This changes the physics, so ratings do not carry across the boundary. The live
# gauge is defined per environment -- random at 0, scripted at 1000 -- and the
# span between those two points is a property of the game, not a constant, so a
# run under this config cannot be compared on Elo to 719-730 however either was
# measured. Cross-config comparison needs a shared opponent played under one
# physics, which is what `bnb crossover` measures.
SHIP_CONFIG = ShipConfig(bullet_energy_cost=2, bullet_min_damage_frac=0.3)

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
    scripted_live_elo=LIVE_SCRIPTED_ELO,
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

# Interior rungs of the live measurement ladder, as scripted-action
# probabilities.  Their live ratings are *derived* — 1000·p, see config/live_elo
# — so this is the only choice a profile makes about the ladder, and no
# environment carries a fitted gauge of its own any more.
#
# Nine rungs cost nothing to evaluate (the whole stationary ladder is one
# scripted call and one random call) and they cover the climb densely enough
# that the live rating stays identified between random and scripted.  They
# deliberately match ELO_CALIBRATE.reference_probabilities so `bnb semi-random`
# measures the same rungs training rates against, but the two are independent
# settings: post-hoc calibration fits a field, training defines a gauge.
LIVE_REFERENCE_PROBABILITIES: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)

REWARDS = RewardConfig(
    # Five numbers, solved rather than chosen. Every event component follows from
    # them by the balance rule documented on RewardConfig.
    #
    # The target is run 720 -- the only configuration measured that beat run 719,
    # by +58 Elo at matched steps on a joint calibration. Its weights were not
    # derived: they were solved per component as ``w = share / d`` against measured
    # gradient coherence, which is why no two of them are equal. These five numbers
    # are the closest the derivation can come to that vector, by least squares on
    # log weights -- log space because the weights span 0.08 to 1.0 and only ratios
    # matter, so a 10% error on ``facing`` should count like a 10% error on
    # ``ally_win``. The fit is exact in closed form and was checked against a
    # numeric optimiser; it lands within 6% RMS of 720, and the residual is
    # irreducible because the rule forces pairs equal that 720 had unequal
    # (``combat_damage_taken`` 0.32 against ``field_damage_taken`` 0.26 is the
    # worst of them, and that spread came out of 720's per-component solve rather
    # than out of any principle).
    #
    # Run 725 established what this is *not*: 719's own vector, which reproduced
    # 719 exactly -- parity on a joint fit at 133M and 154M -- and did not come
    # near 720. Matching 719 is evidently enough to match 719 and not enough to
    # beat it, so this stops copying 719 and reconstructs 720 instead.
    #
    # Only ratios matter -- the aggregate advantage is divided by its own RMS, so
    # scaling all of these together is a no-op. They are stated against a win of
    # 1.0 for that reason.
    win_weight=1.0,
    death_weight=0.283,
    damage_weight=0.274,
    # The one ratio the balance rule leaves free. Solved at 0.4875 and set even:
    # nothing distinguishes them (6.0% RMS against 5.8%), and an even split is the
    # standing principle.
    kill_shot_fraction=0.5,
    # The two ratios the rule forbids, tied to one another and solved as a single
    # free number. Both tiers in 720 were tilted toward the side that caused the
    # event -- kills 2.15 against deaths, damage dealt 1.86 against damage taken --
    # and one shared ratio is the smaller claim: an event pays the aggressor twice
    # what it charges the victim, everywhere, rather than two independently tuned
    # numbers. The solve returns 1.96 for the shared ratio, which the fit cannot
    # tell from 2.0 (5.97% RMS against 6.01%), so it is 2.0 -- also the value runs
    # 725 and 726 carry, which keeps one ratio across all three.
    kill_payout_ratio=2.0,
    damage_payout_ratio=2.0,
    # Shaping is not an event, so it stays individually weighted, and these two
    # are 720's own values rather than solved -- the derivation has nothing to say
    # about them.
    facing_weight=0.09,
    closing_speed_weight=0.08,
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
    "kill_ally_shot": 0.995,
    "kill_ally_assist": 0.995,
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
    "kill_ally_shot": 0.87,
    "kill_ally_assist": 0.97,
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
    """The current RL schedule, as keypoint tables."""

    return TrainingScheduleSpec(
        # Peak 4.5e-4, decaying to a third of it. The last row holds, so a budget
        # longer than 500M steps trains its tail at the floor rather than
        # continuing to decay.
        learning_rate=(
            (0, 1e-7, "linear"),
            (5_000_000, 4.5e-4, "hold"),
            (100_000_000, 4.5e-4, "exponential"),
            (500_000_000, 1.5e-4, "hold"),
        ),
        policy_gradient_coef=hold(1.0),
        entropy_coef=hold(0.005),
        behavior_cloning_coef=hold(2.0),
        value_function_coef=hold(1.0),
        sigreg_coef=hold(0.00),
        # Tier scales ride on top of the per-component weights. Three of the
        # four hold: the realised tier shares already drift the way a curriculum
        # would move them, with the outcome tier rising about 1.29x over a run
        # and the kill/death tier falling to 0.73x as the policy stops dying in
        # ways it can still learn from. Scheduling those would fight a trend
        # rather than create one.
        outcome_scale=hold(1.0),
        kill_death_scale=hold(1.0),
        damage_scale=hold(1.0),
        # Shaping is the exception, and it is also what run 720 carried -- this is
        # the last config difference between that run and this one. It has to be
        # pushed down rather than left alone: its realised share *grows* about
        # 1.58x over a run. Facing and closing speed are not potential-based, so
        # they bias the optimum for as long as they are on, and they oppose the
        # objective directly -- closing_speed against field_damage_taken measured
        # a mean gradient cosine of -0.446, negative in 99.9% of samples. They
        # exist to stop early passive collapse, and that job is finished long
        # before the budget is. The floor is 0.05 rather than 0 so the components
        # stay measurable to the end: their gradient share and explained variance
        # remain readable, which is how the next run learns whether shaping was
        # still buying anything.
        #
        # Note 720 only reached 127M, so it ran barely 27M steps into this taper
        # and ended near 0.76. Everything the taper does past that point is
        # untested by the run this vector reconstructs.
        shaping_scale=(
            (0, 1.0, "hold"),
            (100_000_000, 1.0, "exponential"),
            (400_000_000, 0.05, "hold"),
        ),
        league_fraction=hold(0.5),
        # Every update.  A save costs ~48 ms of blocking device-to-host copy
        # against an update measured in minutes, and the writer already skips
        # itself rather than queueing when a previous save is still running, so
        # the interval buys no throughput -- it only decides how much progress
        # an interrupted run throws away.
        checkpoint_interval=hold(1),
        num_epochs=hold(4),
        target_kl=hold(0.1),
        high_winrate_threshold=hold(0.8),
        high_winrate_target_kl=hold(0.02),
    )
