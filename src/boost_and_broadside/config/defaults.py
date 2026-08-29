"""Project-level constants shared by independent training profiles."""

from __future__ import annotations

from boost_and_broadside.config.core import ModelConfig, RewardConfig, ShipConfig
from boost_and_broadside.config.live_elo import LIVE_SCRIPTED_ELO
from boost_and_broadside.config.schedule_spec import TrainingScheduleSpec, hold
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
    # Five numbers. Every event component follows from them by the balance rule
    # documented on RewardConfig; nothing else here is a free choice.
    #
    # They are set to reproduce run 719 -- the strongest policy measured -- and
    # not to hit a tier-share target. Solving the free numbers against measured
    # gradient share was tried twice and lost both times: run 721 landed the flat
    # 31/32/31/5 allocation and finished ~60 Elo behind run 720, and run 724
    # re-solved against 719's own measured split, landed it (tiers within 5%),
    # and finished the worst and most passive of the set. Tier share is evidently
    # not the quantity that separates a strong policy from a passive one, so this
    # stops targeting it and copies the vector that worked instead.
    #
    # What 719 *trained under* is the target, not what its config said. Its
    # config read ``ally_win_weight=1.5``, but the lambda rows were normalized
    # after the weight was applied, which divided the weight back out: every
    # global component came out at an effective total of 1.0 no matter what was
    # configured. Local components were unaffected below 1.0, so the rest of
    # 719's vector is its config. Effective 719, and therefore the target here:
    # win 1.0, death 1.0, damage 0.5, kill_shot 1.0, kill_assist 1.0.
    #
    # Only ratios matter -- the aggregate advantage is divided by its own RMS, so
    # scaling all five together is a no-op.
    win_weight=1.0,
    death_weight=1.0,
    damage_weight=0.5,
    # The one ratio the balance rule leaves free: "landed the finishing blow"
    # against "contributed damage". Even, which is also what 719 carried --
    # kill_shot and kill_assist both at 1.0.
    kill_shot_fraction=0.5,
    # And the one the rule forbids. 719 charged a death 1.0 while paying the
    # kill 2.0, because it carried kill_shot and kill_assist at combat_death's
    # weight; k=2 is that ratio, and with U=1.0 it reproduces both numbers
    # exactly. See RewardConfig for why this is the tier that gets a knob.
    kill_payout_ratio=2.0,
    # And the same for damage, which is the one thing 725 did not copy from 720.
    # 725 reproduced 719 exactly -- parity at 133M and 154M -- and did not reach
    # 720's +58, so matching 719's vector is evidently enough to match 719 and
    # not enough to beat it. 720's damage tier was tilted 1.69:1 toward damage
    # dealt; 2.0 here for symmetry with the kill ratio rather than to chase a
    # number one run cannot resolve. See RewardConfig for the tier-share caveat.
    damage_payout_ratio=2.0,
    # Shaping is not an event, so it stays individually weighted. 719 carried
    # both at 0.10 and did not taper them; the taper argument is unaffected and
    # still recorded in the schedule, but it would be one more difference than
    # this comparison can carry.
    facing_weight=0.1,
    closing_speed_weight=0.1,
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
        # Shaping holds flat here for the same reason the weights copy 719's:
        # 719 carried its shaping undecayed, and this run exists to reproduce
        # that vector with one change in it. The argument for tapering is
        # unaffected and still stands -- shaping's realised share *grows* about
        # 1.58x over a run, and facing and closing speed are not potential-based,
        # so they bias the optimum for as long as they are on, opposing the
        # objective directly (closing_speed against field_damage_taken measured a
        # mean gradient cosine of -0.446, negative in 99.9% of samples). Restore
        # the taper below once the reward vector is settled:
        #
        #     (0, 1.0, "hold"), (100M, 1.0, "exponential"), (400M, 0.05, "hold")
        #
        # with the 0.05 floor rather than 0 so the components stay measurable to
        # the end.
        shaping_scale=hold(1.0),
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
