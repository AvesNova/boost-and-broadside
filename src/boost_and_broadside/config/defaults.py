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
    # Projectile physics remains authoritative, but normal Frontline training
    # does not spend policy memory/compute on per-bullet K/V observations.
    n_bullet_cross_per_block=0,
    grad_checkpoint=False,
    # Two 64-wide spatial heads rather than four 32-wide ones. This is not a
    # preference, it is what ``spatial_rope`` costs: the Frontline basis rotates
    # 8 x + 8 y + 4 attitude frequency pairs = 40 head dimensions, which does not
    # fit a 32-wide head at all. The temporal sublayers and the value head's
    # TeamPMA keep ``n_heads=4``, so this buys the rotation without changing
    # anything else about the trunk's shape.
    n_spatial_heads=2,
    # Per-entity-type first projection, with a shared second layer. A field token
    # otherwise spends most of its input width on ship-only channels that are hard
    # zeros for it. Each projection runs over its own contiguous span of the token
    # axis rather than over all N+M tokens with a ``torch.where``: measured on the
    # 26-token Frontline layout, 1.48x faster at the 960-env training batch and
    # 1.22x at 256, crossing over near 150. Below that the slicing overhead wins
    # and it is slower -- 0.78x at 64 -- which is the interactive case, where
    # throughput does not matter.
    encoder_split=True,
    # Rotate spatial Q/K by world geometry, so displacement enters the attention
    # score directly instead of being reconstructed by the trunk from absolute
    # position features. The rotation reuses the encoder's own position basis
    # verbatim -- ``tests/models/test_spatial_geometry.py`` pins the two
    # frequency vectors to bitwise equality -- so the features and the rotations
    # cannot describe different geometry.
    spatial_rope=True,
    # Smooth ally and enemy presence within a fixed physical radius. Attention
    # returns proportions and cannot report cardinality, so without these two
    # scalars nothing in the observation says how crowded a neighbourhood is.
    local_presence=True,
    # Shared pairwise bias on spatial attention scores from proximity, ego-frame
    # bearing and range rate. Zero-initialised, so it starts as the identity and
    # earns its contribution.
    #
    # Off while the spatial block is being reworked. It materialises an explicit
    # (N+M, N+M) bias, which is what stops attention reaching the maskless SDPA
    # flash kernel, and the rotation now carries displacement into the score
    # directly -- so the two overlap in what they encode. Measure it again
    # against the fused block rather than carrying it through the change.
    relational_bias=False,
)

ELO_EVAL = EloEvalConfig(
    # Five 1024-env slices advance every *second* rollout step; a floating ladder
    # policy must settle for 1000 games before promotion.
    #
    # The width and the cadence are one decision, and the product is what buys
    # measurements: rated games are proportional to environments x calls. Cost
    # is not. The evaluator issues around twenty thousand operations per call
    # whatever batch they cover, and this pipeline is dispatch-bound, so its
    # bill tracks *calls alone* -- measured at 44.0, 21.5 and 10.7 seconds per
    # update at intervals of 1, 2 and 4, and flat to within noise when the
    # environment count moved by 4x at a fixed interval.
    #
    # So trading cadence for width is very close to free. Against the previous
    # 512 every step, this simulates exactly the same 983,040 evaluator
    # env-decisions per update, finishes the same number of episodes (93 against
    # 108 over three updates, which is sampling noise at those counts), and
    # costs 26.8 s per update instead of 44.0 -- **+15.7% end-to-end training
    # throughput** on an RTX 4070 Laptop. Going further (2048 every fourth step)
    # does not pay: the per-environment term starts to bite and peak memory
    # jumps 1.4 GB.
    #
    # What it does change: an evaluation episode now spans twice as many
    # training updates, so a rated game is played by a slightly more
    # heterogeneous mixture of live-policy versions. The rating is a filtered
    # online estimate either way, and `elo_diag/movement_z` is the series that
    # would show it if the filter became noisier than the games support.
    # See docs/engineering/rl-throughput.md.
    #
    # Back to every rollout step. The +15.7% above was measured against a
    # 1024-step episode, where the evaluator finished a game every 8 updates and
    # the extra lag was 8 updates rather than a meaningful fraction of the run.
    # Frontline episodes are 9,000 steps, which makes the same trade land very
    # differently: the eval environment advances `num_steps / step_interval`
    # steps per update, so at interval 2 it ran one step per 7,680 training
    # steps and a rated game cost 69.1M of them -- about seven games' worth in a
    # 500M-step run, with nothing rated at all for the first 141 updates. Since
    # live Elo gates opponent selection, milestone placement, the
    # behavior-cloning decay and the trust region, all four spent that time on
    # their defaults. Interval 1 halves it to 34.6M steps per rated game and
    # doubles the rate to 14.6 games per update, and the throughput it costs is
    # the cheapest thing on the table to pay with.
    envs_per_matchup=1024,
    step_interval=1,
    k_factor=4.0,
    scripted_live_elo=LIVE_SCRIPTED_ELO,
    # 500 games rather than 100. This window is a boxcar over rated games, and it
    # is read by three things that are not just display: the behavior-cloning
    # decay, the high-win-rate trust-region tightening, and the logged win rate.
    # At the steady-state rate of roughly fifty rated games per update it held
    # about two updates of evidence, so a sampling swing of a few points in the
    # win rate moved `bc_coef` by 40% from one update to the next -- visible as a
    # 1.2-to-2.0 sawtooth on run 732. That oscillates the objective itself, and
    # through a shared trunk it reaches the critic and next-state heads, not only
    # the actor.
    #
    # A boxcar rather than an EMA on purpose: this gate should fire once and stay
    # fired, and an exponential tail keeps re-admitting stale low win rates long
    # after the policy has passed the bar. Five hundred games is roughly a five
    # to ten million step horizon at the steady-state rate, with a hard cutoff.
    # It costs no device memory -- the windows are host-side deques of Python
    # floats, filled after `.cpu().tolist()`.
    window_size=500,
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
    # Event weights carry forward the previous Frontline tier balance as a
    # starting point. Shield-game gradient shares still need a long training run.
    # The balance rule derives paired component weights from these event costs.
    win_weight=1.0,
    death_weight=0.283,
    damage_weight=0.274,
    # Divide kill credit evenly between the finishing shot and prior damage.
    kill_shot_fraction=0.5,
    # Initial offensive premium; offensive_bias tapers all three payout ratios
    # to 1:1, including captures, for the final zero-sum training phase.
    kill_payout_ratio=2.0,
    damage_payout_ratio=2.0,
    # Both off. These were 720's values, carried over from a deathmatch where the
    # only thing to do was fight. They are not potential-based, so they bias the
    # optimum for as long as they are on, and in Frontline the bias points away
    # from the objective: holding a point means breaking off a chase and sitting
    # still, which costs both. Run 735 made the consequence concrete -- once
    # behavior cloning decayed at 50M steps and stopped supplying the scripted
    # prior, zone occupancy fell from 0.075 of live ship-steps to 0.0006, front
    # advances from 220 an update to 1, and 90% of matches ended level with the
    # clock run out while total reward rose 63%. The policy was not failing to
    # capture; it had stopped entering the zones at all.
    facing_weight=0.0,
    closing_speed_weight=0.0,
    proximity_radius=400.0,
    shoot_quality_radius=200.0,
    enemy_neg_lambda_components=frozenset(
        {
            "enemy_combat_damage",
            "enemy_combat_death",
            "enemy_win",
        }
    ),
    ally_zero_components=frozenset(
        {
            "enemy_combat_damage",
            "enemy_combat_death",
            "enemy_win",
        }
    ),
    shooting_penalty_weight=0.0,
    # The strategic tier, stated as what the *absent* side is charged.
    # ``capture_payout_ratio`` then pays the side holding the point twice that,
    # the same 2:1 the kill and damage tiers carry -- but expressed differently.
    # Kills and deaths are separate components, so their ratio lives in the
    # weights; a capture component carries both sides internally, so its ratio
    # lives in the reward and the scaler normalizes the component as a whole.
    # The ratio therefore shapes offense against defense *within* the tier and
    # does not change the tier's share of the gradient.
    #
    # Capture and capture progress are now equal rather than the former 2:1
    # ladder between them. They are two of the five tiers the balance rule
    # names, and the rule asks for equal pressure across tiers; the ordering
    # that used to separate them is what the ladder inside each tier is for.
    #
    # Both are totals rather than rates: a meter runs 0 -> 1 over one capture, so
    # these compare to the kill payout without further arithmetic.
    capture_payout_ratio=2.0,
    # Token weight: trains the head, does not move the policy. Outside the tier
    # balance by design -- it is a value-head probe, not a fifth of the update.
    outcome_weight=0.01,
    capture_progress_weight=2.0,
    front_advance_weight=2.0,
    speed_weight=0.0,
    speed_penalty_min=10.0,
)

# Values are expressed per 60 Hz physics tick.  The resolver raises them to
# action_repeat so decision-step horizons remain normalized to game time.
# Gamma buckets are win=.999, kill/death=.995, damage=.991, shaping=.975;
# their approximate horizons are full episode, engagement, exchange, and
# immediate geometry respectively.
COMPONENT_GAMMAS_PER_TICK: dict[str, float] = {
    # Undiscounted. A win is terminal in a finite-horizon game with a hard step
    # cap, and GAE cuts every trace at the episode boundary, so nothing can
    # diverge. At 0.999/tick a win at the start of a typical match was worth
    # 0.999^2557 = 7.7% of one at the end -- an artifact of a rate inherited from
    # 1024-step episodes, not a statement about the game. At 1.0 the critic head
    # learns P(win) itself. Expect its explained variance to *fall*: a discounted
    # terminal target is about zero for most of an episode and trivially
    # predictable, where P(win) early is genuinely uncertain.
    # 0.9997: a 3,333-step horizon against ~8,600-step episodes, which reaches
    # well down the match while keeping a real contraction per rollout segment
    # (0.9997^128 = 0.962). Undiscounted was worse here than the theory suggested:
    # a 128-step rollout against that episode length bootstraps the value roughly
    # 67 times before any terminal grounds it, and at gamma 1 there is no
    # contraction to damp error across those hops. Run 737 showed it -- win
    # explained variance fell from 0.994 to 0.42, and since the win pair carried
    # 70% of the gradient weight, most of the update became noise: KL pinned at
    # target and `epochs_completed` collapsed to 1.0 for most of the run.
    "ally_win": 0.9997,
    "enemy_win": 0.9997,
    # The single stream stays markovian, which is the thing being measured.
    "outcome": 1.0,
    "front_advance": 0.999,
    "capture_progress": 0.999,
    "ally_combat_death": 0.995,
    "enemy_combat_death": 0.995,
    "combat_death": 0.995,
    "kill_shot": 0.995,
    "kill_assist": 0.995,
    "kill_ally_shot": 0.995,
    "kill_ally_assist": 0.995,
    "shield_recharge": 0.991,
    "boundary": 0.995,
    "boundary_damage": 0.991,
    "ally_combat_damage": 0.991,
    "enemy_combat_damage": 0.991,
    "combat_damage_taken": 0.991,
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
    "front_advance": 0.97,
    "capture_progress": 0.97,
    "outcome": 0.97,
    "ally_combat_death": 0.95,
    "enemy_combat_death": 0.95,
    "combat_death": 0.95,
    "kill_shot": 0.87,
    "kill_assist": 0.97,
    "kill_ally_shot": 0.87,
    "kill_ally_assist": 0.97,
    "shield_recharge": 0.90,
    "boundary": 0.90,
    "boundary_damage": 0.90,
    "ally_combat_damage": 0.90,
    "enemy_combat_damage": 0.90,
    "combat_damage_taken": 0.90,
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
        # Peak 3e-4, decaying to half of it. The last row holds, so a budget
        # longer than 500M steps trains its tail at the floor rather than
        # continuing to decay.
        #
        # 3e-4 rather than the 4.5e-4 that run 731 used, because `target_kl` is
        # what actually bounds an update and Frontline reaches that bound in
        # about half the passes. 731 completed a mean 3.33 epochs of four and
        # pinned 4.0 from 12M steps on; runs 732 and 733 sit at 2.2 and 2.4. The
        # trust region spends a fixed KL budget either way, so a smaller step
        # does not buy less movement -- it buys the same movement in more, finer
        # optimizer steps, which is what the critic is short of. Explained
        # variance is the series to read: 731 reached 0.45 by 13M steps where 733
        # was at 0.35.
        #
        # This is a hypothesis with a mechanism, not a measured result. Nothing
        # here has been run at 3e-4. `train/epochs_completed` rising toward four
        # and explained variance improving per step are what would confirm it;
        # if epochs stay near two, the bound is the task rather than the step
        # size and the peak should go back up rather than lower again.
        learning_rate=(
            (0, 1e-7, "linear"),
            (5_000_000, 3e-4, "hold"),
            (100_000_000, 3e-4, "exponential"),
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
        offensive_bias=((0, 1.0, "hold"), (50_000_000, 1.0, "linear"), (300_000_000, 0.0, "hold")),
        # No unpaired shaping remains during the final zero-sum phase.
        shaping_scale=(
            (0, 1.0, "hold"),
            (50_000_000, 1.0, "linear"),
            (300_000_000, 0.0, "hold"),
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
