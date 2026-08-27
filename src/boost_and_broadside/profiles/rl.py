"""The reinforcement-learning profile.

Designed to follow an explicit BC pretraining checkpoint, but also runnable
from scratch.  The opponent mix stays half self-play and half an Elo-proximity
league draw; roster membership, rather than a schedule phase, provides the
curriculum.

There is one RL profile rather than a field-free one and a fielded one.
``num_fields`` is a sequence length, not an architecture: it sets the token
count ``N + M`` and no weight shape depends on it, so a "model that does not
support fields" was never a thing this registry had to represent.  Zero fields
remains a reachable *configuration* -- it is what run 682 trained under and how
that run is still evaluated -- but it is a value to set, not a profile to pick.
"""

from boost_and_broadside.config.defaults import (
    COMPONENT_GAMMAS_PER_TICK,
    COMPONENT_LAMBDAS_PER_TICK,
    ELO_EVAL,
    LIVE_REFERENCE_PROBABILITIES,
    MODEL_CONFIG,
    REWARDS,
    SHIP_CONFIG,
    make_rl_schedule_spec,
)
from boost_and_broadside.config.schema import LaunchSizingSpec, ProfileSpec
from boost_and_broadside.config.training import FieldMapConfig
from boost_and_broadside.constants import DEFAULT_MAX_BULLETS_PER_SHIP

RL_PROFILE = ProfileSpec(
    name="rl",
    ship_config=SHIP_CONFIG,
    model_config=MODEL_CONFIG,
    # --- Environment ---
    num_ships=8,
    num_fields=4,
    max_bullets=DEFAULT_MAX_BULLETS_PER_SHIP,
    max_episode_steps=1024,
    # Physics stays at 60 Hz; the policy chooses at 30 Hz.  The 128-step rollout
    # therefore spans 4.3 seconds, close to a full episode.
    action_repeat=2,
    spawn_resource_spread=0.25,
    field_map=FieldMapConfig(
        cache_size=512,
        max_generation_attempts=256,
        nesting_probability=0.35,
    ),
    # --- Rollout shape ---
    logical_batch_tokens=12_000_000,
    num_steps=128,
    num_minibatches=32,
    # --- Objective ---
    paradigm="ego_pass",
    schedule_spec=make_rl_schedule_spec(),
    rewards=REWARDS,
    # Set from measured gradient pressure, not from the loss values, and aimed at
    # run 719's split rather than at a round number.  The previous 0.02/0.07 held
    # the two families to ~0.05 of the trunk gradient each on the theory that an
    # auxiliary at 719's 0.39 was crowding out the objective.  Run 723 tested that
    # and refuted it: cutting the auxiliary 4.5x against run 722 moved head-to-head
    # strength by 3.5 Elo, inside the 15.4 measurement error, and both arms stayed
    # behind 719.  Auxiliary share was never the variable.
    #
    # So restore 719's budget.  Its two state losses together took 0.371 of the
    # trunk gradient; that total is split evenly here, which is the ratio 723 was
    # already solved for.  A term's gradient norm is linear in its coefficient, so
    # the scale factors follow directly, and they stay asymmetric because the state
    # family still produces about 3.5x the gradient per unit coefficient.
    #
    # First-order only: the shares are measured against a policy trained at the old
    # reward weights, which this run also changes, so check
    # ``grad_share/trunk_top_level/*`` once the run is past the behavior-cloning
    # decay and adjust if it landed wide.
    predictive_state_coef=0.125,
    predictive_action_coef=0.44,
    # Each rollout step decodes one horizon rather than all twelve, with the
    # horizons split evenly across the step axis.  Same loss in expectation, and
    # a step that decodes at depth d pays for d transitions and one pair of
    # heads instead of eleven and twelve -- about a quarter of the block's
    # compute.  ``full`` restores the exhaustive version for comparison.
    predictive_mode="sampled",
    # Twelve decisions is 0.4 s of game time at the 30 Hz decision rate -- about
    # one bullet flight, and long enough that where a ship will be and what it
    # will do are genuinely open questions.
    prediction_horizon=12,
    # --- Discounts, per physics tick ---
    gamma_per_tick=0.99,
    gae_lambda_per_tick=0.95,
    component_gammas_per_tick=COMPONENT_GAMMAS_PER_TICK,
    component_lambdas_per_tick=COMPONENT_LAMBDAS_PER_TICK,
    # --- Optimizer, scalers, budget ---
    clip_coef=0.15,
    max_grad_norm=1.0,
    total_timesteps=1_000_000_000,
    return_ema_alpha=0.005,
    # A divide-by-zero guard, and nothing more.  At the previous 1.0 it bound 8
    # of 12 components on every update of run 719 -- including the win pair --
    # compressing their critic targets by up to 121x and their critic gradients
    # by four orders of magnitude.  The outlier problem that motivated the large
    # floor is now handled where it belongs, by ``value_huber_delta``.
    #
    # 1e-3 rather than 1e-2 because the guard has to sit far below every live
    # component's spread, not just below it.  Measured on run 719's logged return
    # histograms, the narrowest component (field_death) has a 4-sigma span of
    # 0.0127: twelve times this floor, but only 1.3x a floor of 1e-2.
    return_min_span=1e-3,
    advantage_min_rms=1e-4,
    # Squared error inside one normalized unit, linear outside.  Per-component
    # normalization necessarily exposes heavy tails -- a sparse component is a
    # spike at zero with rare large excursions -- and bounding their gradient
    # here keeps one component's tail from setting the whole critic's step.
    value_huber_delta=1.0,
    # --- League and live evaluation ---
    league_size=20,
    league_slots=4,
    # Rung ratings are derived, not stated: the live gauge pins random at 0 and
    # scripted at 1000 and places a rung at 1000·p.  It is defined rather than
    # fitted, so it says nothing about how hard any particular field count is,
    # and live ratings do not compare across environments.
    live_reference_probabilities=LIVE_REFERENCE_PROBABILITIES,
    elo_milestone_gap=200.0,
    elo_temperature=200.0,
    league_uniform_sampling=False,
    elo_eval=ELO_EVAL,
    bc_winrate_target=0.45,
    # --- Persistence and logging ---
    checkpoint_dir="checkpoints",
    histogram_interval=10,
    log_interval=10,
    # --- Machine sizing ---
    launch=LaunchSizingSpec(rollout_tokens=4_000_000, microbatches_per_minibatch=5),
)
