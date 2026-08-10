"""Independent behavior-cloning profile intent.

BC pretrains the policy against the stochastic scripted controller: the
controller supplies supervised action targets on every environment, no policy
gradient is taken, and no roster opponent plays a rollout.  The critic and the
next-state head train alongside so RL inherits more than an actor.

Every value the behavior-cloning objective does not require is the current
project value, restated here rather than borrowed -- this module never imports
another profile.  The differences that remain are named and enforced by the
allowed-difference invariant in ``tests/config/test_bc_profile.py``:

* ``objective.next_state_coef`` -- BC weights next-state prediction at full
  strength while it has a dense supervised signal to learn the trunk from.
* ``optimizer.total_timesteps`` -- BC owns its budget and stops when imitation
  saturates, not when RL's curriculum ends.
* five schedule entries -- see :func:`make_bc_schedule_spec`.
"""

from boost_and_broadside.config.defaults import (
    COMPONENT_GAMMAS_PER_TICK,
    COMPONENT_LAMBDAS_PER_TICK,
    ELO_EVAL,
    LIVE_REFERENCE_PROBABILITIES,
    MODEL_CONFIG,
    REWARDS,
    SHIP_CONFIG,
    make_bc_schedule_spec,
)
from boost_and_broadside.config.schema import (
    DiscountSpec,
    EnvironmentSpec,
    LaunchSizingSpec,
    LeagueSpec,
    ObjectiveSpec,
    OptimizerSpec,
    ProfileSpec,
    RolloutSpec,
)
from boost_and_broadside.constants import DEFAULT_MAX_BULLETS_PER_SHIP

BC_PROFILE = ProfileSpec(
    name="bc",
    ship_config=SHIP_CONFIG,
    model_config=MODEL_CONFIG,
    environment=EnvironmentSpec(
        # Imitation has to happen in the environment RL continues in: eight
        # ships (4v4), the same 30 Hz decision rate over 60 Hz physics, and the
        # same spawn spread.
        num_ships=8,
        num_fields=0,
        max_bullets=DEFAULT_MAX_BULLETS_PER_SHIP,
        max_episode_steps=1024,
        action_repeat=2,
        spawn_resource_spread=0.25,
    ),
    rollout=RolloutSpec(
        logical_batch_tokens=12_000_000,
        num_steps=128,
        num_minibatches=32,
    ),
    launch_defaults=LaunchSizingSpec(
        # Memory launch choices, deliberately excluded from profile_fingerprint.
        rollout_tokens=4_000_000,
        microbatches_per_minibatch=5,
    ),
    discounts=DiscountSpec(
        gamma_per_tick=0.99,
        gae_lambda_per_tick=0.95,
        component_gammas_per_tick=dict(COMPONENT_GAMMAS_PER_TICK),
        component_lambdas_per_tick=dict(COMPONENT_LAMBDAS_PER_TICK),
    ),
    objective=ObjectiveSpec(
        paradigm="ego_pass",
        schedule=make_bc_schedule_spec(),
        rewards=REWARDS,
        # Named difference: full-strength next-state prediction.  A separate
        # experiment, not this correction, is what would justify changing it.
        next_state_coef=1.0,
        windowed_loss_coef=0.1,
    ),
    optimizer=OptimizerSpec(
        clip_coef=0.15,
        max_grad_norm=1.0,
        # Named difference: BC's own budget.
        total_timesteps=2_000_000_000,
        return_ema_alpha=0.005,
        return_min_span=1.0,
        advantage_min_rms=1e-4,
        return_quantile_samples=262_144,
        checkpoint_dir="checkpoints",
        histogram_interval=10,
        log_interval=10,
    ),
    league=LeagueSpec(
        league_size=20,
        league_slots=4,
        # league_fraction is 0.0 throughout, so no roster entry ever plays a
        # rollout.  The Elo evaluator still runs -- BC's own scripted win rate
        # is what decays the cloning weight -- and it rates the live policy
        # against the same derived rungs RL continues on.
        live_reference_probabilities=LIVE_REFERENCE_PROBABILITIES,
        elo_milestone_gap=200.0,
        elo_temperature=200.0,
        league_uniform_sampling=False,
        elo_eval=ELO_EVAL,
        bc_winrate_target=0.45,
    ),
    field_map=None,
)
