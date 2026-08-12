"""Independent reinforcement-learning intent for four refractive fields.

This is a smoke/future-training combat profile, not an obstacle-avoidance
objective.  Field value emerges through combat, navigation, speed, handling,
and health outcomes rather than universal proximity shaping.
"""

from boost_and_broadside.config.defaults import (
    COMPONENT_GAMMAS_PER_TICK,
    COMPONENT_LAMBDAS_PER_TICK,
    ELO_EVAL,
    FIELD_REWARDS,
    LIVE_REFERENCE_PROBABILITIES,
    MODEL_CONFIG,
    SHIP_CONFIG,
    make_rl_schedule_spec,
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
from boost_and_broadside.config.training import FieldMapConfig
from boost_and_broadside.constants import DEFAULT_MAX_BULLETS_PER_SHIP

RL_FIELDS_PROFILE = ProfileSpec(
    name="rl-fields",
    ship_config=SHIP_CONFIG,
    model_config=MODEL_CONFIG,
    environment=EnvironmentSpec(
        num_ships=8,
        num_fields=4,
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
        schedule=make_rl_schedule_spec(),
        rewards=FIELD_REWARDS,
        next_state_coef=0.2,
        windowed_loss_coef=0.1,
    ),
    optimizer=OptimizerSpec(
        clip_coef=0.15,
        max_grad_norm=1.0,
        total_timesteps=500_000_000,
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
        # The live gauge is defined, not fitted, so the field environment rates
        # on the same rungs as the zero-field one.  That is a deliberate
        # simplification: it says nothing about the two being equally hard, and
        # cross-environment live ratings are not comparable.
        live_reference_probabilities=LIVE_REFERENCE_PROBABILITIES,
        elo_milestone_gap=200.0,
        elo_temperature=200.0,
        league_uniform_sampling=False,
        elo_eval=ELO_EVAL,
        bc_winrate_target=0.45,
    ),
    field_map=FieldMapConfig(
        cache_size=512,
        max_generation_attempts=256,
        nesting_probability=0.35,
    ),
)
