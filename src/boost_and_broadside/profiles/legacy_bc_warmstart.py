"""Transitional pretrain stage for the legacy command retired in S08.

This is deliberately not in ``PROFILES`` and does not import another profile.
"""

from boost_and_broadside.config.defaults import (
    ELO_EVAL,
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

LEGACY_BC_WARMSTART_PRETRAIN_PROFILE = ProfileSpec(
    name="legacy-bc-warmstart-pretrain",
    ship_config=SHIP_CONFIG,
    model_config=MODEL_CONFIG,
    environment=EnvironmentSpec(
        num_ships=4,
        num_fields=0,
        max_bullets=DEFAULT_MAX_BULLETS_PER_SHIP,
        max_episode_steps=1024,
        action_repeat=1,
        spawn_resource_spread=0.0,
    ),
    rollout=RolloutSpec(
        logical_batch_tokens=245_760,
        num_steps=128,
        num_minibatches=4,
    ),
    launch_defaults=LaunchSizingSpec(num_envs=480),
    discounts=DiscountSpec(
        gamma_per_tick=0.99,
        gae_lambda_per_tick=0.95,
        component_gammas_per_tick={},
        component_lambdas_per_tick={},
    ),
    objective=ObjectiveSpec(
        paradigm="ego_pass",
        schedule=make_bc_schedule_spec(),
        rewards=REWARDS,
        next_state_coef=1.0,
        windowed_loss_coef=0.1,
    ),
    optimizer=OptimizerSpec(
        clip_coef=0.2,
        max_grad_norm=1.0,
        total_timesteps=20_000_000,
        return_ema_alpha=0.005,
        return_min_span=1.0,
        advantage_min_rms=1e-4,
        return_quantile_samples=None,
        checkpoint_dir="checkpoints",
        histogram_interval=10,
        log_interval=10,
    ),
    league=LeagueSpec(
        league_size=20,
        league_slots=4,
        reference_ladder=(),
        random_elo=0.0,
        elo_milestone_gap=200.0,
        elo_temperature=200.0,
        league_uniform_sampling=False,
        elo_eval=ELO_EVAL,
        bc_winrate_target=0.45,
    ),
    field_map=None,
)
