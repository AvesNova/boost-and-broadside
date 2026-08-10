"""Independent zero-field reinforcement-learning profile intent.

This is designed to follow an explicit BC pretraining checkpoint, but it is
also runnable from scratch.  The opponent mix stays half self-play and half an
Elo-proximity league draw; roster membership, rather than a schedule phase,
provides the curriculum.
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

RL_PROFILE = ProfileSpec(
    name="rl",
    ship_config=SHIP_CONFIG,
    model_config=MODEL_CONFIG,
    environment=EnvironmentSpec(
        num_ships=8,
        num_fields=0,
        max_bullets=DEFAULT_MAX_BULLETS_PER_SHIP,
        max_episode_steps=1024,
        # Physics stays at 60 Hz; the policy chooses at 30 Hz.  The 128-step
        # rollout therefore spans 4.3 seconds, close to a full episode.
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
        schedule=make_rl_schedule_spec(),
        rewards=REWARDS,
        next_state_coef=0.2,
        windowed_loss_coef=0.1,
    ),
    optimizer=OptimizerSpec(
        clip_coef=0.15,
        max_grad_norm=1.0,
        total_timesteps=500_000_000,
        return_ema_alpha=0.005,
        # This is intentionally 1.0, not an epsilon.  It binds sparse return
        # components; reducing it makes critic outliers dominate the squared
        # value loss and needs Huber/tail-aware retuning as a separate change.
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
        # Rung ratings are derived, not stated: the live gauge pins random at 0
        # and scripted at 1000 and places a rung at 1000·p.
        live_reference_probabilities=LIVE_REFERENCE_PROBABILITIES,
        elo_milestone_gap=200.0,
        elo_temperature=200.0,
        league_uniform_sampling=False,
        elo_eval=ELO_EVAL,
        bc_winrate_target=0.45,
    ),
    field_map=None,
)
