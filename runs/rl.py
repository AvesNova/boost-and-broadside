"""RL training profile.

Designed to be used after loading a pretrained BC checkpoint (via --pretrain_from),
but also runnable from scratch.

Phase structure:
  Step 0 → 5M:   LR warmup 1e-7 → 3e-4. 50% envs vs scripted opponent.
                  All reward group scales active (pretrained value function handles this).
  Step 5M:        LR at cruise.
  ELO 1000:       avg-model starts accumulating (avg_model_elo_threshold gate,
                  ELO-based rather than step-based).
  Step 50M:       avg-model activates as opponent (20% envs).
                  Reduce scripted to 30% to make room.
                  League opponents activate (20% envs).
"""

from boost_and_broadside.config import (
    EnvConfig,
    ObstacleCacheConfig,
    ScaleConfig,
    TrainConfig,
    TrainingSchedule,
    constant,
    linear,
    stepped,
)
from boost_and_broadside.config.schedule import exponential, join
from runs.shared import COMPONENT_GAMMAS, COMPONENT_LAMBDAS, LEAGUE_EVAL, REWARDS

_MAX_TOKENS = 6_000_000
_NUM_SHIPS = 8
_NUM_OBSTACLES = 0
_NUM_STEPS = 128
_NUM_MINIBATCHES = 32
_MICROBATCH_TOKENS = _MAX_TOKENS // _NUM_MINIBATCHES // 4

RL_SCHEDULE = TrainingSchedule(
    learning_rate=join(
        (0, linear((0, 1e-7), (5_000_000, 3e-4))),
        (5_000_000, constant(3e-4)),
        (100_000_000, exponential((100_000_000, 3e-4), (500_000_000, 1e-4))),
    ),
    policy_gradient_coef=constant(1.0),
    entropy_coef=constant(0.005),
    behavior_cloning_coef=constant(2.0),
    value_function_coef=constant(1.0),
    sigreg_coef=constant(0.00),
    true_reward_scale=constant(1.0),
    global_scale=constant(1.0),
    local_scale=constant(1.0),
    opponent_fraction=stepped((0, 0.5), (50_000_000, 0.7)),
    checkpoint_interval=constant(50),
    num_epochs=stepped((0, 4)),
    target_kl=stepped((0, 0.1)),
    high_elo_threshold=constant(900.0),
    high_elo_target_kl=constant(0.02),
)

RL_TRAIN_CONFIG = TrainConfig(
    paradigm="ego_pass",
    scales=(
        ScaleConfig(
            env_config=EnvConfig(
                num_ships=_NUM_SHIPS,
                num_obstacles=_NUM_OBSTACLES,
                max_bullets=20,
                max_episode_steps=1024,
            ),
            num_envs=_MAX_TOKENS
            // (_NUM_SHIPS + _NUM_OBSTACLES)
            // _NUM_STEPS
            // _NUM_MINIBATCHES
            * _NUM_MINIBATCHES,
        ),
    ),
    schedule=RL_SCHEDULE,
    rewards=REWARDS,
    num_steps=_NUM_STEPS,
    num_minibatches=_NUM_MINIBATCHES,
    microbatch_tokens=_MICROBATCH_TOKENS,
    next_state_coef=0.2,
    gamma=0.990,
    gae_lambda=0.95,
    component_gammas=COMPONENT_GAMMAS,
    component_lambdas=COMPONENT_LAMBDAS,
    clip_coef=0.15,
    max_grad_norm=1.0,
    total_timesteps=500_000_000,
    return_ema_alpha=0.005,
    return_min_span=1.0,
    checkpoint_dir="checkpoints",
    league_size=20,
    league_k=4,
    league_admission_interval=25,
    pfsp_mode="hard",
    pfsp_exponent=2.0,
    live_rating_decay=0.9,
    avg_rating_decay=0.995,
    bt_prior_draws=1.0,
    admission_prior_games=10.0,
    league_eval=LEAGUE_EVAL,
    bc_elo_target=950.0,
    bc_elo_scale=200.0,
    histogram_interval=10,
    # Avg-model accumulation starts once normalized training ELO reaches this.
    avg_model_elo_threshold=1000.0,
    obstacle_cache=ObstacleCacheConfig(
        num_cache_envs=4096,
        cache_size=512,
        max_steps=6000,
    ),
)
