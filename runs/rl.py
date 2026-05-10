"""RL training profile.

Designed to be used after loading a pretrained BC checkpoint (via --pretrain_from),
but also runnable from scratch.

Phase structure:
  Step 0 → 5M:   LR warmup 1e-7 → 3e-4. 50% envs vs scripted opponent.
                  avg_model starts accumulating immediately (allow_avg_model_updates=True).
                  All reward group scales active (pretrained value function handles this).
  Step 5M:        LR at cruise.
  Step 25M:       avg-model is ready — activate as opponent (20% envs).
                  Reduce scripted to 30% to make room.
  Step 50M:       League opponents activate (20% envs).
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
from boost_and_broadside.config.schedule import exponential
from runs.shared import REWARDS, COMPONENT_GAMMAS, COMPONENT_LAMBDAS

_MAX_TOKENS = 2_000_000
_NUM_SHIPS = 4
_NUM_OBSTACLES = 0
_NUM_STEPS = 256
_NUM_MINIBATCHES = 32

RL_SCHEDULE = TrainingSchedule(
    learning_rate=linear((0, 1e-7), (5_000_000, 3e-4)),
    policy_gradient_coef=constant(1.0),
    entropy_coef=constant(0.005),
    behavior_cloning_coef=constant(2.0),
    value_function_coef=constant(1.0),
    sigreg_coef=constant(0.00),
    true_reward_scale=constant(1.0),
    global_scale=constant(1.0),
    local_scale=constant(1.0),
    # Scripted at 50% from step 0 — stable, strong signal from the start.
    # At step 50M avg-model is ready; reduce scripted to make room.
    scripted_fraction=stepped((0, 0.5), (50_000_000, 0.3)),
    # avg-model not used as opponent until step 50M (needs time to diverge from init).
    avg_model_fraction=stepped((0, 0.0), (50_000_000, 0.2)),
    # League activates at step 50M once the policy has meaningful ELO.
    league_fraction=stepped((0, 0.0), (50_000_000, 0.2)),
    # Accumulate avg-model immediately so it is ready at step 50M.
    allow_avg_model_updates=stepped((0, False), (40_000_000, True)),
    allow_scripted_in_roster=stepped((0, True)),
    elo_eval_games=stepped((0, 512)),
    elo_eval_interval=stepped((0, 1), (1_000_000, 10)),
    checkpoint_interval=stepped((0, 1), (1_000_000, 10)),
    num_epochs=stepped((0, 4)),
    target_kl=stepped((0, 0.1)),
)

RL_TRAIN_CONFIG = TrainConfig(
    scales=(
        # ScaleConfig(
        #     env_config=EnvConfig(num_ships=2, num_obstacles=4, max_bullets=20, max_episode_steps=1024),
        #     num_envs=_MAX_TOKENS // 4 // 2,
        # ),
        ScaleConfig(
            env_config=EnvConfig(num_ships=_NUM_SHIPS, num_obstacles=_NUM_OBSTACLES, max_bullets=20, max_episode_steps=1024,),
            num_envs=_MAX_TOKENS // (_NUM_SHIPS + _NUM_OBSTACLES) // _NUM_STEPS // _NUM_MINIBATCHES * _NUM_MINIBATCHES,
        ),
    ),
    schedule=RL_SCHEDULE,
    rewards=REWARDS,
    num_steps=_NUM_STEPS,
    num_minibatches=_NUM_MINIBATCHES,
    next_state_coef=0.2,
    gamma=0.990,
    gae_lambda=0.95,
    component_gammas=COMPONENT_GAMMAS,
    component_lambdas=COMPONENT_LAMBDAS,
    clip_coef=0.15,
    max_grad_norm=1.0,
    total_timesteps=2_000_000_000,
    return_ema_alpha=0.005,
    return_min_span=1.0,
    checkpoint_dir="checkpoints",
    league_size=20,
    elo_milestone_gap=100.0,
    elo_k_factor=32.0,
    elo_temperature=200.0,
    league_uniform_sampling=False,
    scripted_roster_min_steps=300_000_000,
    obstacle_cache=ObstacleCacheConfig(
        num_cache_envs=4096,
        cache_size=512,
        max_steps=6000,
    ),
)
