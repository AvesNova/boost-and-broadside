"""RL training profile.

Designed to be used after loading a pretrained BC checkpoint (via --pretrain_from),
but also runnable from scratch.

Phase structure:
  Step 0 → 5M:   LR warmup 1e-7 → 3e-4. 50% envs vs scripted opponent.
                  avg_model starts accumulating immediately (allow_avg_model_updates=True).
                  aux_scale=0.0 — value function is assumed pretrained; shaping off.
  Step 5M:        LR at cruise.
  Step 25M:       avg-model is ready — activate as opponent (20% envs).
                  Reduce scripted to 30% to make room.
  Step 50M:       League opponents activate (20% envs).
"""

from boost_and_broadside.config import (
    EnvConfig,
    ScaleConfig,
    TrainConfig,
    TrainingSchedule,
    constant,
    linear,
    stepped,
)
from runs.shared import REWARDS

_MAX_TOKENS = 3840

RL_SCHEDULE = TrainingSchedule(
    # Warmup from 1e-7 to 3e-4 over 5M steps, then hold.
    learning_rate             = linear((0, 1e-7), (5_000_000, 3e-4)),
    policy_gradient_coef      = constant(1.0),
    entropy_coef              = constant(0.01),
    behavior_cloning_coef     = constant(0.0),
    value_function_coef       = constant(1.0),
    # aux shaping off — value function is assumed pretrained and already knows the game.
    true_reward_scale         = constant(1.0),
    important_scale           = constant(1.0),
    aux_scale                 = constant(0.0),
    # Scripted at 50% from step 0 — stable, strong signal from the start.
    # At step 25M avg-model is ready; reduce scripted to make room.
    scripted_fraction         = stepped((0, 0.5), (25_000_000, 0.3)),
    # avg-model not used as opponent until step 25M (needs time to diverge from init).
    avg_model_fraction        = stepped((0, 0.0), (25_000_000, 0.2)),
    # League activates at step 50M once the policy has meaningful ELO.
    league_fraction           = stepped((0, 0.0), (50_000_000, 0.2)),
    # Accumulate avg-model immediately so it is ready at step 25M.
    allow_avg_model_updates   = stepped((0, True)),
    allow_scripted_in_roster  = stepped((0, True)),
    elo_eval_games            = stepped((0, 256)),
    elo_eval_interval         = stepped((0, 10)),
    checkpoint_interval       = stepped((0, 10)),
)

RL_TRAIN_CONFIG = TrainConfig(
    scales=(
        ScaleConfig(
            env_config=EnvConfig(num_ships=2, max_bullets=20, max_episode_steps=1024),
            num_envs=3 * _MAX_TOKENS // 3 // 8,
        ),
    ),
    schedule=RL_SCHEDULE,
    rewards=REWARDS,
    num_steps=128,
    num_epochs=4,
    num_minibatches=4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_coef=0.2,
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
)
