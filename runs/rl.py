"""RL training profile.

Designed to be used after loading a pretrained BC checkpoint (via --pretrain_from),
but also runnable from scratch.

Phase structure:
  Step 0 → 5M:   LR warmup 1e-7 → 3e-4. 50% envs vs scripted opponent.
                  All reward group scales active (pretrained value function handles this).
  Step 5M:        LR at cruise.
  BC cutoff:      avg-model starts accumulating — the handoff is exact, the avg
                  model picks up as the BC aux loss reaches zero (scripted win
                  rate at bc_winrate_target). Outcome-based, not step-based.
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
from runs.shared import COMPONENT_GAMMAS, COMPONENT_LAMBDAS, ELO_EVAL, REWARDS

_MAX_TOKENS = 12_000_000
_ROLLOUT_TOKENS = 6_000_000
_NUM_SHIPS = 8
_NUM_OBSTACLES = 0
_NUM_STEPS = 128
_NUM_MINIBATCHES = 32
# // 5: split each minibatch into 5 gradient-accumulation micro-batches — the
# headroom needed to fit this scale's attention activations in VRAM on the target GPU.
# Memory-only knob (see TrainConfig.microbatch_tokens); retune the divisor per GPU.
_MICROBATCH_TOKENS = _ROLLOUT_TOKENS // _NUM_MINIBATCHES // 5
_ROLLOUTS_PER_UPDATE = _MAX_TOKENS // _ROLLOUT_TOKENS
assert _MAX_TOKENS % _ROLLOUT_TOKENS == 0

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
    # Scripted at 50% from step 0 — stable, strong signal from the start.
    # At step 50M avg-model is ready; reduce scripted to make room.
    scripted_fraction=stepped((0, 0.5), (50_000_000, 0.3)),
    # avg-model not used as opponent until step 50M (needs time to diverge from init).
    avg_model_fraction=stepped((0, 0.0), (50_000_000, 0.2)),
    # League activates at step 50M once the policy has meaningful ELO.
    league_fraction=stepped((0, 0.0), (50_000_000, 0.2)),
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
            num_envs=_ROLLOUT_TOKENS
            // (_NUM_SHIPS + _NUM_OBSTACLES)
            // _NUM_STEPS
            // _NUM_MINIBATCHES
            * _NUM_MINIBATCHES,
        ),
    ),
    schedule=RL_SCHEDULE,
    rewards=REWARDS,
    num_steps=_NUM_STEPS,
    rollouts_per_update=_ROLLOUTS_PER_UPDATE,
    num_minibatches=_NUM_MINIBATCHES,
    microbatch_tokens=_MICROBATCH_TOKENS,
    next_state_coef=0.2,
    gamma=0.990,
    gae_lambda=0.95,
    component_gammas=COMPONENT_GAMMAS,
    component_lambdas=COMPONENT_LAMBDAS,
    clip_coef=0.15,
    max_grad_norm=1.0,
    total_timesteps=1_000_000_000,
    return_ema_alpha=0.005,
    return_min_span=1.0,
    return_quantile_samples=262_144,
    checkpoint_dir="checkpoints",
    league_size=20,
    elo_milestone_gap=200.0,
    elo_temperature=200.0,
    league_uniform_sampling=False,
    elo_eval=ELO_EVAL,
    bc_winrate_target=0.45,
    histogram_interval=10,
    obstacle_cache=ObstacleCacheConfig(
        num_cache_envs=4096,
        cache_size=512,
        max_steps=6000,
    ),
)
