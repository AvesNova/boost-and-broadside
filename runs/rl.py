"""RL training profile.

Designed to be used after loading a pretrained BC checkpoint (via --pretrain_from),
but also runnable from scratch.

Phase structure:
  Step 0 → 5M:   LR warmup 1e-7 → 3e-4. All reward group scales active
                  (pretrained value function handles this).
  Step 5M:        LR at cruise.
  BC cutoff:      avg-model starts accumulating — the handoff is exact, the avg
                  model picks up as the BC aux loss reaches zero (scripted win
                  rate at bc_winrate_target). Outcome-based, not step-based.

The opponent mix has no phases. Half the batch is self-play and half faces a
league entry drawn by Elo proximity, so the curriculum follows the ratings: the
scripted agent is the only draw at first, the average policy joins at the BC
cutoff, frozen checkpoints join at each Elo milestone, and the scripted agent
falls out of contention as the live rating leaves it behind.
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
from boost_and_broadside.config.schedule import exponential, join
from boost_and_broadside.constants import DEFAULT_MAX_BULLETS_PER_SHIP
from runs.shared import COMPONENT_GAMMAS, COMPONENT_LAMBDAS, ELO_EVAL, REWARDS

_MAX_TOKENS = 12_000_000
_ROLLOUT_TOKENS = 4_000_000
_NUM_SHIPS = 8
_NUM_FIELDS = 0
_NUM_STEPS = 128
# Physics ticks per decision. dt is 1/60, so this is 30 Hz. Measured with the
# *fixed* scripted controller at equal game time, coarsening the decision rate
# costs combat effectiveness monotonically: damage per live ship-step runs
# 0.2965 / 0.2814 / 0.2656 / 0.2475 at 60 / 30 / 20 / 15 Hz. 20 Hz gave up 10%
# of it; 30 Hz gives up 5% and still halves the tokens per second of game time.
# num_steps is deliberately unchanged: a 128-step rollout spans 4.3 s rather
# than 2.1 s, so BPTT covers close to a whole ~4.7 s episode.
_ACTION_REPEAT = 2
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
    # Half the batch is self-play, half faces a league entry. Which entry is
    # decided by Elo proximity, not by a schedule: at step 0 the roster's only
    # sampleable entry is the scripted agent, so this starts as the 50/50
    # scripted split the run wants, then diversifies on its own as the average
    # policy joins and checkpoints freeze, and the scripted agent fades once the
    # live rating outruns it.
    league_fraction=constant(0.5),
    checkpoint_interval=constant(50),
    num_epochs=stepped((0, 4)),
    target_kl=stepped((0, 0.1)),
    # Tighten the trust region once the policy wins 80% against scripted.
    high_winrate_threshold=constant(0.8),
    high_winrate_target_kl=constant(0.02),
)

RL_TRAIN_CONFIG = TrainConfig(
    paradigm="ego_pass",
    scales=(
        ScaleConfig(
            env_config=EnvConfig(
                num_ships=_NUM_SHIPS,
                num_fields=_NUM_FIELDS,
                max_bullets=DEFAULT_MAX_BULLETS_PER_SHIP,
                # In physics ticks, so 17.1 s of game time whatever the repeat.
                max_episode_steps=1024,
                action_repeat=_ACTION_REPEAT,
                # +/-25% on spawn health and power, uniform cooldown.
                spawn_resource_spread=0.25,
            ),
            num_envs=_ROLLOUT_TOKENS
            // (_NUM_SHIPS + _NUM_FIELDS)
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
    # Fallback horizons for components without an entry in the per-component
    # tables; stated per decision at _ACTION_REPEAT (see runs/shared.py).
    gamma=0.990**_ACTION_REPEAT,
    gae_lambda=0.95**_ACTION_REPEAT,
    component_gammas=COMPONENT_GAMMAS,
    component_lambdas=COMPONENT_LAMBDAS,
    clip_coef=0.15,
    max_grad_norm=1.0,
    # Environment steps, which are decisions rather than physics ticks. Half of
    # the previous budget is the same span of game time as the 1e9-step
    # reference run, so runs stay comparable in experience rather than in tokens.
    total_timesteps=500_000_000,
    return_ema_alpha=0.005,
    # Held at 1.0 deliberately, and it *does* bind on six of the eleven active
    # components (watch scaler/floor_bound_span/*). Lowering it to an epsilon is
    # not a free bug fix: ReturnScaler divides the whole return distribution by a
    # robust p5-p95 half-span, so a component whose central 90% is tight but whose
    # tails are not — every sparse terminal reward — produces very large normalized
    # targets, and the value loss squares them. Measured: loss/value rises ~11x at
    # production spans and ~400x in --smoke, which against max_grad_norm=1.0 (grad
    # norm currently 0.65) makes clipping bind every step and silently cuts the
    # effective learning rate. Fixing it properly means bounding the critic's
    # outlier sensitivity (Huber value loss, or a tail-aware span) and re-tuning
    # value_function_coef / max_grad_norm alongside — its own change, with its own
    # measurement, not this one.
    return_min_span=1.0,
    # The actor-side counterpart is a true epsilon. Its floor was pinning
    # ally_win/enemy_win/kill_shot/kill_assist/combat_death/shoot_quality at
    # 0.1 against true RMS values of 0.0075-0.027, downweighting the win signal
    # ~13x in the policy gradient. No loss-magnitude risk here: the aggregated
    # advantage is renormalized to unit RMS again after lambda aggregation
    # (see _compute_minibatch_loss), so this changes the mix, not the scale.
    advantage_min_rms=1e-4,
    return_quantile_samples=262_144,
    checkpoint_dir="checkpoints",
    league_size=20,
    league_slots=4,
    # Fitted by `--mode semi_random --profile rl` on 4v4 at action_repeat=2,
    # 128 games/pair (checkpoints/rl/semi_random_tournament.json).
    # Re-run it if the tick rate, ship config or fleet size changes -- the
    # ratings move by hundreds of Elo when it does.
    reference_ladder=(
        (0.2, -236.0),
        (0.3, -96.2),
        (0.4, 114.9),
        (0.5, 270.7),
        (0.6, 461.1),
        (0.7, 589.2),
        (0.8, 733.0),
        (0.9, 861.3),
        (0.95, 942.5),
    ),
    random_elo=-363.9,
    elo_milestone_gap=200.0,
    elo_temperature=200.0,
    league_uniform_sampling=False,
    elo_eval=ELO_EVAL,
    bc_winrate_target=0.45,
    histogram_interval=10,
)
