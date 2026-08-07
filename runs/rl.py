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
    # 0.29, not 1.0. The value loss and the policy loss land on the same trunk
    # and max_grad_norm renormalizes them together, so whichever sends more
    # gradient takes a larger share of every clipped step. Cross-entropy sends
    # 3.44x the trunk gradient that squared error does at convergence (measured
    # offline on real returns; its loss value is 24x larger, but that is mostly
    # the CE entropy floor and only 1.38x reaches the head's own parameters).
    #
    # Run 711 paid for that: solving the observed gradient norms for an
    # actor/critic split gives actor 0.68 / critic 0.73 under MSE against actor
    # 0.68 / critic 2.51 under CE, so the actor's share of each clipped update
    # fell from 68% to 26%. Its critic fit better and its policy was ~65 Elo
    # worse. 0.29 puts the critic term back at 0.73 and the actor back at 68%.
    #
    # Watch train/actor_grad_share -- this is a convergence-regime number, and
    # MSE's gradient is larger early, so the critic may be under-weighted for
    # the first ~20M steps.
    value_function_coef=constant(0.29),
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
    # NOT an epsilon, and it must not be dropped to one. It divides the return
    # distribution by a robust p5-p95 half-span, and for a sparse component that
    # span measures the noise floor of nothing happening rather than the signal:
    # field_death's central 90% spans 0.0009 while its events reach -0.12. Drop
    # the floor and those events land at |z| up to 2000 against a +/-5 bin grid,
    # where every one of them encodes to the same end bin. Run 710 did exactly
    # that and the affected components collapsed -- measured representation
    # ceilings of 0.085 (field_death), 0.151 (kill_shot), 0.415
    # (damage_dealt_ally) against observed EV of 0.024/0.032/0.109.
    #
    # At 1.0 the ceiling is 1.000 on every component. The compression that used
    # to cost those components their share of the *critic* gradient is no longer
    # a concern under cross-entropy, whose loss does not scale with the target,
    # and two-hot stays exact well below one bin width because it interpolates
    # linearly rather than quantizing.
    return_min_span=1.0,
    # Two-hot. An offline head-only comparison on frozen trunk features measured
    # HL-Gauss at sigma=0.75 as a wash (mean held-out EV 0.613 vs 0.611), so the
    # sharper target is kept for being the one whose mean is exact.
    value_sigma=0.0,
    # The actor-side counterpart is a true epsilon. Its floor was pinning
    # win/kill_shot/kill_assist/combat_death/shoot_quality at
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
