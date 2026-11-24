"""Settings of the learner: the BC stage, PPO, the optimiser, weight averages, reward
levels, the hit head, team-size draws, self-play and the league."""

from dataclasses import dataclass

from lean.contract import LEVELS

# Indices of v2's reward levels (src/bnb/contracts/trajectory.py at 4cfd1e86), lean.contract's
# order.
DAMAGE, KILL_DEATH, OBJECTIVE_PROGRESS, OBJECTIVE, OUTCOME = range(LEVELS)


@dataclass
class LearnConfig:
    """Every number the learner uses.

    Discounts are per tick. Both v2 and this game tick at 30 Hz with one decision per tick,
    so v2's per-level discounts and GAE lambdas carry over unchanged
    (v2 src/bnb/learn/ppo.py:172-173 at 4cfd1e86). A different tick rate `f` would use
    `discount ** (30 / f)`.
    """

    # Returns, per level in LEVEL_NAMES order.
    discounts: tuple[float, ...] = (0.982, 0.990, 0.995, 0.999, 0.9997)
    gae_lambdas: tuple[float, ...] = (0.81, 0.90, 0.95, 0.97, 0.97)
    # Advantages are summed over levels and divided by their standard deviation over the
    # update, the mean not subtracted (docs/rewrite/literature-recipe-1.md section 4).

    # PPO.
    clip: float = 0.2
    epochs: int = 4
    # False (author 2026-11-23, after the literature review,
    # docs/rewrite/literature-ppo-targets-minibatches.md): advantages and value targets once
    # per update from the values stored at collection, as v2 (ppo.py). True: recomputed from
    # the current critic at every epoch (a streamed target pass over the rollout per epoch).
    targets_every_epoch: bool = False
    # The update's size, at parity with v2 (author 2026-11-23): `environments` x
    # `segments_per_update` x `segment_ticks` environment ticks (2,048 x 4 x 128 = 1,048,576).
    # A view is one seat's segment (1.5 seats per environment with half self-play: 12,288
    # views); `minibatches` split the views, one optimizer step each.
    environments: int = 2048
    segments_per_update: int = 4
    segment_ticks: int = 128
    minibatches: int = 32
    # Views per micro-batch: memory follows this, not the minibatch (gradients accumulate
    # over a minibatch's micro-batches; every loss term is divided by its whole-minibatch
    # denominator). Also the batch of the critic's target pass. Sized for an 8 GB GPU
    # (docs/rewrite/scoreboard.md, "Streaming and micro-batches").
    micro_batch_views: int = 64
    # The rollout's large records in pinned host memory, streamed (lean.learn.store); False
    # keeps everything on the device. Ignored on the CPU.
    stream_rollout: bool = True
    # Optimiser (docs/rewrite/literature-recipe-2.md section 4): Adam without weight decay,
    # fused, global gradient clip; a step whose gradient norm is not finite is skipped. The
    # learning rate rises linearly over `warmup_steps` Adam steps (2 / (1 - beta2) at beta2
    # 0.999: about 16 updates at 128 steps each), then stays constant (no cooldown yet).
    learning_rate: float = 3e-4
    warmup_steps: int = 2000
    adam_epsilon: float = 1e-5
    gradient_clip: float = 1.0
    # Loss coefficients, fixed: every term is scale-free (the surrogate on normalised
    # advantages, every other term a cross-entropy or entropy in nats).
    entropy_coefficient: float = 0.003  # on the joint 42-way entropy (RL only)
    value_coefficient: float = 0.05
    next_state_coefficient: float = 0.1
    hit_coefficient: float = 0.05  # cross-entropy of the hit head, nats per labelled row
    # A warning when an epoch's mean approximate KL to the collecting policy exceeds this
    # (a monitor: no early stop, docs/rewrite/literature-recipe-1.md section 3).
    kl_warning: float = 0.02

    # The BC stage (lean.train --stage bc): the student plays its own games (the same
    # rollouts as RL, the league holding only the scripted agent) and every head learns:
    # the policy by the cross-entropy to the scripted teacher's full distribution
    # (the marginals of lean.scripted.decide) with weight bc_coefficient, no surrogate and no
    # entropy
    # bonus; the critics, next-state and hit heads as in RL. 96 updates at parity
    # (2,048 x 4 x 128) is 100.7M environment ticks: v2's 100M-tick BC stage, and the
    # v1/v2 target of BC within 100M.
    bc_updates: int = 96
    bc_coefficient: float = 1.0
    # The averaged weights the BC stage hands to RL: "ema", "uniform" or "live".
    handoff: str = "ema"
    # RL's length in updates (lean.train --stage rl or all; --updates overrides it).
    rl_updates: int = 1000
    # RL's teacher term: the same cross-entropy with weight teacher_coefficient *
    # teacher_decay ** k at RL update k (a reverse warm-up, as VPT's KL to the cloned
    # policy), switched off for good once the win rate against the scripted agent (the
    # league's tally, 0 until a game is played) reaches teacher_gate, or once the weight
    # falls below teacher_cutoff.
    teacher_coefficient: float = 0.5
    teacher_decay: float = 0.95
    teacher_gate: float = 0.45
    teacher_cutoff: float = 0.005

    # Weight averages kept beside the live weights, never fed back into training
    # (lean.learn.weights): an EMA with this decay per update, and a uniform average of
    # every update's weights from the end of the learning-rate warm-up.
    ema_decay: float = 0.95
    # The policy's forward passes (rollout, update, and evaluation of its checkpoints) under
    # bf16 autocast on a GPU (lean.model.agent.precision); the model keeps its position phases
    # and recurrence in float32. Ignored on the CPU.
    bf16: bool = True

    # Categorical critic: the outermost bin centres per level, in return units (LEVEL_NAMES
    # order). Each holds the level's measured returns with margin (lean/learn/NOTES.md,
    # "Reward levels"). The learner owns the support.
    # Measured 2026-11-23 (scripted against scripted, whole episodes, 4v4 to 8v8, v2 and
    # core games; and the learner's first rollouts): damage -4.9 to 5.9, kill_death -5.7 to
    # 7.9, outcome +-10 (its bound). Symmetric, so an untrained critic (uniform softmax)
    # reads 0. Objective levels: placeholders, no mode pays them yet.
    value_low: tuple[float, ...] = (-7.0, -8.0, -1.0, -1.0, -12.0)
    value_high: tuple[float, ...] = (7.0, 8.0, 1.0, 1.0, 12.0)
    value_sigma_ratio: float = 0.75  # Gaussian width of a target, in bin widths
    # Range monitoring (per level, per epoch's targets): a warning when more than this share
    # of the valued targets lies beyond the outermost centres (clamped), and when the
    # targets occupy fewer than this many bins (a bin counts when it holds at least
    # range_bin_share of them). A level that paid no reward in the rollout is not warned narrow.
    range_clamped_warning: float = 0.01
    range_narrow_bins: int = 8
    range_bin_share: float = 0.001

    # Rewards: v2's level weights in LEVEL_NAMES order (v2 src/bnb/learn/rewards.py:74-79,
    # v1's times 10): damage per damage_unit points, kill_death, objective_progress,
    # objective, outcome.
    reward_weights: tuple[float, ...] = (2.74, 2.83, 20.0, 20.0, 10.0)
    damage_unit: float = 100.0  # points, for every ship type

    # Scenario draws at reset.
    team_size_minimum: int = 4
    team_size_maximum: int = 4
    equal_teams: bool = True
    # The scripted agent's personas, drawn per ship at every deal (lean.scripted.personas):
    # the spread of the persona box around the yardstick; 0 plays the yardstick itself.
    persona_spread: float = 0.0

    # Self-play and the league (lean.learn.pool.League). Environments 0 to E/2 - 1 are
    # self-play, the rest league games.
    self_play_share: float = 0.5  # share of environments where the live policy flies both teams
    # League snapshots: v2's league schedule (v2 src/bnb/runtime/snapshots.py LEAGUE_SNAPSHOTS,
    # about 100 to 200 Elo apart on rl-v20's curve): these environment steps, then every
    # league_snapshot_period (None: the last gap, 30M).
    league_snapshot_steps: tuple[int, ...] = tuple(
        step * 1_000_000 for step in (5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 110, 130, 160)
    )
    league_snapshot_period: int | None = None
    # Retention (v2's): the newest recent_snapshots, then the one nearest each age 2^k
    # snapshots back for k >= snapshot_history_from, at most snapshot_members (the oldest go
    # past it; on v2's schedule the rule keeps 11 to 500M steps, 12 to about 1.5B).
    recent_snapshots: int = 8
    snapshot_history_from: int = 3
    snapshot_members: int = 12
    # Slots for dropped snapshots that still play their running games. Retention drops at
    # most 2 snapshots at one addition, and drops start at the 11th snapshot (75M), from
    # where additions are at least 15M steps (14 updates at parity) apart while a game lasts
    # at most clock_ticks + segment_ticks = 5,528 ticks (11 updates): 2 suffice at parity; 4
    # leave room for smaller updates. With none free, the games of the retiring slot with
    # the fewest are re-dealt (League.abandoned counts them).
    retiring_slots: int = 4
    # Each member's row capacity: ceil(league_capacity_factor * L / members present).
    league_capacity_factor: float = 1.25
    league_floor: float = 0.125  # share of league draws uniform over the members
    league_decay: float = 0.99  # each game decays its member's win tallies by this first
    # Checkpoints to disk (lean.train): every this many environment steps, none deleted.
    checkpoint_every: int = 10_000_000


def learning_rate(config: LearnConfig, step: int) -> float:
    """Learning rate of Adam step `step` (from 0, counted from the stage's start): linear
    warm-up over config.warmup_steps steps, then constant."""
    return config.learning_rate * min(1.0, (step + 1) / max(1, config.warmup_steps))
