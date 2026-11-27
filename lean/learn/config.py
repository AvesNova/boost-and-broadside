"""Settings of the learner: the BC stage, PPO, the optimiser, weight averages, reward
levels, the hit head, team-size draws, self-play and the league."""

from dataclasses import dataclass

from lean.contract import LEVELS

# Indices of v2's reward levels (src/bnb/contracts/trajectory.py at 4cfd1e86), lean.contract's
# order.
DAMAGE, KILL_DEATH, OBJECTIVE_PROGRESS, OBJECTIVE, OUTCOME = range(LEVELS)


# Placeholder league personas (LearnConfig.league_personas): round 1's spread-1 personas by
# label, every sampled row (skill.json's values to 4 significant figures).
# fmt: off
PLACEHOLDER_PERSONAS: tuple = (
    ("persona14", (
        ("threat_weight", 0.8996), ("angles_weight", 0.3918), ("energy_weight", 0.2691),
        ("aggression", 0.6698), ("speed_weight", 0.3253), ("collision_weight", 2.245),
        ("boundary_weight", 1.675), ("field_weight", 0.3781), ("horizon", 1.649),
        ("preferred_speed", 0.7812), ("reply_temperature", 0.1839), ("caution", 0.6477),
        ("temperature", 0.0496), ("stickiness", 0.04844), ("store_weight", 0.6157),
        ("power_horizon", 3.786), ("power_temperature", 0.03979), ("power_stickiness", 0.2119),
        ("disengaged_energy_weight", 0.7228), ("seek_weight", 0.09279), ("fire_tolerance", 0.9744),
    )),
    ("persona21", (
        ("threat_weight", 1.073), ("angles_weight", 0.3343), ("energy_weight", 0.2875),
        ("aggression", 0.5946), ("speed_weight", 0.4091), ("collision_weight", 2.304),
        ("boundary_weight", 2.002), ("field_weight", 0.4414), ("horizon", 1.231),
        ("preferred_speed", 0.9202), ("reply_temperature", 0.2821), ("caution", -1.672),
        ("temperature", 0.03741), ("stickiness", 0.06428), ("store_weight", 0.6364),
        ("power_horizon", 3.544), ("power_temperature", 0.03567), ("power_stickiness", 0.3877),
        ("disengaged_energy_weight", 0.5387), ("seek_weight", 0.07699), ("fire_tolerance", 1.109),
    )),
    ("persona27", (
        ("threat_weight", 0.8968), ("angles_weight", 0.3951), ("energy_weight", 0.2277),
        ("aggression", 0.3064), ("speed_weight", 0.2856), ("collision_weight", 2.485),
        ("boundary_weight", 1.716), ("field_weight", 0.3017), ("horizon", 1.227),
        ("preferred_speed", 0.917), ("reply_temperature", 0.1228), ("caution", -1.611),
        ("temperature", 0.0495), ("stickiness", 0.05535), ("store_weight", 0.6149),
        ("power_horizon", 3.437), ("power_temperature", 0.06214), ("power_stickiness", 0.2195),
        ("disengaged_energy_weight", 0.3051), ("seek_weight", 0.1704), ("fire_tolerance", 1.242),
    )),
    ("persona09", (
        ("threat_weight", 0.7911), ("angles_weight", 0.4138), ("energy_weight", 0.2779),
        ("aggression", 0.7431), ("speed_weight", 0.3778), ("collision_weight", 2.505),
        ("boundary_weight", 1.744), ("field_weight", 0.3421), ("horizon", 1.207),
        ("preferred_speed", 0.9899), ("reply_temperature", 0.3288), ("caution", -0.7859),
        ("temperature", 0.0345), ("stickiness", 0.04401), ("store_weight", 0.6504),
        ("power_horizon", 3.741), ("power_temperature", 0.04244), ("power_stickiness", 0.224),
        ("disengaged_energy_weight", 0.4842), ("seek_weight", 0.1532), ("fire_tolerance", 0.7681),
    )),
)
# fmt: on


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
    # (lean.scripted.decide's joint over the 42 actions) with weight bc_coefficient, no
    # surrogate and no entropy
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
    # How the compiled path (lean.learn.compiled) compiles: "static" specialises every graph
    # to the run's sizes (environments, ships, fields, league rows, micro-batch: all fixed in a
    # run) and makes a second compile an error, so a stray dynamic shape fails loudly;
    # "dynamic" compiles once for every size (duck sizing off), for runs whose sizes change.
    compile_shapes: str = "static"

    # Categorical critic: the measured range of each level's returns, in return units
    # (LEVEL_NAMES order); the support adds the buffer columns beyond both ends
    # (lean.model.value.buffered_centres), 63 bins uniform in each level's bin space
    # (lean.model.value.symlog with value_knee; 0 linear). The learner owns the support.
    # Measured 2026-11-25 (scripted against scripted, the training game, 4v4, 35.5M living
    # ship-ticks of finished episodes; docs/rewrite/sections/scoreboard/target_ranges.py):
    # extremes damage -3.9 to 5.4, kill_death -6.5 to 7.5, outcome +-10 (its bound); the
    # 0.1st to 99.9th percentiles -2.1 to 2.1, -2.8 to 3.7 and +-10. Symmetric, so an
    # untrained critic (uniform softmax) reads 0. Objective levels: placeholders, no mode
    # pays them yet.
    value_low: tuple[float, ...] = (-5.4, -7.5, -1.0, -1.0, -10.0)
    value_high: tuple[float, ...] = (5.4, 7.5, 1.0, 1.0, 10.0)
    # Symlog knees by the event rule (author 2026-11-25): one reward event's size x e^-3 (the
    # event three discount horizons ahead). A hit: 21 points / damage_unit x 2.74 = 0.575
    # damage units, so 0.0286; a kill or a death: 2.83 units, so 0.141. The objective
    # placeholders and the outcome (bimodal at its bounds) stay linear.
    value_knee: tuple[float, ...] = (0.0286, 0.141, 0.0, 0.0, 0.0)
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
    # The two-level league (lean.learn.pool): K loaded slots take new games, R spare slots let
    # level 1 load a member while an unloaded slot drains its running games (at most about 4
    # updates at parity: a game lasts at most clock_ticks + segment_ticks = 1,928 ticks). Each
    # slot holds a member: a network's weights or a scripted agent's parameters; a member may
    # hold several. Level 1 (between updates) sets each member's slots to its PFSP share of K
    # (the rounding carried from update to update, so the long-run share is exact), at most
    # one load per free slot; level 2 (each game's end) draws uniformly over the loaded slots
    # with a free row.
    league_slots: int = 12
    league_spare_slots: int = 4
    # Each slot's row capacity C = ceil(league_capacity_factor * L / league_slots), fixed for
    # the run: every network slot runs on C rows.
    league_capacity_factor: float = 1.25
    # The scripted personas in the league from RL's start, each (label, ((row, value), ...)):
    # named points of the scripted agent's parameters (lean.scripted.PARAMETERS; rows not
    # named keep the yardstick's value). Placeholders (2026-11-27): four of round 1's
    # spread-1 personas (docs/rewrite/sections/lookahead-personas/results.md, skill.json),
    # the strongest in 1v1 (persona14, 1319 / 974 Elo 1v1 / 4v4), in 4v4 (persona21,
    # 1079 / 1107), in both (persona27, 1248 / 1046) and persona09 (1284 / 1010), until the
    # persona study picks the league's personas.
    league_personas: tuple = PLACEHOLDER_PERSONAS
    league_floor: float = 0.125  # share of league draws uniform over the members
    league_decay: float = 0.99  # each game decays its member's win tallies by this first
    # Checkpoints to disk (lean.train): every this many environment steps, none deleted.
    checkpoint_every: int = 10_000_000


def learning_rate(config: LearnConfig, step: int) -> float:
    """Learning rate of Adam step `step` (from 0, counted from the stage's start): linear
    warm-up over config.warmup_steps steps, then constant."""
    return config.learning_rate * min(1.0, (step + 1) / max(1, config.warmup_steps))
