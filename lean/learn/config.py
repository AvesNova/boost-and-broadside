"""Settings of the learner: PPO, reward levels, the hit head, team-size draws, self-play
and the league."""

from dataclasses import dataclass

import torch

# v2's reward levels (src/bnb/contracts/trajectory.py at 4cfd1e86), in v2's order.
LEVEL_NAMES = ("damage", "kill_death", "objective_progress", "objective", "outcome")
LEVELS = len(LEVEL_NAMES)
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
    advantage_normaliser: str = "std"  # "std" or "percentile"
    advantage_percentiles: tuple[float, float] = (0.05, 0.95)

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
    learning_rate: float = 3e-4
    warmup_updates: int = 10
    adam_epsilon: float = 1e-5
    gradient_clip: float = 1.0
    entropy_coefficient: float = 0.003
    value_coefficient: float = 0.05
    next_state_coefficient: float = 0.1
    bc_coefficient: float = 2.0
    bc_gate: float = 0.45
    hit_coefficient: float = 0.05  # cross-entropy of the hit head, nats per labelled row
    # The policy's forward passes (rollout and update) under bf16 autocast on a GPU; the
    # model keeps its position phases and recurrence in float32. Ignored on the CPU.
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

    # Rewards: v2's level weights (v2 src/bnb/learn/rewards.py:74-79, v1's times 10).
    damage_weight: float = 2.74  # per damage_unit points
    kill_death_weight: float = 2.83
    objective_progress_weight: float = 20.0
    objective_weight: float = 20.0
    outcome_weight: float = 10.0
    damage_unit: float = 100.0  # points, for every ship type

    @property
    def reward_weights(self) -> tuple[float, ...]:
        """The level weights in LEVEL_NAMES order."""
        return tuple(getattr(self, f"{name}_weight") for name in LEVEL_NAMES)

    # Scenario draws at reset.
    team_size_minimum: int = 4
    team_size_maximum: int = 4
    equal_teams: bool = True

    # Self-play and the league (lean.learn.pool.League).
    self_play_share: float = 0.5  # share of environments where the live policy flies both teams
    snapshot_every: int = 10  # updates between additions to the league
    snapshot_capacity: int = 8  # snapshot members; the scripted agent is one more
    league_floor: float = 0.125  # share of league draws uniform over the members
    league_decay: float = 0.99  # each game decays its member's win tallies by this first
    persona_spread: float = 0.0

def policy_precision(config: LearnConfig, device) -> torch.autocast:
    """The autocast context of the policy's forward passes: bf16 on a GPU when config.bf16."""
    device = torch.device(device)
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=config.bf16 and device.type == "cuda")


def bc_weight(config: LearnConfig, scripted_win_rate: float) -> float:
    """Coefficient of the BC term: bc_coefficient * max(0, 1 - win_rate / gate)."""
    return config.bc_coefficient * max(0.0, 1.0 - scripted_win_rate / config.bc_gate)


def learning_rate(config: LearnConfig, update_index: int) -> float:
    """Learning rate of update `update_index` (from 0): linear warm-up, then constant."""
    return config.learning_rate * min(1.0, (update_index + 1) / max(1, config.warmup_updates))
