"""PPO with v1's math (learner spec 3, 4): per-level returns, the categorical critic, the
return scale and the auxiliary losses with their labels, over any model with `unroll`.

An update reads its segments from a store (`Segments`) and works on the device one shard or
one micro-batch at a time:
  1. the returns over the whole update's ticks, the shards in order and then the lookahead
     (the ticks played after the last shard, never trained on): per-level GAE, team mixing,
     the outcome's targets, bootstrapped only at the end; then the return scale, once per
     batch;
  2. one pass over the views: the shot re-simulation (each view's segment and the bullet
     flight after it), whose labels go into the store beside the segments for every epoch,
     each view's token counts and, when set, the old log-probabilities recomputed by
     `unroll`;
  3. epochs of minibatches over the views (a view keeps its whole segment); a minibatch is
     several micro-batches, each term divided by its minibatch-total denominator, so the
     accumulated gradient is the unsplit minibatch's.
Results stay on the device: the update reads the epoch's mean KL once per epoch (the early
stop) and everything it reports once, at its end.
The learner owns the optimizer and updates the model's parameters in place, which keeps the
rollout's captured graph pointing at the current weights. A model lacking a head gives no
such field (or None) and that term is skipped.
"""

import dataclasses
import functools
import math
from collections.abc import Callable, Iterator
from typing import ClassVar, Protocol

import torch
from torch.utils import _pytree

from bnb.contracts import REWARD_LEVELS, Trajectory
from bnb.contracts.action import to_joint
from bnb.contracts.obs import GLOBAL, SHIP
from bnb.contracts.series import Mean, Stopwatch
from bnb.game.rules import Rules
from bnb.game.ship_types import ShipTypes
from bnb.models.yemong import codes, critic, geometry, policy
from bnb.models.yemong import shots as shot_codes
from bnb.models.yemong.density import HexGrid, poisson_terms
from bnb.streams import stream_generator

from . import advantages as level_advantages
from . import gradients, head_statistics, labels, losses, returns, shots, statistics

# The entropy's terms, one per action factor by the chain rule (`policy.factor_entropies`).
ENTROPY_TERMS = ("entropy_power", "entropy_stick", "entropy_shoot")
TERMS = (
    "policy",
    *ENTROPY_TERMS,
    "value",
    "next_state",
    "correction",
    "enemy_action",
    "density",
    "shot_trajectory",
    "shot_outcome",
    "geometry",
)
# Token counts per view, the denominators' parts: own ships alive; own ships valued (alive or
# waiting to respawn); view-steps; valid beliefs at non-final steps; enemies alive; shots;
# the geometry head's pairs (`geometry.pairs`).
COUNTS = ("policy", "valued", "steps", "next_state", "enemy_action", "shots", "pairs")
# The per-tick records the returns read, staged whole per shard (the outcome only with an
# outcome level: `Learner.small_names`).
SMALL = ("reward", "values", "outcome", "done", "win_loss", "log_probability", "alive", "valued")
# What the returns read of the lookahead (likewise: `Learner.lookahead_names`).
LOOKAHEAD = ("reward", "values", "outcome", "done", "win_loss")
# What the shot re-simulation reads of the ticks after a segment.
FOLLOWING = ("capture", "obs", "done")
# The auxiliary heads' losses (nats per token), as their groups' series.
HEAD_SERIES = {
    "next_state": "next_state/loss",
    "correction": "correction/loss",
    "enemy_action": "enemy_action/loss",
    "density": "density/loss",
    "shot_trajectory": "shot/loss/trajectory",
    "shot_outcome": "shot/loss/outcome",
    "geometry": "geometry/loss",
}
# The policy term's trust regions (`LearnerConfig.trust_region`).
TRUST_REGIONS = ("clip", "clip_higher", "dppo_tv", "dppo_binary_tv")
# The policy's divergence histogram (`trust_region` statistics): log10 bins over
# [DIVERGENCE_FLOOR, 1], a value below the floor in the first bin.
DIVERGENCE_BINS = 32
DIVERGENCE_FLOOR = 1e-4
DIVERGENCE_QUANTILES = (0.5, 0.9, 0.99)
# The policy's entropy buckets for the trust region's rate, as fractions of ln NUM_JOINT.
ENTROPY_BUCKETS = ("low", "middle", "high")
# The stored shot labels' name in the store: per view, {"shots": Shots, "mask"}, (V, L, ...).
SHOT_LABELS = "shot_labels"
# `chained_returns`' sums of the levels' raw advantages (`advantages.raw_sums`), not per view
RAW_SUMS = "raw_advantage_sums"


class Adam(torch.optim.Adam):
    """`torch.optim.Adam`, whose graph-capture check runs only for parameters on an
    accelerator. The check reads the accelerator's current stream, which initialises CUDA on
    a machine that has it even when every parameter is on the CPU; a CPU run never
    captures, so skipping it there changes nothing the step computes."""

    def _accelerator_graph_capture_health_check(self) -> None:
        if any(
            parameter.device.type != "cpu"
            for group in self.param_groups
            for parameter in group["params"]
        ):
            super()._accelerator_graph_capture_health_check()


@dataclasses.dataclass(frozen=True)
class LearnerConfig:
    """Learner spec 3-4, v1's values. The per-level tuples (discount, trace_decay,
    team_spirit) give one value per value level of the critic it trains, in its order (the
    model's `config.levels`: by default REWARD_LEVELS; one for the single value level,
    `bnb.contracts.SUMMED_LEVEL`). A coefficient
    multiplies its term's mean in nats per token. The entropy has one coefficient per action
    factor, on its chain-rule part (`entropy_power` on H(power), `entropy_stick` on
    H(stick | power), `entropy_shoot` on H(shoot | power, stick)), so three equal coefficients
    weigh the joint entropy; a negative coefficient rewards it.

    old_log_probabilities: "rollout" (the rollout's) or "recomputed" (by `unroll` before
    the first epoch). minibatches: per epoch, as equal in views as the views allow;
    micro_batch_rows: ship rows per micro-batch, so micro_batch_rows // N views (at least
    one): memory only, the result is the same, and the memory a micro-batch needs follows
    its rows at any fleet size.

    trust_region: the policy term's trust region. "clip", PPO's ratio clip at 1 +- clip;
    "clip_higher", DAPO's asymmetric clip, the ratio in [1 - clip, 1 + clip_high];
    "dppo_tv" and "dppo_binary_tv", DPPO's mask (Qi et al. 2026; `losses.dppo_keep`): the
    unclipped r A, its gradient zeroed where the divergence between the rollout's policy and
    the current one at the state is past dppo_delta and the update moves further away. The
    divergence of "dppo_tv" is the exact total variation of the 42-way joint action
    distribution (the rollout records its probabilities); "dppo_binary_tv" is |mu(a) - pi(a)|
    of the sampled action, a lower bound of it.
    """

    # Per value level (REWARD_LEVELS: damage, kill_death, objective_progress, objective,
    # outcome): the 2026-11-09 champion's (0.991, 0.995, 0.995, 0.999, 0.9997 and 0.90,
    # 0.95, 0.95, 0.97, 0.97) with the two combat levels' squared for the x2 game (game v13:
    # a fight takes half the ticks); the objective and outcome levels follow the mode
    # clocks, which kept their seconds. One value for every level (tried 2026-11-13) let a
    # run learn to stop fighting.
    discount: tuple[float, ...] = (0.982, 0.990, 0.995, 0.999, 0.9997)
    trace_decay: tuple[float, ...] = (0.81, 0.90, 0.95, 0.97, 0.97)
    team_spirit: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    return_scale_decay: float = 0.99
    return_scale_floor: float = 1.0
    clip: float = 0.2
    trust_region: str = "clip"
    clip_high: float = 0.28  # "clip_higher"'s upper margin (DAPO's); its lower one is `clip`
    dppo_delta: float = 0.15  # the DPPO modes' divergence threshold (the paper's, for TV)
    policy: float = 27.0
    entropy_power: float = -0.0081
    entropy_stick: float = -0.0081
    entropy_shoot: float = -0.0081
    value: float = 0.007
    # the delta physics head's (the champion's; the absolute head took 0.0041 for the same
    # trunk share: docs/explanation/learner.md, round2-log.md 03:50-04:13)
    next_state: float = 0.0033
    # the error head's (`yemong.delta_split`); its term and the physics head's share one
    # denominator (every next-state row), so equal coefficients weigh a row equally
    correction: float = 0.0033
    enemy_action: float = 0.11
    density: float = 0.021
    shot_trajectory: float = 0.007
    shot_outcome: float = 0.019
    # the geometry head's (`yemong.geometry_head`), nats per target per pair
    geometry: float = 0.007
    learning_rate: float = 3e-4
    warmup_start: float = 1e-7
    warmup_steps: int = 10_000_000  # environment steps (decisions: round four)
    max_grad_norm: float = 1.0
    epochs: int = 4
    minibatches: int = 32
    micro_batch_rows: int = 160
    target_kl: float | None = 0.02  # None: no early stop
    old_log_probabilities: str = "rollout"

    # the rotary read-out's coefficient (2026-11-07 to 2026-11-12), retired with the head
    RETIRED: ClassVar[dict] = {"rotary_readout": None}

    def __post_init__(self) -> None:
        if self.old_log_probabilities not in ("rollout", "recomputed"):
            raise ValueError(f"old_log_probabilities {self.old_log_probabilities!r}")
        if self.trust_region not in TRUST_REGIONS:
            raise ValueError(f"trust_region {self.trust_region!r}; one of {TRUST_REGIONS}")
        if self.trust_region == "dppo_tv" and self.old_log_probabilities != "rollout":
            raise ValueError(
                "trust_region 'dppo_tv' compares against the rollout's recorded distribution: "
                "old_log_probabilities must be 'rollout'"
            )
        if not 0.0 < self.clip < 1.0 or self.clip_high <= 0.0:
            raise ValueError(
                f"clip {self.clip} must be in (0, 1) and clip_high {self.clip_high} positive"
            )
        if not 0.0 < self.dppo_delta < 1.0:
            raise ValueError(f"dppo_delta {self.dppo_delta} must be in (0, 1)")

    @property
    def records_distribution(self) -> bool:
        """Whether the rollout records the learner's whole joint action distribution."""
        return self.trust_region == "dppo_tv"


# The entropy coefficients of a warm start from a policy file (`bnb train --init-from`), in
# place of LearnerConfig's `entropy_*`. The BC clone already samples every action (its labels'
# uniform floor), and at its advantage scale the bonus outweighs the policy gradient on rare
# actions: from the first clone, v1's -0.0081 drove the shoot share from 1.7% to 6.8% and the
# score against the scripted agent from 0.23 to 0.03 in 24 updates, while 0 held it (0.25).
WARM_START_ENTROPY = 0.0


class Segments(Protocol):
    """Where the update's segments live (`bnb.runtime.store.SegmentStore`): `shards`
    consecutive segments of `length` ticks of the same `views // shards` rows, every
    Trajectory field by name (`names`), and the lookahead after the last
    (`lookahead_ticks`: the ticks it keeps of each name)."""

    views: int
    shards: int
    length: int
    names: tuple[str, ...]
    lookahead_ticks: dict[str, int]

    def shard(self, names: tuple[str, ...], shard: int) -> dict:
        """Whole-shard copies on the device of the per-tick leaves names, (L, R, ...) with
        R = views // shards."""

    def lookahead(self, names: tuple[str, ...]) -> dict:
        """The lookahead's leaves names on the device, (n, R, ...)."""

    def following(self, names: tuple[str, ...], views: torch.Tensor, ticks: int):
        """The first `ticks` ticks after each view's segment on the device, and whether each
        view has them: ({name: (ticks, n, ...)}, (n,) bool)."""

    def batches(self, names: tuple[str, ...], view_batches: list[torch.Tensor]) -> Iterator[dict]:
        """Yield {name: pytree} on the device for each list of views in view_batches."""

    def write_views(self, source: dict, views: slice) -> None:
        """Copy {name: pytree (n, ...)} into the per-view leaves of the n views of the slice
        views."""


def learning_rate(config: LearnerConfig, steps: int) -> float:
    """Linear warmup from warmup_start over the first warmup_steps environment steps."""
    fraction = min(1.0, steps / config.warmup_steps) if config.warmup_steps else 1.0
    return config.warmup_start + (config.learning_rate - config.warmup_start) * fraction


def _head(unrolled, name: str):
    return getattr(unrolled, name, None)


def _physics(label: dict) -> torch.Tensor:
    """The rows the next-state (physics) head trains on: with `yemong.delta_split` the own
    rows and the enemies seen at t, else every next-state row."""
    return label.get("physics_mask", label["next_state_mask"])


def physics_sum(cross_entropy: torch.Tensor, label: dict, rows: torch.Tensor | None = None):
    """The physics (next-state) head's cross-entropy (L, R, N) summed over its rows
    (`_physics`, within rows if given); with `yemong.physics_hidden_decay` each row times
    its weight (`labels.split_masks`: 1 on the seen rows, exp(-age / tau) on the hidden
    ones). The term divides it by the next-state mask's count, as without the switch: the
    count already holds the hidden rows (the error head's term shares it), so a seen row
    weighs exactly what it did and the hidden rows add on top."""
    weight = label.get("physics_weight")
    if weight is None:
        mask = _physics(label)
        return losses.masked_sum(cross_entropy, mask if rows is None else mask & rows)
    mask = weight > 0
    mask = mask if rows is None else mask & rows
    return torch.where(mask, cross_entropy * weight.to(cross_entropy.dtype), 0.0).sum()


def factor_sums(entropies: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(3,): the factor entropies (..., 3) summed over the tokens of `mask` (...)."""
    return torch.where(mask[..., None], entropies, 0.0).flatten(0, -2).sum(0)


def _own_joint(trajectory: Trajectory) -> torch.Tensor:
    return to_joint(trajectory.action)[..., : trajectory.log_probability.shape[-1]]


# The views the heads' sums (`head_statistics`) read on a cadence update: the first
# micro-batches of the first epoch until they hold this many (a sample; the sums' series
# are ratios of totals, so their meaning does not change with its size). Eager, at about
# 0.35 s a 32-view micro-batch, the whole epoch cost about 8% of RL's steady time at 1024
# environments (`steady-speed` notes); 256 views is 8 such micro-batches.
HEAD_SUM_VIEWS = 256


def micro_batch_sizes(count: int, most: int, family: tuple[int, ...] = ()) -> list[int]:
    """The micro-batch sizes (in order, summing to `count`) of `count` views, each at most
    `most`, chosen with `family`, the other view counts split alongside it, so that the
    counts together take as few sizes as they can: under a compile of fixed shapes each size
    is one compile of every learner stage. One size where one divides every count in up to
    twice its fewest micro-batches (255 views at 64: five of 51); otherwise each count in
    as many micro-batches as the largest needs, as equal as they can be, which gives
    counts that differ by one (the minibatches when they do not divide the views) the same
    two sizes (383 and 382 at 64: six of 64 or 63). The split is memory and speed only: the
    micro-batches' gradients add up to the minibatch's."""
    counts = {count, *family}
    for size in range(min(most, min(counts)), 0, -1):
        if all(n % size == 0 and n // size <= 2 * -(-n // most) for n in counts):
            return [size] * (count // size)
    parts = min(count, -(-max(counts) // most))
    return [len(part) for part in torch.arange(count).tensor_split(parts)]


class Learner:
    """model: the learner's module, with `unroll(carry, obs, replay, reset)`, the critic's
    `bins` and `config.levels` (one of `returns.VALUE_LAYOUTS`), `outcome_level` (None: no
    outcome head) and `local_levels`; Yemong's auxiliary heads
    also need `state_code`, `shot_code`, `shot_trajectory` and `config.hex_rings`. rules and
    types: the game's (for the shot re-simulation). outcome_weight: the outcome level's
    reward weight (its value in reward units is P(win) - P(loss) times it). stage: how the
    runtime compiles the micro-batch loss, the shot re-simulation's tick and the label codes
    (identity: eager). split: compile the micro-batch loss as two units, the model's unroll
    and the loss over its outputs, rather than one graph (the same arithmetic; a
    compile-time choice, `bnb.runtime.train.TrainConfig.compile_split`)."""

    # whether the diagnostics measure the levels' advantages and policy gradients
    # (`bnb.learn.advantages`): PPO's policy term reads the advantage, BC's does not
    measures_advantages = True

    def __init__(
        self,
        model: torch.nn.Module,
        config: LearnerConfig,
        rules: Rules,
        types: ShipTypes,
        outcome_weight: float,
        device: torch.device,
        stage: Callable = lambda function: function,
        seed: int = 0,
        diagnostics: bool = True,
        split: bool = False,
        reward_weights: tuple[float, ...] | None = None,
        view_modes: tuple[str, ...] = (),
        view_mode: torch.Tensor | None = None,
    ) -> None:
        """seed: the learner's random streams (the shot spread, the minibatch order).
        diagnostics: measure the diagnostic series beyond the losses and the early stop's
        KL (none of them changes the update). reward_weights: the reward levels' weights
        (`bnb.learn.rewards.RewardConfig.weights`), for the levels' raw advantage (None: not
        measured). view_modes and view_mode (V,): the modes of the batch's games and each
        stored view's index into them, for the advantage series per mode (none without)."""
        self.model, self.config, self.rules, self.types = model, config, rules, types
        self.device, self.outcome_weight = device, outcome_weight
        self.reward_weights = reward_weights
        self.view_modes = view_modes
        self.view_mode = view_mode
        self.view_members: dict = {}  # `_view_member`'s, by (slice names, views)
        levels = tuple(model.config.levels)
        if levels not in returns.VALUE_LAYOUTS:
            raise ValueError(f"the model's levels {levels} are not one of {returns.VALUE_LAYOUTS}")
        for name in ("discount", "trace_decay", "team_spirit"):
            if len(getattr(config, name)) != len(levels):
                raise ValueError(
                    f"learner {name} {getattr(config, name)} needs one value per value level "
                    f"of the model, {levels}"
                )
        self.levels = levels
        outcome = model.config.outcome_level
        self.outcome_level = None if outcome is None else levels.index(outcome)
        # what the returns read of the store: the outcome's records only with its head
        self.small_names, self.lookahead_names = (
            tuple(name for name in names if outcome is not None or name != "outcome")
            for names in (SMALL, LOOKAHEAD)
        )
        self.moments = 5 * len(levels)  # the fit's moments at the end of the diagnostics
        self.local_levels = model.config.local_levels
        self.local_index = torch.tensor(self.local_levels, device=device)  # no copy per use
        self.optimizer = Adam(
            model.parameters(), lr=config.learning_rate, fused=device.type == "cuda"
        )
        self.return_scale = critic.ReturnScale(
            config.return_scale_decay, config.return_scale_floor, device
        )
        self.spread_generator = stream_generator(seed, "shot_spread", device)  # spread, ages
        self.order_generator = stream_generator(seed, "minibatch_order")  # host indices
        self.auxiliary = hasattr(model, "state_code")
        if self.auxiliary:
            self.grid = HexGrid(model.config.hex_rings, device)
            self.shot_runs = model.shot_code.trajectory_runs
        self.discount = torch.tensor(config.discount, device=device)
        self.trace_decay = torch.tensor(config.trace_decay, device=device)
        self.every_trace = torch.ones_like(self.trace_decay)  # λ = 1: the rewards themselves
        self.team_spirit = torch.tensor(config.team_spirit, device=device)
        if split:  # the unroll one unit, shared by the loss and the cadence terms
            unroll, loss_of = stage(self.unroll), stage(self._loss_of)
            terms_of = stage(self._terms_of)
            self.loss = lambda trajectory, *rest: loss_of(unroll(trajectory), trajectory, *rest)
            self.cadence_terms = lambda trajectory, *rest: terms_of(
                unroll(trajectory), trajectory, *rest
            )
        else:
            self.loss = stage(self._loss)
            self.cadence_terms = stage(self._terms)  # the cadence groups' own compiled function
        # Eager: a diagnostic on the cadence's updates only, of about 40k traced ops; compiled
        # it was the largest single compile of a cold start (554 s, `compile-split` notes).
        self.head_sums = self._head_sums
        self.cadence_stopwatch = Stopwatch(device)
        self.shot_tick = stage(shots.tick)
        self.codes = stage(self._codes)
        self.environment_steps = 0
        self.diagnostics = diagnostics
        # the loss returns what the heads' sums read (one graph for every epoch)
        self.measures_heads = diagnostics and self.auxiliary
        self.parameter_groups = statistics.parameter_groups(model)
        # made once: a tensor built from host values inside the update would be a copy
        self.return_quantiles = torch.tensor([0.05, 0.5, 0.95], device=device)
        self.entropy_quantiles = torch.tensor(statistics.ENTROPY_QUANTILES, device=device)
        self.divergence_quantiles = torch.tensor(DIVERGENCE_QUANTILES, device=device)

    # ------------------------------------------------------------------ returns

    def _in_reward_units(self, values: torch.Tensor) -> torch.Tensor:
        """The outcome level's value is P(win) - P(loss); times its weight it is in reward
        units like the others. Without an outcome level every value already is."""
        if self.outcome_level is None:
            return values
        scale = torch.ones(values.shape[-1], device=values.device)
        scale[self.outcome_level] = self.outcome_weight
        return values * scale

    def chained_returns(self, chain: dict, alive: torch.Tensor, bootstrap) -> dict:
        """Per-level GAE, team mixing and the outcome targets (with an outcome level) over
        consecutive ticks of the same rows: `chain` the LOOKAHEAD records (T, R, ...), the
        trained ticks first, then any that are played only for their targets; `alive`
        (T', R, N) the trained ticks' (T' <= T), which the results cover; bootstrapped from
        `bootstrap` (the learner's Outputs after the last tick). Each value level's reward
        is `returns.value_rewards`'."""
        level, config = self.outcome_level, self.config
        trained = alive.shape[0]
        values = self._in_reward_units(chain["values"])
        done = chain["done"]
        rewards = returns.value_rewards(chain["reward"], self.levels)
        following = self._in_reward_units(bootstrap.values)
        advantages, level_returns = returns.per_level_gae(
            rewards, values, done, following, done[-1], self.discount, self.trace_decay
        )
        advantages, level_returns = advantages[:trained], level_returns[:trained]
        own_alive = alive[..., : values.shape[-2]]
        team = torch.zeros_like(own_alive, dtype=torch.int64)  # a view's own ships: one team
        advantage, total = returns.team_aggregate(
            team, own_alive, advantages, level_returns, self.team_spirit
        )
        out = dict(
            advantage=advantage,
            total=total,
            level_returns=level_returns,
            values=values[:trained],
        )
        if self.diagnostics and self.measures_advantages:  # each level's part of A
            out["level_advantage"] = returns.team_mixed_levels(
                team, own_alive, advantages, self.team_spirit
            )
            out[RAW_SUMS] = level_advantages.raw_sums(advantages, own_alive)
        if self.diagnostics:  # the critic against the rewards, not against itself
            _, monte_carlo = returns.per_level_gae(
                rewards, values, done, following, done[-1], self.discount, self.every_trace
            )
            out["monte_carlo_returns"] = monte_carlo[:trained]
        if level is not None:
            targets = critic.outcome_targets(
                chain["outcome"],
                chain["win_loss"],
                done,
                bootstrap.outcome,
                config.discount[level],
                config.trace_decay[level],
            )
            out["outcome_targets"] = targets[:trained]
        return out

    @staticmethod
    def _counts(small: dict) -> torch.Tensor:
        """(R, COUNTS) the tokens of each view that the small records give; the pass over
        the views fills in the rest."""
        own = small["log_probability"].shape[-1]
        alive, length = small["alive"], small["done"].shape[0]
        counts = torch.zeros((alive.shape[1], len(COUNTS)), device=alive.device)
        counts[:, COUNTS.index("policy")] = alive[..., :own].sum((0, 2))
        counts[:, COUNTS.index("valued")] = small["valued"].sum((0, 2))
        counts[:, COUNTS.index("steps")] = length
        counts[:, COUNTS.index("enemy_action")] = alive[..., own:].sum((0, 2))
        return counts

    # ------------------------------------------------------------------ labels

    def shots(
        self,
        trajectory: Trajectory,
        spread: torch.Tensor,
        drawn: torch.Tensor,
        following: dict | None = None,
        followed: torch.Tensor | None = None,
    ):
        """The re-simulated shots' labels (`shots.ShotLabels`, at `shots.query_ages`, the
        random ages from drawn (L, n, N, 3)) and the loss mask: valid shots of own ships and of
        enemies visible at t. following: the FOLLOWING records of the ticks after the
        segment ({name: (A, n, ...)}), where a shot fired near its end flies on; followed
        (n,) which views have them (None: none do)."""
        own, length = trajectory.log_probability.shape[-1], trajectory.done.shape[0]
        capture, fields, done = trajectory.capture, trajectory.obs.fields, trajectory.done
        available = None
        if following is not None:
            extra = following["done"].shape[0]

            def joined(head, tail):
                return _pytree.tree_map(lambda x, y: torch.cat([x, y], 0), head, tail)

            capture = joined(capture, following["capture"])
            fields = torch.cat([fields, following["obs"].fields], 0)
            done = torch.cat([done, following["done"]], 0)
            spread = torch.cat([spread, spread.new_zeros((extra,) + spread.shape[1:])], 0)
            available = torch.where(followed, length + extra, length)
        shot = shots.resimulate(
            self.rules,
            self.types,
            capture,
            fields,
            done,
            spread,
            own,
            self.shot_tick,
            available,
        )
        shot = _pytree.tree_map_only(torch.Tensor, lambda x: x[:length], shot)
        visible = trajectory.obs.ships[..., SHIP["visible"]] > 0.5
        rows = torch.arange(visible.shape[-1], device=visible.device) < own
        return shots.labels(shot, drawn), shot.valid & (rows | visible)

    @torch.no_grad()
    def labels(self, trajectory: Trajectory, shot_labels: dict | None) -> dict:
        """The auxiliary targets and masks of a micro-batch, from its stored rows and its
        stored shot labels (none without auxiliary heads)."""
        if not self.auxiliary:
            return {}
        return self.codes(trajectory, shot_labels["shots"], shot_labels["mask"])

    def staged(
        self, segments: Segments, view_batches: list[torch.Tensor]
    ) -> Iterator[tuple[Trajectory, dict | None]]:
        """Each batch of views (host indices) on the device: its Trajectory and, with
        auxiliary heads, its shot labels from the store ({"shots", "mask"}, (L, n, ...))."""
        names = segments.names + ((SHOT_LABELS,) if self.auxiliary else ())
        for batch in segments.batches(names, view_batches):
            stored = batch.pop(SHOT_LABELS, None)
            if stored is not None:
                stored = _pytree.tree_map_only(torch.Tensor, lambda x: x.transpose(0, 1), stored)
            yield Trajectory(**batch), stored

    def _codes(
        self, trajectory: Trajectory, shot: shots.ShotLabels, shot_mask: torch.Tensor
    ) -> dict:
        """The targets as the heads' codes, from the stored rows and the re-simulated shots'
        labels (a stage the runtime may compile)."""
        model = self.model
        rows = labels.privileged_rows(trajectory)
        enemy, enemy_mask = labels.enemy_actions(trajectory)
        code = model.shot_code
        pairs = {}
        if getattr(model, "geometry_head", None) is not None:
            partners, mask, values = labels.geometry_pairs(trajectory)
            pairs = dict(geometry_partners=partners, geometry_mask=mask, geometry=values)
        mask = labels.next_state_mask(trajectory, self.model)
        split = {}
        if getattr(model, "correction_code", None) is not None:
            split = labels.split_masks(trajectory, mask, model)
        return dict(
            **pairs,
            **split,
            **labels.next_state_targets(model, trajectory, rows),
            next_state_mask=mask,
            enemy_action=enemy,
            enemy_mask=enemy_mask,
            density=labels.density_targets(self.grid, trajectory, rows),
            shot_snapshots=shot.snapshots,
            shot_ages=shot.ages.long() + 1,  # in moves
            shot_mask=shot_mask,
            shot_outcome=shot_codes.outcome_code(code, shot.hit, shot.distance, shot.side),
        )

    # ------------------------------------------------------------------ the loss

    def unroll(self, trajectory: Trajectory):
        """The model's outputs over a stored segment (L, R, ...), from its initial carry and replay,
        with the resets read from the observation's globals."""
        reset = trajectory.obs.globals[..., GLOBAL["reset"]] > 0.5
        return self.model.unroll(trajectory.initial_carry, trajectory.obs, trajectory.replay, reset)

    def _loss(
        self,
        trajectory: Trajectory,
        label: dict,
        derived: dict,
        denominator: dict,
        scale: torch.Tensor,
    ):
        """One micro-batch: (loss, terms (TERMS,), value loss per level (levels,), statistics
        (STATISTICS,), diagnostics (the policy's sums, `statistics.POLICY_SUMS` and the
        entropy histogram, then the critic targets' entropy per level; empty without
        diagnostics), what the heads' sums read (the unrolled outputs and the shot parts,
        detached; None unless the learner measures the heads)), every sum over its tokens
        divided by its minibatch-total denominator. The heads' sums are their own function
        (`_head_sums`, eager), so every epoch runs this one graph."""
        unrolled = self.unroll(trajectory)
        return self._loss_of(unrolled, trajectory, label, derived, denominator, scale)

    def _loss_of(
        self,
        unrolled,
        trajectory: Trajectory,
        label: dict,
        derived: dict,
        denominator: dict,
        scale: torch.Tensor,
    ):
        """`_loss` given the model's outputs over the micro-batch (`unroll`)."""
        terms, per_level, statistic, unrolled, shot_parts = self._terms_of(
            unrolled, trajectory, label, derived, denominator, scale
        )
        loss = sum(getattr(self.config, term) * terms[term] for term in TERMS)
        heads_read = None
        if self.measures_heads:
            heads_read = _pytree.tree_map_only(
                torch.Tensor, lambda x: x.detach(), (unrolled, shot_parts)
            )
        return (
            loss,
            torch.stack([terms[term] for term in TERMS]).detach(),
            per_level.detach(),
            statistic.detach(),
            self._diagnostics(unrolled, trajectory, derived, denominator),
            heads_read,
        )

    def _head_sums(self, heads_read, trajectory: Trajectory, label: dict) -> dict:
        """The heads' raw totals over one micro-batch (`head_statistics.sums`), from what
        the loss gave (`_loss`)."""
        unrolled, shot_parts = heads_read
        return head_statistics.sums(
            self.model, unrolled, trajectory, label, shot_parts, self.shot_runs
        )

    def _terms(
        self,
        trajectory: Trajectory,
        label: dict,
        derived: dict,
        denominator: dict,
        scale: torch.Tensor,
    ):
        """One micro-batch's terms, before their coefficients: (terms {TERMS: tensor}, the
        value loss per level (levels,), statistics (STATISTICS,), the model's outputs, the
        shot trajectory's cross-entropy per part (L, R, N, ages, parts) or None), with their
        graph."""
        return self._terms_of(
            self.unroll(trajectory), trajectory, label, derived, denominator, scale
        )

    def _terms_of(
        self,
        unrolled,
        trajectory: Trajectory,
        label: dict,
        derived: dict,
        denominator: dict,
        scale: torch.Tensor,
    ):
        """`_terms` given the model's outputs over the micro-batch (`unroll`)."""
        model = self.model
        own = trajectory.log_probability.shape[-1]
        terms = dict.fromkeys(TERMS, unrolled.policy_logits.new_zeros(()))
        terms["policy"], entropies, statistic = self._policy(
            unrolled, trajectory, derived, denominator, scale
        )
        terms.update(zip(ENTROPY_TERMS, entropies.unbind(), strict=True))
        per_level = losses.critic_losses(
            unrolled.value_logits,
            unrolled.outcome_logits,
            derived["level_returns"].index_select(-1, self.local_index),
            derived.get("outcome_targets"),
            trajectory.valued,
            model.bins,
            self.local_levels,
            self.outcome_level,
            denominator["valued"],
            denominator["steps"],
        )
        terms["value"] = per_level.sum()
        parts = None
        if _head(unrolled, "next_state_logits") is not None:
            terms["next_state"] = (
                physics_sum(self._next_state_cross_entropy(unrolled, label), label)
                / denominator["next_state"]
            )
        if _head(unrolled, "correction_logits") is not None:
            target, _, _ = labels.correction_targets(model, trajectory, unrolled.next_state_logits)
            cross_entropy = codes.grouped_cross_entropy(
                unrolled.correction_logits, target, model.correction_code.runs
            ).sum(-1)
            terms["correction"] = (
                losses.masked_sum(cross_entropy, label["correction_mask"])
                / denominator["next_state"]
            )
        if _head(unrolled, "enemy_action_logits") is not None:
            enemy = losses.index_cross_entropy(
                unrolled.enemy_action_logits[..., own:, :], label["enemy_action"]
            )
            terms["enemy_action"] = (
                losses.masked_sum(enemy, label["enemy_mask"]) / denominator["enemy_action"]
            )
        if _head(unrolled, "density_log_rates") is not None:
            per_cell, saturated = poisson_terms(unrolled.density_log_rates, label["density"])
            valued = trajectory.valued.sum(-1).clamp_min(1)
            terms["density"] = ((per_cell - saturated).sum(-1) / valued).sum() / denominator[
                "steps"
            ]
        if _head(unrolled, "shot_latents") is not None:
            parts, outcome = losses.shot_cross_entropy(
                model.shot_trajectory(unrolled.shot_latents, label["shot_ages"]),
                shot_codes.trajectory_coordinates(model.shot_code, label["shot_snapshots"]),
                unrolled.shot_outcome_logits,
                label["shot_outcome"],
                self.shot_runs,
                functools.partial(shot_codes.trajectory_code_at, model.shot_code),
            )
            per_shot = parts.sum(-1).mean(-1)  # the mean over the queried ages
            terms["shot_trajectory"] = (
                losses.masked_sum(per_shot, label["shot_mask"]) / denominator["shots"]
            )
            terms["shot_outcome"] = (
                losses.masked_sum(outcome, label["shot_mask"]) / denominator["shots"]
            )
        if _head(unrolled, "geometry_latents") is not None:
            per_pair = losses.geometry_cross_entropy(
                model.geometry(unrolled.geometry_latents, label["geometry_partners"]),
                label["geometry"],
            )
            terms["geometry"] = (
                losses.masked_sum(per_pair, label["geometry_mask"]) / denominator["pairs"]
            )
        return terms, per_level, statistic, unrolled, parts

    def _next_state_cross_entropy(self, unrolled, label: dict) -> torch.Tensor:
        """(L, R, N) the next-state code's cross-entropy per row, summed over its groups."""
        return codes.grouped_cross_entropy(
            unrolled.next_state_logits, label["next_state"], self.model.next_state_runs
        ).sum(-1)

    @torch.no_grad()
    def _diagnostics(
        self, unrolled, trajectory: Trajectory, derived: dict, denominator: dict
    ) -> torch.Tensor:
        """The micro-batch's diagnostic sums over its denominators (see `_loss`); empty
        without diagnostics."""
        if not self.diagnostics:
            return unrolled.policy_logits.new_zeros(0, dtype=torch.float32)
        own = trajectory.log_probability.shape[-1]
        own_alive = trajectory.alive[..., :own]
        logits = unrolled.policy_logits.float()
        log_probability, entropy = policy.evaluate(logits, _own_joint(trajectory))
        ratio = torch.exp(log_probability - derived["old_log_probability"])
        sums = statistics.policy_sums(logits, ratio, entropy, self.config.clip, own_alive)
        local = statistics.target_entropy(
            critic.two_hot(
                derived["level_returns"].index_select(-1, self.local_index), self.model.bins
            )
        )  # (L, R, S, per-ship levels)
        local = torch.where(trajectory.valued[..., None], local, 0.0).flatten(0, -2).sum(0)
        columns = dict(
            zip(self.local_levels, (local / denominator["valued"]).unbind(), strict=True)
        )
        if self.outcome_level is not None:
            columns[self.outcome_level] = (
                statistics.target_entropy(derived["outcome_targets"]).sum() / denominator["steps"]
            )
        targets = torch.stack([columns[level] for level in range(len(columns))])
        # the fit's moments per level over the valued ships (raw sums): count, target, its
        # square, residual, its square
        target = derived["level_returns"]
        residual = target - self._in_reward_units(unrolled.values.float())
        weight = trajectory.valued[..., None].float().expand_as(target)
        moments = (
            torch.stack(
                [
                    weight,
                    weight * target,
                    weight * target**2,
                    weight * residual,
                    weight * residual**2,
                ]
            )
            .flatten(1, -2)
            .sum(1)
        )  # (5, levels)
        return torch.cat([sums / denominator["policy"], targets, moments.flatten()])

    # what `_policy` reports per minibatch, each a sum over the own living ships over the
    # minibatch's count: the trust region's rate (the ratio past the clip's bounds, or the
    # samples DPPO masks), the approximate KL, the divergence's mean and share past
    # dppo_delta, its histogram, and per entropy bucket the tokens and the rate's.
    STATISTICS = (
        "clip_fraction",
        "approximate_kl",
        "divergence/mean",
        "divergence/above_delta",
        *(f"divergence/bin_{index}" for index in range(DIVERGENCE_BINS)),
        *(f"entropy_bucket/{bucket}/tokens" for bucket in ENTROPY_BUCKETS),
        *(f"entropy_bucket/{bucket}/limited" for bucket in ENTROPY_BUCKETS),
    )

    def _divergence(
        self, unrolled, trajectory: Trajectory, log_probability: torch.Tensor, old: torch.Tensor
    ) -> torch.Tensor:
        """The trust region's divergence per token between the rollout's policy and the
        current one: the joint's exact total variation ("dppo_tv"), else the binary form
        (DPPO's, and the clip modes' reading of the step's size)."""
        if self.config.trust_region == "dppo_tv":
            if trajectory.probabilities is None:
                raise ValueError("trust_region 'dppo_tv' needs the rollout's probabilities")
            return losses.total_variation(trajectory.probabilities, unrolled.policy_logits)
        return losses.binary_total_variation(log_probability, old)

    def _surrogate(
        self,
        log_probability: torch.Tensor,
        old: torch.Tensor,
        ratio: torch.Tensor,
        advantage: torch.Tensor,
        scale: torch.Tensor | float,
        divergence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The policy's surrogate per token under the trust region, and where the region
        limits the step (the ratio past the clip's bounds, or the samples DPPO masks)."""
        config = self.config
        region = config.trust_region
        if region in ("dppo_tv", "dppo_binary_tv"):
            keep = losses.dppo_keep(divergence, ratio, advantage, config.dppo_delta)
            surrogate = losses.dppo_surrogate(log_probability, old, advantage, scale, keep)
            return surrogate, ~keep
        high = config.clip if region == "clip" else config.clip_high
        surrogate = losses.surrogate(
            log_probability,
            old,
            advantage,
            scale,
            config.clip,
            None if region == "clip" else config.clip_high,
        )
        detached = ratio.detach()
        if region == "clip":
            limited = (detached - 1.0).abs() > config.clip
        else:
            limited = (detached < 1.0 - config.clip) | (detached > 1.0 + high)
        return surrogate, limited

    def _level_policy_terms(
        self,
        unrolled,
        trajectory: Trajectory,
        old: torch.Tensor,
        level_advantage: torch.Tensor,
        denominator: dict,
    ) -> list[torch.Tensor]:
        """The weighted policy term with each level's advantage alone, one per level:
        level_advantage (L, R, S, levels) already over the return scale; old (L, R, S) the
        old log-probabilities."""
        own_alive = trajectory.alive[..., : trajectory.log_probability.shape[-1]]
        log_probability, _ = policy.evaluate(unrolled.policy_logits, _own_joint(trajectory))
        ratio = torch.exp(log_probability - old)
        divergence = self._divergence(unrolled, trajectory, log_probability, old)
        terms = []
        for advantage in level_advantage.unbind(-1):
            surrogate, _ = self._surrogate(log_probability, old, ratio, advantage, 1.0, divergence)
            terms.append(
                self.config.policy * losses.masked_sum(surrogate, own_alive) / denominator["policy"]
            )
        return terms

    def _policy(
        self,
        unrolled,
        trajectory: Trajectory,
        derived: dict,
        denominator: dict,
        scale: torch.Tensor,
    ):
        """The policy's terms of a micro-batch: (the surrogate under the trust region,
        the mean entropy's factor parts (3,) (`ENTROPY_TERMS`), statistics (STATISTICS,)),
        each over own ships alive."""
        config = self.config
        own_alive = trajectory.alive[..., : trajectory.log_probability.shape[-1]]
        log_probability, _ = policy.evaluate(unrolled.policy_logits, _own_joint(trajectory))
        entropies = policy.factor_entropies(unrolled.policy_logits)
        old, advantage = derived["old_log_probability"], derived["advantage"]
        ratio = torch.exp(log_probability - old)
        divergence = self._divergence(unrolled, trajectory, log_probability, old)
        surrogate, limited = self._surrogate(
            log_probability, old, ratio, advantage, scale, divergence
        )
        clipped = losses.masked_sum(limited.float(), own_alive)
        approximate = losses.masked_sum((ratio - 1.0) - (log_probability - old), own_alive)
        weight = own_alive.float()
        position = (
            (
                (torch.log10(divergence.clamp_min(DIVERGENCE_FLOOR)) - math.log10(DIVERGENCE_FLOOR))
                / -math.log10(DIVERGENCE_FLOOR)
                * DIVERGENCE_BINS
            )
            .floor()
            .clamp(0, DIVERGENCE_BINS - 1)
        )
        histogram = (
            (torch.nn.functional.one_hot(position.long(), DIVERGENCE_BINS) * weight[..., None])
            .flatten(0, -2)
            .sum(0)
        )
        joint_entropy = entropies.detach().float().sum(-1) / statistics.MAXIMUM_ENTROPY
        bucket = (joint_entropy * len(ENTROPY_BUCKETS)).floor().clamp(0, len(ENTROPY_BUCKETS) - 1)
        buckets = (
            torch.nn.functional.one_hot(bucket.long(), len(ENTROPY_BUCKETS)) * weight[..., None]
        )
        tokens = buckets.flatten(0, -2).sum(0)
        limited_by_bucket = (buckets * limited[..., None]).flatten(0, -2).sum(0)
        statistic = torch.cat(
            [
                torch.stack(
                    [
                        clipped,
                        approximate,
                        losses.masked_sum(divergence, own_alive),
                        losses.masked_sum((divergence > config.dppo_delta).float(), own_alive),
                    ]
                ),
                histogram,
                tokens,
                limited_by_bucket,
            ]
        )
        return (
            losses.masked_sum(surrogate, own_alive) / denominator["policy"],
            factor_sums(entropies, own_alive) / denominator["policy"],
            statistic.detach() / denominator["policy"],
        )

    # ------------------------------------------------------------------ update

    @property
    def micro_views(self) -> int:
        """Views per micro-batch: micro_batch_rows ship rows."""
        return max(1, self.config.micro_batch_rows // self.rules.num_ships)

    def _micro_batches(
        self, views: torch.Tensor, family: tuple[int, ...] = ()
    ) -> list[torch.Tensor]:
        """The views in micro-batches of at most micro_views, of as few sizes as the views
        and `family` (the other view counts split alongside these, such as the update's
        other minibatch size) allow: under a compile of fixed shapes every size is one
        compile of every learner stage (`micro_batch_sizes`)."""
        sizes = micro_batch_sizes(len(views), self.micro_views, family)
        return list(views.split(sizes))

    def minibatch_sizes(self, views: int) -> tuple[int, ...]:
        """The view counts of an update's minibatches (`update`'s split of `views`): one,
        or two that differ by one when the minibatches do not divide the views."""
        minibatches = self.config.minibatches
        return tuple(sorted({views // minibatches, -(-views // minibatches)} - {0}))

    def first_micro_batch(self, views: int) -> int:
        """The views of an update's first micro-batch over `views` views (the first
        minibatch's first, `update`'s split): what the drift unrolls."""
        sizes = self.minibatch_sizes(views)
        return micro_batch_sizes(max(sizes), self.micro_views, sizes)[0]

    def micro_batch_shapes(self, views: int, cadence: bool = False) -> tuple[int, ...]:
        """The micro-batch sizes an update over `views` views runs its loss on (with
        `cadence`, also those of the cadence's gradient decomposition, the first minibatch
        in halves): under a compile of fixed shapes each is one compile of every compiled
        learner stage. One unless the minibatches do not divide the views."""
        groups = [self.minibatch_sizes(views)]
        if cadence:
            first = max(groups[0])
            groups.append(tuple({first - first // 2, first // 2} - {0}))
        return tuple(
            sorted(
                {
                    size
                    for group in groups
                    for count in group
                    for size in micro_batch_sizes(count, self.micro_views, group)
                }
            )
        )

    def _on_device(self, views: torch.Tensor) -> torch.Tensor:
        """Host view indices on the device, from pinned memory: no wait on the queued work
        (a copy from pageable memory, or `.cpu()`, would wait for it and stall the
        staging's overlap)."""
        if self.device.type != "cuda":
            return views
        return views.pin_memory().to(self.device, non_blocking=True)

    def prepare(self, segments: Segments, bootstrap) -> dict:
        """Steps 1 and 2: per view, the advantage, the returns, the outcome targets and the
        old log-probabilities (L, V, ...), each view's token counts (V, COUNTS), the return
        scale and the explained variance per level (levels,); the shot labels go into the
        store. bootstrap: the learner's Outputs after the last tick the store holds (the
        lookahead's, or the last shard's without one)."""
        config, device = self.config, self.device
        shards, length = segments.shards, segments.length
        lookahead = self.lookahead_names
        small = [segments.shard(self.small_names, shard) for shard in range(shards)]
        chain = {name: torch.cat([part[name] for part in small], 0) for name in lookahead}
        if all(segments.lookahead_ticks.get(name, 0) > 0 for name in lookahead):
            tail = segments.lookahead(lookahead)
            chain = {name: torch.cat([chain[name], tail[name]], 0) for name in lookahead}
        chained = self.chained_returns(
            chain, torch.cat([part["alive"] for part in small], 0), bootstrap
        )
        raw_sums = chained.pop(RAW_SUMS, None)

        def by_view(x: torch.Tensor) -> torch.Tensor:
            """(shards L, R, ...) as the store's views, (L, shards R, ...)."""
            return x.unflatten(0, (shards, length)).transpose(0, 1).flatten(1, 2)

        parts = [{name: by_view(value) for name, value in chained.items()}]

        def joined(name: str, source: list[dict]) -> torch.Tensor:
            return torch.cat([part[name] for part in source], 1)

        derived = {
            name: joined(name, parts)
            for name in ("advantage", "level_returns", "outcome_targets")
            if name in chained
        }
        rollout_log_probability = joined("log_probability", small)
        derived["old_log_probability"] = rollout_log_probability.clone()
        own_alive = joined("alive", small)[..., : rollout_log_probability.shape[-1]]
        scale = self.return_scale.update(joined("total", parts), own_alive)
        counts = torch.cat([self._counts(part) for part in small], 0)
        recompute = config.old_log_probabilities == "recomputed"
        belief_total: dict = {}
        if self.auxiliary or recompute:
            # One spread and one draw of uniform ages per update: each epoch reads the same
            # stored labels.
            spread = torch.randn(
                (segments.length, segments.views, self.rules.num_ships, 2),
                device=device,
                generator=self.spread_generator,
            )
            drawn = torch.randint(
                self.rules.longest_bullet_moves,
                (segments.length, segments.views, self.rules.num_ships, 3),
                dtype=torch.uint8,
                device=device,
                generator=self.spread_generator,
            )
            order = self._micro_batches(torch.arange(segments.views))  # host indices
            for host_views, batch in zip(
                order, segments.batches(segments.names, order), strict=True
            ):
                views = self._on_device(host_views)
                trajectory = Trajectory(**batch)
                with torch.no_grad():
                    if self.auxiliary:
                        following, followed = segments.following(
                            FOLLOWING, host_views, self.rules.longest_bullet_moves
                        )
                        shot, shot_mask = self.shots(
                            trajectory, spread[:, views], drawn[:, views], following, followed
                        )
                        counts[views, COUNTS.index("next_state")] = (
                            labels.next_state_mask(trajectory, self.model).sum((0, 2)).float()
                        )
                        counts[views, COUNTS.index("shots")] = shot_mask.sum((0, 2)).float()
                        if getattr(self.model, "geometry_head", None) is not None:
                            _, pair_mask, _ = labels.geometry_pairs(trajectory)
                            counts[views, COUNTS.index("pairs")] = pair_mask.sum((0, 2, 3)).float()
                        stored = _pytree.tree_map_only(
                            torch.Tensor,
                            lambda x: x.transpose(0, 1).contiguous(),
                            dict(shots=shot, mask=shot_mask),
                        )  # (n, L, ...)
                        first = int(host_views[0])
                        segments.write_views(
                            {SHOT_LABELS: stored}, slice(first, first + host_views.numel())
                        )
                        if self.diagnostics:
                            belief_total = head_statistics.add(
                                belief_total,
                                {
                                    **head_statistics.belief_sums(
                                        trajectory,
                                        self.rules.tick_length,
                                        getattr(self.model, "state_code", None),
                                    ),
                                    **head_statistics.input_sums(self.model.state_code, trajectory),
                                    **head_statistics.acceleration_sums(self.model, trajectory),
                                },
                            )
                    if recompute:
                        log_probability, _ = policy.evaluate(
                            self.unroll(trajectory).policy_logits, _own_joint(trajectory)
                        )
                        derived["old_log_probability"][:, views] = log_probability
        level_returns, values = joined("level_returns", parts), joined("values", parts)
        valued = joined("valued", small)
        variance = explained_variance(level_returns, values, valued)
        series = (
            self._target_series(
                level_returns, values, valued, derived["advantage"], own_alive, scale
            )
            if self.diagnostics
            else {}
        )
        cadence_advantage = None
        if "level_advantage" in chained:
            level_advantage = joined("level_advantage", parts) / scale
            series.update(
                self._advantage_series(
                    level_advantage, derived["advantage"] / scale, own_alive, raw_sums
                )
            )
            # kept for the cadence's minibatch (`cadence`): the first views
            cadence_views = len(torch.arange(segments.views).tensor_split(config.minibatches)[0])
            cadence_advantage = level_advantage[:, :cadence_views].clone()
        if "monte_carlo_returns" in chained:  # the λ = 1 returns, bootstrapped at the end
            monte_carlo = explained_variance(joined("monte_carlo_returns", parts), values, valued)
            for index, name in enumerate(self.levels):
                series[f"critic/explained_variance_monte_carlo/{name}"] = monte_carlo[index]
        series.update(head_statistics.belief_series(belief_total))
        series.update(head_statistics.input_series(belief_total))
        series.update(head_statistics.acceleration_series(belief_total))
        return dict(
            derived=derived,
            rollout_log_probability=rollout_log_probability,
            counts=counts,
            scale=scale,
            explained_variance=variance,
            series=series,
            level_advantage=cadence_advantage,
        )

    def _view_member(self, names: tuple[str | None, ...], views: int) -> torch.Tensor:
        """(views, slices) each view's slices for the advantage series: the batch (the
        first, None), then each mode of `names` (`view_mode`, on the host). Made on the host
        once, so the update reads it without a synchronisation."""
        key = (names, views)
        if key not in self.view_members:
            member = torch.ones((views, len(names)))
            if len(names) > 1:
                modes = self.view_mode.cpu().long()
                member[:, 1:] = torch.nn.functional.one_hot(modes, len(names) - 1).float()
            self.view_members[key] = member.to(self.device)
        return self.view_members[key]

    def _advantage_series(
        self,
        level_advantage: torch.Tensor,
        advantage: torch.Tensor,
        own_alive: torch.Tensor,
        raw_sums: torch.Tensor | None,
    ) -> dict:
        """The levels' advantage series (`bnb.learn.advantages`), over the batch and per mode
        of its games: level_advantage (L, V, S, levels) and advantage (L, V, S) over the
        return scale, own_alive (L, V, S) the policy's tokens; raw_sums the levels' raw sums
        (`advantages.raw_sums`)."""
        views = advantage.shape[1]
        names: tuple[str | None, ...] = (None,)
        mode = self.view_mode
        if mode is not None and mode.shape[0] == views and self.view_modes:
            names += self.view_modes
        member = self._view_member(names, views)
        per_level, whole = level_advantages.level_sums(
            level_advantage, advantage, own_alive, member
        )
        out = level_advantages.level_series(per_level, whole, self.levels, names)
        if raw_sums is not None and self.reward_weights is not None:
            weights = dict(zip(REWARD_LEVELS, self.reward_weights, strict=True))
            if all(level in weights for level in self.levels):  # not the single value level
                out.update(
                    level_advantages.raw_series(
                        raw_sums, tuple(weights[level] for level in self.levels), self.levels
                    )
                )
        return out

    def _target_series(
        self,
        level_returns: torch.Tensor,
        values: torch.Tensor,
        valued: torch.Tensor,
        advantage: torch.Tensor,
        own_alive: torch.Tensor,
        scale: torch.Tensor,
    ) -> dict:
        """The targets' series, measured before any step (diagnostics audit, group 5): per
        level over the valued ships, the return's mean and percentiles (5, 50, 95), the
        rollout critic's bias (prediction less target) and the share of returns outside the
        categorical critic's support; the normalised advantage's mean and RMS over the
        living ships; the return scale's percentile EMAs and whether the floor holds it.
        level_returns, values (L, V, S, levels) in reward units; valued, own_alive and
        advantage (L, V, S)."""
        out = {}
        mask = valued[..., None].expand_as(level_returns)
        count = mask.float().sum((0, 1, 2))  # (levels,)
        bins = self.model.bins
        outside = ((level_returns < bins[0]) | (level_returns > bins[-1])).float()
        flat = torch.where(mask, level_returns, torch.nan).flatten(0, 2)  # (tokens, levels)
        quantiles = torch.nanquantile(flat, self.return_quantiles, dim=0)  # (3, levels)
        for index, level in enumerate(self.levels):

            def total(x: torch.Tensor) -> torch.Tensor:
                return torch.where(mask[..., index], x[..., index], 0.0).sum()

            out[f"critic/return_mean/{level}"] = Mean(total(level_returns), count[index])
            out[f"critic/value_bias/{level}"] = Mean(total(values - level_returns), count[index])
            if index != self.outcome_level:  # the outcome's head is not over the bins
                out[f"critic/support_clip/{level}"] = Mean(total(outside), count[index])
            for row, name in enumerate(("p5", "p50", "p95")):
                out[f"returns/{name}/{level}"] = quantiles[row, index]
        normalised = advantage / scale
        living = own_alive.float()
        out["returns/advantage_mean"] = Mean((normalised * living).sum(), living.sum())
        out["returns/advantage_rms"] = torch.sqrt(
            (normalised.square() * living).sum() / living.sum().clamp_min(1.0)
        )
        out["returns/scale_p5"] = self.return_scale.low.clone()
        out["returns/scale_p95"] = self.return_scale.high.clone()
        out["returns/scale_floored"] = (
            self.return_scale.high - self.return_scale.low < self.return_scale.floor
        )
        return out

    def gradient(
        self, segments: Segments, prepared: dict, chosen: torch.Tensor, head_views: int = 0
    ) -> tuple:
        """Accumulate one minibatch's gradient (views `chosen`, host indices) into the
        parameters' .grad, micro-batch by micro-batch: (terms (TERMS,), value loss per
        level, statistics (STATISTICS,), diagnostics, the heads' sums (see `_loss`) over
        its first micro-batches until they hold `head_views` views (0: none)), each the
        minibatch's."""
        device, derived = self.device, prepared["derived"]
        total = prepared["counts"][self._on_device(chosen)].sum(0).clamp_min(1.0)
        denominator = dict(zip(COUNTS, total.unbind()))
        sizes = self.minibatch_sizes(len(prepared["counts"]))
        micro = self._micro_batches(chosen, sizes if len(chosen) in sizes else ())
        terms_sum = torch.zeros(len(TERMS), device=device)
        levels_sum = torch.zeros(len(self.levels), device=device)
        statistic_sum = torch.zeros(len(self.STATISTICS), device=device)
        diagnostic_sum = None
        head_sum: dict = {}
        self.optimizer.zero_grad(set_to_none=True)
        for host_views, (trajectory, shot_labels) in zip(
            micro, self.staged(segments, micro), strict=True
        ):
            views = self._on_device(host_views)
            label = self.labels(trajectory, shot_labels)
            part = {name: value.index_select(1, views) for name, value in derived.items()}
            loss, terms, per_level, statistic, diagnostic, heads_read = self.loss(
                trajectory, label, part, denominator, prepared["scale"]
            )
            loss.backward()
            head_part = {}
            if head_views > 0 and heads_read is not None:
                with torch.no_grad():
                    head_part = self.head_sums(heads_read, trajectory, label)
                head_views -= len(host_views)
            terms_sum += terms
            levels_sum += per_level
            statistic_sum += statistic
            diagnostic_sum = diagnostic if diagnostic_sum is None else diagnostic_sum + diagnostic
            head_sum = head_statistics.add(head_sum, head_part)
        return terms_sum, levels_sum, statistic_sum, diagnostic_sum, head_sum

    def _step(self, finite: torch.Tensor) -> None:
        """The optimizer's step, kept only where `finite` (a device flag) holds: a step with
        a non-finite gradient leaves the parameters and the optimizer's state as they were
        (learner spec 4d: skipped and counted), with no host synchronisation. A state the
        step creates (the first step) is zero before it, as Adam starts it.

        Fused (on a GPU), the step reads the flag itself (`found_inf`, as gradient scaling
        passes it): a flagged step leaves the parameters and moments untouched and takes its
        step count back, in the one kernel, the same result as the copy-back below without
        its clone and select per tensor (about 1,400 kernels a step at Yemong's size).
        Unfused (the CPU), every tensor is saved and put back where the flag fails."""
        if all(group.get("fused") for group in self.optimizer.param_groups):
            self.optimizer.found_inf = (~finite).to(torch.float32).reshape(())
            try:
                self.optimizer.step()
            finally:
                del self.optimizer.found_inf
            return
        tensors = [p for group in self.optimizer.param_groups for p in group["params"]]
        tensors += [
            value
            for state in self.optimizer.state.values()
            for value in state.values()
            if isinstance(value, torch.Tensor)
        ]
        saved = [tensor.detach().clone() for tensor in tensors]
        known = set(map(id, tensors))
        self.optimizer.step()
        created = [
            value
            for state in self.optimizer.state.values()
            for value in state.values()
            if isinstance(value, torch.Tensor) and id(value) not in known
        ]
        with torch.no_grad():
            for tensor, before in zip(tensors, saved, strict=True):
                tensor.copy_(torch.where(finite.to(tensor.device), tensor, before))
            for tensor in created:
                tensor.copy_(torch.where(finite.to(tensor.device), tensor, 0))

    def update(
        self,
        segments: Segments,
        bootstrap,
        environment_steps: int = 0,
        cadence: bool = False,
        heads: bool = True,
    ) -> dict:
        """One update from the store's shards and the lookahead after them, and the bootstrap
        (the learner's Outputs on the observation after the last tick). environment_steps:
        environment steps this update collected (the warmup counts them). cadence: also
        measure the cadence groups (`cadence`), after the steps. heads: measure the heads' sums
        (a sample: the first epoch's first micro-batches until they hold `HEAD_SUM_VIEWS`
        views, when the learner measures them; the trainer passes its cadence). Returns the
        update's series on the device (`bnb.contracts.series`), for the caller to read back
        once."""
        config, device = self.config, self.device
        views = segments.views
        if views < config.minibatches:
            raise ValueError(f"{views} views do not make {config.minibatches} minibatches")
        prepared = self.prepare(segments, bootstrap)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate(config, self.environment_steps)
        self.environment_steps += environment_steps
        before = (
            {name: p.detach().clone() for name, p in self.model.named_parameters()}
            if self.diagnostics
            else None
        )
        terms_total = torch.zeros(len(TERMS), device=device)
        levels_total = torch.zeros(len(self.levels), device=device)
        statistics_total = torch.zeros(len(self.STATISTICS), device=device)
        diagnostics_total = None
        last_moments = None  # the last completed epoch's fit moments
        # the heads' sums: a sample, the first micro-batches of the first epoch until they
        # hold HEAD_SUM_VIEWS views (eager; every micro-batch cost ~8% of RL's steady time)
        head_total: dict = {}
        head_views = HEAD_SUM_VIEWS if self.measures_heads and heads else 0
        drift = self.drift(segments, prepared)
        skipped = torch.zeros((), dtype=torch.int64, device=device)
        nonfinite_terms = torch.zeros(len(TERMS), dtype=torch.int64, device=device)
        norm_total = torch.zeros((), device=device)  # the finite steps' pre-clip norms
        clipped = torch.zeros((), dtype=torch.int64, device=device)
        steps, stopped, epochs = 0, False, 0
        first_kl = first_epoch = epoch_statistics = None
        for epoch in range(config.epochs):
            # Minibatches as equal as the views allow (micro-batches of at most micro_views
            # views, `_micro_batches`).
            order = torch.randperm(views, generator=self.order_generator)
            epoch_statistics = torch.zeros(len(self.STATISTICS), device=device)
            epoch_moments = None
            for chosen in order.tensor_split(config.minibatches):
                terms, per_level, statistic, diagnostic, head_part = self.gradient(
                    segments, prepared, chosen, head_views if epoch == 0 else 0
                )
                if epoch == 0:
                    head_views = max(0, head_views - len(chosen))
                head_total = head_statistics.add(head_total, head_part)
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.max_grad_norm)
                finite = torch.isfinite(norm)
                self._step(finite)
                skipped += ~finite
                nonfinite_terms += ~torch.isfinite(terms)
                norm_total += torch.where(finite, norm, 0.0)
                clipped += finite & (norm > config.max_grad_norm)
                if first_kl is None and "approximate_kl" in self.STATISTICS:
                    first_kl = statistic[self.STATISTICS.index("approximate_kl")]
                terms_total += terms
                levels_total += per_level
                epoch_statistics += statistic
                statistics_total += statistic
                if self.diagnostics:
                    diagnostic, moments = diagnostic[: -self.moments], diagnostic[-self.moments :]
                    epoch_moments = moments if epoch_moments is None else epoch_moments + moments
                diagnostics_total = (
                    diagnostic if diagnostics_total is None else diagnostics_total + diagnostic
                )
                steps += 1
            epochs += 1
            last_moments = epoch_moments
            epoch_statistics = epoch_statistics / config.minibatches  # the epoch's means
            if epoch == 0:
                first_epoch = epoch_statistics
            # The early stop reads the epoch's mean KL: one host sync per epoch.
            if (
                config.target_kl is not None
                and epoch + 1 < config.epochs
                and epoch_statistics[self.STATISTICS.index("approximate_kl")].item()
                > config.target_kl
            ):
                stopped = True
                break
        out = {
            HEAD_SERIES[term]: Mean(terms_total[index], steps)
            for index, term in enumerate(TERMS)
            if term in HEAD_SERIES
        }
        for index, level in enumerate(self.levels):
            out[f"critic/value_loss/{level}"] = Mean(levels_total[index], steps)
            out[f"critic/explained_variance/{level}"] = prepared["explained_variance"][index]
        out.update(self.report(first_epoch, epoch_statistics, statistics_total / steps))
        out.update(
            {
                "policy/ratio_drift/max": drift[0],
                "policy/ratio_drift/mean": drift[1],
                "policy/minibatch_steps": steps,
                "policy/epochs_completed": epochs,
                "policy/stopped_early": stopped,
                "returns/scale": prepared["scale"],
                "train/skipped_steps": skipped,
                "train/gradient_norm": Mean(norm_total, steps - skipped),
                "train/clip_fire_rate": Mean(clipped, steps - skipped),
                "schedule/learning_rate": self.optimizer.param_groups[0]["lr"],
            }
        )
        if first_kl is not None:  # the behaviour policy against the learner's, before a step
            out["policy/kl_first"] = first_kl
        for index, term in enumerate(TERMS):
            out[f"train/nonfinite_steps/{term}"] = nonfinite_terms[index]
        if self.diagnostics:
            out.update(self._diagnostic_series(prepared, diagnostics_total, steps, levels_total))
            out.update(fit_series(last_moments, self.levels))
            out.update(self._parameter_series(before))
            out.update(head_statistics.series(head_total, self.model))
            if cadence:
                out.update(self.cadence(segments, prepared))
        return out

    def report(
        self, first_epoch: torch.Tensor, last_epoch: torch.Tensor, mean: torch.Tensor
    ) -> dict:
        """The policy's series from its STATISTICS: the first and the last completed
        epoch's means and the mean over every minibatch step. `policy/kl` is the last
        epoch's (what the early stop reads), the clip fraction every step's."""
        index = self.STATISTICS.index
        out = {
            "policy/kl": last_epoch[index("approximate_kl")],
            "policy/clip_fraction": mean[index("clip_fraction")],
            "policy/divergence/mean": mean[index("divergence/mean")],
            "policy/divergence/above_delta": mean[index("divergence/above_delta")],
        }
        first = index("divergence/bin_0")
        histogram = mean[first : first + DIVERGENCE_BINS]
        for quantile, value in zip(
            DIVERGENCE_QUANTILES,
            divergence_quantiles(histogram, self.divergence_quantiles).unbind(),
            strict=True,
        ):
            out[f"policy/divergence/p{round(quantile * 100)}"] = value
        for bucket in ENTROPY_BUCKETS:
            tokens = mean[index(f"entropy_bucket/{bucket}/tokens")]
            limited = mean[index(f"entropy_bucket/{bucket}/limited")]
            out[f"policy/clip_fraction/entropy_{bucket}"] = Mean(limited, tokens)
        return out

    def _diagnostic_series(
        self, prepared: dict, total: torch.Tensor, steps: int, levels_total: torch.Tensor
    ) -> dict:
        """The policy's diagnostic means over the update's steps (`statistics.POLICY_SUMS`,
        the entropy quantiles from its histogram, the factor coupling), the value KL per
        level, and the series `prepare` measured before any step."""
        levels = len(self.levels)
        policy_part, targets = total[:-levels], total[-levels:]
        sums, histogram = (
            policy_part[: len(statistics.POLICY_SUMS)],
            policy_part[len(statistics.POLICY_SUMS) :],
        )
        means = dict(zip(statistics.POLICY_SUMS, (sums / steps).unbind(), strict=True))
        out = {f"policy/{name}": value for name, value in means.items()}
        out["policy/factor_coupling"] = (
            means["entropy/power"]
            + means["entropy/stick"]
            + means["entropy/shoot"]
            - means["entropy/joint"]
        )
        for quantile, value in zip(
            statistics.ENTROPY_QUANTILES,
            statistics.histogram_quantiles(histogram, self.entropy_quantiles).unbind(),
            strict=True,
        ):
            out[f"policy/entropy_quantile/p{round(quantile * 100)}"] = value
        for index, level in enumerate(self.levels):
            out[f"critic/value_kl/{level}"] = Mean(levels_total[index] - targets[index], steps)
        out.update(prepared["series"])
        return out

    @torch.no_grad()
    def _parameter_series(self, before: dict[str, torch.Tensor]) -> dict:
        """Per module (`statistics.parameter_groups`): the parameters' norm after the update
        and the update's size relative to the norm before it."""
        after = dict(self.model.named_parameters())
        change = {name: after[name].detach() - before[name] for name in before}
        norms = statistics.group_norms(after, self.parameter_groups)
        previous = statistics.group_norms(before, self.parameter_groups)
        moved = statistics.group_norms(change, self.parameter_groups)
        out = {}
        for group in self.parameter_groups:
            out[f"train/parameter_norm/{group}"] = norms[group]
            out[f"train/update_ratio/{group}"] = Mean(moved[group], previous[group])
        return out

    # ------------------------------------------------------------------ cadence

    def cadence(self, segments: Segments, prepared: dict) -> dict:
        """The cadence groups (`bnb.learn.gradients`): the gradient decomposition over the
        first minibatch's views, split into two halves, and the representation's series on
        its first micro-batch, after the update's steps. Leaves .grad empty; timed by
        `cadence_stopwatch`."""
        self.cadence_stopwatch.start()
        views = torch.arange(segments.views).tensor_split(self.config.minibatches)[0]
        out = self._gradient_decomposition(segments, prepared, views)
        with torch.no_grad(), gradients.block_taps(self.model) as taps:
            for batch in segments.batches(segments.names, [views[: self.micro_views]]):
                self.unroll(Trajectory(**batch))
            out.update(gradients.representation_series(taps))
        self.cadence_stopwatch.stop()
        return out

    def _basic_terms(
        self, terms: dict, per_level: torch.Tensor, unrolled, trajectory, label, denominator
    ):
        """{name: the weighted term} of the gradients the decomposition takes one by one: the
        top-level terms, with the value split by level, the next state by population and the
        geometry by target (`geometry.TARGETS`, each its share of the mean over targets, so
        the parts sum to the term)."""
        config, own = self.config, trajectory.log_probability.shape[-1]
        basic = {}
        for term in TERMS:
            if term == "value":
                for level, loss in zip(self.levels, per_level.unbind(), strict=True):
                    basic[f"value_{level}"] = config.value * loss
            elif term == "next_state" and terms[term].requires_grad:
                cross_entropy = self._next_state_cross_entropy(unrolled, label)
                visible = trajectory.obs.ships[..., SHIP["visible"]] > 0.5
                rows = torch.arange(visible.shape[-1], device=visible.device) < own
                members = dict(
                    ally=rows.expand(visible.shape),
                    enemy_visible=~rows & visible,
                    enemy_hidden=~rows & ~visible,
                )
                for name, mask in members.items():
                    part = physics_sum(cross_entropy, label, mask)
                    # the term's normalisation, so the populations sum to the term
                    basic[f"next_state_{name}"] = (
                        config.next_state * part / denominator["next_state"]
                    )
            elif term in ("geometry", "correction") and not terms[term].requires_grad:
                continue  # a model without the geometry head or the error head
            elif term == "geometry":
                logits = self.model.geometry(
                    unrolled.geometry_latents, label["geometry_partners"]
                ).float()
                per_target = codes.grouped_cross_entropy(
                    logits, label["geometry"].float(), geometry.RUNS, geometry.encode
                )  # (L, R, S, K, targets)
                for name, part in zip(geometry.TARGETS, per_target.unbind(-1), strict=True):
                    # the term's normalisation: the mean over targets, over the pairs
                    basic[f"geometry_{name}"] = (
                        config.geometry
                        * losses.masked_sum(part, label["geometry_mask"])
                        / (len(geometry.TARGETS) * denominator["pairs"])
                    )
            else:
                basic[term] = getattr(config, term) * terms[term]
        return basic

    def _gradient_decomposition(
        self, segments: Segments, prepared: dict, views: torch.Tensor
    ) -> dict:
        """`gradients.decomposition` over the minibatch `views` (host indices)."""
        derived, device = prepared["derived"], self.device
        parameters, trunk_size = gradients.trunk_first(self.model)
        size = sum(p.numel() for p in parameters)
        total = prepared["counts"][self._on_device(views)].sum(0).clamp_min(1.0)
        denominator = dict(zip(COUNTS, total.unbind()))
        halves = [half for half in views.tensor_split(2) if half.numel()]
        family = tuple(len(half) for half in halves)
        names = None
        accumulated = None
        level_advantage = prepared.get("level_advantage") if self.measures_advantages else None
        level_gradients = (
            None
            if level_advantage is None
            else torch.zeros((len(self.levels), size), device=device)
        )
        for half_index, half in enumerate(halves):
            micro = self._micro_batches(half, family)
            for host_views, (trajectory, shot_labels) in zip(
                micro, self.staged(segments, micro), strict=True
            ):
                device_views = self._on_device(host_views)
                label = self.labels(trajectory, shot_labels)
                part = {
                    name: value.index_select(1, device_views) for name, value in derived.items()
                }
                terms, per_level, _, unrolled, _ = self.cadence_terms(
                    trajectory, label, part, denominator, prepared["scale"]
                )
                basic = self._basic_terms(
                    terms, per_level, unrolled, trajectory, label, denominator
                )
                if accumulated is None:
                    names = tuple(basic)
                    accumulated = torch.zeros((2, len(names), size), device=device)
                for index, name in enumerate(names):
                    accumulated[half_index, index] += gradients.flat_gradient(
                        basic[name], parameters, size
                    )
                if level_gradients is not None:  # the policy term with one level's A alone
                    level_terms = self._level_policy_terms(
                        unrolled,
                        trajectory,
                        part["old_log_probability"],
                        level_advantage.index_select(1, device_views),
                        denominator,
                    )
                    for index, term in enumerate(level_terms):
                        level_gradients[index] += gradients.flat_gradient(term, parameters, size)
        self.optimizer.zero_grad(set_to_none=True)
        populations = tuple(name for name in names if name.startswith("next_state_"))
        levels = tuple(name for name in names if name.startswith("value_"))
        targets = tuple(name for name in names if name.startswith("geometry_"))
        derived_terms = dict(value=levels, next_state=populations, geometry=targets)
        top_level = tuple(term for term in TERMS if term in names or derived_terms.get(term, ()))
        everything = top_level + levels + populations + targets
        combination = torch.zeros((len(everything), len(names)), device=device)
        for row, name in enumerate(everything):
            for column, basic_name in enumerate(names):
                if basic_name == name or basic_name in derived_terms.get(name, ()):
                    combination[row, column] = 1.0
        groups = {"terms": top_level, "value": levels}
        if populations:
            groups["next_state"] = populations
        if targets:
            groups["geometry"] = targets
        grams = gradients.grams(accumulated.flatten(0, 1), trunk_size)
        if len(halves) < 2:  # one view: no halves to compare
            grams = {"model": grams["model"]}
        out = gradients.decomposition(
            grams,
            combination,
            everything,
            top_level,
            groups,
            populations,
            actor=("policy", *ENTROPY_TERMS),
            critic=("value",),
        )
        if level_gradients is not None:
            whole = accumulated[:, names.index("policy")].sum(0)
            out.update(level_advantages.policy_level_series(level_gradients, whole, self.levels))
        return out

    @torch.no_grad()
    def drift(self, segments: Segments, prepared: dict) -> torch.Tensor:
        """(2,): the step/unroll ratio drift before any optimizer step, on the first
        micro-batch of views: |exp(log_probability_unroll - log_probability_rollout) - 1|
        over living own ships, max and mean (0 without any). The micro-batch's views are
        `first_micro_batch`'s: never more than a loss micro-batch holds (an eager unroll of
        `micro_views` views, when the minibatches are smaller, cost memory and time the
        update never otherwise needs)."""
        views = torch.arange(self.first_micro_batch(segments.views))
        # The loop runs the store's generator out: it releases the batch's staging buffers
        # for reuse only once the compute below is queued (an abandoned generator never does).
        for batch in segments.batches(segments.names, [views]):
            trajectory = Trajectory(**batch)
            log_probability, _ = policy.evaluate(
                self.unroll(trajectory).policy_logits, _own_joint(trajectory)
            )
            mask = trajectory.alive[..., : log_probability.shape[-1]]
            rollout = prepared["rollout_log_probability"][:, self._on_device(views)]
            error = torch.where(mask, (torch.exp(log_probability - rollout) - 1.0).abs(), 0.0)
        return torch.stack([error.amax(), error.sum() / mask.sum().clamp_min(1)])


def divergence_quantiles(counts: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    """(Q,) the quantiles (Q,) of the divergence histogram `counts` (DIVERGENCE_BINS,), log
    linear within a bin, on the device; NaN for an empty histogram."""
    total = counts.sum()
    cumulative = counts.cumsum(0)
    targets = quantiles * total
    index = torch.searchsorted(cumulative, targets).clamp(max=counts.shape[0] - 1)
    below = cumulative[index] - counts[index]
    fraction = ((targets - below) / counts[index].clamp_min(1e-30)).clamp(0.0, 1.0)
    exponent = math.log10(DIVERGENCE_FLOOR) * (1.0 - (index + fraction) / counts.shape[0])
    return torch.where(total > 0, 10.0**exponent, torch.nan)


def fit_series(moments: torch.Tensor, levels: tuple[str, ...]) -> dict:
    """`critic/explained_variance_fit/{level}`: the explained variance of the learner's own
    values during the last completed epoch (moments (5 * levels,) from `_diagnostics`, for
    the critic's value levels `levels`), against the returns it fits (0 where the returns
    do not vary)."""
    count, target, target_square, residual, residual_square = moments.view(5, -1)

    def variance(total: torch.Tensor, square: torch.Tensor) -> torch.Tensor:
        return (square - total**2 / count.clamp_min(1.0)) / (count - 1.0).clamp_min(1.0)

    total = variance(target, target_square)
    fitted = variance(residual, residual_square)
    varies = total > 0
    fit = torch.where(varies, 1.0 - fitted / torch.where(varies, total, 1.0), 0.0)
    return {
        f"critic/explained_variance_fit/{level}": value
        for level, value in zip(levels, fit.unbind(), strict=True)
    }


def explained_variance(
    level_returns: torch.Tensor, values: torch.Tensor, valued: torch.Tensor
) -> torch.Tensor:
    """(levels,) per level, 1 - Var(return - value) / Var(return) over the valued own ships
    (unbiased variances; 0 where the returns do not vary). level_returns, values
    (L, V, S, levels); valued (L, V, S)."""
    mask = valued[..., None].to(level_returns.dtype)
    count = mask.sum((0, 1, 2))

    def variance(x: torch.Tensor) -> torch.Tensor:
        mean = (x * mask).sum((0, 1, 2)) / count.clamp_min(1)
        return ((x - mean) ** 2 * mask).sum((0, 1, 2)) / (count - 1).clamp_min(1)

    total = variance(level_returns)
    residual = variance(level_returns - values)
    varies = total > 0
    return torch.where(varies, 1.0 - residual / torch.where(varies, total, 1.0), 0.0)
