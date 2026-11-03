"""PPO with v1's math (learner spec 3, 4): per-level returns, the categorical critic, the
return scale and the auxiliary losses with their labels, over any model with `unroll`.

An update reads its segments from a store (`Segments`) and works on the device one shard or
one micro-batch at a time:
  1. per shard, the returns: per-level GAE, team mixing, the outcome's targets; then the
     return scale, once per batch;
  2. one pass over the views: the shot re-simulation, whose labels go into the store
     beside the segments for every epoch, each view's token counts and, when set, the old
     log-probabilities recomputed by `unroll`;
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
import zlib
from collections.abc import Callable, Iterator
from typing import Protocol

import torch
from torch.utils import _pytree

from bnb.contracts import REWARD_LEVELS, Trajectory
from bnb.contracts.action import to_joint
from bnb.contracts.obs import GLOBAL, SHIP
from bnb.contracts.series import Mean
from bnb.game.rules import Rules
from bnb.game.ship_types import ShipTypes
from bnb.models.yemong import codes, critic, policy
from bnb.models.yemong import shots as shot_codes
from bnb.models.yemong.density import HexGrid, poisson_terms

from . import labels, losses, returns, shots, statistics

TERMS = (
    "policy",
    "entropy",
    "value",
    "next_state",
    "enemy_action",
    "density",
    "shot_trajectory",
    "shot_outcome",
)
# Token counts per view, the denominators' parts: own ships alive; own ships valued (alive or
# waiting to respawn); view-steps; valid beliefs at non-final steps; enemies alive; shots.
COUNTS = ("policy", "valued", "steps", "next_state", "enemy_action", "shots")
# The per-tick records the returns read, staged whole per shard.
SMALL = ("reward", "values", "outcome", "done", "result", "log_probability", "alive", "valued")
# The auxiliary heads' losses (nats per token), as their groups' series.
HEAD_SERIES = {
    "next_state": "next_state/loss",
    "enemy_action": "enemy_action/loss",
    "density": "density/loss",
    "shot_trajectory": "shot/loss/trajectory",
    "shot_outcome": "shot/loss/outcome",
}
# The stored shot labels' name in the store: per view, {"shots": Shots, "mask"}, (V, L, ...).
SHOT_LABELS = "shot_labels"


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
    """Learner spec 3-4, v1's values. Per-level tuples follow REWARD_LEVELS. A coefficient
    multiplies its term's mean in nats per token (entropy: the mean entropy, so a negative
    coefficient rewards it).

    old_log_probabilities: "rollout" (the rollout's) or "recomputed" (by `unroll` before
    the first epoch). minibatches: per epoch, as equal in views as the views allow;
    micro_batch_rows: ship rows per micro-batch, so micro_batch_rows // N views (at least
    one): memory only, the result is the same, and the memory a micro-batch needs follows
    its rows at any fleet size.
    """

    discount: tuple[float, ...] = (0.991, 0.995, 0.995, 0.999, 0.9997)
    trace_decay: tuple[float, ...] = (0.90, 0.95, 0.95, 0.97, 0.97)
    team_spirit: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    return_scale_decay: float = 0.99
    return_scale_floor: float = 1.0
    clip: float = 0.2
    policy: float = 27.0
    entropy: float = -0.0081
    value: float = 0.007
    next_state: float = 0.0041
    enemy_action: float = 0.11
    density: float = 0.021
    shot_trajectory: float = 0.007
    shot_outcome: float = 0.019
    learning_rate: float = 3e-4
    warmup_start: float = 1e-7
    warmup_steps: int = 1_000_000  # environment steps
    max_grad_norm: float = 1.0
    epochs: int = 4
    minibatches: int = 32
    micro_batch_rows: int = 160
    target_kl: float | None = 0.02  # None: no early stop
    old_log_probabilities: str = "rollout"

    def __post_init__(self) -> None:
        if self.old_log_probabilities not in ("rollout", "recomputed"):
            raise ValueError(f"old_log_probabilities {self.old_log_probabilities!r}")
        for name in ("discount", "trace_decay", "team_spirit"):
            if len(getattr(self, name)) != len(REWARD_LEVELS):
                raise ValueError(f"{name} needs one value per reward level")


class Segments(Protocol):
    """Where the update's segments live (`bnb.runtime.store.SegmentStore`): `shards`
    segments of `length` ticks of `views // shards` views each, every Trajectory field by
    name (`names`)."""

    views: int
    shards: int
    length: int
    names: tuple[str, ...]

    def shard(self, names: tuple[str, ...], shard: int) -> dict:
        """Whole-shard copies on the device of the per-tick leaves names, (L, R, ...) with
        R = views // shards."""

    def batches(self, names: tuple[str, ...], view_batches: list[torch.Tensor]) -> Iterator[dict]:
        """Yield {name: pytree} on the device for each list of views in view_batches."""

    def write_views(self, source: dict, views: slice) -> None:
        """Copy {name: pytree (n, ...)} into the per-view leaves of the n views of the slice
        views."""


def random_stream(seed: int, name: str, device: torch.device | None = None) -> torch.Generator:
    """The learner's random stream `name`: its own generator, seeded from the seed and the
    name, so a new stream shifts no other's draws and the global generator none of them."""
    return torch.Generator(device).manual_seed(seed << 32 | zlib.crc32(name.encode()))


def learning_rate(config: LearnerConfig, steps: int) -> float:
    """Linear warmup from warmup_start over the first warmup_steps environment steps."""
    fraction = min(1.0, steps / config.warmup_steps) if config.warmup_steps else 1.0
    return config.warmup_start + (config.learning_rate - config.warmup_start) * fraction


def trajectory_runs(code: shot_codes.ShotCodeParameters) -> codes.Runs:
    """The shot trajectory code's parts as (groups, size): displacement, direction, speed,
    refractive index."""
    return (
        (code.displacement_levels, 9),
        (code.direction_levels, 4),
        (1, code.speed_bins),
        (1, code.index_bins),
    )


def _head(unrolled, name: str):
    return getattr(unrolled, name, None)


def _own_joint(trajectory: Trajectory) -> torch.Tensor:
    return to_joint(trajectory.action)[..., : trajectory.log_probability.shape[-1]]


class Learner:
    """model: the learner's module, with `unroll(carry, obs, replay, reset)`, the critic's
    `bins` and `config.levels`, `outcome_level` and `local_levels`; Yemong's auxiliary heads
    also need `state_code`, `shot_code`, `shot_trajectory` and `config.hex_rings`. rules and
    types: the game's (for the shot re-simulation). outcome_weight: the outcome level's
    reward weight (its value in reward units is P(win) - P(loss) times it). stage: how the
    runtime compiles the micro-batch loss, the shot re-simulation's tick and the label codes
    (identity: eager)."""

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
    ) -> None:
        """seed: the learner's random streams (the shot spread, the minibatch order).
        diagnostics: measure the diagnostic series beyond the losses and the early stop's
        KL (none of them changes the update)."""
        self.model, self.config, self.rules, self.types = model, config, rules, types
        self.device, self.outcome_weight = device, outcome_weight
        levels = model.config.levels
        if tuple(levels) != REWARD_LEVELS:
            raise ValueError(f"the model's levels {levels} are not {REWARD_LEVELS}")
        self.outcome_level = levels.index(model.config.outcome_level)
        self.local_levels = model.config.local_levels
        self.local_index = torch.tensor(self.local_levels, device=device)  # no copy per use
        self.optimizer = Adam(
            model.parameters(), lr=config.learning_rate, fused=device.type == "cuda"
        )
        self.return_scale = critic.ReturnScale(
            config.return_scale_decay, config.return_scale_floor, device
        )
        self.spread_generator = random_stream(seed, "shot_spread", device)
        self.order_generator = random_stream(seed, "minibatch_order")  # host indices
        self.auxiliary = hasattr(model, "state_code")
        if self.auxiliary:
            self.grid = HexGrid(model.config.hex_rings, device)
            self.shot_runs = trajectory_runs(model.shot_code)
        self.discount = torch.tensor(config.discount, device=device)
        self.trace_decay = torch.tensor(config.trace_decay, device=device)
        self.team_spirit = torch.tensor(config.team_spirit, device=device)
        self.ages = torch.arange(1, rules.bullet_moves + 1, dtype=torch.float32, device=device)
        self.loss = stage(self._loss)
        self.shot_tick = stage(shots.tick)
        self.codes = stage(self._codes)
        self.environment_steps = 0
        self.diagnostics = diagnostics
        self.parameter_groups = statistics.parameter_groups(model)
        # made once: a tensor built from host values inside the update would be a copy
        self.return_quantiles = torch.tensor([0.05, 0.5, 0.95], device=device)
        self.entropy_quantiles = torch.tensor(statistics.ENTROPY_QUANTILES, device=device)

    # ------------------------------------------------------------------ returns

    def _in_reward_units(self, values: torch.Tensor) -> torch.Tensor:
        """The outcome level's value is P(win) - P(loss); times its weight it is in reward
        units like the others."""
        scale = torch.ones(values.shape[-1], device=values.device)
        scale[self.outcome_level] = self.outcome_weight
        return values * scale

    def shard_returns(self, small: dict, bootstrap) -> dict:
        """Per-level GAE, team mixing and the outcome targets of one shard, (L, R, ...)."""
        level, config = self.outcome_level, self.config
        values = self._in_reward_units(small["values"])
        done = small["done"]
        advantages, level_returns = returns.per_level_gae(
            small["reward"],
            values,
            done,
            self._in_reward_units(bootstrap.values),
            done[-1],
            self.discount,
            self.trace_decay,
        )
        own_alive = small["alive"][..., : values.shape[-2]]
        team = torch.zeros_like(own_alive, dtype=torch.int64)  # a view's own ships: one team
        advantage, total = returns.team_aggregate(
            team, own_alive, advantages, level_returns, self.team_spirit
        )
        targets = critic.outcome_targets(
            small["outcome"],
            small["result"],
            done,
            bootstrap.outcome,
            config.discount[level],
            config.trace_decay[level],
        )
        return dict(
            advantage=advantage,
            total=total,
            level_returns=level_returns,
            values=values,
            outcome_targets=targets,
        )

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

    def shots(self, trajectory: Trajectory, spread: torch.Tensor):
        """The re-simulated shots and the loss mask: valid shots of own ships and of enemies
        visible at t."""
        own = trajectory.log_probability.shape[-1]
        shot = shots.resimulate(
            self.rules,
            self.types,
            trajectory.capture,
            trajectory.obs.fields,
            trajectory.done,
            spread,
            own,
            self.shot_tick,
        )
        visible = trajectory.obs.ships[..., SHIP["visible"]] > 0.5
        rows = torch.arange(visible.shape[-1], device=visible.device) < own
        return shot, shot.valid & (rows | visible)

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

    def _codes(self, trajectory: Trajectory, shot: shots.Shots, shot_mask: torch.Tensor) -> dict:
        """The targets as the heads' codes, from the stored rows and the re-simulated shots
        (a stage the runtime may compile)."""
        model = self.model
        rows = labels.privileged_rows(trajectory)
        enemy, enemy_mask = labels.enemy_actions(trajectory)
        code = model.shot_code
        return dict(
            next_state=labels.next_state_targets(model.state_code, trajectory, rows),
            next_state_mask=labels.next_state_mask(trajectory),
            enemy_action=enemy,
            enemy_mask=enemy_mask,
            density=labels.density_targets(self.grid, trajectory, rows),
            shot_trajectory=shot_codes.trajectory_code(code, shot.snapshots),
            shot_ages=shot.age_valid,
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
        diagnostics)), every sum over its tokens divided by its minibatch-total
        denominator."""
        config, model = self.config, self.model
        unrolled = self.unroll(trajectory)
        own = trajectory.log_probability.shape[-1]
        terms = dict.fromkeys(TERMS, unrolled.policy_logits.new_zeros(()))
        terms["policy"], terms["entropy"], statistic = self._policy(
            unrolled, trajectory, derived, denominator, scale
        )
        per_level = losses.critic_losses(
            unrolled.value_logits,
            unrolled.outcome_logits,
            derived["level_returns"].index_select(-1, self.local_index),
            derived["outcome_targets"],
            trajectory.valued,
            model.bins,
            self.local_levels,
            self.outcome_level,
            denominator["valued"],
            denominator["steps"],
        )
        terms["value"] = per_level.sum()
        if _head(unrolled, "next_state_logits") is not None:
            next_state = codes.grouped_cross_entropy(
                unrolled.next_state_logits, label["next_state"], model.state_code.runs
            ).sum(-1)
            terms["next_state"] = (
                losses.masked_sum(next_state, label["next_state_mask"]) / denominator["next_state"]
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
                model.shot_trajectory(unrolled.shot_latents, self.ages),
                label["shot_trajectory"],
                unrolled.shot_outcome_logits,
                label["shot_outcome"],
                self.shot_runs,
            )
            ages = label["shot_ages"]
            per_shot = torch.where(ages, parts.sum(-1), 0.0).sum(-1) / ages.sum(-1).clamp_min(1)
            terms["shot_trajectory"] = (
                losses.masked_sum(per_shot, label["shot_mask"]) / denominator["shots"]
            )
            terms["shot_outcome"] = (
                losses.masked_sum(outcome, label["shot_mask"]) / denominator["shots"]
            )
        loss = sum(getattr(config, term) * terms[term] for term in TERMS)
        return (
            loss,
            torch.stack([terms[term] for term in TERMS]).detach(),
            per_level.detach(),
            statistic.detach(),
            self._diagnostics(unrolled, trajectory, derived, denominator),
        )

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
        )  # (L, R, S, levels - 1)
        local = torch.where(trajectory.valued[..., None], local, 0.0).flatten(0, -2).sum(0)
        columns = dict(
            zip(self.local_levels, (local / denominator["valued"]).unbind(), strict=True)
        )
        columns[self.outcome_level] = (
            statistics.target_entropy(derived["outcome_targets"]).sum() / denominator["steps"]
        )
        targets = torch.stack([columns[level] for level in range(len(columns))])
        return torch.cat([sums / denominator["policy"], targets])

    STATISTICS = ("clip_fraction", "approximate_kl")  # what `_policy` reports per minibatch

    def _policy(
        self,
        unrolled,
        trajectory: Trajectory,
        derived: dict,
        denominator: dict,
        scale: torch.Tensor,
    ):
        """The policy's terms of a micro-batch: (the clipped surrogate, the mean entropy,
        statistics (STATISTICS,)), each over own ships alive."""
        own_alive = trajectory.alive[..., : trajectory.log_probability.shape[-1]]
        log_probability, entropy = policy.evaluate(unrolled.policy_logits, _own_joint(trajectory))
        old = derived["old_log_probability"]
        surrogate = losses.surrogate(
            log_probability, old, derived["advantage"], scale, self.config.clip
        )
        ratio = torch.exp(log_probability - old)
        clipped = losses.masked_sum(((ratio - 1.0).abs() > self.config.clip).float(), own_alive)
        divergence = losses.masked_sum((ratio - 1.0) - (log_probability - old), own_alive)
        return (
            losses.masked_sum(surrogate, own_alive) / denominator["policy"],
            losses.masked_sum(entropy, own_alive) / denominator["policy"],
            torch.stack([clipped, divergence]) / denominator["policy"],
        )

    # ------------------------------------------------------------------ update

    @property
    def micro_views(self) -> int:
        """Views per micro-batch: micro_batch_rows ship rows."""
        return max(1, self.config.micro_batch_rows // self.rules.num_ships)

    def _micro_batches(self, views: torch.Tensor) -> list[torch.Tensor]:
        return list(views.split(self.micro_views))

    def _on_device(self, views: torch.Tensor) -> torch.Tensor:
        """Host view indices on the device, from pinned memory: no wait on the queued work
        (a copy from pageable memory, or `.cpu()`, would wait for it and stall the
        staging's overlap)."""
        if self.device.type != "cuda":
            return views
        return views.pin_memory().to(self.device, non_blocking=True)

    def prepare(self, segments: Segments, bootstraps: list) -> dict:
        """Steps 1 and 2: per view, the advantage, the returns, the outcome targets and the
        old log-probabilities (L, V, ...), each view's token counts (V, COUNTS), the return
        scale and the explained variance per level (levels,); the shot labels go into the
        store."""
        config, device = self.config, self.device
        small = [segments.shard(SMALL, shard) for shard in range(segments.shards)]
        parts = [self.shard_returns(part, bootstraps[shard]) for shard, part in enumerate(small)]

        def joined(name: str, source: list[dict]) -> torch.Tensor:
            return torch.cat([part[name] for part in source], 1)

        derived = {
            name: joined(name, parts) for name in ("advantage", "level_returns", "outcome_targets")
        }
        rollout_log_probability = joined("log_probability", small)
        derived["old_log_probability"] = rollout_log_probability.clone()
        own_alive = joined("alive", small)[..., : rollout_log_probability.shape[-1]]
        scale = self.return_scale.update(joined("total", parts), own_alive)
        counts = torch.cat([self._counts(part) for part in small], 0)
        recompute = config.old_log_probabilities == "recomputed"
        if self.auxiliary or recompute:
            # One spread draw per update: each epoch reads the same stored labels.
            spread = torch.randn(
                (segments.length, segments.views, self.rules.num_ships, 2),
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
                        shot, shot_mask = self.shots(trajectory, spread[:, views])
                        counts[views, COUNTS.index("next_state")] = (
                            labels.next_state_mask(trajectory).sum((0, 2)).float()
                        )
                        counts[views, COUNTS.index("shots")] = shot_mask.sum((0, 2)).float()
                        stored = _pytree.tree_map_only(
                            torch.Tensor,
                            lambda x: x.transpose(0, 1).contiguous(),
                            dict(shots=shot, mask=shot_mask),
                        )  # (n, L, ...)
                        first = int(host_views[0])
                        segments.write_views(
                            {SHOT_LABELS: stored}, slice(first, first + host_views.numel())
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
        return dict(
            derived=derived,
            rollout_log_probability=rollout_log_probability,
            counts=counts,
            scale=scale,
            explained_variance=variance,
            series=series,
        )

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
        for index, level in enumerate(REWARD_LEVELS):

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

    def gradient(self, segments: Segments, prepared: dict, chosen: torch.Tensor) -> tuple:
        """Accumulate one minibatch's gradient (views `chosen`, host indices) into the
        parameters' .grad, micro-batch by micro-batch: (terms (TERMS,), value loss per
        level, statistics (STATISTICS,), diagnostics (see `_loss`)), each the minibatch's."""
        device, derived = self.device, prepared["derived"]
        total = prepared["counts"][self._on_device(chosen)].sum(0).clamp_min(1.0)
        denominator = dict(zip(COUNTS, total.unbind()))
        micro = self._micro_batches(chosen)
        terms_sum = torch.zeros(len(TERMS), device=device)
        levels_sum = torch.zeros(len(REWARD_LEVELS), device=device)
        statistic_sum = torch.zeros(len(self.STATISTICS), device=device)
        diagnostic_sum = None
        self.optimizer.zero_grad(set_to_none=True)
        for host_views, (trajectory, shot_labels) in zip(
            micro, self.staged(segments, micro), strict=True
        ):
            views = self._on_device(host_views)
            label = self.labels(trajectory, shot_labels)
            part = {name: value.index_select(1, views) for name, value in derived.items()}
            loss, terms, per_level, statistic, diagnostic = self.loss(
                trajectory, label, part, denominator, prepared["scale"]
            )
            loss.backward()
            terms_sum += terms
            levels_sum += per_level
            statistic_sum += statistic
            diagnostic_sum = diagnostic if diagnostic_sum is None else diagnostic_sum + diagnostic
        return terms_sum, levels_sum, statistic_sum, diagnostic_sum

    def _step(self, finite: torch.Tensor) -> None:
        """The optimizer's step, kept only where `finite` (a device flag) holds: a step with
        a non-finite gradient leaves the parameters and the optimizer's state as they were
        (learner spec 4d: skipped and counted), with no host synchronisation. A state the
        step creates (the first step) is zero before it, as Adam starts it."""
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
        bootstraps: list,
        environment_steps: int = 0,
    ) -> dict:
        """One update from the store's shards and their bootstraps (each the learner's
        Outputs on the observation after the shard's last tick). environment_steps:
        environment steps this update collected (the warmup counts them). Returns the update's
        series on the device (`bnb.contracts.series`), for the caller to read back once."""
        config, device = self.config, self.device
        views = segments.views
        if views < config.minibatches:
            raise ValueError(f"{views} views do not make {config.minibatches} minibatches")
        prepared = self.prepare(segments, bootstraps)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate(config, self.environment_steps)
        self.environment_steps += environment_steps
        before = (
            {name: p.detach().clone() for name, p in self.model.named_parameters()}
            if self.diagnostics
            else None
        )
        terms_total = torch.zeros(len(TERMS), device=device)
        levels_total = torch.zeros(len(REWARD_LEVELS), device=device)
        statistics_total = torch.zeros(len(self.STATISTICS), device=device)
        diagnostics_total = None
        drift = self.drift(segments, prepared)
        skipped = torch.zeros((), dtype=torch.int64, device=device)
        nonfinite_terms = torch.zeros(len(TERMS), dtype=torch.int64, device=device)
        norm_total = torch.zeros((), device=device)  # the finite steps' pre-clip norms
        clipped = torch.zeros((), dtype=torch.int64, device=device)
        steps, stopped, epochs = 0, False, 0
        first_kl = first_epoch = epoch_statistics = None
        for epoch in range(config.epochs):
            # Minibatches as equal as the views allow (micro-batches of micro_views views,
            # each minibatch's last one smaller where they do not divide).
            order = torch.randperm(views, generator=self.order_generator)
            epoch_statistics = torch.zeros(len(self.STATISTICS), device=device)
            for chosen in order.tensor_split(config.minibatches):
                terms, per_level, statistic, diagnostic = self.gradient(segments, prepared, chosen)
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
                diagnostics_total = (
                    diagnostic if diagnostics_total is None else diagnostics_total + diagnostic
                )
                steps += 1
            epochs += 1
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
        for index, level in enumerate(REWARD_LEVELS):
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
            out.update(self._parameter_series(before))
        return out

    def report(
        self, first_epoch: torch.Tensor, last_epoch: torch.Tensor, mean: torch.Tensor
    ) -> dict:
        """The policy's series from its STATISTICS: the first and the last completed
        epoch's means and the mean over every minibatch step. `policy/kl` is the last
        epoch's (what the early stop reads), the clip fraction every step's."""
        index = self.STATISTICS.index
        return {
            "policy/kl": last_epoch[index("approximate_kl")],
            "policy/clip_fraction": mean[index("clip_fraction")],
        }

    def _diagnostic_series(
        self, prepared: dict, total: torch.Tensor, steps: int, levels_total: torch.Tensor
    ) -> dict:
        """The policy's diagnostic means over the update's steps (`statistics.POLICY_SUMS`,
        the entropy quantiles from its histogram, the factor coupling), the value KL per
        level, and the series `prepare` measured before any step."""
        policy_part, targets = total[: -len(REWARD_LEVELS)], total[-len(REWARD_LEVELS) :]
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
        for index, level in enumerate(REWARD_LEVELS):
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

    @torch.no_grad()
    def drift(self, segments: Segments, prepared: dict) -> torch.Tensor:
        """(2,): the step/unroll ratio drift before any optimizer step, on the first
        micro-batch of views: |exp(log_probability_unroll - log_probability_rollout) - 1|
        over living own ships, max and mean (0 without any)."""
        views = torch.arange(min(self.micro_views, segments.views))
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
