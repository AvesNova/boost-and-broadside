"""PPO with v1's math (learner spec 3, 4): per-level returns, the categorical critic, the
return scale and the auxiliary losses with their labels, over any model with `unroll`.

An update reads its segments from a store (`Segments`) and works on the device one shard or
one micro-batch at a time:
  1. per shard, the returns: per-level GAE, team mixing, the outcome's targets; then the
     return scale, once per batch;
  2. one pass over the views for each view's token counts (the shot labels need the
     re-simulation) and, when set, the old log-probabilities recomputed by `unroll`;
  3. epochs of minibatches over the views (a view keeps its whole segment); a minibatch is
     several micro-batches, each term divided by its minibatch-total denominator, so the
     accumulated gradient is the unsplit minibatch's.
The learner owns the optimizer and updates the model's parameters in place, which keeps the
rollout's captured graph pointing at the current weights. A model lacking a head gives no
such field (or None) and that term is skipped.
"""

import dataclasses
from collections.abc import Callable, Iterator
from typing import Protocol

import torch

from bnb.contracts import REWARD_LEVELS, Trajectory
from bnb.contracts.action import to_joint
from bnb.contracts.obs import GLOBAL, SHIP
from bnb.game.rules import Rules
from bnb.game.ship_types import ShipTypes
from bnb.models.yemong import codes, critic, policy
from bnb.models.yemong import shots as shot_codes
from bnb.models.yemong.density import HexGrid, poisson_terms

from . import labels, losses, returns, shots

TERMS = ("policy", "entropy", "value", "next_state", "enemy_action", "density",
         "shot_trajectory", "shot_outcome")
# Token counts per view, the denominators' parts: own ships alive; own ships valued (alive or
# waiting to respawn); view-steps; valid beliefs at non-final steps; enemies alive; shots.
COUNTS = ("policy", "valued", "steps", "next_state", "enemy_action", "shots")
# The per-tick records the returns read, staged whole per shard.
SMALL = ("reward", "values", "outcome", "done", "result", "logp", "alive", "valued")


@dataclasses.dataclass(frozen=True)
class LearnerConfig:
    """Learner spec 3-4, v1's values. Per-level tuples follow REWARD_LEVELS. A coefficient
    multiplies its term's mean in nats per token (entropy: the mean entropy, so a negative
    coefficient rewards it).

    old_log_probabilities: "rollout" (the rollout's) or "recomputed" (by `unroll` before
    the first epoch). minibatches: per epoch, as equal in views as the views allow;
    micro_batch_rows: ship rows per micro-batch, so micro_batch_rows // N views (at least
    one): memory only, the result is the same, and the memory a micro-batch needs follows
    its rows at any fleet size. seed: the shot re-simulation's spread generator.
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
    target_kl: float = 0.02
    old_log_probabilities: str = "rollout"
    seed: int = 0

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

    def shard(self, names: tuple[str, ...], shard: int) -> dict: ...

    def batches(self, names: tuple[str, ...],
                view_batches: list[torch.Tensor]) -> Iterator[dict]: ...


def learning_rate(config: LearnerConfig, steps: int) -> float:
    """Linear warmup from warmup_start over the first warmup_steps environment steps."""
    fraction = min(1.0, steps / config.warmup_steps) if config.warmup_steps else 1.0
    return config.warmup_start + (config.learning_rate - config.warmup_start) * fraction


def trajectory_runs(code: shot_codes.ShotCodeParameters) -> codes.Runs:
    """The shot trajectory code's parts as (groups, size): displacement, direction, speed,
    refractive index."""
    return ((code.displacement_levels, 9), (code.direction_levels, 4), (1, code.speed_bins),
            (1, code.index_bins))


def _head(unrolled, name: str):
    return getattr(unrolled, name, None)


def _own_joint(trajectory: Trajectory) -> torch.Tensor:
    return to_joint(trajectory.action)[..., :trajectory.logp.shape[-1]]


class Learner:
    """model: the learner's module, with `unroll(carry, obs, replay, reset)`, the critic's
    `bins` and `config.levels`, `outcome_level` and `local_levels`; Yemong's auxiliary heads
    also need `state_code`, `shot_code`, `shot_trajectory` and `config.hex_rings`. rules and
    types: the game's (for the shot re-simulation). outcome_weight: the outcome level's
    reward weight (its value in reward units is P(win) - P(loss) times it). stage: how the
    runtime compiles the micro-batch loss, the shot re-simulation's tick and the label codes
    (identity: eager)."""

    def __init__(self, model: torch.nn.Module, config: LearnerConfig, rules: Rules,
                 types: ShipTypes, outcome_weight: float, device: torch.device,
                 stage: Callable = lambda function: function) -> None:
        self.model, self.config, self.rules, self.types = model, config, rules, types
        self.device, self.outcome_weight = device, outcome_weight
        levels = model.config.levels
        if tuple(levels) != REWARD_LEVELS:
            raise ValueError(f"the model's levels {levels} are not {REWARD_LEVELS}")
        self.outcome_level = levels.index(model.config.outcome_level)
        self.local_levels = model.config.local_levels
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                          fused=device.type == "cuda")
        self.return_scale = critic.ReturnScale(config.return_scale_decay,
                                               config.return_scale_floor, device)
        self.generator = torch.Generator(device).manual_seed(config.seed)
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
        self.env_steps = 0

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
            small["reward"], values, done, self._in_reward_units(bootstrap.values), done[-1],
            self.discount, self.trace_decay)
        own_alive = small["alive"][..., :values.shape[-2]]
        team = torch.zeros_like(own_alive, dtype=torch.int64)  # a view's own ships: one team
        advantage, total = returns.team_aggregate(team, own_alive, advantages, level_returns,
                                                  self.team_spirit)
        targets = critic.outcome_targets(small["outcome"], small["result"], done,
                                         bootstrap.outcome,
                                         config.discount[level], config.trace_decay[level])
        return dict(advantage=advantage, total=total, level_returns=level_returns,
                    values=values, outcome_targets=targets)

    @staticmethod
    def _counts(small: dict) -> torch.Tensor:
        """(R, COUNTS) the tokens of each view that the small records give; the pass over
        the views fills in the rest."""
        own = small["logp"].shape[-1]
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
        own = trajectory.logp.shape[-1]
        shot = shots.resimulate(self.rules, self.types, trajectory.capture, trajectory.obs.fields,
                                trajectory.done, spread, own, self.shot_tick)
        visible = trajectory.obs.ships[..., SHIP["visible"]] > 0.5
        rows = torch.arange(visible.shape[-1], device=visible.device) < own
        return shot, shot.valid & (rows | visible)

    @torch.no_grad()
    def labels(self, trajectory: Trajectory, spread: torch.Tensor) -> dict:
        """The auxiliary targets and masks of a micro-batch (none without auxiliary heads)."""
        if not self.auxiliary:
            return {}
        shot, shot_mask = self.shots(trajectory, spread)
        return self.codes(trajectory, shot, shot_mask)

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
            enemy_action=enemy, enemy_mask=enemy_mask,
            density=labels.density_targets(self.grid, trajectory, rows),
            shot_trajectory=shot_codes.trajectory_code(code, shot.snapshots),
            shot_ages=shot.age_valid, shot_mask=shot_mask,
            shot_outcome=shot_codes.outcome_code(code, shot.hit, shot.distance, shot.side))

    # ------------------------------------------------------------------ the loss

    def unroll(self, trajectory: Trajectory):
        reset = trajectory.obs.globals[..., GLOBAL["reset"]] > 0.5
        return self.model.unroll(trajectory.initial_carry, trajectory.obs, trajectory.replay,
                                 reset)

    def _loss(self, trajectory: Trajectory, label: dict, derived: dict, denominator: dict,
              scale: torch.Tensor):
        """One micro-batch: (loss, terms (TERMS,), value loss per level (K,), statistics
        (clip fraction, approximate KL)), every sum over its tokens divided by its
        minibatch-total denominator."""
        config, model = self.config, self.model
        unrolled = self.unroll(trajectory)
        own = trajectory.logp.shape[-1]
        own_alive = trajectory.alive[..., :own]
        log_probability, entropy = policy.evaluate(unrolled.policy_logits,
                                                   _own_joint(trajectory))
        old = derived["old_log_probability"]
        surrogate = losses.surrogate(log_probability, old, derived["advantage"], scale,
                                     config.clip)
        terms = dict.fromkeys(TERMS, log_probability.new_zeros(()))
        terms["policy"] = losses.masked_sum(surrogate, own_alive) / denominator["policy"]
        terms["entropy"] = losses.masked_sum(entropy, own_alive) / denominator["policy"]
        per_level = losses.critic_losses(
            unrolled.value_logits, unrolled.outcome_logits,
            derived["level_returns"][..., list(self.local_levels)], derived["outcome_targets"],
            trajectory.valued, model.bins, self.local_levels, self.outcome_level,
            denominator["valued"], denominator["steps"])
        terms["value"] = per_level.sum()
        if _head(unrolled, "next_state_logits") is not None:
            next_state = codes.grouped_cross_entropy(unrolled.next_state_logits,
                                                     label["next_state"],
                                                     model.state_code.runs).sum(-1)
            terms["next_state"] = losses.masked_sum(next_state, label["next_state_mask"]) \
                / denominator["next_state"]
        if _head(unrolled, "enemy_action_logits") is not None:
            enemy = losses.index_cross_entropy(unrolled.enemy_action_logits[..., own:, :],
                                               label["enemy_action"])
            terms["enemy_action"] = losses.masked_sum(enemy, label["enemy_mask"]) \
                / denominator["enemy_action"]
        if _head(unrolled, "density_log_rates") is not None:
            per_cell, saturated = poisson_terms(unrolled.density_log_rates, label["density"])
            valued = trajectory.valued.sum(-1).clamp_min(1)
            terms["density"] = ((per_cell - saturated).sum(-1) / valued).sum() \
                / denominator["steps"]
        if _head(unrolled, "shot_latents") is not None:
            parts, outcome = losses.shot_cross_entropy(
                model.shot_trajectory(unrolled.shot_latents, self.ages),
                label["shot_trajectory"], unrolled.shot_outcome_logits, label["shot_outcome"],
                self.shot_runs)
            ages = label["shot_ages"]
            per_shot = torch.where(ages, parts.sum(-1), 0.0).sum(-1) / ages.sum(-1).clamp_min(1)
            terms["shot_trajectory"] = losses.masked_sum(per_shot, label["shot_mask"]) \
                / denominator["shots"]
            terms["shot_outcome"] = losses.masked_sum(outcome, label["shot_mask"]) \
                / denominator["shots"]
        loss = sum(getattr(config, term) * terms[term] for term in TERMS)
        ratio = torch.exp(log_probability - old)
        clipped = losses.masked_sum(((ratio - 1.0).abs() > config.clip).float(), own_alive)
        divergence = losses.masked_sum((ratio - 1.0) - (log_probability - old), own_alive)
        statistics = torch.stack([clipped, divergence]) / denominator["policy"]
        return loss, torch.stack([terms[term] for term in TERMS]).detach(), \
            per_level.detach(), statistics.detach()

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
        scale, the shot spread draw and the explained variance."""
        config, device = self.config, self.device
        small = [segments.shard(SMALL, shard) for shard in range(segments.shards)]
        parts = [self.shard_returns(part, bootstraps[shard]) for shard, part in enumerate(small)]

        def joined(name: str, source: list[dict]) -> torch.Tensor:
            return torch.cat([part[name] for part in source], 1)

        derived = {name: joined(name, parts)
                   for name in ("advantage", "level_returns", "outcome_targets")}
        rollout_log_probability = joined("logp", small)
        derived["old_log_probability"] = rollout_log_probability.clone()
        own_alive = joined("alive", small)[..., :rollout_log_probability.shape[-1]]
        scale = self.return_scale.update(joined("total", parts), own_alive)
        counts = torch.cat([self._counts(part) for part in small], 0)
        spread = torch.randn((segments.length, segments.views, self.rules.num_ships, 2),
                             device=device, generator=self.generator)
        recompute = config.old_log_probabilities == "recomputed"
        if self.auxiliary or recompute:
            order = self._micro_batches(torch.arange(segments.views))  # host indices
            for host_views, batch in zip(order, segments.batches(segments.names, order),
                                         strict=True):
                views = self._on_device(host_views)
                trajectory = Trajectory(**batch)
                with torch.no_grad():
                    if self.auxiliary:
                        label = self.labels(trajectory, spread[:, views])
                        counts[views, COUNTS.index("next_state")] = \
                            label["next_state_mask"].sum((0, 2)).float()
                        counts[views, COUNTS.index("shots")] = \
                            label["shot_mask"].sum((0, 2)).float()
                    if recompute:
                        log_probability, _ = policy.evaluate(self.unroll(trajectory).policy_logits,
                                                  _own_joint(trajectory))
                        derived["old_log_probability"][:, views] = log_probability
        variance = _explained_variance(joined("level_returns", parts), joined("values", parts),
                                       joined("valued", small))
        return dict(derived=derived, rollout_log_probability=rollout_log_probability,
                    counts=counts, scale=scale,
                    spread=spread, explained_variance=variance)

    def gradient(self, segments: Segments, prepared: dict, chosen: torch.Tensor) -> tuple:
        """Accumulate one minibatch's gradient (views `chosen`, host indices) into the
        parameters' .grad,
        micro-batch by micro-batch: (terms (TERMS,), value loss per level, statistics
        (clip fraction, approximate KL)), each the minibatch's."""
        device, derived = self.device, prepared["derived"]
        total = prepared["counts"][self._on_device(chosen)].sum(0).clamp_min(1.0)
        denominator = dict(zip(COUNTS, total.unbind()))
        micro = self._micro_batches(chosen)
        terms_sum = torch.zeros(len(TERMS), device=device)
        levels_sum = torch.zeros(len(REWARD_LEVELS), device=device)
        statistics = torch.zeros(2, device=device)
        self.optimizer.zero_grad(set_to_none=True)
        for host_views, batch in zip(micro, segments.batches(segments.names, micro),
                                     strict=True):
            views = self._on_device(host_views)
            trajectory = Trajectory(**batch)
            label = self.labels(trajectory, prepared["spread"][:, views])
            part = {name: value.index_select(1, views) for name, value in derived.items()}
            loss, terms, per_level, statistic = self.loss(trajectory, label, part, denominator,
                                                          prepared["scale"])
            loss.backward()
            terms_sum += terms
            levels_sum += per_level
            statistics += statistic
        return terms_sum, levels_sum, statistics

    def update(self, segments: Segments, bootstraps: list, env_steps: int = 0) -> dict:
        """One update from the store's shards and their bootstraps (each the learner's
        Outputs on the observation after the shard's last tick). env_steps: environment
        steps this update collected (the warmup counts them)."""
        config, device = self.config, self.device
        views = segments.views
        if views < config.minibatches:
            raise ValueError(f"{views} views do not make {config.minibatches} minibatches")
        prepared = self.prepare(segments, bootstraps)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate(config, self.env_steps)
        self.env_steps += env_steps
        terms_total = torch.zeros(len(TERMS), device=device)
        levels_total = torch.zeros(len(REWARD_LEVELS), device=device)
        first_epoch = torch.zeros(2, device=device)
        steps, first_epoch_steps, stopped = 0, 0, False
        drift = self.drift(segments, prepared)
        for epoch in range(config.epochs):
            # Minibatches as equal as the views allow (micro-batches of micro_views views,
            # each minibatch's last one smaller where they do not divide).
            for chosen in torch.randperm(views).tensor_split(config.minibatches):
                terms, per_level, statistics = self.gradient(segments, prepared, chosen)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.max_grad_norm)
                self.optimizer.step()
                terms_total += terms
                levels_total += per_level
                steps += 1
                if epoch == 0:
                    first_epoch += statistics
                    first_epoch_steps += 1
                if statistics[1].item() > config.target_kl:  # one host sync per minibatch
                    stopped = True
                    break
            if stopped:
                break
        first = (first_epoch / first_epoch_steps).tolist()
        out = {f"loss/{term}": value for term, value in zip(TERMS, (terms_total / steps).tolist())}
        out.update({f"value_loss/{name}": value
                    for name, value in zip(REWARD_LEVELS, (levels_total / steps).tolist())})
        out.update(prepared["explained_variance"])
        out.update(drift)
        out.update(clip_fraction=first[0], approximate_kl=first[1], minibatch_steps=steps,
                   stopped_early=stopped, return_scale=prepared["scale"].item(),
                   learning_rate=self.optimizer.param_groups[0]["lr"])
        return out

    @torch.no_grad()
    def drift(self, segments: Segments, prepared: dict) -> dict:
        """The step/unroll ratio drift before any optimizer step, on the first micro-batch
        of views: |exp(log_probability_unroll - log_probability_rollout) - 1| over living
        own ships, max and mean."""
        views = torch.arange(min(self.micro_views, segments.views))
        batch = next(iter(segments.batches(segments.names, [views])))
        trajectory = Trajectory(**batch)
        log_probability, _ = policy.evaluate(self.unroll(trajectory).policy_logits,
                                             _own_joint(trajectory))
        mask = trajectory.alive[..., :log_probability.shape[-1]]
        rollout = prepared["rollout_log_probability"][:, self._on_device(views)]
        error = (torch.exp(log_probability - rollout) - 1.0).abs()[mask]
        if error.numel() == 0:
            return dict(ratio_drift_max=0.0, ratio_drift_mean=0.0)
        return dict(ratio_drift_max=error.max().item(), ratio_drift_mean=error.mean().item())


def _explained_variance(level_returns: torch.Tensor, values: torch.Tensor,
                        valued: torch.Tensor) -> dict:
    """1 - Var(return - value) / Var(return) per level, over valued own ships."""
    out = {}
    for index, name in enumerate(REWARD_LEVELS):
        target, value = level_returns[..., index][valued], values[..., index][valued]
        variance = target.var() if target.numel() > 1 else target.new_zeros(())
        out[f"explained_variance/{name}"] = \
            (1.0 - (target - value).var() / variance).item() if variance > 0 else 0.0
    return out
