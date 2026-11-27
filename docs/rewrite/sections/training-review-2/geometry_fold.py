"""Prototype (queue item 4, measurement only): the geometry heads' pair columns folded into
padded tensors instead of 8c's per-column loop: one last Linear for the six ship-pair columns
(their weights concatenated, the same numbers), one for the two both-order columns on the
reversed pairs, one HL-Gauss + cross-entropy over the four linear / symlog columns (per-column
centres and knees stacked) and one over the two circular columns (both orders).

    PYTHONPATH=. python docs/rewrite/sections/training-review-2/geometry_fold.py check|time

check: the folded losses equal the trainer's on one eager micro-batch (CPU-free GPU check).
time: the compiled update (as heads-8c/cost.py, 128 environments) with the fold, a JSON line
appended to DATA/geometry_fold.jsonl.
"""

import sys

sys.path.insert(0, "docs/rewrite/sections/training-review-2")
import common  # noqa: E402

WHAT = sys.argv[1]
cache = common.cache_dir("geometry-fold")

import dataclasses  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from lean import game, model, scripted  # noqa: E402
from lean.game.core import FIELD_RADIUS_SCALE, GEOMETRY_COLUMNS  # noqa: E402
from lean.learn import LearnConfig, Opponent, collect, geometry, start, training  # noqa: E402
from lean.learn.store import RolloutStore  # noqa: E402
from lean.learn.training import make_optimiser, update  # noqa: E402
from lean.model import geometry as model_geometry  # noqa: E402
from lean.model.geometry import take  # noqa: E402
from lean.model.value import symlog  # noqa: E402

ORIGINAL_FORWARD = model_geometry.PairHead.forward
ORIGINAL_LOSSES = geometry.geometry_losses


def folded_forward(self, ships, memory, query):
    """PairHead.forward returning (forward logits (T, E, K, 6, bins), reverse logits of the
    both-order columns (T, E, K, 2, bins), field logits)."""
    normed = self.ship_norm(ships)
    this, other = self._gated(self.ship_side, normed), self._gated(self.other_ship, normed)
    forward = self._pair(take(this, query.first), take(other, query.second))
    reverse = self._pair(take(this, query.second), take(other, query.first))
    weight = torch.cat([last.weight for last in self.ship_lasts])
    bias = torch.cat([last.bias for last in self.ship_lasts])
    both = [i for i, column in enumerate(self.columns) if column.both_orders]
    bins = self.columns[0].bins
    forward_logits = F.linear(forward, weight, bias).unflatten(-1, (len(self.columns), bins))
    reverse_weight = torch.cat([self.ship_lasts[i].weight for i in both])
    reverse_bias = torch.cat([self.ship_lasts[i].bias for i in both])
    reverse_logits = F.linear(reverse, reverse_weight, reverse_bias).unflatten(-1, (len(both), bins))
    field_logits = None
    if query.field_ship is not None:
        fields = self._gated(self.other_field, self.field_norm(memory))
        field_logits = self.field_last(
            self._pair(take(this, query.field_ship), take(fields, query.field_index))
        )
    return (forward_logits, reverse_logits), field_logits


def linear_targets(values, centres, knees, sigma):
    """HL-Gauss over per-column supports: values (..., C), centres (C, bins), knees (C,)."""
    width = centres[:, 1:2] - centres[:, 0:1]
    edges = torch.cat([centres - 0.5 * width, centres[:, -1:] + 0.5 * width], -1)
    values = symlog(values, knees)
    values = torch.maximum(torch.minimum(values, centres[:, -1]), centres[:, 0])
    cumulative = torch.erf((edges - values[..., None]) / (sigma * width * math.sqrt(2.0)))
    mass = cumulative[..., 1:] - cumulative[..., :-1]
    return mass / (cumulative[..., -1:] - cumulative[..., :1])


def circular_targets(values, centres, knees, sigma):
    """circular_hl_gauss_targets over per-column circles: values (..., C)."""
    width = centres[:, 1:2] - centres[:, 0:1]
    period = width * centres.shape[-1]
    wrapped = torch.atan2(torch.sin(values), torch.cos(values))
    position = symlog(wrapped, knees)
    offset = torch.remainder(centres - position[..., None] + 0.5 * period, period) - 0.5 * period
    scale = sigma * width * math.sqrt(2.0)
    mass = 0.5 * (torch.erf((offset + 0.5 * width) / scale) - torch.erf((offset - 0.5 * width) / scale))
    return mass / mass.sum(-1, keepdim=True)


def folded_losses(policy, outputs, observation, geometry_record, query, ships, field_sample, denominators):
    config = policy.config
    sigma = config.next_state_sigma
    observed_count, pair_count, field_count = denominators.clamp(min=1.0).unbind()
    observed = geometry.observed_rows(observation)
    edge_column = config.edge_column
    edge_target = geometry_record[..., GEOMETRY_COLUMNS.index("edge")]
    edge_cross = model_geometry.column_cross_entropy(
        edge_column, outputs.edge, torch.where(observed, edge_target, 0.0),
        policy.geometry_centres(edge_column), sigma,
    )
    edge = torch.where(observed, edge_cross, 0.0).sum() / observed_count

    position, heading = observation.pos, observation.heading
    first = (take(position, query.first), take(heading, query.first), take(geometry_record, query.first))
    second = (take(position, query.second), take(heading, query.second), take(geometry_record, query.second))
    forward = game.pair_geometry(*first, second[0], second[2])
    reverse = game.pair_geometry(*second, first[0], first[2])
    columns = config.pair_columns
    linear = [i for i, c in enumerate(columns) if not c.circular]
    circular = [i for i, c in enumerate(columns) if c.circular]
    assert all(columns[i].both_orders for i in circular) and not any(columns[i].both_orders for i in linear)
    forward_logits, reverse_logits = outputs.pairs
    valid = ships.valid[..., None]
    device = forward_logits.device

    def stacked(indices):
        centres = torch.stack([policy.geometry_centres(columns[i]) for i in indices])
        knees = torch.tensor([columns[i].knee for i in indices], device=device)
        return centres, knees

    centres, knees = stacked(linear)
    values = torch.stack([forward[columns[i].name] for i in linear], -1)
    target = linear_targets(torch.where(valid, values, 0.0), centres, knees, sigma)
    cross = -(target * torch.log_softmax(forward_logits[..., linear, :].float(), -1)).sum(-1)
    total = (cross * ships.weight[..., None]).sum()
    centres, knees = stacked(circular)
    values = torch.stack(
        [torch.stack([forward[columns[i].name] for i in circular], -1),
         torch.stack([reverse[columns[i].name] for i in circular], -1)], -2
    )  # (T, B, K, 2 orders, C)
    logits = torch.stack([forward_logits[..., circular, :], reverse_logits], -3)
    target = circular_targets(torch.where(valid[..., None], values, 0.0), centres, knees, sigma)
    cross = -(target * torch.log_softmax(logits.float(), -1)).sum(-1)
    total = total + (cross * ships.weight[..., None, None]).sum() / 2.0
    pair = total / (pair_count * len(columns))

    field = edge.new_zeros(())
    if field_sample is not None:
        centre = take(observation.field_pos, query.field_index)
        radius = take(observation.field[..., 0] * FIELD_RADIUS_SCALE, query.field_index)
        at = take(position, query.field_ship)
        distance = game.field_edge_distance(centre[..., 0], centre[..., 1], radius, at[..., 0], at[..., 1])
        column = config.field_column
        cross = model_geometry.column_cross_entropy(
            column, outputs.fields, torch.where(field_sample.valid, distance, 0.0),
            policy.geometry_centres(column), sigma,
        )
        field = (cross * field_sample.weight).sum() / field_count
    return pair, field, edge


def fold(on: bool) -> None:
    model_geometry.PairHead.forward = folded_forward if on else ORIGINAL_FORWARD
    training.geometry_losses = folded_losses if on else ORIGINAL_LOSSES


def setup(envs):
    device = torch.device("cuda")
    torch.manual_seed(0)
    generator = torch.Generator(device=device).manual_seed(0)
    config = dataclasses.replace(LearnConfig(), epochs=1, minibatches=1)
    game_config = game.GameConfig(training_spawn=True)
    policy = model.Policy(model.ModelConfig.for_arena(game_config.arena_radius)).to(device)
    opponent = Opponent(scripted.ScriptedConfig(), None)
    state, carry = start(config, game_config, policy, opponent, envs, generator)
    store = RolloutStore(1)
    collect(config, game_config, policy, opponent, state, carry, 128, generator, store=store)
    return config, policy, store, generator


def check() -> None:
    config, policy, store, generator = setup(64)
    index, _ = store.batch_index(torch.arange(store.views), config.micro_batch_views)
    training.target_pass(config, policy, store, store.chain_index(config.micro_batch_views), None)
    counts = store.counts()
    batch = next(store.batches(store.observation_names + training.LOSS_NAMES, index[:1]))
    observation, offset = store.observation(batch)
    ticks, width, slots = observation.key_mask.shape
    keys = geometry.draw_keys(8, 8, ticks, width, slots, observation.field_pos.shape[-2], generator, "cuda")
    denominators = counts.index_select(0, batch["views"]).sum(0)
    views = batch["views"]
    arguments = (
        config, policy, observation, batch["carry"], batch["reset"], offset, batch["action"],
        batch["log_probability"], store.advantage.index_select(1, views), store.target.index_select(1, views),
        batch["teacher"], batch["next_state"], batch["next_valid"], batch["hit_label"], batch["geometry"],
        *keys, torch.tensor([1.0, 0.003, 0.0], device="cuda"), denominators,
    )
    results = {}
    for on in (False, True):
        fold(on)
        policy.zero_grad()
        loss, measures = training.minibatch_loss(*arguments)
        loss.backward()
        grads = torch.cat([p.grad.flatten() for p in policy.parameters() if p.grad is not None])
        results[on] = (measures, grads)
    record = {"job": "geometry_fold_check"}
    for name in ("pair", "field", "edge", "loss"):
        record[name] = [results[False][0][name].item(), results[True][0][name].item()]
    difference = (results[False][1] - results[True][1]).abs().max().item()
    record["max_gradient_difference"] = difference
    record["max_gradient"] = results[False][1].abs().max().item()
    common.emit(record, os.path.join(common.DATA, "geometry_fold.jsonl"))


def timing() -> None:
    record = {"job": "geometry_fold_time", **common.machine()}
    fold(True)
    config, policy, store, generator = setup(128)
    optimiser = make_optimiser(config, policy)
    began = time.perf_counter()
    update(config, policy, optimiser, store, generator=generator, compiled=True)
    torch.cuda.synchronize()
    record["first_update_s"] = round(time.perf_counter() - began, 1)
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(6):
        torch.cuda.synchronize()
        began = time.perf_counter()
        update(config, policy, optimiser, store, generator=generator, compiled=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - began)
    steady = statistics.median(times)
    record.update(
        steady_update_ms=round(1000 * steady, 1),
        per_micro_batch_ms=round(1000 * steady / 3, 2),
        peak_allocated_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3),
    )
    common.emit(record, os.path.join(common.DATA, "geometry_fold.jsonl"))
    shutil.rmtree(cache, ignore_errors=True)


if __name__ == "__main__":
    check() if WHAT == "check" else timing()
