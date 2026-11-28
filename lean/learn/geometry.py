"""The relative-geometry losses (queue item 8c, author 2026-11-28): which pairs the pairwise
head evaluates, their targets from the game's own functions, and the losses.

Rows the team sees (`observed`): its own living ships and the enemies visible this tick
(Obs.key_mask and Obs.visible). A pair trains when both its ships are observed (a ship-field
pair when its ship is); a hidden or dead ship's pairs are masked. The targets come from the
geometry the team sees: the observation's positions and headings (exact for observed rows)
and the rollout's per-ship geometry record (lean.game.geometry_record), through
lean.game.pair_geometry and lean.game.field_edge_distance.

Cost control: per view and tick, K valid ship pairs (of the N (N - 1) / 2 with first <
second) and K_f valid ship-field pairs (of N F) are drawn uniformly without replacement,
before the pair layers, so the pair layers' cost scales with K. The draw is an input (`keys`,
uniform numbers drawn outside the compiled loss): the top K keys among the valid pairs, a
static shape. With c valid pairs on a tick, each sampled one counts c / min(K, c): the sum
over the sampled pairs divided by the minibatch's valid pairs is an unbiased estimate of the
mean over all of them (exact when K >= c). K at or above the pair count takes every pair, no
draw.
"""

from typing import NamedTuple

import torch
from torch import Tensor

from lean import game
from lean.contract import random_keywords
from lean.game.core import FIELD_RADIUS_SCALE, GEOMETRY_COLUMNS
from lean.model.geometry import PairQuery, column_cross_entropy, take


def observed_rows(observation) -> Tensor:
    """(..., N) bool: the rows the team sees (its living ships and the visible living enemies):
    the rows the next-state, hit and geometry losses train."""
    return observation.key_mask & observation.visible


def ship_pairs(slots: int, device) -> tuple[Tensor, Tensor]:
    """The upper triangle's pairs (first < second), (N (N - 1) / 2,) long each."""
    first, second = torch.triu_indices(slots, slots, 1, device=device)
    return first, second


def field_pairs(slots: int, fields: int, device) -> tuple[Tensor, Tensor]:
    """Every ship-field pair, ship-major: (N F,) long ship slots and field slots."""
    ship = torch.arange(slots, device=device).repeat_interleave(fields)
    field = torch.arange(fields, device=device).repeat(slots)
    return ship, field


def pair_counts(observed: Tensor, fields: int) -> tuple[Tensor, Tensor]:
    """Valid ship pairs and ship-field pairs per row of observed (..., N): c (c - 1) / 2 and c F,
    c the observed ships. Float."""
    count = observed.sum(-1).float()
    return 0.5 * count * (count - 1.0), count * fields


def same_device(first, second) -> bool:
    """Whether two devices are one, a CUDA device without an index being the current one."""

    def resolved(device) -> torch.device:
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            return torch.device("cuda", torch.cuda.current_device())
        return device

    return resolved(first) == resolved(second)


def draw_keys(
    samples: int,
    field_samples: int,
    ticks: int,
    views: int,
    slots: int,
    fields: int,
    generator,
    device,
) -> tuple[Tensor | None, Tensor | None]:
    """The pair draw's uniform keys of one micro-batch, (T, B, N (N - 1) / 2) and (T, B, N F)
    float32, each None where K covers every pair (no draw). Drawn on `device` from `generator`
    when it lives there (a device without an index is the current one: "cuda" and "cuda:0"
    are one device), else from the device's default generator."""
    if generator is not None and not same_device(generator.device, device):
        generator = None
    keywords = random_keywords(generator, device)
    pairs, field_count = slots * (slots - 1) // 2, slots * fields
    keys = torch.rand((ticks, views, pairs), **keywords) if samples < pairs else None
    field_keys = (
        torch.rand((ticks, views, field_count), **keywords)
        if 0 < field_count and field_samples < field_count
        else None
    )
    return keys, field_keys


class Sampled(NamedTuple):
    """Sampled pairs of one family: their indices into the family's pair table (T, B, K), whether
    each is a valid pair, and each valid one's weight c / min(K, c) (0 elsewhere)."""

    index: Tensor
    valid: Tensor
    weight: Tensor


def sample(valid: Tensor, keys: Tensor | None, samples: int) -> Sampled:
    """K pairs per row of valid (T, B, Q) bool: every pair when keys is None (K >= Q), else the
    K valid pairs with the largest keys (T, B, Q) in [0, 1) (invalid pairs score -1, so they
    are taken only when fewer than K are valid, and then marked invalid)."""
    count = valid.sum(-1, keepdim=True).float()
    if keys is None:
        index = torch.arange(valid.shape[-1], device=valid.device).expand(valid.shape)
        return Sampled(index, valid, valid.float())
    index = torch.where(valid, keys, -1.0).topk(samples, -1).indices
    chosen = valid.gather(-1, index)
    weight = count / count.clamp(max=float(samples)).clamp(min=1.0)
    return Sampled(index, chosen, torch.where(chosen, weight, 0.0))


def pair_query(
    observed: Tensor,
    fields: int,
    keys: Tensor | None,
    field_keys: Tensor | None,
    samples: int,
    field_samples: int,
) -> tuple[PairQuery, Sampled, Sampled | None]:
    """The pairs of one micro-batch (observed (T, B, N)): the PairQuery for the model and the
    two families' samples."""
    slots, device = observed.shape[-1], observed.device
    first, second = ship_pairs(slots, device)
    ships = sample(observed[..., first] & observed[..., second], keys, samples)
    query_first, query_second = first[ships.index], second[ships.index]
    if fields == 0:
        return PairQuery(query_first, query_second, None, None), ships, None
    ship, field = field_pairs(slots, fields, device)
    field_sample = sample(observed[..., ship], field_keys, field_samples)
    query = PairQuery(
        query_first, query_second, ship[field_sample.index], field[field_sample.index]
    )
    return query, ships, field_sample


def geometry_losses(
    policy,
    outputs,
    observation,
    geometry: Tensor,
    query: PairQuery,
    ships: Sampled,
    field_sample: Sampled | None,
    denominators: Tensor,
    by_column: bool = False,
) -> tuple[Tensor, ...]:
    """The pair, field and map-edge losses of one micro-batch, each nats per column summed over
    its valid rows and divided by the minibatch's count (`denominators` = (observed rows, ship
    pairs, ship-field pairs)): the sampled pairs weighted by Sampled.weight. outputs: the
    policy's Sequence for `query`; geometry (T, B, N, GEOMETRY_COLUMNS) the record. Targets of
    rows that do not train are set to 0 before the loss (the spare view's zero record gives
    0 / 0 in the time of flight). `by_column` (the gradient diagnostic, lean.learn.gradients)
    also returns the pair loss's part from each of `pair_columns` (a list summing to it)."""
    config = policy.config
    sigma = config.next_state_sigma
    observed_count, pair_count, field_count = denominators.clamp(min=1.0).unbind()
    observed = observed_rows(observation)

    edge_column = config.edge_column
    edge_target = geometry[..., GEOMETRY_COLUMNS.index("edge")]
    edge_cross = column_cross_entropy(
        edge_column,
        outputs.edge,
        torch.where(observed, edge_target, 0.0),
        policy.geometry_centres(edge_column),
        sigma,
    )
    edge = torch.where(observed, edge_cross, 0.0).sum() / observed_count

    position, heading = observation.pos, observation.heading
    first_rows = (
        take(position, query.first),
        take(heading, query.first),
        take(geometry, query.first),
    )
    second_rows = (
        take(position, query.second),
        take(heading, query.second),
        take(geometry, query.second),
    )
    forward = game.pair_geometry(*first_rows, second_rows[0], second_rows[2])
    reverse = None
    pair_total = edge.new_zeros(())
    column_totals = []
    for column, logits in zip(config.pair_columns, outputs.pairs, strict=True):
        if column.both_orders:
            if reverse is None:
                reverse = game.pair_geometry(*second_rows, first_rows[0], first_rows[2])
            values = torch.cat([forward[column.name], reverse[column.name]], 2)
            valid = torch.cat([ships.valid, ships.valid], 2)
            weight = torch.cat([ships.weight, ships.weight], 2)
            orders = 2.0
        else:
            values, valid, weight, orders = forward[column.name], ships.valid, ships.weight, 1.0
        cross = column_cross_entropy(
            column, logits, torch.where(valid, values, 0.0), policy.geometry_centres(column), sigma
        )
        column_total = (cross * weight).sum() / orders
        pair_total = pair_total + column_total
        if by_column:
            column_totals.append(column_total)
    pair = pair_total / (pair_count * len(config.pair_columns))

    field = edge.new_zeros(())
    if field_sample is not None:
        field_pos = observation.field_pos
        centre = take(field_pos, query.field_index)
        radius = take(observation.field[..., 0] * FIELD_RADIUS_SCALE, query.field_index)
        at = take(position, query.field_ship)
        distance = game.field_edge_distance(
            centre[..., 0], centre[..., 1], radius, at[..., 0], at[..., 1]
        )
        column = config.field_column
        cross = column_cross_entropy(
            column,
            outputs.fields,
            torch.where(field_sample.valid, distance, 0.0),
            policy.geometry_centres(column),
            sigma,
        )
        field = (cross * field_sample.weight).sum() / field_count
    if by_column:
        scale = pair_count * len(config.pair_columns)
        return pair, field, edge, [total / scale for total in column_totals]
    return pair, field, edge
