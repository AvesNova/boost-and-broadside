"""The update of both stages over a RolloutStore in micro-batches: PPO with a shrinking
teacher term (RL), or behaviour cloning of the teacher's full distribution (the BC stage)."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from lean.contract import JOINT
from lean.learn.config import LearnConfig, learning_rate
from lean.learn.geometry import draw_keys, geometry_losses, observed_rows, pair_query
from lean.learn.returns import (
    ValueSupport,
    advantage_shares,
    advantage_sums,
    fallen_team_values,
    generalised_advantages,
    join_segments,
    normalise_advantages,
    split_segments,
    usage_counts,
    usage_flags,
    usage_metrics,
    value_support,
)
from lean.learn.rollout import TEACHER_WIDTH, Rollout, own_slots, teacher_joint
from lean.learn.store import RolloutStore
from lean.model.agent import precision
from lean.model.policy import next_state_cross_entropy
from lean.model.value import decode_value, hl_gauss_targets

# What each pass reads from the store besides the carry and the observation's per-tick fields
# (the stored-target pass reads no observation).
OBSERVED = ("deal_row", "reset")
TARGET_NAMES = OBSERVED + ("reward", "done", "fallen")
STORED_TARGET_NAMES = ("own", "reward", "done", "value", "fallen")
LOSS_NAMES = OBSERVED + (
    "action",
    "log_probability",
    "teacher",
    "next_state",
    "next_valid",
    "hit_label",
    "geometry",
)
MEASURES = (
    "loss",
    "surrogate",
    "value",
    "entropy",
    "next_state",
    "hit",
    "pair",
    "field",
    "edge",
    "bc",
    "bc_kl",
    "approximate_kl",
    "clip_fraction",
)


def to_device(tensor: Tensor, device) -> Tensor:
    """A small host tensor on `device` without a host wait (through pinned memory on a GPU;
    a plain copy from pageable memory waits for the stream)."""
    device = torch.device(device)
    return (
        tensor.pin_memory().to(device, non_blocking=True)
        if device.type == "cuda"
        else tensor.to(device)
    )


def make_optimiser(config: LearnConfig, policy) -> torch.optim.Optimizer:
    """Adam without weight decay, the configured epsilon, fused (one kernel per step on the
    GPU; the fused path also skips a step through `found_inf` without a host read, which
    `update` uses for non-finite gradients). `update` sets the learning rate per step."""
    return torch.optim.Adam(
        policy.parameters(), lr=config.learning_rate, eps=config.adam_epsilon, fused=True
    )


@dataclass
class TeacherTerm:
    """RL's teacher term: weight teacher_coefficient * teacher_decay ** k at RL update k (a
    reverse warm-up), off for good once the win rate against the scripted agent reaches
    teacher_gate or the weight falls below teacher_cutoff. The win rate reads 0 until a game
    against the scripted agent has been played (the policy is assumed to lose them all)."""

    updates: int = 0
    off: bool = False

    def weight(self, config: LearnConfig, scripted_win_rate: float) -> float:
        """The term's weight for this update; advances the schedule by one update."""
        weight = config.teacher_coefficient * config.teacher_decay**self.updates
        if scripted_win_rate >= config.teacher_gate or weight < config.teacher_cutoff:
            self.off = True
        self.updates += 1
        return 0.0 if self.off else weight


def masked_sum(values: Tensor, mask: Tensor) -> Tensor:
    """Sum of `values` over the tokens where `mask` (broadcast from the left) is set."""
    weight = mask.reshape(mask.shape + (1,) * (values.dim() - mask.dim())).to(values.dtype)
    return (values * weight).sum()


def hit_loss(
    hit_logits: Tensor, label: Tensor, rows: Tensor, denominator: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """Cross-entropy (nats) of the hit head summed over the rows in `rows` that are labelled
    (label >= 0), divided by `denominator` (default: their count, so a mean), and that mask.
    hit_logits (..., HIT_CLASSES), label and rows (...) (the learner passes the observed rows,
    lean.learn.geometry.observed_rows). Other rows contribute nothing, to the loss or to the
    gradient; no labelled row gives 0."""
    labelled = rows & (label >= 0)
    log_probability = (
        torch.log_softmax(hit_logits.float(), -1)
        .gather(-1, label.clamp_min(0)[..., None])
        .squeeze(-1)
    )
    denominator = labelled.sum().clamp(min=1) if denominator is None else denominator
    return torch.where(labelled, -log_probability, 0.0).sum() / denominator, labelled


def teacher_cross_entropy(log_policy: Tensor, teacher: Tensor) -> tuple[Tensor, Tensor]:
    """Per row, the cross-entropy (nats) of the policy's joint distribution against the
    teacher's (its recorded joint distribution), and the teacher's own entropy (their
    difference is the KL from the teacher to the policy). log_policy and teacher (..., JOINT)."""
    target = teacher.float()
    cross_entropy = -(target * log_policy).sum(-1)
    entropy = -(target * torch.log(target.clamp(min=1e-30))).sum(-1)
    return cross_entropy, entropy


def minibatch_values(
    config: LearnConfig,
    policy,
    observation,
    carry: Tensor,
    reset: Tensor,
    position_offset: Tensor,
    support: ValueSupport | None = None,
) -> tuple[Tensor, Tensor]:
    """The critic's decoded values (T, B, N, LEVELS) of one batch of views, zero outside
    `own`, and the support's centres (LEVELS, BINS). `support` (returns.value_support) the
    critic's support on the device, else built here."""
    with precision(carry.device, config.bf16):
        value_logits = policy.sequence(observation, carry, reset, position_offset).value_logits
    value_logits = value_logits.float()
    if support is None:
        support = value_support(config, value_logits.shape[-1], value_logits.device)
    value = decode_value(value_logits, support.centres, support.knees)
    return value * observation.own[..., None], support.centres


def minibatch_loss(
    config: LearnConfig,
    policy,
    observation,
    carry: Tensor,
    reset: Tensor,
    position_offset: Tensor,
    action: Tensor,
    old_log_probability: Tensor,
    advantage: Tensor,
    target: Tensor,
    teacher: Tensor,
    next_target: Tensor,
    next_valid: Tensor,
    hit_label: Tensor,
    geometry: Tensor | None,
    pair_keys: Tensor | None,
    field_keys: Tensor | None,
    weights: Tensor,
    denominators: Tensor,
    support: ValueSupport | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """One micro-batch's share of its minibatch's loss (see `update`) and of its measures,
    all 0-d tensors: every term is a sum over this micro-batch's valid tokens divided by
    the whole minibatch's count, `denominators` = (own tokens, next_valid tokens, labelled
    observed tokens, observed tokens, ship pairs, ship-field pairs; RolloutStore.counts), so
    the micro-batches' losses and gradients add up to the minibatch's.
    teacher (T, B, M, TEACHER_WIDTH) the teacher's record on each view's own rows
    (Rollout.teacher; zeros when its weight is 0 and nothing was copied, `update`).
    weights = (surrogate, entropy bonus, teacher cross-entropy) coefficients. weights and
    denominators are tensors so that a compiled graph does not take each new value as a
    constant. `support` (returns.value_support) the critic's support on the device, else
    built here. geometry (T, B, N, GEOMETRY_COLUMNS) the geometry record, pair_keys and
    field_keys the pair draw (lean.learn.geometry.draw_keys); all unused without the
    policy's geometry heads."""
    own = observation.own
    observed = observed_rows(observation)
    own_count, next_count, labelled_count = denominators[:3].clamp(min=1.0).unbind()
    surrogate_weight, entropy_weight, teacher_weight = weights.unbind()
    geometry_heads = policy.config.geometry_heads
    query = ships = field_sample = None
    if geometry_heads:
        query, ships, field_sample = pair_query(
            observed,
            observation.field_pos.shape[-2],
            pair_keys,
            field_keys,
            config.pair_samples,
            config.field_pair_samples,
        )
    with precision(carry.device, config.bf16):
        outputs = policy.sequence(observation, carry, reset, position_offset, query)
    # The losses in float32.
    logits, value_logits, next_state, hit_logits = (output.float() for output in outputs[:4])
    log_policy = torch.log_softmax(logits, -1)
    log_probability = log_policy.gather(-1, action[..., None]).squeeze(-1)
    log_ratio = log_probability - old_log_probability
    ratio = log_ratio.exp()
    clipped = ratio.clamp(1.0 - config.clip, 1.0 + config.clip)
    surrogate = masked_sum(-torch.minimum(ratio * advantage, clipped * advantage), own) / own_count
    entropy = masked_sum(-(log_policy.exp() * log_policy).sum(-1), own) / own_count
    # The teacher's rows are the own rows in own_slots' order (Rollout.teacher).
    slots = own_slots(own, teacher.shape[-2])
    log_own = log_policy.gather(-2, slots[..., None].expand(*slots.shape, log_policy.shape[-1]))
    teacher_own = own.gather(-1, slots)
    teacher_cross, teacher_entropy = teacher_cross_entropy(log_own, teacher_joint(teacher))
    bc = masked_sum(teacher_cross, teacher_own) / own_count

    if support is None:
        support = value_support(config, value_logits.shape[-1], value_logits.device)
    target_distribution = hl_gauss_targets(
        target, support.centres, support.knees, config.value_sigma_ratio
    )
    cross_entropy = -(target_distribution * torch.log_softmax(value_logits, -1)).sum(-1)
    value_loss = masked_sum(cross_entropy, own) / (own_count * cross_entropy.shape[-1])

    if next_state.shape[:-1] != next_target.shape:
        raise ValueError(
            f"next-state head gives {tuple(next_state.shape)}, the recorded target is "
            f"{tuple(next_target.shape)}"
        )
    next_cross_entropy = next_state_cross_entropy(
        policy.config, next_state, next_target, policy.next_state_centres, policy.next_state_knees
    )
    next_state_loss = masked_sum(next_cross_entropy, next_valid) / (
        next_count * next_cross_entropy.shape[-1]
    )

    hit, _ = hit_loss(hit_logits, hit_label, observed, labelled_count)

    pair = field = edge = torch.zeros((), device=logits.device)
    if geometry_heads:
        pair, field, edge = geometry_losses(
            policy, outputs, observation, geometry, query, ships, field_sample, denominators[3:]
        )

    loss = (
        surrogate_weight * surrogate
        + config.value_coefficient * value_loss
        - entropy_weight * entropy
        + config.next_state_coefficient * next_state_loss
        + config.hit_coefficient * hit
        + config.pair_coefficient * pair
        + config.field_coefficient * field
        + config.edge_coefficient * edge
        + teacher_weight * bc
    )
    with torch.no_grad():
        measures = {
            "loss": loss,
            "surrogate": surrogate,
            "value": value_loss,
            "entropy": entropy,
            "next_state": next_state_loss,
            "hit": hit,
            "pair": pair,
            "field": field,
            "edge": edge,
            "bc": bc,
            "bc_kl": masked_sum(teacher_cross - teacher_entropy, teacher_own) / own_count,
            "approximate_kl": masked_sum(ratio - 1.0 - log_ratio, own) / own_count,
            "clip_fraction": masked_sum(((ratio - 1.0).abs() > config.clip).float(), own)
            / own_count,
        }
    return loss, {name: measure.detach() for name, measure in measures.items()}


@torch.no_grad()
def target_pass(
    config: LearnConfig, policy, store: RolloutStore, index: Tensor, values=minibatch_values
) -> tuple[Tensor, Tensor]:
    """Advantages and value targets of every view, into store.targets(): the critic's
    values (`values`, minibatch_values or its compiled form, over the micro-batches of
    `index`, rows of store.chain_index; None takes the values stored at collection), zero
    outside `own`, run through GAE per level (each level its own discount and lambda),
    summed over levels and normalised over the whole rollout. GAE runs over each seat's
    segments chained in time (returns.join_segments: the value after a segment's last tick
    is the next segment's first value), bootstrapped only after the last segment from the
    value stored at collection; an episode end cuts as anywhere else.
    Returns, on the device, the targets' range measures, usage_flags (4, LEVELS), and the
    advantages' tallies before scaling (returns.advantage_sums, for each level's share)."""
    levels = len(config.level_names)
    advantage, target = store.targets(levels)
    support = value_support(config, policy.config.bins, store.device)
    discounts = to_device(torch.tensor(config.per_level(config.discounts)), store.device)
    lambdas = to_device(torch.tensor(config.per_level(config.gae_lambdas)), store.device)
    names = store.observation_names + TARGET_NAMES if values is not None else STORED_TARGET_NAMES
    counts = paid = sums = None
    for batch in store.batches(names, index):
        views = batch["views"]
        if values is not None:
            observation, offset = store.observation(batch)
            value, centres = values(
                config, policy, observation, batch["carry"], batch["reset"], offset, support
            )
        else:
            value, centres = batch["value"], store.centres
        value = fallen_team_values(value, batch["own"], batch["fallen"])
        segments = store.segments
        last = views[views.numel() - views.numel() // segments :]  # each seat's last segment
        advantages, targets = generalised_advantages(
            join_segments(batch["reward"], segments),
            join_segments(value, segments),
            store.bootstrap.index_select(0, last),
            join_segments(batch["done"], segments),
            discounts,
            lambdas,
        )
        advantages, targets = (
            split_segments(advantages, segments),
            split_segments(targets, segments),
        )
        # Padding views all write the spare view, which owns nothing.
        advantage.index_copy_(1, views, advantages.sum(-1))
        target.index_copy_(1, views, targets)
        tally, part = (
            usage_counts(targets, batch["own"], centres, support.knees),
            advantage_sums(advantages, batch["own"]),
        )
        pays = (batch["reward"] != 0).flatten(0, -2).any(0)
        if counts is None:
            counts, paid, sums = tally, pays, part
        else:
            counts, paid, sums = counts + tally, paid | pays, sums + part
    advantage.copy_(normalise_advantages(advantage, store.resident["own"]))
    return usage_flags(counts, paid, config), sums


def update(
    config: LearnConfig,
    policy,
    optimiser,
    data: RolloutStore | Rollout,
    *,
    teacher_weight: float = 0.0,
    cloning: bool = False,
    step_index: int = 0,
    generator=None,
    compiled: bool = False,
) -> dict[str, float]:
    """One update on the views of `data` (a RolloutStore, or one collected Rollout). Returns
    means over all minibatches as floats.

    RL (`cloning` False):
        loss = surrogate - entropy_coefficient * entropy + teacher_weight * bc
               + value_coefficient * value + next_state_coefficient * next_state
               + hit_coefficient * hit
    The BC stage (`cloning` True): no surrogate and no entropy bonus, the teacher's
    cross-entropy with weight bc_coefficient, the other terms as in RL (the critics learn
    the stage's own returns).

    Every term is a mean over its valid tokens in the minibatch: the learner's living ships
    (`own`) for surrogate, entropy, bc and value (value also averaged over levels, so it is
    nats per softmax group); bc is the cross-entropy of the policy's 42-way distribution
    against the scripted teacher's full distribution (Rollout.teacher, the joint of its
    factors; with its weight 0 in RL the record is neither copied nor read: a zero record of
    the same shape goes into the same graph); `next_valid` rows
    and their columns for the next-state head's HL-Gauss cross-entropy (nats per column,
    lean.model.policy.next_state_cross_entropy; a head whose columns differ from the
    recorded target's raises ValueError); the hit head's cross-entropy (nats) over the `own`
    rows that have a hit label.

    A minibatch is a random set of views (one seat's whole segment each), run as
    micro-batches of config.micro_batch_views views through Policy.sequence from the stored
    carry, the last padded with the store's empty spare view; gradients accumulate over a
    minibatch's micro-batches (each term divided by its whole-minibatch denominator), then
    one Adam step: the global gradient norm clipped to config.gradient_clip, and the step
    skipped (counted in `skipped_steps`) when that norm is not finite, through the fused
    optimiser's `found_inf` (no host read). The learning rate of the k-th step of the update
    is learning_rate(config, step_index + k); `adam_steps` reports the steps attempted.

    Advantages and value targets (target_pass) come once per update from the values stored
    at collection; with config.targets_every_epoch they are recomputed from the current
    critic at the start of every epoch. Advantages are summed over levels and divided by
    their standard deviation over the update (mean kept); `advantage_share_<level>` is each
    level's share of the summed advantage's variance and `advantage_std` its scale before
    the division. The targets' range metrics (returns.usage_metrics) are averaged over the
    epochs; a warning flag reads above 0 when any epoch raised it. Each epoch's mean
    approximate KL is `approximate_kl_epoch_<k>`; `kl_warning` is 1 when one exceeds
    config.kl_warning (a monitor: nothing stops early). `entropy_fraction` is the entropy
    over ln 42. `generator` fixes the order of the data. `compiled` runs the loss (forward
    and backward) and the critic's target pass as lean.learn.compiled's graphs.

    Host reads and waits: the data order once and the wait for the store's copies (both at
    the start), the measures once at the end. Streaming adds a host wait per micro-batch
    only for a staging buffer's copy of two micro-batches before (RolloutStore.batches).
    """
    if not optimiser.defaults.get("fused"):
        raise ValueError(
            "update needs a fused optimiser (it skips non-finite steps through found_inf)"
        )
    store = data if isinstance(data, RolloutStore) else RolloutStore.of(data)
    if compiled:
        from lean.learn.compiled import graphs

        compiled_graphs = graphs(config.compile_shapes)
        loss_of, values = compiled_graphs.minibatch_loss, compiled_graphs.minibatch_values
    else:
        loss_of, values = minibatch_loss, minibatch_values
    if cloning:
        weights = (0.0, 0.0, config.bc_coefficient)
    else:
        weights = (1.0, config.entropy_coefficient, teacher_weight)
    device = store.device
    weights = to_device(torch.tensor(weights), device)
    support = value_support(config, policy.config.bins, device)
    views = store.views
    size = min(config.micro_batch_views, views)

    # Every epoch's data order, drawn on the generator's device and read in one go, with the
    # host's wait for the store's copies.
    draw_device = generator.device if generator is not None else torch.device("cpu")
    orders = torch.stack(
        [
            torch.randperm(views, generator=generator, device=draw_device)
            for _ in range(config.epochs)
        ]
    )
    store.ready()
    orders = orders.cpu()
    target_index = store.chain_index(size)
    counts = store.counts()
    loss_names = store.observation_names + LOSS_NAMES
    # The teacher's record only where its term weighs (audit finding 6): otherwise a zero
    # record of its shape (the same graph), and nothing copied.
    blank_teacher = None
    if not cloning and teacher_weight == 0.0:
        loss_names = tuple(name for name in loss_names if name != "teacher")
        blank_teacher = torch.zeros(
            (store.ticks, size, min(config.team_size_maximum, store.resident["own"].shape[-1]))
            + (TEACHER_WIDTH,),
            device=device,
        )

    totals = {
        name: torch.zeros((), device=device)
        for name in MEASURES + ("gradient_norm", "skipped_steps", "hit_labelled")
    }
    epoch_kl = torch.zeros(config.epochs, device=device)
    epoch_steps = []
    usage, shares = [], []
    steps = 0
    rate = learning_rate(config, step_index)
    for epoch in range(config.epochs):
        if config.targets_every_epoch or epoch == 0:
            flags, sums = target_pass(
                config, policy, store, target_index, values if config.targets_every_epoch else None
            )
            share, deviation = advantage_shares(sums)
            usage.append(flags)
            shares.append(torch.cat([share, deviation[None]]))
        else:
            usage.append(usage[-1])
            shares.append(shares[-1])
        advantage, target = store.advantage, store.target
        index, owners = store.batch_index(orders[epoch], size, config.minibatches)
        index_device = to_device(index, device)
        epoch_steps.append(len(set(owners)))
        for row, batch in enumerate(store.batches(loss_names, index)):
            part = owners[row]
            first, last = (
                row == 0 or owners[row - 1] != part,
                row + 1 == len(owners) or owners[row + 1] != part,
            )
            if first:
                members = index_device[
                    row : row + owners.count(part)
                ].flatten()  # a minibatch's rows are consecutive
                denominators = counts.index_select(0, members).sum(0)
            views_now = batch["views"]
            observation, offset = store.observation(batch)
            pair_keys, field_keys = None, None
            if policy.config.geometry_heads:
                ticks, width, slots = observation.key_mask.shape
                pair_keys, field_keys = draw_keys(
                    config.pair_samples,
                    config.field_pair_samples,
                    ticks,
                    width,
                    slots,
                    observation.field_pos.shape[-2],
                    generator,
                    device,
                )
            loss, measures = loss_of(
                config,
                policy,
                observation,
                batch["carry"],
                batch["reset"],
                offset,
                batch["action"],
                batch["log_probability"],
                advantage.index_select(1, views_now),
                target.index_select(1, views_now),
                batch["teacher"] if blank_teacher is None else blank_teacher,
                batch["next_state"],
                batch["next_valid"],
                batch["hit_label"],
                batch["geometry"],
                pair_keys,
                field_keys,
                weights,
                denominators,
                support,
            )
            loss.backward()
            for name, measure in measures.items():
                totals[name] += measure
            epoch_kl[epoch] += measures["approximate_kl"]
            if last:
                rate = learning_rate(config, step_index + steps)
                for group in optimiser.param_groups:
                    group["lr"] = rate
                norm = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), config.gradient_clip
                ).detach()
                finite = torch.isfinite(norm)
                optimiser.found_inf = (~finite).float()
                optimiser.step()
                optimiser.zero_grad(set_to_none=True)
                totals["gradient_norm"] += torch.where(finite, norm, 0.0)
                totals["skipped_steps"] += (~finite).float()
                totals["hit_labelled"] += denominators[2] / denominators[3].clamp(min=1.0)
                steps += 1
    optimiser.found_inf = None

    names = list(totals)
    levels = len(config.level_names)
    read = torch.cat(
        [
            torch.stack([totals[name] for name in names]),
            torch.stack(usage).mean(0).flatten(),
            torch.stack(shares).mean(0),
            epoch_kl / torch.tensor(epoch_steps, device=device).clamp(min=1),
        ]
    ).tolist()
    metrics = {
        name: total / max(1, steps) for name, total in zip(names, read[: len(names)], strict=True)
    }
    metrics["skipped_steps"] = read[names.index("skipped_steps")]
    metrics["gradient_norm"] = read[names.index("gradient_norm")] / max(
        1, steps - metrics["skipped_steps"]
    )
    position = len(names)
    rows = [read[position + row * levels : position + (row + 1) * levels] for row in range(4)]
    metrics.update(usage_metrics(rows, config.level_names))
    position += 4 * levels
    for level, name in enumerate(config.level_names):
        metrics[f"advantage_share_{name}"] = read[position + level]
    metrics["advantage_std"] = read[position + levels]
    position += levels + 1
    kl = read[position:]
    for epoch, value in enumerate(kl):
        metrics[f"approximate_kl_epoch_{epoch}"] = value
    metrics["kl_warning"] = float(max(kl) > config.kl_warning)
    metrics["entropy_fraction"] = metrics["entropy"] / math.log(JOINT)
    metrics["teacher_weight"] = config.bc_coefficient if cloning else teacher_weight
    metrics["learning_rate"] = rate
    metrics["adam_steps"] = steps
    return metrics
