"""The PPO update, over a RolloutStore in micro-batches."""

import torch
from torch import Tensor

from lean.learn.config import LearnConfig, bc_weight, learning_rate, policy_precision
from lean.learn.returns import (
    generalised_advantages,
    join_segments,
    normalise_advantages,
    split_segments,
    usage_counts,
    usage_flags,
    usage_metrics,
    value_centres,
)
from lean.learn.rollout import Rollout
from lean.learn.store import RolloutStore
from lean.model.policy import next_state_cross_entropy
from lean.model.value import decode_value, hl_gauss_targets

# What each pass reads from the store besides the carry and the observation's per-tick fields
# (the stored-target pass reads no observation).
OBSERVED = ("deal_row", "reset")
TARGET_NAMES = OBSERVED + ("reward", "done")
STORED_TARGET_NAMES = ("own", "reward", "done", "value")
LOSS_NAMES = OBSERVED + ("action", "log_probability", "teacher", "next_state", "next_valid", "hit_label")
MEASURES = ("loss", "surrogate", "value", "entropy", "next_state", "hit", "bc", "approximate_kl", "clip_fraction")


def to_device(tensor: Tensor, device) -> Tensor:
    """A small host tensor on `device` without a host wait (through pinned memory on a GPU;
    a plain copy from pageable memory waits for the stream)."""
    device = torch.device(device)
    return tensor.pin_memory().to(device, non_blocking=True) if device.type == "cuda" else tensor.to(device)


def make_optimiser(config: LearnConfig, policy) -> torch.optim.Optimizer:
    """Adam with the configured rate and epsilon; `update` sets the warm-up rate itself."""
    return torch.optim.Adam(policy.parameters(), lr=config.learning_rate, eps=config.adam_epsilon)


def masked_sum(values: Tensor, mask: Tensor) -> Tensor:
    """Sum of `values` over the tokens where `mask` (broadcast from the left) is set."""
    weight = mask.reshape(mask.shape + (1,) * (values.dim() - mask.dim())).to(values.dtype)
    return (values * weight).sum()


def hit_loss(hit_logits: Tensor, label: Tensor, own: Tensor, denominator: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Cross-entropy (nats) of the hit head summed over the rows that are `own` and labelled
    (label >= 0), divided by `denominator` (default: their count, so a mean), and that mask.
    hit_logits (..., HIT_CLASSES), label and own (...). Other rows contribute nothing, to
    the loss or to the gradient; no labelled row gives 0."""
    labelled = own & (label >= 0)
    log_probability = torch.log_softmax(hit_logits.float(), -1).gather(-1, label.clamp_min(0)[..., None]).squeeze(-1)
    denominator = labelled.sum().clamp(min=1) if denominator is None else denominator
    return torch.where(labelled, -log_probability, 0.0).sum() / denominator, labelled


def minibatch_values(
    config: LearnConfig, policy, observation, carry: Tensor, reset: Tensor, position_offset: Tensor
) -> tuple[Tensor, Tensor]:
    """The critic's decoded values (T, B, N, LEVELS) of one batch of views, zero outside
    `own`, and the support's centres (LEVELS, BINS)."""
    with policy_precision(config, carry.device):
        _, value_logits, _, _, _ = policy.sequence(observation, carry, reset, position_offset)
    value_logits = value_logits.float()
    centres = value_centres(config, value_logits.shape[-1], value_logits.device)
    return decode_value(value_logits, centres) * observation.own[..., None], centres


def minibatch_loss(
    config: LearnConfig, policy, observation, carry: Tensor, reset: Tensor, position_offset: Tensor, action: Tensor,
    old_log_probability: Tensor, advantage: Tensor, target: Tensor, teacher: Tensor, next_target: Tensor,
    next_valid: Tensor, hit_label: Tensor, bc_coefficient: Tensor, denominators: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    """One micro-batch's share of its minibatch's loss (see `update`) and of its measures,
    all 0-d tensors: every term is a sum over this micro-batch's valid tokens divided by
    the whole minibatch's count, `denominators` = (own tokens, next_valid tokens, labelled
    own tokens), so the micro-batches' losses and gradients add up to the minibatch's.
    bc_coefficient and denominators are tensors so that a compiled graph does not take each
    new value as a constant."""
    own = observation.own
    own_count, next_count, labelled_count = denominators.clamp(min=1.0).unbind()
    with policy_precision(config, carry.device):
        outputs = policy.sequence(observation, carry, reset, position_offset)
    # The losses in float32.
    logits, value_logits, next_state, hit_logits = (output.float() for output in outputs[:4])
    log_policy = torch.log_softmax(logits, -1)
    log_probability = log_policy.gather(-1, action[..., None]).squeeze(-1)
    log_ratio = log_probability - old_log_probability
    ratio = log_ratio.exp()
    clipped = ratio.clamp(1.0 - config.clip, 1.0 + config.clip)
    surrogate = masked_sum(-torch.minimum(ratio * advantage, clipped * advantage), own) / own_count
    entropy = masked_sum(-(log_policy.exp() * log_policy).sum(-1), own) / own_count
    bc = masked_sum(-log_policy.gather(-1, teacher[..., None]).squeeze(-1), own) / own_count

    centres = value_centres(config, value_logits.shape[-1], value_logits.device)
    target_distribution = hl_gauss_targets(target, centres, config.value_sigma_ratio)
    cross_entropy = -(target_distribution * torch.log_softmax(value_logits, -1)).sum(-1)
    value_loss = masked_sum(cross_entropy, own) / (own_count * cross_entropy.shape[-1])

    if next_state.shape[:-1] != next_target.shape:
        raise ValueError(
            f"next-state head gives {tuple(next_state.shape)}, the recorded target is {tuple(next_target.shape)}"
        )
    next_cross_entropy = next_state_cross_entropy(policy.config, next_state, next_target)
    next_state_loss = masked_sum(next_cross_entropy, next_valid) / (next_count * next_cross_entropy.shape[-1])

    hit, _ = hit_loss(hit_logits, hit_label, own, labelled_count)

    loss = (
        surrogate
        + config.value_coefficient * value_loss
        - config.entropy_coefficient * entropy
        + config.next_state_coefficient * next_state_loss
        + config.hit_coefficient * hit
        + bc_coefficient * bc
    )
    with torch.no_grad():
        measures = {
            "loss": loss, "surrogate": surrogate, "value": value_loss, "entropy": entropy,
            "next_state": next_state_loss, "hit": hit, "bc": bc,
            "approximate_kl": masked_sum(ratio - 1.0 - log_ratio, own) / own_count,
            "clip_fraction": masked_sum(((ratio - 1.0).abs() > config.clip).float(), own) / own_count,
        }
    return loss, {name: measure.detach() for name, measure in measures.items()}


@torch.no_grad()
def target_pass(config: LearnConfig, policy, store: RolloutStore, index: Tensor, values=minibatch_values) -> Tensor:
    """Advantages and value targets of every view, into store.targets(): the critic's
    values (`values`, minibatch_values or its compiled form, over the micro-batches of
    `index`, rows of store.chain_index; None takes the values stored at collection), zero
    outside `own`, run through GAE per level (each level its own discount and lambda),
    summed over levels and normalised over the whole rollout. GAE runs over each seat's
    segments chained in time (returns.join_segments: the value after a segment's last tick
    is the next segment's first value), bootstrapped only after the last segment from the
    value stored at collection; an episode end cuts as anywhere else.
    Returns the targets' range measures, usage_flags (4, LEVELS), on the device."""
    levels = len(config.discounts)
    advantage, target = store.targets(levels)
    discounts = to_device(torch.tensor(config.discounts), store.device)
    lambdas = to_device(torch.tensor(config.gae_lambdas), store.device)
    names = store.observation_names + TARGET_NAMES if values is not None else STORED_TARGET_NAMES
    counts = paid = None
    for batch in store.batches(names, index):
        views = batch["views"]
        if values is not None:
            observation, offset = store.observation(batch)
            value, centres = values(config, policy, observation, batch["carry"], batch["reset"], offset)
        else:
            value, centres = batch["value"], store.centres
        segments = store.segments
        last = views[views.numel() - views.numel() // segments:]  # each seat's last segment
        advantages, targets = generalised_advantages(
            join_segments(batch["reward"], segments), join_segments(value, segments),
            store.bootstrap.index_select(0, last), join_segments(batch["done"], segments), discounts, lambdas,
        )
        advantages, targets = split_segments(advantages, segments), split_segments(targets, segments)
        # Padding views all write the spare view, which owns nothing.
        advantage.index_copy_(1, views, advantages.sum(-1))
        target.index_copy_(1, views, targets)
        tally = usage_counts(targets, batch["own"], centres)
        pays = (batch["reward"] != 0).flatten(0, -2).any(0)
        counts, paid = (tally, pays) if counts is None else (counts + tally, paid | pays)
    advantage.copy_(normalise_advantages(advantage, store.resident["own"], config))
    return usage_flags(counts, paid, config)


def update(
    config: LearnConfig, policy, optimiser, data: RolloutStore | Rollout,
    *, scripted_win_rate: float = 0.0, update_index: int = 0, generator=None, compiled: bool = False,
) -> dict[str, float]:
    """One PPO update on the views of `data` (a RolloutStore, or one collected Rollout).
    Returns means over all minibatches as floats.

    loss = surrogate + value_coefficient * value - entropy_coefficient * entropy
           + next_state_coefficient * next_state + hit_coefficient * hit
           + bc_weight(scripted_win_rate) * bc

    Every term is a mean over its valid tokens in the minibatch: the learner's living ships
    (`own`) for surrogate, entropy, BC and value (value also averaged over levels, so it is
    nats per softmax group); `next_valid` rows and their columns for the next-state head's
    HL-Gauss cross-entropy against the target recorded at collection (nats per column,
    lean.model.policy.next_state_cross_entropy); the hit head's cross-entropy (nats)
    over the `own` rows that have a hit label (rows without one contribute nothing). A
    next-state head whose columns differ from the recorded target's raises ValueError.

    A minibatch is a random set of views (one seat's whole segment each), run as
    micro-batches of config.micro_batch_views views through Policy.sequence from the stored
    carry, the last padded with the store's empty spare view; gradients accumulate over a
    minibatch's micro-batches (each term divided by its whole-minibatch denominator, so the
    sum is the minibatch's gradient), then one clipped Adam step. Advantages and value
    targets (target_pass) come once per update from the values stored at collection; with
    config.targets_every_epoch they are recomputed from the current critic at the start of
    every epoch (a streamed forward pass over the rollout, in micro-batches of whole seats
    so that each seat's segments are chained). The targets' range metrics
    (returns.support_usage) are averaged over the epochs; a warning flag reads above 0 when any epoch raised it. `generator`
    fixes the order of the data; `update_index` sets the warm-up learning rate. `compiled`
    runs the loss (forward and backward) and the critic's target pass as
    lean.learn.compiled's graphs; Adam and the clipping stay eager.

    Host reads and waits: the data order once and the wait for the store's copies (both at
    the start), the measures once at the end; the "percentile" normaliser reads a count per
    target pass. Streaming adds a host wait per micro-batch only for a staging buffer's
    copy of two micro-batches before (RolloutStore.batches).
    """
    store = data if isinstance(data, RolloutStore) else RolloutStore.of(data)
    if compiled:
        from lean.learn.compiled import graphs

        loss_of, values = graphs().minibatch_loss, graphs().minibatch_values
    else:
        loss_of, values = minibatch_loss, minibatch_values
    rate = learning_rate(config, update_index)
    for group in optimiser.param_groups:
        group["lr"] = rate
    bc_coefficient = bc_weight(config, scripted_win_rate)
    device = store.device
    bc_tensor = to_device(torch.tensor(bc_coefficient), device)
    views = store.views
    size = min(config.micro_batch_views, views)

    # Every epoch's data order, drawn on the generator's device and read in one go, with the
    # host's wait for the store's copies.
    draw_device = generator.device if generator is not None else torch.device("cpu")
    orders = torch.stack([torch.randperm(views, generator=generator, device=draw_device) for _ in range(config.epochs)])
    store.ready()
    orders = orders.cpu()
    target_index = store.chain_index(size)
    counts = store.counts()
    loss_names = store.observation_names + LOSS_NAMES

    totals = {name: torch.zeros((), device=device) for name in MEASURES + ("gradient_norm", "hit_labelled")}
    usage = []
    steps = 0
    for epoch in range(config.epochs):
        if config.targets_every_epoch or epoch == 0:
            usage.append(target_pass(config, policy, store, target_index, values if config.targets_every_epoch else None))
        else:
            usage.append(usage[-1])
        advantage, target = store.advantage, store.target
        index, owners = store.batch_index(orders[epoch], size, config.minibatches)
        index_device = to_device(index, device)
        for row, batch in enumerate(store.batches(loss_names, index)):
            part = owners[row]
            first, last = row == 0 or owners[row - 1] != part, row + 1 == len(owners) or owners[row + 1] != part
            if first:
                members = index_device[row:row + owners.count(part)].flatten()  # a minibatch's rows are consecutive
                denominators = counts.index_select(0, members).sum(0)
            views_now = batch["views"]
            observation, offset = store.observation(batch)
            loss, measures = loss_of(
                config, policy, observation, batch["carry"], batch["reset"], offset, batch["action"],
                batch["log_probability"], advantage.index_select(1, views_now), target.index_select(1, views_now),
                batch["teacher"], batch["next_state"], batch["next_valid"], batch["hit_label"], bc_tensor, denominators,
            )
            loss.backward()
            for name, measure in measures.items():
                totals[name] += measure
            if last:
                totals["gradient_norm"] += torch.nn.utils.clip_grad_norm_(policy.parameters(), config.gradient_clip).detach()
                totals["hit_labelled"] += denominators[2] / denominators[0].clamp(min=1.0)
                optimiser.step()
                optimiser.zero_grad(set_to_none=True)
                steps += 1
    names = list(totals)
    read = torch.cat([torch.stack([totals[name] for name in names]), torch.stack(usage).mean(0).flatten()]).tolist()
    metrics = {name: total / max(1, steps) for name, total in zip(names, read[:len(names)], strict=True)}
    levels = len(config.discounts)
    rows = [read[len(names) + row * levels:len(names) + (row + 1) * levels] for row in range(4)]
    metrics.update(usage_metrics(rows))
    metrics["bc_weight"] = bc_coefficient
    metrics["learning_rate"] = rate
    return metrics
