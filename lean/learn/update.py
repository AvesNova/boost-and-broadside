"""The PPO update."""

import torch
from torch import Tensor

from lean.learn.collect import Rollout, index_carry, map_fields
from lean.learn.config import LearnConfig, bc_weight, learning_rate, policy_precision
from lean.learn.returns import (
    generalised_advantages,
    normalise_advantages,
    support_usage,
    value_centres,
)
from lean.model.value import decode_value, hl_gauss_targets


def make_optimiser(config: LearnConfig, policy) -> torch.optim.Optimizer:
    """Adam with the configured rate and epsilon; `update` sets the warm-up rate itself."""
    return torch.optim.Adam(policy.parameters(), lr=config.learning_rate, eps=config.adam_epsilon)


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Mean of `values` over the tokens where `mask` (broadcast from the left) is set."""
    weight = mask.reshape(mask.shape + (1,) * (values.dim() - mask.dim())).to(values.dtype).expand_as(values)
    return (values * weight).sum() / weight.sum().clamp(min=1.0)


def hit_loss(hit_logits: Tensor, label: Tensor, own: Tensor) -> tuple[Tensor, Tensor]:
    """Cross-entropy (nats) of the hit head over the rows that are `own` and labelled
    (label >= 0), and that mask. hit_logits (..., HIT_CLASSES), label and own (...). Other
    rows contribute nothing, to the loss or to the gradient; no labelled row gives 0."""
    labelled = own & (label >= 0)
    log_probability = torch.log_softmax(hit_logits.float(), -1).gather(-1, label.clamp_min(0)[..., None]).squeeze(-1)
    loss = torch.where(labelled, -log_probability, 0.0).sum() / labelled.sum().clamp(min=1)
    return loss, labelled


def environment_batches(envs: int, minibatches: int, generator) -> list[Tensor]:
    """A random split of the environments into min(minibatches, E) index sets."""
    device = generator.device if generator is not None else "cpu"
    order = torch.randperm(envs, generator=generator, device=device)
    return [part for part in torch.tensor_split(order, min(minibatches, envs)) if part.numel() > 0]


def minibatch_values(
    config: LearnConfig, policy, observation, carry: Tensor, reset: Tensor, position_offset: Tensor
) -> tuple[Tensor, Tensor]:
    """The critic's decoded values (T, B, N, LEVELS) of one minibatch, zero outside `own`,
    and the support's centres (LEVELS, BINS)."""
    with policy_precision(config, carry.device):
        _, value_logits, _, _, _ = policy.sequence(observation, carry, reset, position_offset)
    value_logits = value_logits.float()
    centres = value_centres(config, value_logits.shape[-1], value_logits.device)
    return decode_value(value_logits, centres) * observation.own[..., None], centres


@torch.no_grad()
def fresh_targets(config: LearnConfig, policy, rollout: Rollout, batches: list[Tensor], values=minibatch_values) -> tuple[Tensor, Tensor, dict]:
    """Advantages (T, E, N), value targets (T, E, N, LEVELS) from the current critic, and
    the targets' range metrics (returns.support_usage).

    Values are recomputed with Policy.sequence (`values` is minibatch_values or its
    compiled form), zeroed outside `own`, run through GAE per level (each level its own
    discount and lambda) and the advantages normalised over the whole rollout. The
    bootstrap value is the one stored at collection.
    """
    own = rollout.observation.own
    value = torch.zeros_like(rollout.reward)
    for index in batches:
        index = index.to(own.device)
        observation = map_fields(rollout.observation, lambda tensor: tensor[:, index])
        value[:, index], centres = values(
            config, policy, observation, index_carry(rollout.initial_carry, index), rollout.reset[:, index],
            rollout.position_offset[:, index],
        )
    return targets_from(config, rollout, value, centres)


@torch.no_grad()
def targets_from(config: LearnConfig, rollout: Rollout, value: Tensor, centres: Tensor) -> tuple[Tensor, Tensor, dict]:
    """Normalised advantages, value targets and range metrics (see fresh_targets) from
    the values `value` (T, E, N, LEVELS), zero outside `own`."""
    own = rollout.observation.own
    discounts = torch.tensor(config.discounts, dtype=value.dtype, device=value.device)
    lambdas = torch.tensor(config.gae_lambdas, dtype=value.dtype, device=value.device)
    advantage, target = generalised_advantages(rollout.reward, value, rollout.bootstrap_value, rollout.done, discounts, lambdas)
    usage = support_usage(target, rollout.reward, own, centres, config)
    return normalise_advantages(advantage, own, config), target, usage


def minibatch_loss(
    config: LearnConfig, policy, observation, carry: Tensor, reset: Tensor, position_offset: Tensor, action: Tensor,
    old_log_probability: Tensor, advantage: Tensor, target: Tensor, teacher: Tensor, next_target: Tensor,
    next_valid: Tensor, hit_label: Tensor, bc_coefficient: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    """The loss of one minibatch (see `update`) and its measures, all 0-d tensors.
    bc_coefficient is a 0-d tensor so that a compiled graph does not take each new value
    as a constant."""
    own = observation.own
    with policy_precision(config, carry.device):
        outputs = policy.sequence(observation, carry, reset, position_offset)
    # The losses in float32.
    logits, value_logits, next_state, hit_logits = (output.float() for output in outputs[:4])
    log_policy = torch.log_softmax(logits, -1)
    log_probability = log_policy.gather(-1, action[..., None]).squeeze(-1)
    log_ratio = log_probability - old_log_probability
    ratio = log_ratio.exp()
    clipped = ratio.clamp(1.0 - config.clip, 1.0 + config.clip)
    surrogate = masked_mean(-torch.minimum(ratio * advantage, clipped * advantage), own)
    entropy = masked_mean(-(log_policy.exp() * log_policy).sum(-1), own)
    bc = masked_mean(-log_policy.gather(-1, teacher[..., None]).squeeze(-1), own)

    centres = value_centres(config, value_logits.shape[-1], value_logits.device)
    target_distribution = hl_gauss_targets(target, centres, config.value_sigma_ratio)
    value_loss = masked_mean(-(target_distribution * torch.log_softmax(value_logits, -1)).sum(-1), own)

    if next_state.shape != next_target.shape:
        raise ValueError(
            f"next-state head gives {tuple(next_state.shape)}, the recorded target is {tuple(next_target.shape)}"
        )
    next_state_loss = masked_mean((next_state.float() - next_target) ** 2, next_valid)

    hit, labelled = hit_loss(hit_logits, hit_label, own)

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
            "next_state": next_state_loss, "hit": hit,
            "hit_labelled": labelled.float().sum() / own.float().sum().clamp(min=1.0), "bc": bc,
            "approximate_kl": masked_mean(ratio - 1.0 - log_ratio, own),
            "clip_fraction": masked_mean(((ratio - 1.0).abs() > config.clip).float(), own),
        }
    return loss, {name: measure.detach() for name, measure in measures.items()}


def update(
    config: LearnConfig, policy, optimiser, rollout: Rollout,
    *, scripted_win_rate: float = 0.0, update_index: int = 0, generator=None, compiled: bool = False,
) -> dict[str, float]:
    """One PPO update on one rollout. Returns means over all minibatches as floats.

    loss = surrogate + value_coefficient * value - entropy_coefficient * entropy
           + next_state_coefficient * next_state + hit_coefficient * hit
           + bc_weight(scripted_win_rate) * bc

    Every term is a mean over its valid tokens: the learner's living ships (`own`) for
    surrogate, entropy, BC and value (value also averaged over levels, so it is nats per
    softmax group); `next_valid` rows and their columns for the next-state squared error
    against the target recorded at collection; the hit head's cross-entropy (nats) over
    the `own` rows that have a hit label (rows without one contribute nothing). A
    next-state head whose width differs from the recorded target raises ValueError.
    The value targets' range metrics (returns.support_usage) are averaged over the epochs;
    a warning flag reads above 0 when any epoch raised it.
    A minibatch is a set of whole environments with the full segment, run through
    Policy.sequence from the stored carry. Advantages and value targets are recomputed
    from the current weights at the start of every epoch. `generator` fixes the order of
    the data; `update_index` sets the warm-up learning rate. With
    config.targets_every_epoch False they come once from the values stored at collection.
    `compiled` runs the loss
    (forward and backward) and the critic's target pass as lean.learn.compiled's graphs;
    Adam and the clipping stay eager. The measures stay on the device until the end: one
    read-back per update.
    """
    if compiled:
        from lean.learn import compiled as graphs

        loss_of, values = graphs.minibatch_loss, graphs.minibatch_values
    else:
        loss_of, values = minibatch_loss, minibatch_values
    rate = learning_rate(config, update_index)
    for group in optimiser.param_groups:
        group["lr"] = rate
    bc_coefficient = bc_weight(config, scripted_win_rate)
    device = rollout.done.device
    bc_tensor = torch.tensor(bc_coefficient, device=device)
    envs = rollout.done.shape[1]
    totals: dict[str, Tensor] = {}
    range_totals: dict[str, float] = {}
    steps = 0
    stored = None
    for _ in range(config.epochs):
        batches = environment_batches(envs, config.minibatches, generator)
        if config.targets_every_epoch:
            advantage, target, usage = fresh_targets(config, policy, rollout, batches, values)
        elif stored is None:
            stored = advantage, target, usage = targets_from(config, rollout, rollout.value, rollout.value_centres)
        for name, measure in usage.items():
            range_totals[name] = range_totals.get(name, 0.0) + measure
        for index in batches:
            index = index.to(device)
            loss, measures = loss_of(
                config, policy, map_fields(rollout.observation, lambda tensor: tensor[:, index]),
                index_carry(rollout.initial_carry, index), rollout.reset[:, index], rollout.position_offset[:, index],
                rollout.action[:, index], rollout.log_probability[:, index], advantage[:, index], target[:, index],
                rollout.teacher[:, index], rollout.next_state[:, index], rollout.next_valid[:, index],
                rollout.hit_label[:, index], bc_tensor,
            )
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            measures["gradient_norm"] = torch.nn.utils.clip_grad_norm_(policy.parameters(), config.gradient_clip).detach()
            optimiser.step()
            for name, measure in measures.items():
                totals[name] = totals[name] + measure if name in totals else measure
            steps += 1
    names = list(totals)
    values_read = torch.stack([totals[name].float() for name in names]).tolist() if names else []
    metrics = {name: total / max(1, steps) for name, total in zip(names, values_read, strict=True)}
    metrics.update({name: total / config.epochs for name, total in range_totals.items()})
    metrics["bc_weight"] = bc_coefficient
    metrics["learning_rate"] = rate
    return metrics
