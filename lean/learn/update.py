"""The PPO update."""

import torch
from torch import Tensor

from lean.learn.collect import Rollout, map_fields
from lean.learn.config import LearnConfig, bc_weight, learning_rate
from lean.learn.returns import expected_value, generalised_advantages, hl_gauss, normalise_advantages, value_centres


def make_optimiser(config: LearnConfig, policy) -> torch.optim.Optimizer:
    """Adam with the configured rate and epsilon; `update` sets the warm-up rate itself."""
    return torch.optim.Adam(policy.parameters(), lr=config.learning_rate, eps=config.adam_epsilon)


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Mean of `values` over the tokens where `mask` (broadcast from the left) is set."""
    weight = mask.reshape(mask.shape + (1,) * (values.dim() - mask.dim())).to(values.dtype).expand_as(values)
    return (values * weight).sum() / weight.sum().clamp(min=1.0)


def environment_batches(envs: int, minibatches: int, generator) -> list[Tensor]:
    """A random split of the environments into min(minibatches, E) index sets."""
    device = generator.device if generator is not None else "cpu"
    order = torch.randperm(envs, generator=generator, device=device)
    return [part for part in torch.tensor_split(order, min(minibatches, envs)) if part.numel() > 0]


@torch.no_grad()
def fresh_targets(config: LearnConfig, policy, rollout: Rollout, batches: list[Tensor]) -> tuple[Tensor, Tensor]:
    """Advantages (T, E, N) and value targets (T, E, N, LEVELS) from the current critic.

    Values are recomputed with Policy.sequence, zeroed outside `own`, run through GAE per
    level and the advantages normalised over the whole rollout. The bootstrap value is
    the one stored at collection.
    """
    own = rollout.observation.own
    value = torch.zeros_like(rollout.reward)
    for index in batches:
        index = index.to(own.device)
        observation = map_fields(rollout.observation, lambda tensor: tensor[:, index])
        _, value_logits, _, _ = policy.sequence(observation, rollout.initial_carry[index], rollout.reset[:, index])
        centres = value_centres(config, value_logits.shape[-1], value_logits.device)
        value[:, index] = expected_value(value_logits, centres) * own[:, index, :, None]
    discounts = torch.tensor(config.discounts, dtype=value.dtype, device=value.device)
    advantage, target = generalised_advantages(
        rollout.reward, value, rollout.bootstrap_value, rollout.done, discounts, config.gae_lambda
    )
    return normalise_advantages(advantage, own, config), target


def update(
    config: LearnConfig, policy, optimiser, rollout: Rollout,
    *, scripted_win_rate: float = 0.0, update_index: int = 0, generator=None,
) -> dict[str, float]:
    """One PPO update on one rollout. Returns means over all minibatches as floats.

    loss = surrogate + value_coefficient * value - entropy_coefficient * entropy
           + next_state_coefficient * next_state + bc_weight(scripted_win_rate) * bc

    Every term is a mean over its valid tokens: the learner's living ships (`own`) for
    surrogate, entropy, BC and value (value also averaged over levels, so it is nats per
    softmax group); `next_valid` rows and their features for the next-state squared error.
    A minibatch is a set of whole environments with the full segment, run through
    Policy.sequence from the stored carry. Advantages and value targets are recomputed
    from the current weights at the start of every epoch. `generator` fixes the order of
    the data; `update_index` sets the warm-up learning rate.
    """
    rate = learning_rate(config, update_index)
    for group in optimiser.param_groups:
        group["lr"] = rate
    bc_coefficient = bc_weight(config, scripted_win_rate)
    envs = rollout.done.shape[1]
    totals: dict[str, float] = {}
    steps = 0
    for _ in range(config.epochs):
        batches = environment_batches(envs, config.minibatches, generator)
        advantage, target = fresh_targets(config, policy, rollout, batches)
        for index in batches:
            index = index.to(rollout.done.device)
            observation = map_fields(rollout.observation, lambda tensor: tensor[:, index])
            own = observation.own
            logits, value_logits, next_state, _ = policy.sequence(
                observation, rollout.initial_carry[index], rollout.reset[:, index]
            )
            log_policy = torch.log_softmax(logits, -1)
            log_probability = log_policy.gather(-1, rollout.action[:, index, :, None]).squeeze(-1)
            log_ratio = log_probability - rollout.log_probability[:, index]
            ratio = log_ratio.exp()
            batch_advantage = advantage[:, index]
            clipped = ratio.clamp(1.0 - config.clip, 1.0 + config.clip)
            surrogate = masked_mean(-torch.minimum(ratio * batch_advantage, clipped * batch_advantage), own)
            entropy = masked_mean(-(log_policy.exp() * log_policy).sum(-1), own)
            bc = masked_mean(-log_policy.gather(-1, rollout.teacher[:, index, :, None]).squeeze(-1), own)

            centres = value_centres(config, value_logits.shape[-1], value_logits.device)
            target_distribution = hl_gauss(target[:, index], centres, config.value_sigma_ratio)
            value_loss = masked_mean(-(target_distribution * torch.log_softmax(value_logits, -1)).sum(-1), own)

            next_ship = rollout.next_ship[:, index]
            if next_state.shape[-1] == next_ship.shape[-1]:
                change = next_ship - observation.ship
                next_state_loss = masked_mean((next_state - change) ** 2, rollout.next_valid[:, index])
            else:
                next_state_loss = logits.new_zeros(())

            loss = (
                surrogate
                + config.value_coefficient * value_loss
                - config.entropy_coefficient * entropy
                + config.next_state_coefficient * next_state_loss
                + bc_coefficient * bc
            )
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), config.gradient_clip)
            optimiser.step()

            with torch.no_grad():
                measures = {
                    "loss": loss, "surrogate": surrogate, "value": value_loss, "entropy": entropy,
                    "next_state": next_state_loss, "bc": bc, "gradient_norm": gradient_norm,
                    "approximate_kl": masked_mean(ratio - 1.0 - log_ratio, own),
                    "clip_fraction": masked_mean(((ratio - 1.0).abs() > config.clip).float(), own),
                }
            for name, measure in measures.items():
                totals[name] = totals.get(name, 0.0) + float(measure)
            steps += 1
    metrics = {name: total / max(1, steps) for name, total in totals.items()}
    metrics["bc_weight"] = bc_coefficient
    metrics["learning_rate"] = rate
    return metrics
