"""The joint policy over NUM_JOINT actions: Gumbel-max sampling, log-probabilities and entropy."""

import torch
import torch.nn.functional as F

from bnb.contracts.action import NUM_POWER, NUM_SHOOT, NUM_STICK

GUMBEL_FLOOR = 1e-20


def sample(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A Gumbel-max sample of (..., NUM_JOINT) logits: (joint int32, log-prob, entropy), each (...).

    Draws from the default generator, so a captured graph draws fresh noise on every replay.
    """
    uniform = torch.rand_like(logits).clamp_min(GUMBEL_FLOOR)
    joint = torch.argmax(logits - torch.log(-torch.log(uniform)), -1)
    log_probability, entropy = evaluate(logits, joint)
    return joint.to(torch.int32), log_probability, entropy


def tempered(logits: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    """(R, ..., NUM_JOINT) logits at each view's sampling temperature (R,): logits / T over
    the whole joint action, so every factor of it (power, stick, shoot) is tempered alike.
    Divided in float32 and returned in the logits' dtype, so T = 1 gives the logits exactly
    (and the sample drawn from them is the one drawn without a temperature)."""
    scale = temperature.reshape(temperature.shape + (1,) * (logits.dim() - temperature.dim()))
    return (logits.float() / scale).to(logits.dtype)


def evaluate(logits: torch.Tensor, joint: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(log-prob of `joint`, entropy), each (...)."""
    log_probabilities = F.log_softmax(logits, -1)
    log_probability = log_probabilities.gather(-1, joint.long()[..., None])[..., 0]
    return log_probability, -(log_probabilities.exp() * log_probabilities).sum(-1)


def factor_entropies(logits: torch.Tensor) -> torch.Tensor:
    """(..., 3): the joint entropy split by the chain rule in `bnb.contracts.action.to_joint`'s
    order, H(power), H(stick | power) and H(shoot | power, stick), from (..., NUM_JOINT)
    logits. They sum to the joint entropy, so equal coefficients on them weigh the joint."""
    log_joint = F.log_softmax(logits, -1).unflatten(-1, (NUM_POWER, NUM_STICK, NUM_SHOOT))
    log_pair = torch.logsumexp(log_joint, -1)  # (..., power, stick)
    log_power = torch.logsumexp(log_pair, -1)  # (..., power)

    def entropy(log_probabilities: torch.Tensor, axes: int) -> torch.Tensor:
        return -(log_probabilities.exp() * log_probabilities).flatten(-axes).sum(-1)

    joint, pair, power = entropy(log_joint, 3), entropy(log_pair, 2), entropy(log_power, 1)
    return torch.stack([power, pair - power, joint - pair], -1)
