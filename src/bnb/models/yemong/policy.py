"""The 42-way joint policy: Gumbel-max sampling, log-probabilities and entropy."""

import torch
import torch.nn.functional as F

GUMBEL_FLOOR = 1e-20


def sample(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A Gumbel-max sample of (..., 42) logits: (joint int32, log-prob, entropy), each (...).

    Draws from the default generator, so a captured graph draws fresh noise on every replay.
    """
    uniform = torch.rand_like(logits).clamp_min(GUMBEL_FLOOR)
    joint = torch.argmax(logits - torch.log(-torch.log(uniform)), -1)
    logp, entropy = evaluate(logits, joint)
    return joint.to(torch.int32), logp, entropy


def evaluate(logits: torch.Tensor, joint: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(log-prob of `joint`, entropy), each (...)."""
    log_probabilities = F.log_softmax(logits, -1)
    logp = log_probabilities.gather(-1, joint.long()[..., None])[..., 0]
    return logp, -(log_probabilities.exp() * log_probabilities).sum(-1)
