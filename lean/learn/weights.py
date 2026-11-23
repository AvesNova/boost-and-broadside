"""Weight averages kept beside the live weights, and checkpoints.

The averages are side copies: training reads and steps the live weights only, and the
averages are updated from them once per update, under no_grad, into their own tensors.

- EMA: ema = decay * ema + (1 - decay) * live, from the stage's starting weights
  (LearnConfig.ema_decay per update, v2's 0.95: a half-life of about 13.5 updates).
- Uniform: the plain mean of the live weights after every update that ends at or past the
  learning-rate warm-up (LearnConfig.warmup_steps Adam steps), the running mean
  uniform += (live - uniform) / n. Empty (count 0) until then.

Buffers are not averaged; a policy loaded with averaged weights keeps its own buffers.
"""

import dataclasses

import torch
from torch import Tensor, nn

from lean.learn.config import LearnConfig


class WeightAverages:
    """The EMA and the uniform average of a policy's parameters (float32, on its device)."""

    def __init__(self, policy: nn.Module, config: LearnConfig):
        self.decay = config.ema_decay
        self.warmup_steps = config.warmup_steps
        self.ema = {name: parameter.detach().float().clone() for name, parameter in policy.named_parameters()}
        self.uniform = {name: torch.zeros_like(value) for name, value in self.ema.items()}
        self.count = 0  # updates in the uniform average

    @torch.no_grad()
    def update(self, policy: nn.Module, adam_steps: int) -> None:
        """Fold in the live weights after an update; `adam_steps` the stage's Adam steps so far."""
        parameters = dict(policy.named_parameters())
        names = list(self.ema)
        live = [parameters[name].detach().float() for name in names]
        torch._foreach_lerp_([self.ema[name] for name in names], live, 1.0 - self.decay)
        if adam_steps >= self.warmup_steps:
            self.count += 1
            torch._foreach_lerp_([self.uniform[name] for name in names], live, 1.0 / self.count)

    def weights(self, which: str, policy: nn.Module) -> dict[str, Tensor]:
        """A state dict for `policy`: its own ("live"), or with the "ema" or "uniform"
        parameters in place (the uniform falls back to the live weights while empty)."""
        checkpoint = {"live": policy.state_dict(), "averages": self.state()}
        return {name: value.detach().clone() for name, value in policy_weights(checkpoint, which).items()}

    def state(self) -> dict:
        return {"ema": self.ema, "uniform": self.uniform, "count": self.count}

    def load(self, state: dict) -> None:
        self.ema = {name: value.clone() for name, value in state["ema"].items()}
        self.uniform = {name: value.clone() for name, value in state["uniform"].items()}
        self.count = state["count"]


def save_checkpoint(path, policy: nn.Module, averages: WeightAverages, config: LearnConfig, **extra) -> None:
    """The live weights, both averages, the model's and the learner's configs, and `extra`
    (stage, updates, steps, the teacher term, ...) in one file."""
    torch.save({
        "model_config": policy.config, "learn_config": dataclasses.asdict(config),
        "live": policy.state_dict(), "averages": averages.state(), **extra,
    }, path)


def load_checkpoint(path, device=None) -> dict:
    """A checkpoint written by save_checkpoint (this project's own files: full unpickling)."""
    return torch.load(path, map_location=device, weights_only=False)


def policy_weights(checkpoint: dict, which: str) -> dict[str, Tensor]:
    """The state dict of `which` ("live", "ema" or "uniform") from a loaded checkpoint (the
    uniform falls back to the live weights while empty)."""
    state = dict(checkpoint["live"])
    averages = checkpoint["averages"]
    if which == "live" or (which == "uniform" and averages["count"] == 0):
        return state
    for name, value in averages[which].items():
        state[name] = value.to(state[name].dtype)
    return state
