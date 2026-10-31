"""A model as a controller: sample from its policy, report log-prob, entropy and value."""

import torch

from bnb.contracts import Obs, Outputs
from bnb.models.heads import sample


class ModelController:
    """Stateless models only for now (the carry is None)."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.reads = model.reads

    def init_carry(self, views: int, device: torch.device) -> None:
        return None

    def step(self, carry: None, obs: Obs, done: torch.Tensor):
        logits, value = self.model(obs)
        action, logp, entropy = sample(logits)
        return action, carry, Outputs(logp=logp, entropy=entropy, value=value)
