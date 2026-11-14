"""A model as a controller: sample from its policy, report log-prob, entropy and value."""

import torch

from bnb.contracts import Obs, Outputs
from bnb.models.heads import sample
from bnb.streams import action_uniforms


class ModelController:
    """Stateless models only for now (the carry is None)."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.reads = model.reads

    def init_carry(self, views: int, device: torch.device) -> None:
        """No carry: the model is stateless."""
        return None

    def step(self, carry: None, obs: Obs, reset: torch.Tensor, decision_key: torch.Tensor):
        """One decision for views (R, ...); see `Controller.step`."""
        logits, value = self.model(obs)
        uniforms = action_uniforms(decision_key, logits.shape[-2])
        action, log_probability, entropy = sample(logits, uniforms)
        return action, carry, Outputs(log_probability=log_probability, entropy=entropy, value=value)
