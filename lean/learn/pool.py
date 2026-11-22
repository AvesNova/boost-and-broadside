"""A small pool of past weights for self-play."""

import copy

import torch


class SnapshotPool:
    """Keeps up to `capacity` copies of the policy's weights, oldest dropped first.

    `maybe_add` stores the current weights when update_index is a multiple of `every`.
    `sample` loads one stored set, chosen uniformly, into a frozen copy of the policy and
    returns it, or returns None while the pool is empty (the scripted agent then plays
    every environment). The scripted agent's share of environments is set separately by
    Opponent.scripted_share.
    """

    def __init__(self, policy, every: int, capacity: int):
        self.every = every
        self.capacity = capacity
        self.weights: list[dict] = []
        self.frozen = copy.deepcopy(policy).eval().requires_grad_(False)

    def maybe_add(self, policy, update_index: int) -> bool:
        if update_index % self.every != 0:
            return False
        self.weights.append({name: value.detach().clone() for name, value in policy.state_dict().items()})
        del self.weights[: -self.capacity]
        return True

    def sample(self, generator):
        if not self.weights:
            return None
        index = int(torch.randint(0, len(self.weights), (1,), generator=generator, device=generator.device))
        self.frozen.load_state_dict(self.weights[index])
        return self.frozen
