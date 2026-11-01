"""A captured rollout step: `step` compiled, its carry kept in static buffers and updated in
place, the whole decision replayed as one CUDA graph.

    captured = CapturedStep(model, views, obs_template)
    outputs = captured(obs, reset)   # copies obs and reset in, replays, returns static outputs

The carry lives in `captured.carry` (static tensors, the runtime may `copy_` into them).
Outputs are static too: read or copy them before the next call. Sampling draws fresh
Gumbel noise on every replay (the CUDA generator's offset advances per replay).
"""

import contextlib

import torch

from bnb.contracts.obs import Obs


def obs_leaves(obs: Obs):
    return obs.ships, obs.fields, obs.globals, obs.zones


def static_like(obs: Obs) -> Obs:
    return Obs(*(None if leaf is None else torch.zeros_like(leaf) for leaf in obs_leaves(obs)))


class CapturedStep:
    def __init__(self, model, views: int, obs: Obs, precision: str = "float32",
                 compile: bool = True, graph: bool = True, warmup: int = 3):
        """obs: one decision's observation (R, ...), used for the shapes and the warm-up."""
        self.model = model
        ships = obs.ships.shape[-2]
        device = obs.ships.device
        self.carry = model.initial_carry(views, ships, device=device)
        self.obs = static_like(obs)
        self.reset = torch.zeros(views, dtype=torch.bool, device=device)
        self.precision = precision
        function = self.update
        self.function = torch.compile(function, fullgraph=True) if compile else function
        for leaf, value in zip(obs_leaves(self.obs), obs_leaves(obs), strict=True):
            if leaf is not None:
                leaf.copy_(value)
        self.reset.fill_(True)
        self.graph = None
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                self.outputs = self.function(self.carry, self.obs, self.reset)
        torch.cuda.current_stream().wait_stream(stream)
        if graph:
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.outputs = self.function(self.carry, self.obs, self.reset)
        self.reset.fill_(False)

    def autocast(self):
        if self.precision == "bfloat16":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    @torch.no_grad()
    def update(self, carry, obs, reset):
        """step, then write the new carry into the static one (fused into its producers)."""
        with self.autocast():
            new, outputs = self.model.step(carry, obs, reset)
        for static, value in zip(carry, new, strict=True):
            static.copy_(value)
        return {key: value for key, value in outputs.items() if key != "replay"} | {
            "replay": outputs["replay"]}

    def replay(self):
        """Run one decision on the static obs and reset already in place."""
        if self.graph is None:
            self.outputs = self.function(self.carry, self.obs, self.reset)
        else:
            self.graph.replay()
        return self.outputs

    def __call__(self, obs: Obs, reset: torch.Tensor):
        for leaf, value in zip(obs_leaves(self.obs), obs_leaves(obs), strict=True):
            if leaf is not None:
                leaf.copy_(value)
        self.reset.copy_(reset)
        return self.replay()
