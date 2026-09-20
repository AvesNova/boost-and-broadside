"""Fixed-storage CUDA graph replay for an interactive ``TensorEnv`` tick.

The authoritative physics functions return a new :class:`TensorState` by
rebinding tensor fields. CUDA graph replay cannot reproduce those Python
rebindings, so capture ends by copying every result back into the original
storages. Replays then advance the same stable storages in place.

This boundary is deliberately opt-in and interactive-only. Training keeps the
ordinary eager tick, and callers must fall back to eager execution for modes
whose Python arguments change the captured program (currently unlimited
resources).
"""

from __future__ import annotations

from dataclasses import fields

import torch

from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.state import TensorState


class CapturedTick:
    """Replay one fixed-shape CUDA ``TensorEnv.tick`` without Python dispatch."""

    def __init__(self, env: TensorEnv, action: torch.Tensor) -> None:
        if env.device.type != "cuda":
            raise ValueError("captured tick requires a CUDA TensorEnv")
        if env.state is None:
            raise ValueError("captured tick requires an already-reset TensorEnv")
        expected = tuple(env.state.prev_action.shape)
        if tuple(action.shape) != expected:
            raise ValueError(
                f"captured tick requires action shape {expected}, got {tuple(action.shape)}"
            )
        if action.device != env.state.prev_action.device:
            raise ValueError("captured tick action must be on the environment device")

        self.env = env
        self.static_action = action.clone()
        self.input_storages = {
            field.name: getattr(env.state, field.name) for field in fields(TensorState)
        }
        initial_values = {name: storage.clone() for name, storage in self.input_storages.items()}
        # Graph construction is an implementation detail, not a simulation
        # step. Preserve the generator position consumed by bullet spread and
        # reset randomization so enabling this wrapper does not change a run.
        initial_rng = torch.cuda.get_rng_state(env.device)

        self.graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize(env.device)
        capture_stream = torch.cuda.Stream(device=env.device)
        capture_stream.wait_stream(torch.cuda.current_stream(env.device))
        with torch.cuda.stream(capture_stream):
            self.env.tick(self.static_action)
        torch.cuda.current_stream(env.device).wait_stream(capture_stream)
        for name, storage in self.input_storages.items():
            storage.copy_(initial_values[name])
        self._restore_input_state()
        torch.cuda.set_rng_state(initial_rng, env.device)
        torch.cuda.synchronize(env.device)

        with torch.cuda.graph(self.graph):
            self.done, self.truncated = self.env.tick(self.static_action)
            for field in fields(TensorState):
                self.input_storages[field.name].copy_(getattr(self.env.state, field.name))

        # Capture itself executes the program once. Restore both state and RNG
        # so the first replay is semantically the first tick.
        for name, storage in self.input_storages.items():
            storage.copy_(initial_values[name])
        self._restore_input_state()
        torch.cuda.set_rng_state(initial_rng, env.device)
        torch.cuda.synchronize(env.device)

    def _restore_input_state(self) -> None:
        self.env.state = TensorState(**self.input_storages)

    def replay(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance one tick with a same-shape action."""

        if tuple(action.shape) != tuple(self.static_action.shape):
            raise ValueError("action shape changed after CUDA graph capture")
        self.static_action.copy_(action)
        self.graph.replay()
        self._restore_input_state()
        return self.done, self.truncated

    def load_state(self, state: TensorState) -> None:
        """Load a reset or eager-advanced state into the captured storages."""

        for field in fields(TensorState):
            source = getattr(state, field.name)
            target = self.input_storages[field.name]
            if (
                source.shape != target.shape
                or source.dtype != target.dtype
                or source.device != target.device
            ):
                raise ValueError(f"state changed captured field metadata: {field.name}")
            target.copy_(source)
        self._restore_input_state()
