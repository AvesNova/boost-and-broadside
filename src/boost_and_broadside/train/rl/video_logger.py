"""Async W&B video logging via a dedicated CPU episode thread.

VideoLogger runs a single-env episode in a background thread using a
headless pygame surface, captures frames, and logs them as wandb.Video.

The schedule() method is non-blocking — it drops the request silently
when the worker is already busy, so training is never stalled.
"""

from __future__ import annotations

import copy
import os
import threading
from queue import Queue, Full

import numpy as np
import torch

from boost_and_broadside.config import (
    ShipConfig, EnvConfig, ModelConfig, RewardConfig,
)
from boost_and_broadside.env.wrapper import MVPEnvWrapper
from boost_and_broadside.models.mvp.policy import MVPPolicy
from boost_and_broadside.ui.renderer import GameRenderer, RenderConfig


class VideoLogger:
    """Captures a greedy episode and logs it as a W&B video asynchronously.

    Args:
        ship_config:   Physics constants.
        env_config:    Environment sizing (determines episode length).
        reward_config: Reward weights (needed to build wrapper).
        model_config:  Policy architecture.
        fps:           Frame rate for the logged video.
    """

    def __init__(
        self,
        ship_config:   ShipConfig,
        env_config:    EnvConfig,
        reward_config: RewardConfig,
        model_config:  ModelConfig,
        fps:           int = 20,
    ) -> None:
        self._ship_config   = ship_config
        self._env_config    = env_config
        self._reward_config = reward_config
        self._model_config  = model_config
        self._fps           = fps
        # maxsize=1 — worker drops a request rather than queueing a backlog
        self._queue: Queue = Queue(maxsize=1)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def schedule(self, policy: MVPPolicy, global_step: int) -> None:
        """Schedule a video episode with the current policy weights.

        Non-blocking. If the worker is still processing the previous request,
        this call is silently dropped.

        Args:
            policy:      Main policy (CPU copy is made here).
            global_step: Current training step (used as W&B x-axis key).
        """
        state_dict = copy.deepcopy({k: v.cpu() for k, v in policy.state_dict().items()})
        try:
            self._queue.put_nowait({"state_dict": state_dict, "step": global_step})
        except Full:
            pass

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break
            try:
                frames = self._run_episode(item["state_dict"])
                if frames:
                    import wandb
                    arr = np.stack(frames)   # (T, H, W, 3)
                    wandb.log(
                        {"video/episode": wandb.Video(arr, fps=self._fps, format="mp4")},
                        step=item["step"],
                    )
            except Exception as e:
                print(f"[VideoLogger] error: {e}", flush=True)

    def _run_episode(self, state_dict: dict) -> list[np.ndarray]:
        """Run one greedy episode on CPU and return frames as (H, W, 3) arrays."""
        import pygame

        # Force headless SDL before any pygame call in this thread
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"

        wrapper = MVPEnvWrapper(
            num_envs=1,
            ship_config=self._ship_config,
            env_config=self._env_config,
            reward_config=self._reward_config,
            device="cpu",
        )
        policy = MVPPolicy(self._model_config, self._ship_config)
        policy.load_state_dict(state_dict)
        policy.eval()

        # Small window size for fast rendering
        render_cfg = RenderConfig(window_size=256, fps=0)
        renderer   = GameRenderer(self._ship_config, render_cfg)

        N      = wrapper.num_ships
        obs    = wrapper.reset()
        hidden = policy.initial_hidden(1, N, torch.device("cpu"))
        frames: list[np.ndarray] = []

        try:
            for _ in range(self._env_config.max_episode_steps):
                with torch.no_grad():
                    action, _, _, hidden = policy.get_action_and_value(obs, hidden)
                obs, _, dones, truncated, _ = wrapper.step(action)

                # Render frame (no display flip needed with dummy driver)
                renderer.render(wrapper.state)
                # pygame.surfarray.array3d returns (W, H, 3); transpose to (H, W, 3)
                frame = pygame.surfarray.array3d(renderer._screen).transpose(1, 0, 2)
                frames.append(frame.copy())

                if (dones | truncated).any():
                    break
        finally:
            renderer.close()

        return frames
