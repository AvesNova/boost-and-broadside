"""capture mode: headless gameplay video for a finished run.

Runs single-environment matches with no display (SDL dummy driver), renders each
frame to an offscreen surface, and pipes raw RGB to ffmpeg to write an mp4. Every
clip is seeded, so a good one can be regenerated or extended later.

Scenarios, all with the run's final checkpoint:
    self         — the final policy against itself. The team-1 side sees a
                   team-flipped observation, because the ego_pass policy only ever
                   learned to act as team 0; without the flip its team-1 play is off.
    vs_scripted  — the final policy (team 0) against the stochastic scripted agent.

The same weights play any team size (1v1 … 64v64) zero-shot, since the model is
token-based and nothing in it is sized by the ship count.

Writes ``<out>/<scenario>_<AvA>_seed<NN>.mp4``, one clip per (scenario, size, seed).
"""

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pygame
import torch

from boost_and_broadside.config import EnvConfig, ModelConfig, RewardConfig, ShipConfig
from boost_and_broadside.env.wrapper import MVPEnvWrapper
from boost_and_broadside.modes.agent_factory import (
    ResolvedAgent,
    get_actions,
    init_hidden,
    resolve_agent_spec,
)
from boost_and_broadside.ui.renderer import GameRenderer, RenderConfig

SCENARIOS = ("self", "vs_scripted")


def parse_seeds(spec: str) -> list[int]:
    """Parse a seed spec: a range like '0-7' or a list like '0,3,9'."""
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",") if s]


def parse_matchup(spec: str) -> tuple[int, int]:
    """Team-0 and team-1 ship counts from a size spec.

    '4v4' -> (4, 4); the asymmetric '8v12' -> (8, 12); a bare '4' -> (4, 4). The
    model is token-based and scale-invariant, so the same weights play any size or
    imbalance zero-shot; the spec only sets how many ship tokens each side spawns.
    """
    if "v" in spec:
        a, b = spec.split("v", 1)
        return int(a), int(b)
    n = int(spec)
    return n, n


def _find_run_dir(run_spec: str, checkpoint_dir: str) -> Path:
    root = Path(checkpoint_dir)
    if run_spec in ("latest", "none"):
        runs = [p for p in root.iterdir() if p.is_dir() and any(p.glob("step_*.pt"))]
        if not runs:
            sys.exit(f"no run with step_*.pt under {root}")
        return max(runs, key=lambda p: max(f.stat().st_mtime for f in p.glob("step_*.pt")))
    run_dir = root / run_spec
    if not run_dir.exists():
        sys.exit(f"run directory not found: {run_dir}")
    return run_dir


def _final_checkpoint(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob("step_*.pt"))
    if not candidates:
        sys.exit(f"no step_*.pt checkpoint in {run_dir}")
    return candidates[-1]


def _open_encoder(out: Path, size: int, fps: int) -> subprocess.Popen:
    """Start an ffmpeg process reading raw rgb24 frames on stdin."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size}x{size}", "-r", str(fps),
        "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        "-preset", "medium", str(out),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def _capture_match(
    scenario: str,
    seed: int,
    n0: int,
    n1: int,
    policy: ResolvedAgent,
    scripted: ResolvedAgent,
    ship_config: ShipConfig,
    env_config: EnvConfig,
    rewards: RewardConfig,
    renderer: GameRenderer,
    device: torch.device,
    out: Path,
    max_steps: int,
    fps: int,
) -> int:
    """Play one seeded match, encode it to ``out``; return the frame count.

    team-0 (n0 ships) is always the policy; team-1 (n1 ships) is a second policy
    view of the same weights (self) or the scripted agent (vs_scripted).
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    N = env_config.num_ships
    num_tokens = N + env_config.num_obstacles
    wrapper = MVPEnvWrapper(
        num_envs=1, ship_config=ship_config, env_config=env_config,
        rewards=rewards, device=str(device),
    )
    agent0 = policy
    agent1 = ResolvedAgent("policy", policy.agent) if scenario == "self" else scripted

    obs = wrapper.reset(options={"team_sizes": (n0, n1)}, seed=seed)
    init_hidden(agent0, 1, num_tokens, device)
    init_hidden(agent1, 1, num_tokens, device)

    encoder = _open_encoder(out, renderer._render_config.window_size, fps)
    frames = 0
    try:
        for _ in range(max_steps):
            state = wrapper.state
            action0 = get_actions(agent0, obs, state, 1, N, device)
            if scenario == "self":
                action1 = get_actions(agent1, obs.flip_team(N), state, 1, N, device)
            else:
                action1 = get_actions(agent1, None, state, 1, N, device)

            team_id = obs.team_id[:, :N]
            action = torch.where((team_id == 0).unsqueeze(-1), action0, action1)
            obs, _, dones, truncated, _ = wrapper.step(action)

            renderer._draw_frame(wrapper.state)  # no UI, no ghost trajectories
            encoder.stdin.write(pygame.image.tostring(renderer._screen, "RGB"))
            frames += 1
            if bool((dones | truncated).any()):
                break
    finally:
        encoder.stdin.close()
        err = encoder.stderr.read().decode(errors="replace")
        if encoder.wait() != 0:
            sys.exit(f"ffmpeg failed for {out}:\n{err}")
    return frames


def run_capture_mode(
    run_spec: str,
    scenarios: list[str],
    seeds: str,
    ship_config: ShipConfig,
    model_config: ModelConfig,
    rewards: RewardConfig,
    device: str,
    checkpoint_dir: str = "checkpoints",
    out_dir: Path = Path("gameplay_clips"),
    sizes: list[str] | None = None,
    fps: int = 60,
    max_steps: int = 1024,
    window: int = 720,
) -> list[Path]:
    """Capture one seeded mp4 per (scenario, size, seed) for a run's final checkpoint.

    ``sizes`` is a list of team-size specs ('1v1', '4v4', ..., or bare per-side
    counts); each is played zero-shot by the same weights. When omitted, the run's
    native training size is used.
    """
    os.environ.setdefault("HEADLESS", "1")  # dummy SDL driver — set before pygame init

    run_dir = _find_run_dir(run_spec, checkpoint_dir)
    checkpoint = _final_checkpoint(run_dir)
    base_env = EnvConfig(**torch.load(str(checkpoint), map_location="cpu", weights_only=False)[
        "env_config"
    ])
    native = base_env.num_ships // 2
    matchups = [parse_matchup(s) for s in sizes] if sizes else [(native, native)]
    torch_device = torch.device(device)
    out_dir.mkdir(parents=True, exist_ok=True)

    for scenario in scenarios:
        if scenario not in SCENARIOS:
            sys.exit(f"unknown scenario {scenario!r}; choose from {SCENARIOS}")

    renderer = GameRenderer(ship_config, RenderConfig(window_size=window, show_ui=False))
    scripted = resolve_agent_spec("scripted", ship_config, model_config, device)
    seed_list = parse_seeds(seeds)

    written: list[Path] = []
    try:
        for n0, n1 in matchups:
            env_config = replace(base_env, num_ships=n0 + n1)
            # A fresh policy per size so its token slicing matches the env; the same
            # weights load at any size (nothing in the model is sized by ship count).
            policy = resolve_agent_spec(
                str(checkpoint), ship_config, model_config, device, num_ships=env_config.num_ships
            )
            for scenario in scenarios:
                for seed in seed_list:
                    out = out_dir / f"{scenario}_{n0}v{n1}_seed{seed:02d}.mp4"
                    frames = _capture_match(
                        scenario, seed, n0, n1, policy, scripted, ship_config, env_config,
                        rewards, renderer, torch_device, out, max_steps, fps,
                    )
                    written.append(out)
                    print(f"wrote {out}  ({frames} frames, {frames / fps:.1f}s)")
    finally:
        renderer.close()
    return written
