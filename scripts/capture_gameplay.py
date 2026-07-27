"""Headless gameplay video capture for a landmark run.

Runs single-environment matches with no display (SDL dummy driver), renders each
frame to an offscreen surface, and pipes raw RGB to ffmpeg to write an mp4. Every
clip is seeded, so a good one can be regenerated or extended later.

Two matchups, both with the run's final checkpoint:
  self       — the final policy against itself (self-play). The team-1 side sees a
               team-flipped observation, because the ego_pass policy only ever
               learned to act as team 0; without the flip its team-1 play is off.
  vs_scripted — the final policy (team 0) against the stochastic scripted agent.

Usage:
    uv run --no-sync scripts/capture_gameplay.py                 # 682, seeds 0-7, both
    uv run --no-sync scripts/capture_gameplay.py --seeds 0-15 --matchups self
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Scripts run with only the installed package on the path; add the repo root so
# the run-config constants in runs/ import the same as they do for main.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("HEADLESS", "1")  # SDL dummy driver — must precede pygame init

import pygame  # noqa: E402
import torch  # noqa: E402

from boost_and_broadside.config import EnvConfig  # noqa: E402
from boost_and_broadside.env.wrapper import MVPEnvWrapper  # noqa: E402
from boost_and_broadside.modes.agent_factory import (  # noqa: E402
    ResolvedAgent,
    get_actions,
    init_hidden,
    resolve_agent_spec,
)
from boost_and_broadside.ui.renderer import GameRenderer, RenderConfig  # noqa: E402
from runs.shared import MODEL_CONFIG, REWARDS, SHIP_CONFIG  # noqa: E402

DEFAULT_RUN = "resilient-resonance-682"
FINAL_CHECKPOINT = "step_000999424000.pt"


def _final_policy(run_dir: Path, env_config: EnvConfig, device: str) -> ResolvedAgent:
    path = run_dir / FINAL_CHECKPOINT
    if not path.exists():
        candidates = sorted(run_dir.glob("step_*.pt"))
        if not candidates:
            sys.exit(f"no step_*.pt checkpoint in {run_dir}")
        path = candidates[-1]
    return resolve_agent_spec(
        str(path), SHIP_CONFIG, MODEL_CONFIG, device, num_ships=env_config.num_ships
    )


def _open_encoder(out: Path, size: int, fps: int) -> subprocess.Popen:
    """Start an ffmpeg process reading raw rgb24 frames on stdin."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size}x{size}", "-r", str(fps),
        "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        "-preset", "medium", str(out),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def capture_match(
    kind: str,
    seed: int,
    policy: ResolvedAgent,
    scripted: ResolvedAgent,
    ship_config,
    env_config: EnvConfig,
    renderer: GameRenderer,
    device: torch.device,
    out: Path,
    max_steps: int,
    fps: int,
) -> int:
    """Play one seeded match, encode it to ``out``; return the frame count.

    ``kind`` is "self" (policy vs a team-flipped copy of itself) or "vs_scripted"
    (policy on team 0 vs the scripted agent on team 1).
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    N = env_config.num_ships
    num_tokens = N + env_config.num_obstacles
    wrapper = MVPEnvWrapper(
        num_envs=1, ship_config=ship_config, env_config=env_config,
        rewards=REWARDS, device=str(device),
    )

    # team-0 side is always the policy; team-1 side is a second policy view (self)
    # or the scripted agent (vs_scripted).
    agent0 = policy
    agent1 = ResolvedAgent("policy", policy.agent) if kind == "self" else scripted

    obs = wrapper.reset(options={"team_sizes": (N // 2, N - N // 2)}, seed=seed)
    init_hidden(agent0, 1, num_tokens, device)
    init_hidden(agent1, 1, num_tokens, device)

    encoder = _open_encoder(out, renderer._render_config.window_size, fps)
    frames = 0
    try:
        for _ in range(max_steps):
            state = wrapper.state
            action0 = get_actions(agent0, obs, state, 1, N, device)
            if kind == "self":
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


def _parse_seeds(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",") if s]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=DEFAULT_RUN, help="landmark run under checkpoints/")
    parser.add_argument("--out", type=Path, default=Path("gameplay_clips"), help="output dir")
    parser.add_argument("--seeds", default="0-7", help="e.g. '0-7' or '0,3,9'")
    parser.add_argument(
        "--matchups", default="self,vs_scripted", help="comma list: self, vs_scripted"
    )
    parser.add_argument("--max-steps", type=int, default=1024, help="frame cap per clip")
    parser.add_argument("--fps", type=int, default=60, help="playback frame rate")
    parser.add_argument("--window", type=int, default=720, help="video side length in px")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path("checkpoints") / args.run
    env_config = EnvConfig(num_ships=8, max_bullets=20, max_episode_steps=1024, num_obstacles=0)
    args.out.mkdir(parents=True, exist_ok=True)

    render_config = RenderConfig(window_size=args.window, show_ui=False)
    renderer = GameRenderer(SHIP_CONFIG, render_config)

    policy = _final_policy(run_dir, env_config, str(device))
    scripted = resolve_agent_spec("scripted", SHIP_CONFIG, MODEL_CONFIG, str(device))
    seeds = _parse_seeds(args.seeds)
    matchups = [m.strip() for m in args.matchups.split(",") if m.strip()]

    try:
        for kind in matchups:
            for seed in seeds:
                out = args.out / f"{kind}_seed{seed:02d}.mp4"
                frames = capture_match(
                    kind, seed, policy, scripted, SHIP_CONFIG, env_config, renderer,
                    device, out, args.max_steps, args.fps,
                )
                print(f"wrote {out}  ({frames} frames, {frames / args.fps:.1f}s)")
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
