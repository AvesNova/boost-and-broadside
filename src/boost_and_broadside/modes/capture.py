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
from dataclasses import replace
from pathlib import Path

import pygame
import torch

from boost_and_broadside.config import EnvConfig, ModelConfig, ShipConfig
from boost_and_broadside.evaluation.agents import ResolvedAgent, resolve_agent_spec
from boost_and_broadside.evaluation.environment import create_evaluation_env
from boost_and_broadside.evaluation.match import MatchRunner
from boost_and_broadside.evaluation.run_catalog import (
    resolve_exact_run,
    select_final_training_checkpoint,
)
from boost_and_broadside.evaluation.sizes import Matchup, parse_matchup
from boost_and_broadside.train.rl.checkpoint_schema import require_observation_schema
from boost_and_broadside.ui.renderer import GameRenderer, RenderConfig

SCENARIOS = ("self", "vs_scripted")


def parse_seeds(spec: str) -> list[int]:
    """Parse a seed spec: a range like '0-7' or a list like '0,3,9'."""
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",") if s]


def _final_checkpoint(run_dir: Path) -> Path:
    return select_final_training_checkpoint(run_dir).path


def _open_encoder(out: Path, size: int, fps: int) -> subprocess.Popen:
    """Start an ffmpeg process reading raw rgb24 frames on stdin."""
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{size}x{size}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "20",
        "-preset",
        "medium",
        str(out),
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
    renderer: GameRenderer,
    device: torch.device,
    out: Path,
    max_steps: int,
    fps: int,
    hold_ms: int,
) -> int:
    """Play one seeded match, encode it to ``out``; return the frame count.

    team-0 (n0 ships) is always the policy; team-1 (n1 ships) is a second policy
    view of the same weights (self) or the scripted agent (vs_scripted).

    Uses TensorEnv directly (not the training wrapper, which auto-resets a finished
    env): the terminal state persists, so after the outcome is decided the match
    keeps stepping for ``hold_ms`` — the winning team cruises, the eliminated team
    stays gone — letting a viewer see who won before the clip ends.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    N = env_config.num_ships
    env = create_evaluation_env(1, ship_config, env_config, device)
    agent0 = policy
    agent1 = (
        ResolvedAgent("policy", policy.agent, bundle=policy.bundle)
        if scenario == "self"
        else scripted
    )
    # Finished environments are deliberately never reset: the match holds its
    # terminal state on screen so a viewer can see who won.
    runner = MatchRunner(
        env,
        [agent0, agent1],
        team0_index=torch.zeros(1, dtype=torch.long, device=device),
        team1_index=torch.ones(1, dtype=torch.long, device=device),
        ship_config=ship_config,
        num_ships=N,
    )

    env.reset(options={"team_sizes": (n0, n1)}, seed=seed)
    runner.init_hidden()

    hold_frames = round(hold_ms / 1000 * fps)
    encoder = _open_encoder(out, renderer._render_config.window_size, fps)
    frames = 0
    ended_at: int | None = None
    winner = "tie"
    try:
        for step in range(max_steps + hold_frames):
            dones, truncated = runner.step()

            renderer._draw_frame(env.state)  # no UI, no ghost trajectories
            encoder.stdin.write(pygame.image.tostring(renderer._screen, "RGB"))
            frames += 1

            # Once decided, record the winner from the terminal survivors, then
            # keep playing for the hold before stopping.
            if ended_at is None and bool((dones | truncated).any()):
                ended_at = step
                winner = _winner(env.state)
            if ended_at is not None and step - ended_at >= hold_frames:
                break
    finally:
        encoder.stdin.close()
        err = encoder.stderr.read().decode(errors="replace")
        if encoder.wait() != 0:
            raise RuntimeError(f"ffmpeg failed for {out}:\n{err}")
    return frames, winner


def _winner(state) -> str:
    """Which side survives at a terminal state: 'team0', 'team1', or 'tie'."""
    alive, team = state.ship_alive, state.ship_team_id
    team0 = bool((alive & (team == 0)).any())
    team1 = bool((alive & (team == 1)).any())
    if team0 and not team1:
        return "team0"
    if team1 and not team0:
        return "team1"
    return "tie"


def _write_gif(mp4: Path, fps: int, width: int) -> Path:
    """Convert an mp4 to a palette-optimised gif beside it (downscaled for size)."""
    gif = mp4.with_suffix(".gif")
    vf = (
        f"fps={fps},scale={width}:-1:flags=lanczos,"
        "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse"
    )
    result = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-vf", vf, str(gif)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg gif failed for {mp4}:\n{result.stderr}")
    return gif


def run_capture_mode(
    run_spec: str,
    scenarios: list[str],
    seeds: str,
    ship_config: ShipConfig,
    model_config: ModelConfig,
    device: str,
    checkpoint_dir: str = "checkpoints",
    out_dir: Path = Path("gameplay_clips"),
    sizes: list[str] | None = None,
    fps: int = 60,
    max_steps: int = 1024,
    window: int = 720,
    hold_ms: int = 1500,
    gif: bool = False,
    gif_fps: int = 24,
    gif_width: int = 480,
) -> list[Path]:
    """Capture one seeded mp4 per (scenario, size, seed) for a run's final checkpoint.

    ``sizes`` is a list of team-size specs ('1v1', '4v4', ..., or bare per-side
    counts); each is played zero-shot by the same weights. When omitted, the run's
    native training size is used. ``max_steps`` is the match length before it is
    called a tie (also the env's truncation point); most matches end far sooner by
    elimination. With ``gif`` a downscaled gif is written beside each mp4.
    """
    os.environ.setdefault("HEADLESS", "1")  # dummy SDL driver — set before pygame init

    run_dir = resolve_exact_run(run_spec, checkpoint_dir).path
    checkpoint = _final_checkpoint(run_dir)
    checkpoint_data = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    require_observation_schema(checkpoint_data, str(checkpoint))
    base_env = EnvConfig(**checkpoint_data["env_config"])
    native = base_env.num_ships // 2
    matchups = [parse_matchup(s) for s in sizes] if sizes else [Matchup(native, native)]
    torch_device = torch.device(device)
    out_dir.mkdir(parents=True, exist_ok=True)

    for scenario in scenarios:
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}; choose from {SCENARIOS}")

    renderer = GameRenderer(ship_config, RenderConfig(window_size=window, show_ui=False))
    scripted = resolve_agent_spec("scripted", ship_config, model_config, device)
    seed_list = parse_seeds(seeds)

    written: list[Path] = []
    try:
        for n0, n1 in matchups:
            # Override the env's truncation so matches get max_steps to resolve
            # before being called a tie.
            env_config = replace(base_env, num_ships=n0 + n1, max_episode_steps=max_steps)
            # A fresh policy per size so its token slicing matches the env; the same
            # weights load at any size (nothing in the model is sized by ship count).
            policy = resolve_agent_spec(
                str(checkpoint), ship_config, model_config, device, num_ships=env_config.num_ships
            )
            for scenario in scenarios:
                for seed in seed_list:
                    out = out_dir / f"{scenario}_{n0}v{n1}_seed{seed:02d}.mp4"
                    frames, winner = _capture_match(
                        scenario,
                        seed,
                        n0,
                        n1,
                        policy,
                        scripted,
                        ship_config,
                        env_config,
                        renderer,
                        torch_device,
                        out,
                        max_steps,
                        fps,
                        hold_ms,
                    )
                    written.append(out)
                    if gif:
                        written.append(_write_gif(out, gif_fps, gif_width))
                    # team0 is always blue (the trained policy); name the winner plainly.
                    side = {"team0": "blue", "team1": "red", "tie": "tie"}[winner]
                    print(f"wrote {out}  ({frames} frames, {frames / fps:.1f}s)  winner: {side}")
    finally:
        renderer.close()
    return written
