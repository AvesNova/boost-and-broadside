"""A video of one game: a policy against the scripted agent, drawn by the play shell
offscreen (`bnb.play.app.Session` on an EGL context, no window) and encoded to MP4 by
ffmpeg, which reads the raw frames on a pipe.

The policy plays team 0 and the scripted chaser team 1, in the mode of the preset asked for
(`bnb.play.sides.load_agent` with `any_mode`: a policy trained on a mixed batch plays every
mode of the mix). The match runs uncompiled and its agent on whatever device torch sees
(`bnb.play.sides.agent_device`; `bnb figures` hides the GPU from torch unless `--device`
names one). Each video frame advances the match by `time_scale` frames of display time,
so a video of 25 s at the default 4x shows the game's first 100 s. The frame is the
shell's own drawing, with its top bar and kill feed (the fleets card and the minimap
collapsed, the camera fitted to the arena).
"""

import dataclasses
import pathlib
import shutil
import subprocess
import time
from typing import TYPE_CHECKING

from bnb.game.api import PRESETS
from bnb.play.loop import TIME_SCALES, MatchSetup, SideSetup
from bnb.play.sides import Scripted, load_agent

FFMPEG = "ffmpeg"  # looked up on the PATH (`/usr/bin/ffmpeg` on the dev machine)
POLICY_TEAM = 0  # the policy plays team 0, the scripted chaser team 1
COLLAPSED_CARDS = ("fleets", "selection", "minimap")  # the cards a video leaves closed

if TYPE_CHECKING:
    from bnb.play.app import Session


@dataclasses.dataclass(frozen=True)
class VideoConfig:
    """seconds and frame_rate: the video's length and frames per second; time_scale: game
    seconds per video second (one of `bnb.play.loop.TIME_SCALES`); width and height:
    pixels; seed: the match's generator and the agent's; temperature: the agent's sampling
    temperature; interface: draw the shell's top bar and kill feed; quality: ffmpeg's
    constant rate factor for H.264 (lower is larger and sharper)."""

    seconds: float = 25.0
    frame_rate: int = 30
    time_scale: float = 4.0
    width: int = 1280
    height: int = 720
    seed: int = 0
    temperature: float = 1.0
    interface: bool = True
    quality: int = 28


def encoder(path: pathlib.Path, config: VideoConfig) -> subprocess.Popen:
    """An ffmpeg process that encodes raw RGB frames from its stdin to an H.264 MP4 at
    `path` (yuv420p, the moov atom first, so a browser starts it at once)."""
    found = shutil.which(FFMPEG)
    if found is None:
        raise SystemExit(f"bnb figures video needs ffmpeg on the PATH ({FFMPEG!r} not found)")
    command = [
        found,
        "-y",
        "-loglevel",
        "error",
        *("-f", "rawvideo", "-pix_fmt", "rgb24"),
        *("-s", f"{config.width}x{config.height}", "-r", str(config.frame_rate)),
        *("-i", "-"),
        *("-c:v", "libx264", "-preset", "veryfast", "-crf", str(config.quality)),
        *("-pix_fmt", "yuv420p", "-movflags", "+faststart"),
        str(path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def setup_of(policy: pathlib.Path, preset: str, config: VideoConfig) -> MatchSetup:
    """The match: the preset's game with fog, the policy's side and the scripted one."""
    sides = [SideSetup(), SideSetup()]
    sides[POLICY_TEAM] = SideSetup("agent", str(policy), config.temperature)
    return MatchSetup(PRESETS[preset], fogged=True, seed=config.seed, sides=tuple(sides))


class Tally:
    """What happened in the video's game time, by team: kills (the killer's team; a ship
    lost to the boundary counts as a kill for nobody), losses, zone captures and the ends
    of games (the winner, or -1 for a draw)."""

    def __init__(self, teams) -> None:
        self.teams = teams  # (N,) each sim index's team
        self.kills, self.losses, self.captures = [0, 0], [0, 0], [0, 0]
        self.results: list[int] = []

    def count(self, events) -> None:
        """Add one tick's `bnb.play.loop.MatchEvent`s."""
        for event in events:
            if event.kind == "kill":
                self.losses[int(self.teams[event.ship])] += 1
                if event.other >= 0:
                    self.kills[int(self.teams[event.other])] += 1
            elif event.kind == "capture":
                self.captures[event.team] += 1
            elif event.kind == "end":
                self.results.append(event.team)

    def summary(self) -> dict:
        """The tally with the policy's team first: `policy_*` and `scripted_*`, and the
        games ended in the video as wins, losses and draws of the policy."""
        policy, scripted = POLICY_TEAM, 1 - POLICY_TEAM
        return dict(
            policy_kills=self.kills[policy],
            scripted_kills=self.kills[scripted],
            policy_losses=self.losses[policy],
            scripted_losses=self.losses[scripted],
            policy_captures=self.captures[policy],
            scripted_captures=self.captures[scripted],
            policy_wins=self.results.count(policy),
            policy_defeats=self.results.count(scripted),
            draws=self.results.count(-1),
        )


def draw_frame(session: Session, target) -> None:
    """One video frame onto `target`: the shell's battlefield and interface. The play-mode
    overlays will be drawn here, onto the same target, before its pixels are read."""
    session.draw(target)


def render(
    policy: str | pathlib.Path, preset: str, config: VideoConfig, path: str | pathlib.Path
) -> dict:
    """Write the video of one game of `policy` against the scripted agent in the game
    `preset` (`bnb.game.api.PRESETS`) to `path` (MP4); returns its numbers: the tally
    (`Tally.summary`), the frames, the game seconds shown, the file's bytes and the
    seconds it took to make."""
    from bnb.play import app  # OpenGL and ImGui: only for a video

    if config.time_scale not in TIME_SCALES:
        raise SystemExit(f"--time-scale {config.time_scale}: one of {TIME_SCALES}")
    started = time.perf_counter()
    policy, path = pathlib.Path(policy), pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    setup = setup_of(policy, preset, config)
    options = app.Options(
        preset=preset,
        record=False,
        compile=False,
        width=config.width,
        height=config.height,
        seed=config.seed,
    )
    context = app.offscreen_context()
    size = (config.width, config.height)
    target = context.framebuffer(color_attachments=[context.texture(size, 4)])
    # The session opens on a scripted match (its options name no agent); the video's match,
    # with the policy loaded for any mode, replaces it.
    session = app.Session(options, context, *size)
    loop = session.loop
    agent = load_agent(
        policy,
        setup.config,
        loop.rules,
        loop.types,
        config.temperature,
        config.seed + POLICY_TEAM,
        any_mode=True,
    )
    sides = [Scripted(loop.rules.ships_per_team), Scripted(loop.rules.ships_per_team)]
    sides[POLICY_TEAM] = agent
    session.start(setup, whole=True, sides=sides)
    session.show_interface = config.interface
    session.interface.open -= set(COLLAPSED_CARDS)
    session.set_time_scale(config.time_scale)
    tally = Tally(session.loop.current.ships.team)
    shell_on_tick = session.loop.on_tick

    def on_tick(actions, realtime, events) -> None:
        shell_on_tick(actions, realtime, events)
        tally.count(events)

    session.loop.on_tick = on_tick
    frames = round(config.seconds * config.frame_rate)
    process = encoder(path, config)
    try:
        for _ in range(frames):
            session.update(1.0 / config.frame_rate)
            draw_frame(session, target)
            process.stdin.write(app.read_pixels(target).tobytes())
    finally:
        process.stdin.close()
        process.wait()
        session.close()
        context.release()
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (exit {process.returncode}) writing {path}")
    return dict(
        **tally.summary(),
        frames=frames,
        game_seconds=round(session.loop.ticks * session.loop.tick_length, 2),
        bytes=path.stat().st_size,
        seconds_taken=round(time.perf_counter() - started, 1),
    )


def video_name(preset: str, config: VideoConfig) -> str:
    """The video's file name: the preset and the seed."""
    return f"{preset}_seed{config.seed}.mp4"
